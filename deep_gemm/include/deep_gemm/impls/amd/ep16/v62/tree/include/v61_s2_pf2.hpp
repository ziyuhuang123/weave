#pragma once
#include <type_traits>
// V61 Opus 5.5 High -- GEMM2 1x4 BK128 body with a distance-2 software pipeline ("pf2").
//
// Same math, tile shape (BM x 128 per group, each of the 4 waves owns BM rows x 32 cols), MFMA opcode
// (v_mfma_f32_16x16x32_fp8_fp8), LDS layout (2 A stages of BM*128 B, sa stages at kSaBaseOff, tok at kTokOff),
// barriers (one 4-wave LDS barrier per K128 step) and epilogue as weave_s2_1x4_bk128_zerob.
// What changes is only *when* the global loads of a K step are issued and *which registers* they land in:
//
//   zerob (V60):  step kb issues A/B/sa/sb of step kb+1 at its start and consumes them at its END
//                 (ds_write A_{kb+1}, ds_write sa_{kb+1}, copy B_next -> B_cur) -> the step waits for the loads it
//                 issued itself (ISA: s_waitcnt vmcnt(0) at the loop tail) -> exposed latency ~ L - T_mfma(step).
//   pf2 (V61):    step kb issues the loads of step kb+2 at its start into register set (kb+2)%3,
//                 writes A_{kb+1}/sa_{kb+1} (issued one step earlier) to LDS at its end and runs its MFMAs on
//                 B set kb%3 (issued two steps earlier) -> every wait only covers loads >= 1 full step old.
//                 Loads are made uniform (clamped step index, invalid rows read row 0 then zeroed) so the
//                 compiler can count vmcnt exactly instead of draining.
// Cost: 3 register sets of B (3 x 16 VGPR), A (3 x 4*kAPasses VGPR), sa/sb (3 x 2), vs 2 / 1 / 1 sets in zerob.
template <unsigned kBM>
__device__ __forceinline__ void weave_s2_1x4_pf2(
    const int32_t* __restrict__ s2_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    const float* __restrict__ weights_tile,
    uint16_t* __restrict__ c_out_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    static_assert(kBM == 64u, "pf2 is instantiated for BM=64 only");
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kMXDL = kBM / 16u;
    constexpr unsigned kNXDL = 2u;
    constexpr unsigned kAPasses = kBM / 32u;
    constexpr unsigned kAStageBytes = kBM * 128u;
    constexpr unsigned kSaBaseOff = 16384u;
    constexpr unsigned kTokOff = 17408u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu) : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned nk = K >> 7u;                 // number of K128 steps (>= 1)
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * N * K;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * n_blocks + tile_n) * nk;

    const unsigned a_r32 = tid >> 3u;
    const unsigned a_c16 = tid & 7u;
    uint32_t a_roff[kAPasses];
    bool a_ok[kAPasses];
    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 32u + a_r32;
        const uint32_t tok = lds_tok[r];
        a_ok[ap] = (r < valid_rows) && (tok < route_rows);
        a_roff[ap] = a_ok[ap] ? (tok * K + a_c16 * 16u) : (a_c16 * 16u);       // invalid rows read row 0, zeroed
    }
    const bool sa_active = (tid < kBM);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : route_rows;
    const bool sa_ok = (tid < valid_rows) && sa_active && (tok_sa < route_rows);
    const uint32_t sa_roff = sa_ok ? (tok_sa * nk) : 0u;

    WeaveFloat4 acc[kMXDL][kNXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) acc[mx][nx] = {0,0,0,0};
    }

    const uint64_t b_col_off0 = (static_cast<uint64_t>(tile_n * 8u + wave_idx * 2u + 0u) * num_k64 * 64ull + lane) * 16ull;
    const uint64_t b_col_off1 = (static_cast<uint64_t>(tile_n * 8u + wave_idx * 2u + 1u) * num_k64 * 64ull + lane) * 16ull;

    // ---- register sets, rotated with (step % 3) ----
    uint4 ar[3][kAPasses];
    float sar[3], sbr[3];
    WeaveI64x2 b00[3], b01[3], b10[3], b11[3];   // bw0[0], bw0[1], bw1[0], bw1[1] of zerob

    auto issue = [&](auto Sc, unsigned step) {   // all global loads of K step `step` into set S (uniform)
        constexpr unsigned S = decltype(Sc)::value;
        const unsigned s = (step < nk) ? step : (nk - 1u);
        const uint32_t ak_off = s * 128u;
        const uint64_t bk_off = static_cast<uint64_t>(s) * 2048ull;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) ar[S][ap] = weave_gload_u4(a_base + a_roff[ap] + ak_off);
        sar[S] = weave_as1_arr(a_scale_base)[sa_roff + s];
        b00[S] = weave_gload_i64x2(b_exp + b_col_off0 + bk_off);
        b01[S] = weave_gload_i64x2(b_exp + b_col_off1 + bk_off);
        b10[S] = weave_gload_i64x2(b_exp + b_col_off0 + bk_off + 1024ull);
        b11[S] = weave_gload_i64x2(b_exp + b_col_off1 + bk_off + 1024ull);
        sbr[S] = weave_as1_arr(bs_row)[s];
    };
    auto stage_a = [&](auto Sc, unsigned step) { // A/sa of K step `step` (register set S) -> LDS stage step&1
        constexpr unsigned S = decltype(Sc)::value;
        WeaveLdsU8* dst = lds + (step & 1u) * kAStageBytes;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            const unsigned r = ap * 32u + a_r32;
            const uint4 v = a_ok[ap] ? ar[S][ap] : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<WeaveLdsU4*>(dst + r * 128u + (a_c16 ^ (r & 7u)) * 16u) = v;
        }
        if (sa_active) {
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + (step & 1u) * 512u)[tid] = sa_ok ? sar[S] : 0.0f;
        }
    };
    auto mfma_step = [&](auto Sc, unsigned step) {
        constexpr unsigned S = decltype(Sc)::value;
        const WeaveLdsU8* lds_a_cur = lds + (step & 1u) * kAStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + (step & 1u) * 512u);
        const float sb_cur = sbr[S];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned rm = mx * 16u + m_lane;
            const WeaveI64x2 av0 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
            const WeaveI64x2 av1 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
            const float sab = lds_sa_cur[rm] * sb_cur;
            WeaveFloat4 t0 = {0,0,0,0}, t1 = {0,0,0,0};
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b00[S][0], av0[0], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b01[S][0], av0[0], t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b00[S][1], av0[1], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b01[S][1], av0[1], t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b10[S][0], av1[0], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b11[S][0], av1[0], t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b10[S][1], av1[1], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b11[S][1], av1[1], t1, 0, 0, 0);
            acc[mx][0] += t0 * sab;
            acc[mx][1] += t1 * sab;
        }
    };
    using C0 = std::integral_constant<unsigned, 0u>;
    using C1 = std::integral_constant<unsigned, 1u>;
    using C2 = std::integral_constant<unsigned, 2u>;

    // prologue: step 0 -> set 0 (A/sa staged to LDS stage 0), step 1 -> set 1 (in flight)
    issue(C0{}, 0u);
    issue(C1{}, 1u);
    stage_a(C0{}, 0u);
    weave_4wave_lds_barrier_65440(arena);

    // steady state, step kb with S = kb % 3:
    //   issue(kb+2 -> set (S+2)%3)   [that set held step kb-1, fully consumed]
    //   MFMA(step kb: LDS stage kb&1, B set S)
    //   stage_a(step kb+1 from set (S+1)%3 -> LDS stage (kb+1)&1)   [stage held step kb-1, all waves past its barrier]
    //   barrier
    // Steps >= nk are clamped loads / stages of step nk-1 into the stage that is not being read: harmless.
    #pragma clang loop unroll(disable)
    for (unsigned kb = 0u;; kb += 3u) {
        issue(C2{}, kb + 2u);
        mfma_step(C0{}, kb);
        stage_a(C1{}, kb + 1u);
        weave_4wave_lds_barrier_65440(arena);
        if (kb + 1u >= nk) break;
        issue(C0{}, kb + 3u);
        mfma_step(C1{}, kb + 1u);
        stage_a(C2{}, kb + 2u);
        weave_4wave_lds_barrier_65440(arena);
        if (kb + 2u >= nk) break;
        issue(C1{}, kb + 4u);
        mfma_step(C2{}, kb + 2u);
        stage_a(C0{}, kb + 3u);
        weave_4wave_lds_barrier_65440(arena);
        if (kb + 3u >= nk) break;
    }
    // drain the clamped tail loads before the epilogue reuses LDS / registers
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    weave_4wave_lds_barrier_65440(arena);
    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + m_lane;
        const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) {
            const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][1] * weight)) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][2] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][3] * weight)) << 16u);
            const unsigned c8 = (col >> 2u) ^ (r & 7u);
            *reinterpret_cast<WeaveLdsU2*>(lds_c + r * 128u + c8 * 4u) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned c_row = tid >> 4u;
    const unsigned c_lane = tid & 15u;
    #pragma clang loop unroll(disable)
    for (unsigned r0 = 0u; r0 < kBM; r0 += 16u) {
        const unsigned r = r0 + c_row;
        const uint32_t tok_m = lds_tok[r];
        if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
            const unsigned c8_0 = (c_lane * 2u + 0u) ^ (r & 7u);
            const unsigned c8_1 = (c_lane * 2u + 1u) ^ (r & 7u);
            const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_0 * 4u);
            const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_1 * 4u);
            weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + c_lane * 8u, make_uint4(u0.x, u0.y, u1.x, u1.y));
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}
