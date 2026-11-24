#pragma once
#include <type_traits>
// V62 Opus 5.5 High -- GEMM2 1x4 BK128 body with BN=256 ("bn256"): one K loop produces TWO adjacent 128-column
// output blocks (tile_n, tile_n+1) of the same (expert, M tile).
//
// Relative to weave_s2_1x4_bk128_zerob / weave_s2_1x4_pf2 (BN=128):
//   * the A tile (BM x 128 fp8 per K128 step) is loaded from HBM, written to LDS and read back from LDS ONCE for
//     256 columns instead of once per 128 columns  -> A HBM/LDS traffic and ds_read count per output column / 2
//   * one 4-wave LDS counting barrier per K128 step covers 256 columns          -> barrier cost per column / 2
//   * per wave and K128 step: kMXDL x 2 halves x 2 n16 x 4 MFMA = 64 MFMA at BM=64 (zerob BN=128: 32)
//   * LDS layout identical (2 A stages of BM*128 B at 0, sa stages at 16384, tok at 17408): ZERO extra LDS
//   * wave w owns, in each half h, the n16 column groups (tile_n+h)*8 + 2w + {0,1}: exactly zerob's mapping per
//     half, so the epilogue is zerob's epilogue run once per half (C staged through the same 16 KB of LDS).
// Template kD = schedule:  1 -> 2 B register sets (loads of step kb+1 issued at the start of step kb, like zerob);
//   2 -> 3 sets (loads of step kb+2 issued at step kb, like V61 pf2; needs ~100 B VGPRs: spills in the megakernel);
//   3 -> ONE B set reloaded in place per half ("split"): MFMA(half 0) -> reload B half 0 for step kb+1 ->
//        MFMA(half 1) -> reload B half 1; A fragments of all kMXDL row groups are read from LDS once per step and
//        kept in VGPRs for both halves.  Every B load is covered by >= 32 MFMAs (half a step) + the barrier.
// Loads are uniform (clamped step index; invalid rows read row 0 and are zeroed when staged) and use a uniform
// 64-bit base + 32-bit lane offset (global_load saddr form) to keep address VGPRs low.
// Caller contract: tile_n + 1 < N/128 (both column blocks exist); kBM in {16, 32, 64}.
template <unsigned kBM, unsigned kD>
__device__ __forceinline__ void weave_s2_1x4_bn256(
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
    static_assert(kBM == 16u || kBM == 32u || kBM == 48u || kBM == 64u, "bn256: kBM in {16,32,64}");
    static_assert(kD == 1u || kD == 2u || kD == 3u, "bn256: schedule 1, 2 or 3");
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kMXDL = kBM / 16u;
    constexpr unsigned kAPasses = (kBM + 31u) / 32u;
    constexpr unsigned kAStageBytes = kBM * 128u;
    constexpr unsigned kSaBaseOff = 16384u;
    constexpr unsigned kTokOff = 17408u;
    constexpr unsigned kNS = (kD == 3u) ? 1u : (kD + 1u);     // B register sets
    constexpr unsigned kNA = (kD == 3u) ? 1u : kNS;            // A register sets

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu) : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned nk = K >> 7u;                 // number of K128 steps (>= 1)
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * N * K;
    const float* bs_row0 = b_scale_base + (static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * n_blocks + tile_n) * nk;
    const float* bs_row1 = bs_row0 + nk;

    const unsigned a_r32 = tid >> 3u;
    const unsigned a_c16 = tid & 7u;
    const bool a_thr = (kBM >= 32u) || (a_r32 < kBM);     // BM=16: only threads 0..127 stage A
    uint32_t a_roff[kAPasses];
    bool a_ok[kAPasses];
    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 32u + a_r32;
        const uint32_t tok = a_thr ? lds_tok[(r < kBM) ? r : 0u] : route_rows;
        a_ok[ap] = a_thr && (r < valid_rows) && (tok < route_rows);
        a_roff[ap] = a_ok[ap] ? (tok * K + a_c16 * 16u) : (a_c16 * 16u);       // invalid rows read row 0, zeroed
    }
    const bool sa_active = (tid < kBM);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : route_rows;
    const bool sa_ok = (tid < valid_rows) && sa_active && (tok_sa < route_rows);
    const uint32_t sa_roff = sa_ok ? (tok_sa * nk) : 0u;

    WeaveFloat4 acc[kMXDL][2][2];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        #pragma unroll
        for (unsigned h = 0u; h < 2u; ++h) {
            acc[mx][h][0] = {0,0,0,0};
            acc[mx][h][1] = {0,0,0,0};
        }
    }

    // per-expert B is N*K bytes (<= 64 MB for every model): 32-bit lane offsets relative to the expert base
    const uint64_t b_half = static_cast<uint64_t>(8u) * num_k64 * 1024ull;   // one 128-col block of B
    const uint32_t b_col_off0 = ((tile_n * 8u + wave_idx * 2u + 0u) * num_k64 * 64u + lane) * 16u;
    const uint32_t b_col_off1 = b_col_off0 + num_k64 * 1024u;

    // ---- register sets, rotated with (step % kNS) ----
    uint4 ar[kNA][kAPasses];
    float sar[kNA], sb0r[kNS], sb1r[kNS];
    WeaveI64x2 b[kNS][2][2][2];   // [set][half][nx][k64 half]

    auto issue_a = [&](auto Sc, unsigned step) {   // A + sa loads of K step `step` into A set S (uniform)
        constexpr unsigned S = decltype(Sc)::value;
        const unsigned s = (step < nk) ? step : (nk - 1u);
        const uint8_t* pa = a_base + s * 128u;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) ar[S][ap] = weave_gload_u4(pa + a_roff[ap]);
        sar[S] = weave_as1_arr(a_scale_base)[sa_roff + s];
    };
    auto issue_bh = [&](auto Sc, unsigned h, unsigned step) {   // B half h + its scale of K step `step` -> set S
        constexpr unsigned S = decltype(Sc)::value;
        const unsigned s = (step < nk) ? step : (nk - 1u);
        const uint8_t* pb = b_exp + (h ? b_half : 0ull) + static_cast<uint64_t>(s) * 2048ull;
        b[S][h][0][0] = weave_gload_i64x2(pb + b_col_off0);
        b[S][h][1][0] = weave_gload_i64x2(pb + b_col_off1);
        b[S][h][0][1] = weave_gload_i64x2(pb + (b_col_off0 + 1024u));
        b[S][h][1][1] = weave_gload_i64x2(pb + (b_col_off1 + 1024u));
        if (h) sb1r[S] = weave_as1_arr(bs_row1)[s]; else sb0r[S] = weave_as1_arr(bs_row0)[s];
    };
    auto issue = [&](auto Sc, unsigned step) {   // all global loads of K step `step` into set S
        issue_a(Sc, step);
        issue_bh(Sc, 0u, step);
        issue_bh(Sc, 1u, step);
    };
    auto stage_a = [&](auto Sc, unsigned step) { // A/sa of K step `step` (register set S) -> LDS stage step&1
        constexpr unsigned S = decltype(Sc)::value;
        WeaveLdsU8* dst = lds + (step & 1u) * kAStageBytes;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            const unsigned r = ap * 32u + a_r32;
            const uint4 v = a_ok[ap] ? ar[S][ap] : make_uint4(0u, 0u, 0u, 0u);
            if (a_thr) *reinterpret_cast<WeaveLdsU4*>(dst + r * 128u + (a_c16 ^ (r & 7u)) * 16u) = v;
        }
        if (sa_active) {
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + (step & 1u) * 512u)[tid] = sa_ok ? sar[S] : 0.0f;
        }
    };
    auto mfma_step = [&](auto Sc, unsigned step) {
        constexpr unsigned S = decltype(Sc)::value;
        const WeaveLdsU8* lds_a_cur = lds + (step & 1u) * kAStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + (step & 1u) * 512u);
        const float sb0 = sb0r[S], sb1 = sb1r[S];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned rm = mx * 16u + m_lane;
            const WeaveI64x2 av0 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
            const WeaveI64x2 av1 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
            const float sa = lds_sa_cur[rm];
            const float sab0 = sa * sb0, sab1 = sa * sb1;
            WeaveFloat4 t00 = {0,0,0,0}, t01 = {0,0,0,0}, t10 = {0,0,0,0}, t11 = {0,0,0,0};
            t00 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][0][0][0][0], av0[0], t00, 0, 0, 0);
            t01 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][0][1][0][0], av0[0], t01, 0, 0, 0);
            t10 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][1][0][0][0], av0[0], t10, 0, 0, 0);
            t11 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][1][1][0][0], av0[0], t11, 0, 0, 0);
            t00 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][0][0][0][1], av0[1], t00, 0, 0, 0);
            t01 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][0][1][0][1], av0[1], t01, 0, 0, 0);
            t10 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][1][0][0][1], av0[1], t10, 0, 0, 0);
            t11 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][1][1][0][1], av0[1], t11, 0, 0, 0);
            t00 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][0][0][1][0], av1[0], t00, 0, 0, 0);
            t01 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][0][1][1][0], av1[0], t01, 0, 0, 0);
            t10 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][1][0][1][0], av1[0], t10, 0, 0, 0);
            t11 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][1][1][1][0], av1[0], t11, 0, 0, 0);
            t00 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][0][0][1][1], av1[1], t00, 0, 0, 0);
            t01 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][0][1][1][1], av1[1], t01, 0, 0, 0);
            t10 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][1][0][1][1], av1[1], t10, 0, 0, 0);
            t11 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[S][1][1][1][1], av1[1], t11, 0, 0, 0);
            acc[mx][0][0] += t00 * sab0;
            acc[mx][0][1] += t01 * sab0;
            acc[mx][1][0] += t10 * sab1;
            acc[mx][1][1] += t11 * sab1;
        }
    };
    // split schedule (kD == 3): A fragments of step `step` read once, MFMA half 0, reload B half 0 for step+1,
    // MFMA half 1, reload B half 1 for step+1 (single B register set)
    auto split_step = [&](unsigned step) {
        using Z = std::integral_constant<unsigned, 0u>;
        const WeaveLdsU8* lds_a_cur = lds + (step & 1u) * kAStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + (step & 1u) * 512u);
        WeaveI64x2 av0[kMXDL], av1[kMXDL];
        float sa[kMXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned rm = mx * 16u + m_lane;
            av0[mx] = weave_lds_load_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
            av1[mx] = weave_lds_load_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
            sa[mx] = lds_sa_cur[rm];
        }
        #pragma unroll
        for (unsigned h = 0u; h < 2u; ++h) {
            const float sbh = h ? sb1r[0] : sb0r[0];
            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const float sab = sa[mx] * sbh;
                WeaveFloat4 t0 = {0,0,0,0}, t1 = {0,0,0,0};
                t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[0][h][0][0][0], av0[mx][0], t0, 0, 0, 0);
                t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[0][h][1][0][0], av0[mx][0], t1, 0, 0, 0);
                t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[0][h][0][0][1], av0[mx][1], t0, 0, 0, 0);
                t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[0][h][1][0][1], av0[mx][1], t1, 0, 0, 0);
                t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[0][h][0][1][0], av1[mx][0], t0, 0, 0, 0);
                t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[0][h][1][1][0], av1[mx][0], t1, 0, 0, 0);
                t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[0][h][0][1][1], av1[mx][1], t0, 0, 0, 0);
                t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[0][h][1][1][1], av1[mx][1], t1, 0, 0, 0);
                acc[mx][h][0] += t0 * sab;
                acc[mx][h][1] += t1 * sab;
            }
            issue_bh(Z{}, h, step + 1u);
        }
    };
    using C0 = std::integral_constant<unsigned, 0u>;
    using C1 = std::integral_constant<unsigned, 1u>;
    using C2 = std::integral_constant<unsigned, 2u>;

    if constexpr (kD == 3u) {
        // prologue: A/sa + B of step 0 -> LDS stage 0 / B set; step kb: issue A(kb+1), split MFMA(kb) with the
        // in-place B(kb+1) reloads, stage A(kb+1) (its loads are covered by the 64 MFMAs), barrier
        issue(C0{}, 0u);
        stage_a(C0{}, 0u);
        weave_4wave_lds_barrier_65440(arena);
        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u; kb < nk; ++kb) {
            issue_a(C0{}, kb + 1u);
            split_step(kb);
            stage_a(C0{}, kb + 1u);
            weave_4wave_lds_barrier_65440(arena);
        }
    } else if constexpr (kD == 2u) {
        // prologue: step 0 -> set 0 (A/sa staged to LDS stage 0), step 1 -> set 1 (in flight)
        issue(C0{}, 0u);
        issue(C1{}, 1u);
        stage_a(C0{}, 0u);
        weave_4wave_lds_barrier_65440(arena);
        // step kb, S = kb % 3: issue(kb+2 -> (S+2)%3), MFMA(kb, S), stage_a(kb+1 from (S+1)%3), barrier
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
    } else {
        // prologue: step 0 -> set 0, staged; step kb, S = kb % 2: issue(kb+1 -> S^1), MFMA(kb, S),
        // stage_a(kb+1 from S^1) [waits for this step's own loads, like zerob], barrier
        issue(C0{}, 0u);
        stage_a(C0{}, 0u);
        weave_4wave_lds_barrier_65440(arena);
#if V62_BAR_MODE >= 2
        // split arrive/wait: the global loads of step kb+2 are issued between the arrival and the spin so the
        // 4-wave rendezvous latency is hidden behind independent load issue.
        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u;; kb += 2u) {
            issue(C1{}, kb + 1u);
            mfma_step(C0{}, kb);
            stage_a(C1{}, kb + 1u);
            unsigned bar_t = weave_4wave_arrive_65440();
            issue(C0{}, kb + 2u);
            weave_4wave_wait_65440(bar_t);
            if (kb + 1u >= nk) break;
            mfma_step(C1{}, kb + 1u);
            stage_a(C0{}, kb + 2u);
            bar_t = weave_4wave_arrive_65440();
            issue(C1{}, kb + 3u);
            weave_4wave_wait_65440(bar_t);
            if (kb + 2u >= nk) break;
        }
#else
        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u;; kb += 2u) {
            issue(C1{}, kb + 1u);
            mfma_step(C0{}, kb);
            stage_a(C1{}, kb + 1u);
            weave_4wave_lds_barrier_65440(arena);
            if (kb + 1u >= nk) break;
            issue(C0{}, kb + 2u);
            mfma_step(C1{}, kb + 1u);
            stage_a(C0{}, kb + 2u);
            weave_4wave_lds_barrier_65440(arena);
            if (kb + 2u >= nk) break;
        }
#endif
    }
    // drain the clamped tail loads before the epilogue reuses LDS / registers
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    const unsigned c_row = tid >> 4u;
    const unsigned c_lane = tid & 15u;
    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    #pragma unroll
    for (unsigned h = 0u; h < 2u; ++h) {
        weave_4wave_lds_barrier_65440(arena);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned r = mx * 16u + m_lane;
            const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
            #pragma unroll
            for (unsigned nx = 0u; nx < 2u; ++nx) {
                const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
                const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][h][nx][0] * weight)) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][h][nx][1] * weight)) << 16u);
                const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][h][nx][2] * weight)) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][h][nx][3] * weight)) << 16u);
                const unsigned c8 = (col >> 2u) ^ (r & 7u);
                *reinterpret_cast<WeaveLdsU2*>(lds_c + r * 128u + c8 * 4u) = make_uint2(p0, p1);
            }
        }
        weave_4wave_lds_barrier_65440(arena);
        #pragma clang loop unroll(disable)
        for (unsigned r0 = 0u; r0 < kBM; r0 += 16u) {
            const unsigned r = r0 + c_row;
            const uint32_t tok_m = lds_tok[r];
            if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
                const unsigned c8_0 = (c_lane * 2u + 0u) ^ (r & 7u);
                const unsigned c8_1 = (c_lane * 2u + 1u) ^ (r & 7u);
                const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_0 * 4u);
                const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_1 * 4u);
                weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + (tile_n + h) * 128u + c_lane * 8u,
                                make_uint4(u0.x, u0.y, u1.x, u1.y));
            }
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}
