#pragma once
#include <type_traits>
// V62 Opus 5.5 High -- GEMM1 (gate+up, SiLU*up, bf16 -> per-(1,128) fp8 requant) 1x4 BK128 body, generalised to
// BM in {16, 32, 64}, with a selectable load schedule.
//
// Same math / MFMA opcode / B layout / LDS layout (A stages of BM*128 B at 0, sa stages at 16384, tok at 17408,
// C staging through lds[0..BM*256)) / epilogue as weave_s1_1x4_bk128_zerob<64>.  Each of the 4 waves owns BM rows
// x 32 columns of the 128-column block tile_n (n16 groups tile_n*8 + 2w + {0,1}), gate and up.
//   kD = 1: two register sets: ALL loads (A, sa, B gate/up, sb) of step kb+1 are issued at the START of step kb,
//           so every B load is covered by a full step of 64 MFMA (zerob reloads B per nx half: half a step).
//   kD = 3: one B set reloaded in place per nx half (the zerob schedule), for BM 16 / 32.
// Loads are uniform (clamped step index; invalid rows read row 0 then are zeroed) with a uniform 64-bit base +
// 32-bit lane offset.
template <unsigned kBM, unsigned kD>
__device__ __forceinline__ void weave_s1_1x4_v62(
    const int32_t* __restrict__ s_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    uint8_t* __restrict__ quant_out_base,
    float* __restrict__ quant_scale_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned num_tokens,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    static_assert(kBM == 16u || kBM == 32u || kBM == 48u || kBM == 64u, "s1 v62: kBM in {16,32,64}");
    static_assert(kD == 1u || kD == 3u, "s1 v62: schedule 1 or 3");
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
    constexpr unsigned kNS = (kD == 1u) ? 2u : 1u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu) : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned nk = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint64_t expert_b_bytes = static_cast<uint64_t>(N) * K * 2ull;
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_b_bytes;
    const uint8_t* b_up_exp = b_gate_exp + static_cast<uint64_t>(N) * K;
    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * nk;
    const float* bs_gate_row = b_scale_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_bs_stride + static_cast<uint64_t>(tile_n) * nk;
    const float* bs_up_row = bs_gate_row + static_cast<uint64_t>(n_blocks) * nk;

    const unsigned a_r32 = tid >> 3u;
    const unsigned a_c16 = tid & 7u;
    const bool a_thr = (kBM >= 32u) || (a_r32 < kBM);
    uint32_t a_roff[kAPasses];
    bool a_ok[kAPasses];
    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 32u + a_r32;
        const uint32_t tok = a_thr ? lds_tok[(r < kBM) ? r : 0u] : num_tokens;
        a_ok[ap] = a_thr && (r < valid_rows) && (tok < num_tokens);
        a_roff[ap] = a_ok[ap] ? (tok * K + a_c16 * 16u) : (a_c16 * 16u);
    }
    const bool sa_active = (tid < kBM);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : num_tokens;
    const bool sa_ok = (tid < valid_rows) && sa_active && (tok_sa < num_tokens);
    const uint32_t sa_roff = sa_ok ? (tok_sa * nk) : 0u;

    WeaveFloat4 gacc[kMXDL][2], uacc[kMXDL][2];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) { gacc[mx][nx] = {0,0,0,0}; uacc[mx][nx] = {0,0,0,0}; }
    }
    const uint32_t b_col_off0 = ((tile_n * 8u + wave_idx * 2u + 0u) * num_k64 * 64u + lane) * 16u;
    const uint32_t b_col_off1 = b_col_off0 + num_k64 * 1024u;

    uint4 ar[kNS][kAPasses];
    float sar[kNS], sbg[kNS], sbu[kNS];
    WeaveI64x2 bg[kNS][2][2], bu[kNS][2][2];   // [set][nx][k64 half]

    auto issue_a = [&](auto Sc, unsigned step) {
        constexpr unsigned S = decltype(Sc)::value;
        const unsigned s = (step < nk) ? step : (nk - 1u);
        const uint8_t* pa = a_base + s * 128u;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) ar[S][ap] = weave_gload_u4(pa + a_roff[ap]);
        sar[S] = weave_as1_arr(a_scale_base)[sa_roff + s];
    };
    auto issue_bn = [&](auto Sc, unsigned nx, unsigned step) {   // gate+up B of column half nx -> set S
        constexpr unsigned S = decltype(Sc)::value;
        const unsigned s = (step < nk) ? step : (nk - 1u);
        const uint64_t bk = static_cast<uint64_t>(s) * 2048ull;
        const uint32_t o = nx ? b_col_off1 : b_col_off0;
        const uint8_t* pg = b_gate_exp + bk;
        const uint8_t* pu = b_up_exp + bk;
        bg[S][nx][0] = weave_gload_i64x2(pg + o);
        bu[S][nx][0] = weave_gload_i64x2(pu + o);
        bg[S][nx][1] = weave_gload_i64x2(pg + (o + 1024u));
        bu[S][nx][1] = weave_gload_i64x2(pu + (o + 1024u));
    };
    auto issue_sb = [&](auto Sc, unsigned step) {
        constexpr unsigned S = decltype(Sc)::value;
        const unsigned s = (step < nk) ? step : (nk - 1u);
        sbg[S] = weave_as1_arr(bs_gate_row)[s];
        sbu[S] = weave_as1_arr(bs_up_row)[s];
    };
    auto stage_a = [&](auto Sc, unsigned step) {
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
    // MFMAs of column half nx for every row group; B from set SB, scales sbg/sbu from set SS
    auto mfma_nx = [&](auto SBc, unsigned nx, unsigned step, float sb_g, float sb_u) {
        constexpr unsigned SB = decltype(SBc)::value;
        const WeaveLdsU8* lds_a_cur = lds + (step & 1u) * kAStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + (step & 1u) * 512u);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned rm = mx * 16u + m_lane;
            const WeaveI64x2 av0 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
            const WeaveI64x2 av1 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
            const float sa = lds_sa_cur[rm];
            const float sab_g = sa * sb_g, sab_u = sa * sb_u;
            WeaveFloat4 tg = {0,0,0,0}, tu = {0,0,0,0};
            tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[SB][nx][0][0], av0[0], tg, 0, 0, 0);
            tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[SB][nx][0][0], av0[0], tu, 0, 0, 0);
            tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[SB][nx][0][1], av0[1], tg, 0, 0, 0);
            tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[SB][nx][0][1], av0[1], tu, 0, 0, 0);
            tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[SB][nx][1][0], av1[0], tg, 0, 0, 0);
            tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[SB][nx][1][0], av1[0], tu, 0, 0, 0);
            tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[SB][nx][1][1], av1[1], tg, 0, 0, 0);
            tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[SB][nx][1][1], av1[1], tu, 0, 0, 0);
            gacc[mx][nx] += tg * sab_g;
            uacc[mx][nx] += tu * sab_u;
        }
    };
    using C0 = std::integral_constant<unsigned, 0u>;
    using C1 = std::integral_constant<unsigned, 1u>;

    if constexpr (kD == 1u) {
        issue_a(C0{}, 0u); issue_bn(C0{}, 0u, 0u); issue_bn(C0{}, 1u, 0u); issue_sb(C0{}, 0u);
        stage_a(C0{}, 0u);
        weave_4wave_lds_barrier_65440(arena);
#if V62_BAR_MODE >= 3
        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u;; kb += 2u) {
            issue_a(C1{}, kb + 1u); issue_bn(C1{}, 0u, kb + 1u); issue_bn(C1{}, 1u, kb + 1u); issue_sb(C1{}, kb + 1u);
            mfma_nx(C0{}, 0u, kb, sbg[0], sbu[0]);
            mfma_nx(C0{}, 1u, kb, sbg[0], sbu[0]);
            stage_a(C1{}, kb + 1u);
            unsigned bar_t = weave_4wave_arrive_65440();
            issue_a(C0{}, kb + 2u); issue_bn(C0{}, 0u, kb + 2u); issue_bn(C0{}, 1u, kb + 2u); issue_sb(C0{}, kb + 2u);
            weave_4wave_wait_65440(bar_t);
            if (kb + 1u >= nk) break;
            mfma_nx(C1{}, 0u, kb + 1u, sbg[1], sbu[1]);
            mfma_nx(C1{}, 1u, kb + 1u, sbg[1], sbu[1]);
            stage_a(C0{}, kb + 2u);
            bar_t = weave_4wave_arrive_65440();
            issue_a(C1{}, kb + 3u); issue_bn(C1{}, 0u, kb + 3u); issue_bn(C1{}, 1u, kb + 3u); issue_sb(C1{}, kb + 3u);
            weave_4wave_wait_65440(bar_t);
            if (kb + 2u >= nk) break;
        }
#else
        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u;; kb += 2u) {
            issue_a(C1{}, kb + 1u); issue_bn(C1{}, 0u, kb + 1u); issue_bn(C1{}, 1u, kb + 1u); issue_sb(C1{}, kb + 1u);
            mfma_nx(C0{}, 0u, kb, sbg[0], sbu[0]);
            mfma_nx(C0{}, 1u, kb, sbg[0], sbu[0]);
            stage_a(C1{}, kb + 1u);
            weave_4wave_lds_barrier_65440(arena);
            if (kb + 1u >= nk) break;
            issue_a(C0{}, kb + 2u); issue_bn(C0{}, 0u, kb + 2u); issue_bn(C0{}, 1u, kb + 2u); issue_sb(C0{}, kb + 2u);
            mfma_nx(C1{}, 0u, kb + 1u, sbg[1], sbu[1]);
            mfma_nx(C1{}, 1u, kb + 1u, sbg[1], sbu[1]);
            stage_a(C0{}, kb + 2u);
            weave_4wave_lds_barrier_65440(arena);
            if (kb + 2u >= nk) break;
        }
#endif
    } else {
        // zerob schedule: one B set; reload half nx for step kb+1 right after its MFMAs of step kb
        issue_a(C0{}, 0u); issue_bn(C0{}, 0u, 0u); issue_bn(C0{}, 1u, 0u); issue_sb(C0{}, 0u);
        stage_a(C0{}, 0u);
        weave_4wave_lds_barrier_65440(arena);
        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u; kb < nk; ++kb) {
            issue_a(C0{}, kb + 1u);
            const float sg = sbg[0], su = sbu[0];
            mfma_nx(C0{}, 0u, kb, sg, su);
            issue_bn(C0{}, 0u, kb + 1u);
            mfma_nx(C0{}, 1u, kb, sg, su);
            issue_bn(C0{}, 1u, kb + 1u);
            issue_sb(C0{}, kb + 1u);
            stage_a(C0{}, kb + 1u);
            weave_4wave_lds_barrier_65440(arena);
        }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    weave_4wave_lds_barrier_65440(arena);

    WeaveLdsU16* lds_q = reinterpret_cast<WeaveLdsU16*>(lds);
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + m_lane;
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(weave_bk64_silu(gacc[mx][nx][0]) * uacc[mx][nx][0])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(weave_bk64_silu(gacc[mx][nx][1]) * uacc[mx][nx][1])) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(weave_bk64_silu(gacc[mx][nx][2]) * uacc[mx][nx][2])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(weave_bk64_silu(gacc[mx][nx][3]) * uacc[mx][nx][3])) << 16u);
            *reinterpret_cast<WeaveLdsU2*>(lds_q + r * 128u + col) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    unsigned q_tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(q_tid));
    const unsigned q_row = q_tid >> 4u;
    const unsigned q_lane = q_tid & 15u;
    #pragma clang loop unroll(disable)
    for (unsigned r0 = 0u; r0 < kBM; r0 += 16u) {
        const unsigned r = r0 + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        uint4 raw = make_uint4(0,0,0,0);
        if (valid) { raw = *reinterpret_cast<const WeaveLdsU4*>(lds_q + r * 128u + q_lane * 8u); }
        weave_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}
