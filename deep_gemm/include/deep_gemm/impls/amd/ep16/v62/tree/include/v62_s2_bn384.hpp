#pragma once
// V62 DS1017 -- GEMM2 1x4 BK128 body with BN=384: one K loop produces THREE adjacent 128-column output
// blocks of the same (expert, M tile).  Same A LDS stage / barrier as the BN=256 body (no extra LDS).
// Per wave and K128 step: kMXDL * 3 halves * 2 n16 * 4 MFMA = 96 MFMA at BM=64 (BN256: 64).
// Caller contract: tile_n + 2 < N/128 (all three column blocks exist).
template <unsigned kBM, unsigned kD>
__device__ __forceinline__ void weave_s2_1x4_bn384(
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
    static_assert(kBM == 16u || kBM == 32u || kBM == 48u || kBM == 64u, "bn384: kBM in {16,32,64}");
    static_assert(kD == 3u || kD == 4u, "bn384: kD in {3,4}");
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kNH = 3u;
    constexpr unsigned kMXDL = kBM / 16u;
    constexpr unsigned kAPasses = (kBM + 31u) / 32u;
    constexpr unsigned kAStageBytes = kBM * 128u;
    constexpr unsigned kSaBaseOff = 16384u;
    constexpr unsigned kTokOff = 17408u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu) : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned nk = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * N * K;
    const float* bs_row[kNH];
    #pragma unroll
    for (unsigned h = 0u; h < kNH; ++h)
        bs_row[h] = b_scale_base + (static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * n_blocks + tile_n + h) * nk;

    const unsigned a_r32 = tid >> 3u;
    const unsigned a_c16 = tid & 7u;
    const bool a_thr = (kBM >= 32u) || (a_r32 < kBM);
    uint32_t a_roff[kAPasses];
    bool a_ok[kAPasses];
    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 32u + a_r32;
        const uint32_t tok = a_thr ? lds_tok[(r < kBM) ? r : 0u] : route_rows;
        a_ok[ap] = a_thr && (r < valid_rows) && (tok < route_rows);
        a_roff[ap] = a_ok[ap] ? (tok * K + a_c16 * 16u) : (a_c16 * 16u);
    }
    const bool sa_active = (tid < kBM);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : route_rows;
    const bool sa_ok = (tid < valid_rows) && sa_active && (tok_sa < route_rows);
    const uint32_t sa_roff = sa_ok ? (tok_sa * nk) : 0u;

    WeaveFloat4 acc[kMXDL][kNH][2];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx)
        #pragma unroll
        for (unsigned h = 0u; h < kNH; ++h) {
            acc[mx][h][0] = {0,0,0,0};
            acc[mx][h][1] = {0,0,0,0};
        }

    const uint64_t b_half = static_cast<uint64_t>(8u) * num_k64 * 1024ull;   // one 128-col block of B
    const uint32_t b_col_off0 = ((tile_n * 8u + wave_idx * 2u + 0u) * num_k64 * 64u + lane) * 16u;
    const uint32_t b_col_off1 = b_col_off0 + num_k64 * 1024u;

    constexpr unsigned kNB = (kD == 4u) ? 2u : kNH;   // B register sets
    uint4 ar[1][kAPasses];
    float sar[1];
    float sb[kNH];
    WeaveI64x2 b[kNB][2][2];   // [set][nx][k64 half]

    auto issue_a = [&](unsigned step) {
        const unsigned s = (step < nk) ? step : (nk - 1u);
        const uint8_t* pa = a_base + s * 128u;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) ar[0][ap] = weave_gload_u4(pa + a_roff[ap]);
        sar[0] = weave_as1_arr(a_scale_base)[sa_roff + s];
    };
    auto issue_bh = [&](unsigned bsel, unsigned h, unsigned step) {
        const unsigned s = (step < nk) ? step : (nk - 1u);
        const uint8_t* pb = b_exp + static_cast<uint64_t>(h) * b_half + static_cast<uint64_t>(s) * 2048ull;
        b[bsel][0][0] = weave_gload_i64x2(pb + b_col_off0);
        b[bsel][1][0] = weave_gload_i64x2(pb + b_col_off1);
        b[bsel][0][1] = weave_gload_i64x2(pb + (b_col_off0 + 1024u));
        b[bsel][1][1] = weave_gload_i64x2(pb + (b_col_off1 + 1024u));
    };
    auto issue_sb = [&](unsigned h, unsigned step) {
        const unsigned s = (step < nk) ? step : (nk - 1u);
        sb[h] = weave_as1_arr(bs_row[h])[s];
    };
    auto stage_a = [&](unsigned step) {
        WeaveLdsU8* dst = lds + (step & 1u) * kAStageBytes;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            const unsigned r = ap * 32u + a_r32;
            const uint4 v = a_ok[ap] ? ar[0][ap] : make_uint4(0u, 0u, 0u, 0u);
            if (a_thr) *reinterpret_cast<WeaveLdsU4*>(dst + r * 128u + (a_c16 ^ (r & 7u)) * 16u) = v;
        }
        if (sa_active) {
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + (step & 1u) * 512u)[tid] = sa_ok ? sar[0] : 0.0f;
        }
    };
    // A fragments read once per step, then four adjacent halves of MFMA with the in-place B reload of step+1.
    auto split_step = [&](unsigned step) {
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
        const unsigned cur0 = (kD == 4u) ? (step * kNH) : 0u;
        #pragma unroll
        for (unsigned h = 0u; h < kNH; ++h) {
            const unsigned bsel = (kD == 4u) ? ((cur0 + h) & 1u) : h;
            const float sbh = sb[h];
            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const float sab = sa[mx] * sbh;
                WeaveFloat4 t0 = {0,0,0,0}, t1 = {0,0,0,0};
                t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[bsel][0][0][0], av0[mx][0], t0, 0, 0, 0);
                t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[bsel][1][0][0], av0[mx][0], t1, 0, 0, 0);
                t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[bsel][0][0][1], av0[mx][1], t0, 0, 0, 0);
                t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[bsel][1][0][1], av0[mx][1], t1, 0, 0, 0);
                t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[bsel][0][1][0], av1[mx][0], t0, 0, 0, 0);
                t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[bsel][1][1][0], av1[mx][0], t1, 0, 0, 0);
                t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[bsel][0][1][1], av1[mx][1], t0, 0, 0, 0);
                t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[bsel][1][1][1], av1[mx][1], t1, 0, 0, 0);
                acc[mx][h][0] += t0 * sab;
                acc[mx][h][1] += t1 * sab;
            }
            if constexpr (kD == 4u) {
                // two-set rotation: the buffer just consumed is refilled with the slot two halves ahead
                const unsigned nslot = (h + 2u) % kNH;
                const unsigned nstep = step + ((h + 2u) / kNH);
                issue_bh((cur0 + h) & 1u, nslot, nstep);
                sb[nslot] = weave_as1_arr(bs_row[nslot])[(nstep < nk) ? nstep : (nk - 1u)];
            } else {
                issue_bh(h, h, step + 1u);
                sb[h] = weave_as1_arr(bs_row[h])[((step + 1u) < nk) ? (step + 1u) : (nk - 1u)];
            }
        }
    };
    issue_a(0u);
    stage_a(0u);
    if constexpr (kD == 4u) {
        issue_bh(0u, 0u, 0u);
        issue_bh(1u, 1u, 0u);
        #pragma unroll
        for (unsigned h = 0u; h < kNH; ++h) issue_sb(h, 0u);
    } else {
        #pragma unroll
        for (unsigned h = 0u; h < kNH; ++h) { issue_bh(h, h, 0u); issue_sb(h, 0u); }
    }
    weave_4wave_lds_barrier_65440(arena);
    #pragma clang loop unroll(disable)
    for (unsigned kb = 0u; kb < nk; ++kb) {
        issue_a(kb + 1u);
        split_step(kb);
        stage_a(kb + 1u);
        weave_4wave_lds_barrier_65440(arena);
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    const unsigned c_row = tid >> 4u;
    const unsigned c_lane = tid & 15u;
    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    #pragma unroll
    for (unsigned h = 0u; h < kNH; ++h) {
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
