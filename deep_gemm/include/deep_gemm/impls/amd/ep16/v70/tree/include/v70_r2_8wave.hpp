#pragma once
// =============================================================================================================
// V70 MEGA80 (Opus 5.5 R3/R4): the Round-2 standalone GEMM body (delivery/v70_gemm_best.hpp, R2_ILV + WEARLY +
// SCHED=3, 0 spill) fused into the persistent megakernel's pure 8/0 compute CTA as TWO independent 4-wave groups.
//
//   * 512-thread CTA = group 0 (waves 0..3, LDS [0, 26368)) + group 1 (waves 4..7, LDS [26368, 52736)).
//   * Each group runs the unmodified R2 per-K128 schedule on its own 64-row x {128 h | 256 out}-column work unit.
//   * The R2 body's only barrier is `s_waitcnt lgkmcnt(0); s_barrier` (workgroup-wide).  Both groups therefore
//     execute exactly the same number of s_barrier per call: GEMM1 = nk + 2, GEMM2 = nk + 1 (nk = K/128).  A group
//     without a work unit executes the same number of barriers and nothing else, so every s_barrier is matched.
//   * No software LDS-atomic barrier anywhere in the K loop.
//   * GEMM2 output rows are the stage-2 token ids (dense tok-major gemm2_output), optional S1F direct rows.
//   * R4 (V70_R2X_COMPACT): a work unit no longer carries any row address.  Its 64 rows come from the CTA-wide
//     LDS row table R2X_RTAB = { abs stage row, stage id & 0xffffff, GEMM2 route weight, 0 } x 64 (rows >= valid
//     repeat row 0), filled
//     by wave 0 before the unit and published by the caller's barrier.  The table address is an immediate LDS
//     offset, so a contiguous descriptor tile and a cross-pass compacted tile run the identical body with
//     three fewer live scalars (ids pointer, row offset) than R3 and no per-row weight VGPRs across the K loop.
//
// Internal compile-time selector only (V70_R2X); no public WEAVE_* knob is added.
// =============================================================================================================
#include <stdint.h>
#include <type_traits>

#ifndef V70_R2X
#define V70_R2X 0
#endif
#ifndef R2X_ILV_S
#define R2X_ILV_S 5      // GEMM1 B-prefetch stride (MFMA slots)
#endif
#ifndef R2X_ILV_B
#define R2X_ILV_B 24     // GEMM1 first B-prefetch slot
#endif
#ifndef R2X_ILV2_S
#define R2X_ILV2_S 5     // GEMM2 B-prefetch stride
#endif
#ifndef R2X_ILV2_B
#define R2X_ILV2_B 2     // GEMM2 first B-prefetch slot
#endif
#ifndef R2X_MBSET
#define R2X_MBSET 0      // bit0: MB=1, bit1: MB=2, bit2: MB=3 specializations (0 = MB=4 only, as R2 best)
#endif
#ifndef R2X_GRP_LDS
#define R2X_GRP_LDS 26368u
#endif
#ifndef V70_E3_SC1_NOBW
#define V70_E3_SC1_NOBW 0
#endif
#ifndef V70_E3_PRIO
#define V70_E3_PRIO 0
#endif
#ifndef V70_E3_HOIST
#define V70_E3_HOIST 0
#endif
#ifndef V70_E3_QUIET_HT
#define V70_E3_QUIET_HT 0
#endif
#ifndef V70_E3_3BAR
#define V70_E3_3BAR 0
#endif
#ifndef V70_E3_FOLD
#define V70_E3_FOLD 0
#endif
#if V70_E3_SC1_NOBW
typedef int32_t WeaveI32x2 __attribute__((ext_vector_type(2)));
__device__ void v70_raw_buf_store_i32(int32_t vdata, WeaveI32x4 srsrc, int32_t voffset, int32_t soffset,
                                      int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i32");
__device__ void v70_raw_buf_store_i32x2(WeaveI32x2 vdata, WeaveI32x4 srsrc, int32_t voffset, int32_t soffset,
                                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.v2i32");
#endif
#ifndef R2X_RTAB_LDS
#define R2X_RTAB_LDS 56320u   // [56320, 57344): 64 x int4 row table, free on 8/0 compute CTAs during the Atrex
#endif

namespace r2x {
typedef float f4 __attribute__((ext_vector_type(4)));
typedef float f2 __attribute__((ext_vector_type(2)));
static __device__ __forceinline__ void r2x_fold4(f4& c, const f4& t, float sc) {
    f2 s2 = {sc, 0.f};
    f2 c01 = {c[0], c[1]}, c23 = {c[2], c[3]};
    const f2 t01 = {t[0], t[1]}, t23 = {t[2], t[3]};
    asm volatile("v_pk_fma_f32 %0, %2, %4, %0 op_sel_hi:[0,1,1]\n\t"
                 "v_pk_fma_f32 %1, %3, %4, %1 op_sel_hi:[0,1,1]"
                 : "+v"(c01), "+v"(c23) : "v"(t01), "v"(t23), "v"(s2));
    c[0] = c01[0]; c[1] = c01[1]; c[2] = c23[0]; c[3] = c23[1];
}
typedef uint32_t u4 __attribute__((ext_vector_type(4)));
typedef uint32_t u2 __attribute__((ext_vector_type(2)));
#define R2X_DEV static __device__ __forceinline__

R2X_DEV long lo8(const u4& v) { return (long)(((uint64_t)v.y << 32) | (uint64_t)v.x); }
R2X_DEV long hi8(const u4& v) { return (long)(((uint64_t)v.w << 32) | (uint64_t)v.z); }
R2X_DEV f4 mma(long a, long b, f4 c) { return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c, 0, 0, 0); }
R2X_DEV void hw_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
// Barrier that also completes the two scalar B-scale loads issued one step ahead (ties them to the wait).
R2X_DEV void hw_barrier_s(float& a, float& b) {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" : "+s"(a), "+s"(b) :: "memory");
}
// Scalar loads of the per-K128 B scales.  The megakernel's scale pointers come from a kernarg struct (no
// noalias/readonly proof), so plain C++ loads become VMEM loads + vmcnt(0) inside the K loop; asm s_load keeps
// them on the scalar path exactly like the standalone kernel's __restrict__ argument.
R2X_DEV void sload2(float& a, float& b, const float* pa, const float* pb) {
    asm volatile("s_load_dword %0, %1, 0x0" : "=s"(a) : "s"(pa));
    asm volatile("s_load_dword %0, %1, 0x0" : "=s"(b) : "s"(pb));
}
R2X_DEV u4 ldg(const uint8_t* p) { return *(const u4*)p; }
R2X_DEV float bf16r(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}
R2X_DEV uint32_t bf16bits(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}
R2X_DEV void gl4(u4& r, uint32_t voff, const void* sbase) {
    asm volatile("global_load_dwordx4 %0, %1, %2" : "=v"(r) : "v"(voff), "s"(sbase));
}
R2X_DEV void gl4_16(u4& r, uint32_t voff, const void* sbase) {
    asm volatile("global_load_dwordx4 %0, %1, %2 offset:16" : "=v"(r) : "v"(voff), "s"(sbase));
}
R2X_DEV void gl4_1k(u4& r, uint32_t voff, const void* sbase) {
    asm volatile("global_load_dwordx4 %0, %1, %2 offset:1024" : "=v"(r) : "v"(voff), "s"(sbase));
}
R2X_DEV void gl1(float& r, uint32_t voff, const void* sbase) {
    asm volatile("global_load_dword %0, %1, %2" : "=v"(r) : "v"(voff), "s"(sbase));
}
template <int N> R2X_DEV void vmwait8(u4 (&b)[4][2]) {
    asm volatile("s_waitcnt vmcnt(%8)" : "+v"(b[0][0]), "+v"(b[0][1]), "+v"(b[1][0]), "+v"(b[1][1]),
                 "+v"(b[2][0]), "+v"(b[2][1]), "+v"(b[3][0]), "+v"(b[3][1]) : "n"(N));
}
template <int N> R2X_DEV void vmwait3(u4& a, u4& b, float& c) {
    asm volatile("s_waitcnt vmcnt(%3)" : "+v"(a), "+v"(b), "+v"(c) : "n"(N));
}
// SW = 1: four-wave software barrier (W00..W03 only, LDS counters at 65440) for the head prelude, where
// W04..W07 are still draining the remote head and never reach an s_barrier.
template <int SW> R2X_DEV void bar() {
    if constexpr (SW) { asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory"); weave_4wave_barrier_65440(g_v1262_phase_arena); }
    else hw_barrier();
}
template <int SW> R2X_DEV void bar_s(float& a, float& b) {
    if constexpr (SW) {
        asm volatile("s_waitcnt lgkmcnt(0)" : "+s"(a), "+s"(b) :: "memory");
        weave_4wave_barrier_65440(g_v1262_phase_arena);
    } else hw_barrier_s(a, b);
}
template <int SW> R2X_DEV void idle(int n) {
    #pragma clang loop unroll(disable)
    for (int i = 0; i < n; ++i) bar<SW>();
}

R2X_DEV int lane_fresh() {
    int l;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\tv_mbcnt_hi_u32_b32 %0, -1, %0" : "=v"(l));
    return l;
}
// lane ^ m exchange with an address derived from a fresh v_mbcnt (identical result to __shfl_xor(v, m)).
R2X_DEV float xor_fresh(const float v, const int m) {
    const int a = (lane_fresh() ^ m) << 2;
    return __int_as_float(__builtin_amdgcn_ds_bpermute(a, __float_as_int(v)));
}
R2X_DEV const int4* rtab() { return reinterpret_cast<const int4*>(g_v1262_phase_arena + R2X_RTAB_LDS); }
// Row table fill for one work unit (threads 0..63 of the CTA, i.e. wave 0).  pos(r) gives the absolute stage row
// of local row r < valid; rows >= valid repeat row 0 so every gather stays in bounds.  The caller publishes the
// table with its barrier before the unit starts.
template <class F>
R2X_DEV void rtab_fill(const unsigned t, const int32_t* __restrict__ ids, const float* __restrict__ wts,
                       const int valid, F pos) {
    if (t < 64u) {
        const int p = pos(((int)t < valid) ? (int)t : 0);
        const int w = (wts != nullptr) ? __float_as_int(wts[p]) : 0;
        reinterpret_cast<int4*>(g_v1262_phase_arena + R2X_RTAB_LDS)[t] =
            make_int4(p, (int)((uint32_t)ids[p] & 0xffffffu), w, 0);
    }
}

// One 64-row work unit of one 4-wave group.  Identical per-K128 schedule to v70_body_mb<MB, G1, SCHED=3> with
// R2_ILV=1, R2_ILV_A=-1 (A issued at the top), R2_WEARLY=1, no split-K.
//   G1: B = ck_b (gate|up), NO = I, K = H, out = quant_output (FP8, dense row off+row), oscale = quant_scale.
//   G2: B = ck_w2, NO = hidden, K = I, out = gemm2_output rows = stage-2 token id (or S1F direct row).
template <int MB, int G1, int SW = 0>
R2X_DEV void body(uint8_t* __restrict__ lds, const int gtid,
                  const int valid, const int nbk, const int e,
                  const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                  const float* __restrict__ sa, const float* __restrict__ sb,
                  void* __restrict__ out, float* __restrict__ oscale, const float* __restrict__ wts,
                  const unsigned K, const unsigned NO, const unsigned route_rows,
                  const unsigned* row_dir, const uint64_t* peer_tab, const uint64_t* node_tab,
                  unsigned node_row_base, unsigned my_node)
{
    constexpr int BM = 16 * MB;
    constexpr int IL_S = G1 ? R2X_ILV_S : R2X_ILV2_S;
    constexpr int IL_B = G1 ? R2X_ILV_B : R2X_ILV2_B;
    constexpr int STA = 64 * 128;                 // A bytes per K128 slot (64-row ring, MB<=4)
    float* lsa = (float*)(lds + 3 * STA);         // [3][64] activation scales
    float* lrm = (float*)(lds + 3 * STA + 3 * 64 * 4);  // [4][64] row amax exchange (GEMM1)

    const int tid = gtid, lane = tid & 63;
    const int wn = __builtin_amdgcn_readfirstlane(tid >> 6) & 3;
    const unsigned kb = K >> 7;
    const int nk = (int)kb;

    const int lr = tid >> 2, lp = tid & 3;
    uint32_t aw0_off = (uint32_t)(lr * 128 + (((2 * lp) ^ (lr & 7)) << 4));
    uint32_t aw1_off = (uint32_t)(lr * 128 + (((2 * lp + 1) ^ (lr & 7)) << 4));
#if V70_E3_HOIST
    asm volatile("" : "+v"(aw0_off), "+v"(aw1_off));
#endif
    uint8_t* aW0 = lds + aw0_off;
    uint8_t* aW1 = lds + aw1_off;

    const uint8_t* bp[4];
    const float* sbp0;
    const float* sbp1;
    if (G1) {
        const size_t mat = (size_t)NO * K;
        const uint8_t* bg = B + (size_t)(e * 2 + 0) * mat;
        const uint8_t* bu = B + (size_t)(e * 2 + 1) * mat;
        for (int j = 0; j < 2; ++j) {
            const size_t n16 = (size_t)(nbk * 8 + wn * 2 + j);
            bp[j] = bg + n16 * (K * 16);
            bp[2 + j] = bu + n16 * (K * 16);
        }
        sbp0 = sb + ((size_t)(e * 2 + 0) * (NO >> 7) + nbk) * kb;
        sbp1 = sb + ((size_t)(e * 2 + 1) * (NO >> 7) + nbk) * kb;
    } else {
        const uint8_t* bw = B + (size_t)e * NO * K;
        for (int j = 0; j < 4; ++j) {
            const size_t n16 = (size_t)(nbk * 16 + wn * 4 + j);
            bp[j] = bw + n16 * (K * 16);
        }
        sbp0 = sb + ((size_t)e * (NO >> 7) + nbk * 2 + (wn >> 1)) * kb;
        sbp1 = sbp0;
    }
    const uint32_t bvo = lane * 16;
    // rows >= valid reuse row 0 (always a real token): loads stay in bounds, results are never stored
    const int4* rt = rtab();
    const uint32_t avo = (uint32_t)rt[lr].y * K + lp * 32;
    const uint32_t svo = (uint32_t)rt[tid & 63].y * kb * 4;
    int ard00 = (lane & 15) * 128 + (((lane >> 4) ^ (lane & 7)) << 4);
    int ard01 = (lane & 15) * 128 + ((((4 + (lane >> 4)) ^ (lane & 7)) << 4));
    int srd = (lane & 15);
#if V70_E3_HOIST
    asm volatile("" : "+v"(ard00), "+v"(ard01), "+v"(srd));
#endif
    int ard[MB][2];
    for (int mb = 0; mb < MB; ++mb) {
        ard[mb][0] = ard00 + mb * 2048;
        ard[mb][1] = ard01 + mb * 2048;
    }

    f4 acc[4][MB];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < MB; ++j) acc[i][j] = f4{0.f, 0.f, 0.f, 0.f};

    {
        u4 v0 = ldg(A + avo), v1 = ldg(A + avo + 16);
        float s = *(const float*)((const uint8_t*)sa + svo);
        if (lr < BM) { *(u4*)aW0 = v0; *(u4*)aW1 = v1; }
        if (tid < BM) lsa[tid] = s;
        if (nk > 1) {
            u4 w0 = ldg(A + avo + 128), w1 = ldg(A + avo + 144);
            float s1 = *(const float*)((const uint8_t*)sa + svo + 4);
            if (lr < BM) { *(u4*)(aW0 + STA) = w0; *(u4*)(aW1 + STA) = w1; }
            if (tid < BM) lsa[64 + tid] = s1;
        }
    }
    u4 bA[4][2], bB[4][2];
    for (int j = 0; j < 4; ++j) { gl4(bA[j][0], bvo, bp[j]); gl4_1k(bA[j][1], bvo, bp[j]); }
    float sbn0, sbn1;
    sload2(sbn0, sbn1, sbp0, sbp1);
    bar_s<SW>(sbn0, sbn1);                                            // barrier #1

    int slot = 0;
    auto step = [&](int k, u4 (&bc)[4][2], u4 (&bn)[4][2]) {
        const int kb1 = (k + 1) < nk ? k + 1 : nk - 1;
        const int ka2 = (k + 2) < nk ? k + 2 : nk - 1;
        u4 v0, v1; float s2;
        const uint8_t* abI = A + (size_t)ka2 * 128;
        const uint8_t* sbI = (const uint8_t*)sa + (size_t)ka2 * 4;
        const size_t oI = (size_t)kb1 * 2048;
        gl4(v0, avo, abI); gl4_16(v1, avo, abI); gl1(s2, svo, sbI);   // A(k+2) at the top (ILV_A = -1)
        vmwait8<3>(bc);                                               // B(k) landed; A(k+2) in flight
        const float sb0 = sbn0, sb1 = sbn1;
        sload2(sbn0, sbn1, sbp0 + kb1, sbp1 + kb1);                   // next step's scales (scalar path)
        const uint8_t* ls = lds + slot * STA;
        u4 af[MB][2];
        float st[MB];
        for (int mb = 0; mb < MB; ++mb)
            for (int kk = 0; kk < 2; ++kk) af[mb][kk] = *(const u4*)(ls + ard[mb][kk]);
        for (int mb = 0; mb < MB; ++mb) st[mb] = lsa[slot * 64 + srd + mb * 16];
        float sm0[MB], sm1[MB];
        for (int mb = 0; mb < MB; ++mb) { sm0[mb] = st[mb] * sb0; sm1[mb] = G1 ? st[mb] * sb1 : sm0[mb]; }
        __builtin_amdgcn_sched_barrier(0);
        f4 tq[2][MB];
        for (int nb = 0; nb < 4; ++nb) {
            f4 (&tmp)[MB] = tq[nb & 1];
            f4 (&tp)[MB] = tq[(nb + 1) & 1];
            int slotm = 0;
            for (int k32 = 0; k32 < 4; ++k32) {
                const int kk = k32 >> 1;
                const long bv = (k32 & 1) ? hi8(bc[nb][kk]) : lo8(bc[nb][kk]);
                for (int mb = 0; mb < MB; ++mb) {
                    const long av = (k32 & 1) ? hi8(af[mb][kk]) : lo8(af[mb][kk]);
                    tmp[mb] = mma(bv, av, k32 == 0 ? f4{0.f, 0.f, 0.f, 0.f} : tmp[mb]);
                    {
                        constexpr int STRIDE = (IL_S * MB / 4) > 0 ? (IL_S * MB / 4) : 1;
                        constexpr int BPOS = (IL_B * MB / 4) > 0 ? (IL_B * MB / 4) : 1;
                        static_assert(BPOS + 7 * STRIDE < 16 * MB, "ILV positions must fall inside the MFMA stream");
                        const int g = nb * 4 * MB + k32 * MB + mb;
                        #pragma unroll
                        for (int i = 0; i < 8; ++i)
                            if (g == BPOS + i * STRIDE) {
                                __builtin_amdgcn_sched_barrier(0);
                                if (i & 1) gl4_1k(bn[i >> 1][1], bvo, bp[i >> 1] + oI); else gl4(bn[i >> 1][0], bvo, bp[i >> 1] + oI);
                            }
                    }
                    ++slotm;
                    if (nb > 0 && (slotm & 1) == 0 && (slotm >> 1) <= MB) {
                        const int j = (slotm >> 1) - 1;
                        const float* smp = (G1 && (nb - 1) >= 2) ? sm1 : sm0;
#if V70_E3_FOLD
                        r2x_fold4(acc[nb - 1][j], tp[j], smp[j]);
#else
                        acc[nb - 1][j] += tp[j] * smp[j];
#endif
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
        }
        {
            const float* smp = G1 ? sm1 : sm0;
#if V70_E3_FOLD
            for (int mb = 0; mb < MB; ++mb) r2x_fold4(acc[3][mb], tq[1][mb], smp[mb]);
#else
            for (int mb = 0; mb < MB; ++mb) acc[3][mb] += tq[1][mb] * smp[mb];
#endif
        }
        __builtin_amdgcn_sched_barrier(0);
        {
            const int ws = (slot + 2) >= 3 ? slot - 1 : slot + 2;
            vmwait3<8>(v0, v1, s2);                                   // A(k+2) landed (8 B loads after it)
            if (lr < BM) { *(u4*)(aW0 + ws * STA) = v0; *(u4*)(aW1 + ws * STA) = v1; }
            if (tid < BM) lsa[ws * 64 + tid] = s2;
        }
        bar_s<SW>(sbn0, sbn1);                                        // barriers #2 .. #nk+1
        slot = (slot == 2) ? 0 : slot + 1;
    };
    for (int k = 0; k < nk; k += 2) {
        step(k, bA, bB);
        if (k + 1 < nk) step(k + 1, bB, bA);
    }
    vmwait8<0>(bA); vmwait8<0>(bB);

    if (G1) {
        const int le = lane_fresh();
        float h[2][MB][4];
        float am[MB];
        for (int mb = 0; mb < MB; ++mb) {
            float m = 0.f;
            for (int j = 0; j < 2; ++j)
                for (int v = 0; v < 4; ++v) {
                    const float g = acc[j][mb][v], u = acc[2 + j][mb][v];
                    const float x = bf16r(g * u * __builtin_amdgcn_rcpf(1.f + __expf(-g)));
                    h[j][mb][v] = x;
                    m = fmaxf(m, fabsf(x));
                }
            m = fmaxf(m, xor_fresh(m, 16));
            m = fmaxf(m, xor_fresh(m, 32));
            am[mb] = m;
        }
        if (le < 16)
            for (int mb = 0; mb < MB; ++mb) lrm[wn * 64 + mb * 16 + le] = am[mb];
        bar<SW>();                                                    // barrier #nk+2 (GEMM1 only)
        uint8_t* q = (uint8_t*)out;
        const unsigned NS = NO >> 7;
#if V70_E3_SC1_NOBW
        const WeaveI32x4 rq = weave_make_buf_rsrc(q, route_rows * NO);
        const WeaveI32x4 rs = weave_make_buf_rsrc(oscale, route_rows * NS * 4u);
#endif
        for (int mb = 0; mb < MB; ++mb) {
            const int row = mb * 16 + (le & 15);
            const float a = fmaxf(fmaxf(lrm[row], lrm[64 + row]), fmaxf(lrm[128 + row], lrm[192 + row]));
            const float sc = fmaxf(a, 1e-10f) / 240.f;
            const float inv = 1.f / sc;
            const int prow = rtab()[row].x;
            if (row < valid && (unsigned)prow < route_rows) {
                for (int j = 0; j < 2; ++j) {
                    float y[4];
                    for (int v = 0; v < 4; ++v) y[v] = fminf(fmaxf(h[j][mb][v] * inv, -240.f), 240.f);
                    int pk = __builtin_amdgcn_cvt_pk_fp8_f32(y[0], y[1], 0, false);
                    pk = __builtin_amdgcn_cvt_pk_fp8_f32(y[2], y[3], pk, true);
#if V70_E3_SC1_NOBW
                    const int32_t qoff = (int32_t)((size_t)prow * NO + nbk * 128 + wn * 32 + j * 16 + (le >> 4) * 4);
                    v70_raw_buf_store_i32((int32_t)pk, rq, qoff, 0, 16);
#else
                    *(uint32_t*)(q + (size_t)prow * NO + nbk * 128 + wn * 32 + j * 16 + (le >> 4) * 4) = (uint32_t)pk;
#endif
                }
#if V70_E3_SC1_NOBW
                if (wn == 0 && le < 16)
                    v70_raw_buf_store_i32(__float_as_int(sc), rs, (int32_t)(((size_t)prow * NS + nbk) * 4u), 0, 16);
#else
                if (wn == 0 && le < 16) oscale[(size_t)prow * NS + nbk] = sc;
#endif
            }
        }
    } else {
        asm volatile("" ::: "memory");
        uint16_t* o = (uint16_t*)out;
#if V70_E3_SC1_NOBW
        const WeaveI32x4 ro = weave_make_buf_rsrc(o, route_rows * NO * 2u);
#endif
        const int le = lane_fresh();
        for (int mb = 0; mb < MB; ++mb) {
            const int row = mb * 16 + (le & 15);
            const int4 rr = rtab()[row];
            const unsigned tok = (unsigned)rr.y;
            if (row < valid && tok < route_rows && (unsigned)rr.x < route_rows) {
                const float w = __int_as_float(rr.z);
                uint16_t* dp = nullptr;
                if (row_dir != nullptr) {
                    dp = weave_direct_row_dst(row_dir, peer_tab, node_tab, node_row_base, my_node, tok, NO);
                }
#if V70_E3_SC1_NOBW
                if (dp == nullptr) {
                    const size_t base_el = (size_t)tok * NO + nbk * 256 + wn * 64 + (le >> 4) * 4;
                    for (int nb = 0; nb < 4; ++nb) {
                        const f4 c = acc[nb][mb] * w;
                        WeaveI32x2 pk;
                        pk[0] = (int32_t)(bf16bits(c[0]) | (bf16bits(c[1]) << 16));
                        pk[1] = (int32_t)(bf16bits(c[2]) | (bf16bits(c[3]) << 16));
                        v70_raw_buf_store_i32x2(pk, ro, (int32_t)((base_el + nb * 16) * 2u), 0, 16);
                    }
                } else
#endif
                {
                    uint16_t* dst = (dp != nullptr ? dp : (o + (size_t)tok * NO)) + nbk * 256 + wn * 64 + (le >> 4) * 4;
                    for (int nb = 0; nb < 4; ++nb) {
                        const f4 c = acc[nb][mb] * w;
                        u2 pk;
                        pk.x = bf16bits(c[0]) | (bf16bits(c[1]) << 16);
                        pk.y = bf16bits(c[2]) | (bf16bits(c[3]) << 16);
                        *(u2*)(dst + nb * 16) = pk;
                    }
                }
            }
        }
    }
}

// M-clip dispatch: wave-uniform branch once per work unit (no inner-loop divergence).
template <int G1, int SW = 0>
R2X_DEV void unit(uint8_t* __restrict__ lds, const int gtid,
                  const int valid, const int nbk, const int e,
                  const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                  const float* __restrict__ sa, const float* __restrict__ sb,
                  void* __restrict__ out, float* __restrict__ oscale, const float* __restrict__ wts,
                  const unsigned K, const unsigned NO, const unsigned route_rows,
                  const unsigned* row_dir, const uint64_t* peer_tab, const uint64_t* node_tab,
                  unsigned node_row_base, unsigned my_node)
{
#define R2X_FWD lds, gtid, valid, nbk, e, A, B, sa, sb, out, oscale, wts, K, NO, route_rows, \
                row_dir, peer_tab, node_tab, node_row_base, my_node
    const int mbsel = __builtin_amdgcn_readfirstlane((valid + 15) >> 4);
    if (false) {}
#if R2X_MBSET & 1
    else if (mbsel <= 1) body<1, G1, SW>(R2X_FWD);
#endif
#if R2X_MBSET & 2
    else if (mbsel == 2) body<2, G1, SW>(R2X_FWD);
#endif
#if R2X_MBSET & 4
    else if (mbsel == 3) body<3, G1, SW>(R2X_FWD);
#endif
    else body<4, G1, SW>(R2X_FWD);
#undef R2X_FWD
}
}  // namespace r2x
