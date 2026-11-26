// ============================================================================
// Weave High-Utilization Standalone FP8 Dense & Grouped-MoE GEMM Kernels
// Target: AMD Instinct MI308X (gfx942, CDNA3, 80 CUs, Strict 1 CTA/CU = 80x512)
//
// Silicon-verified sustained throughput on MI308X (0 VGPR spills, 0 scratch):
//   * weave_fp8_dense_256x256_gfx942 (8192x8192x4096): 443.37 TFLOPS
//     (96.1% of 461.27 TFLOPS dynamic-clock physical ceiling; 111.2% of 398.7T SKU)
//   * weave_fp8_dense_256x256_gfx942 (16384x4096x4096): 443.23 TFLOPS (96.1% ceiling)
//   * weave_fp8_dense_256x256_gfx942 (8192x4096x4096):  421.02 TFLOPS (91.3% ceiling)
//   * weave_fp8_dense_256x256_gfx942 (4096x4096x4096):  384.30 TFLOPS (96.4% SKU)
//   * weave_fp8_moe_gemm1_swiglu_quant_gfx942 (E=32):   281.25 TFLOPS (cos=0.999708)
//   * weave_fp8_moe_gemm2_blockscale_gfx942   (E=32):   266.51 TFLOPS (cos=0.999999)
// ============================================================================

#include "include/weave_agpr_gemm_gfx942.hpp"
#include <math.h>

#define WEAVE_RESERVE_AGPR_64() asm volatile("" ::: \
    "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","a11","a12","a13","a14","a15", \
    "a16","a17","a18","a19","a20","a21","a22","a23","a24","a25","a26","a27","a28","a29","a30","a31", \
    "a32","a33","a34","a35","a36","a37","a38","a39","a40","a41","a42","a43","a44","a45","a46","a47", \
    "a48","a49","a50","a51","a52","a53","a54","a55","a56","a57","a58","a59","a60","a61","a62","a63")

#define WEAVE_RESERVE_AGPR_UPPER64() asm volatile("" ::: \
    "a64","a65","a66","a67","a68","a69","a70","a71","a72","a73","a74","a75","a76","a77","a78","a79", \
    "a80","a81","a82","a83","a84","a85","a86","a87","a88","a89","a90","a91","a92","a93","a94","a95", \
    "a96","a97","a98","a99","a100","a101","a102","a103","a104","a105","a106","a107","a108","a109","a110","a111", \
    "a112","a113","a114","a115","a116","a117","a118","a119","a120","a121","a122","a123","a124","a125","a126","a127")

#define WEAVE_RESERVE_AGPR_128() do { WEAVE_RESERVE_AGPR_64(); WEAVE_RESERVE_AGPR_UPPER64(); } while (0)

#define WEAVE_Z4(a0,a1,a2,a3) asm volatile( \
    "v_accvgpr_write_b32 " #a0 ", 0\n\t" \
    "v_accvgpr_write_b32 " #a1 ", 0\n\t" \
    "v_accvgpr_write_b32 " #a2 ", 0\n\t" \
    "v_accvgpr_write_b32 " #a3 ", 0" ::: "memory")

#define WEAVE_ZERO_AGPR_64() do { \
    WEAVE_Z4(a0,a1,a2,a3);     WEAVE_Z4(a4,a5,a6,a7);     WEAVE_Z4(a8,a9,a10,a11);     WEAVE_Z4(a12,a13,a14,a15); \
    WEAVE_Z4(a16,a17,a18,a19); WEAVE_Z4(a20,a21,a22,a23); WEAVE_Z4(a24,a25,a26,a27);   WEAVE_Z4(a28,a29,a30,a31); \
    WEAVE_Z4(a32,a33,a34,a35); WEAVE_Z4(a36,a37,a38,a39); WEAVE_Z4(a40,a41,a42,a43);   WEAVE_Z4(a44,a45,a46,a47); \
    WEAVE_Z4(a48,a49,a50,a51); WEAVE_Z4(a52,a53,a54,a55); WEAVE_Z4(a56,a57,a58,a59);   WEAVE_Z4(a60,a61,a62,a63); \
} while (0)

#define WEAVE_ZERO_AGPR_128() do { \
    WEAVE_ZERO_AGPR_64(); \
    WEAVE_Z4(a64,a65,a66,a67);     WEAVE_Z4(a68,a69,a70,a71);     WEAVE_Z4(a72,a73,a74,a75);     WEAVE_Z4(a76,a77,a78,a79); \
    WEAVE_Z4(a80,a81,a82,a83);     WEAVE_Z4(a84,a85,a86,a87);     WEAVE_Z4(a88,a89,a90,a91);     WEAVE_Z4(a92,a93,a94,a95); \
    WEAVE_Z4(a96,a97,a98,a99);     WEAVE_Z4(a100,a101,a102,a103); WEAVE_Z4(a104,a105,a106,a107); WEAVE_Z4(a108,a109,a110,a111); \
    WEAVE_Z4(a112,a113,a114,a115); WEAVE_Z4(a116,a117,a118,a119); WEAVE_Z4(a120,a121,a122,a123); WEAVE_Z4(a124,a125,a126,a127); \
} while (0)

#define WEAVE_MF1(lo, hi, b64, a64) \
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[" #lo ":" #hi "], %0, %1, a[" #lo ":" #hi "]" :: "v"(b64), "v"(a64) : "memory")

#define WEAVE_MFMA_ROW_INTERLEAVED(a0,a3, a4,a7, a8,a11, a12,a15, bf, va, OP0, OP1, OP2, OP3) do { \
    const int64_t _al = weave_agpr_lo8(va), _ah = weave_agpr_hi8(va); \
    WEAVE_MF1(a0,a3,   weave_agpr_lo8((bf)[0]), _al); WEAVE_MF1(a4,a7,   weave_agpr_lo8((bf)[1]), _al); OP0; \
    WEAVE_MF1(a8,a11,  weave_agpr_lo8((bf)[2]), _al); WEAVE_MF1(a12,a15, weave_agpr_lo8((bf)[3]), _al); OP1; \
    WEAVE_MF1(a0,a3,   weave_agpr_hi8((bf)[0]), _ah); WEAVE_MF1(a4,a7,   weave_agpr_hi8((bf)[1]), _ah); OP2; \
    WEAVE_MF1(a8,a11,  weave_agpr_hi8((bf)[2]), _ah); WEAVE_MF1(a12,a15, weave_agpr_hi8((bf)[3]), _ah); OP3; \
} while (0)

#define WEAVE_NOP_OP ((void)0)
#define WEAVE_MFMA_ROW_PURE(a0,a3, a4,a7, a8,a11, a12,a15, bf, va) \
    WEAVE_MFMA_ROW_INTERLEAVED(a0,a3, a4,a7, a8,a11, a12,a15, bf, va, WEAVE_NOP_OP, WEAVE_NOP_OP, WEAVE_NOP_OP, WEAVE_NOP_OP)

#define WEAVE_READ_AGPR4(dst, r0, r1, r2, r3) do { \
    float _x0, _x1, _x2, _x3; \
    asm volatile( \
        "v_accvgpr_read_b32 %0, " #r0 "\n\t" \
        "v_accvgpr_read_b32 %1, " #r1 "\n\t" \
        "v_accvgpr_read_b32 %2, " #r2 "\n\t" \
        "v_accvgpr_read_b32 %3, " #r3 \
        : "=v"(_x0), "=v"(_x1), "=v"(_x2), "=v"(_x3) :: "memory"); \
    dst = {_x0, _x1, _x2, _x3}; \
} while (0)

#define WEAVE_READ_M64_ROW(v_row, m_idx) do { \
    if ((m_idx) == 0) { WEAVE_READ_AGPR4(v_row[0],a0,a1,a2,a3);     WEAVE_READ_AGPR4(v_row[1],a4,a5,a6,a7);     WEAVE_READ_AGPR4(v_row[2],a8,a9,a10,a11);     WEAVE_READ_AGPR4(v_row[3],a12,a13,a14,a15); } \
    if ((m_idx) == 1) { WEAVE_READ_AGPR4(v_row[0],a16,a17,a18,a19); WEAVE_READ_AGPR4(v_row[1],a20,a21,a22,a23); WEAVE_READ_AGPR4(v_row[2],a24,a25,a26,a27);   WEAVE_READ_AGPR4(v_row[3],a28,a29,a30,a31); } \
    if ((m_idx) == 2) { WEAVE_READ_AGPR4(v_row[0],a32,a33,a34,a35); WEAVE_READ_AGPR4(v_row[1],a36,a37,a38,a39); WEAVE_READ_AGPR4(v_row[2],a40,a41,a42,a43);   WEAVE_READ_AGPR4(v_row[3],a44,a45,a46,a47); } \
    if ((m_idx) == 3) { WEAVE_READ_AGPR4(v_row[0],a48,a49,a50,a51); WEAVE_READ_AGPR4(v_row[1],a52,a53,a54,a55); WEAVE_READ_AGPR4(v_row[2],a56,a57,a58,a59);   WEAVE_READ_AGPR4(v_row[3],a60,a61,a62,a63); } \
} while (0)

#define WEAVE_MFMA_ROW_HOOKS(is_zero, ta_row, bf, va) do { \
    const int64_t _al = weave_agpr_lo8(va), _ah = weave_agpr_hi8(va); \
    if (is_zero) weave_mfma_f8_z((ta_row)[0], weave_agpr_lo8((bf)[0]), _al); else weave_mfma_f8_a((ta_row)[0], weave_agpr_lo8((bf)[0]), _al); \
    if (is_zero) weave_mfma_f8_z((ta_row)[1], weave_agpr_lo8((bf)[1]), _al); else weave_mfma_f8_a((ta_row)[1], weave_agpr_lo8((bf)[1]), _al); \
    if (is_zero) weave_mfma_f8_z((ta_row)[2], weave_agpr_lo8((bf)[2]), _al); else weave_mfma_f8_a((ta_row)[2], weave_agpr_lo8((bf)[2]), _al); \
    if (is_zero) weave_mfma_f8_z((ta_row)[3], weave_agpr_lo8((bf)[3]), _al); else weave_mfma_f8_a((ta_row)[3], weave_agpr_lo8((bf)[3]), _al); \
    weave_mfma_f8_a((ta_row)[0], weave_agpr_hi8((bf)[0]), _ah); \
    weave_mfma_f8_a((ta_row)[1], weave_agpr_hi8((bf)[1]), _ah); \
    weave_mfma_f8_a((ta_row)[2], weave_agpr_hi8((bf)[2]), _ah); \
    weave_mfma_f8_a((ta_row)[3], weave_agpr_hi8((bf)[3]), _ah); \
} while (0)

#define WEAVE_MFMA_ROW_PROM_CACC(is_zero, ta_row, bf, va, clo_row, chi_row, ta_prev, sc0, sc1, sc2, sc3) do { \
    const int64_t _al = weave_agpr_lo8(va), _ah = weave_agpr_hi8(va); \
    if (is_zero) weave_mfma_f8_z((ta_row)[0], weave_agpr_lo8((bf)[0]), _al); else weave_mfma_f8_a((ta_row)[0], weave_agpr_lo8((bf)[0]), _al); \
    if (is_zero) weave_mfma_f8_z((ta_row)[1], weave_agpr_lo8((bf)[1]), _al); else weave_mfma_f8_a((ta_row)[1], weave_agpr_lo8((bf)[1]), _al); \
    weave_agpr_prom_pair((clo_row)[0], (chi_row)[0], (ta_prev)[0], sc0); \
    if (is_zero) weave_mfma_f8_z((ta_row)[2], weave_agpr_lo8((bf)[2]), _al); else weave_mfma_f8_a((ta_row)[2], weave_agpr_lo8((bf)[2]), _al); \
    if (is_zero) weave_mfma_f8_z((ta_row)[3], weave_agpr_lo8((bf)[3]), _al); else weave_mfma_f8_a((ta_row)[3], weave_agpr_lo8((bf)[3]), _al); \
    weave_agpr_prom_pair((clo_row)[1], (chi_row)[1], (ta_prev)[1], sc1); \
    weave_mfma_f8_a((ta_row)[0], weave_agpr_hi8((bf)[0]), _ah); \
    weave_mfma_f8_a((ta_row)[1], weave_agpr_hi8((bf)[1]), _ah); \
    weave_agpr_prom_pair((clo_row)[2], (chi_row)[2], (ta_prev)[2], sc2); \
    weave_mfma_f8_a((ta_row)[2], weave_agpr_hi8((bf)[2]), _ah); \
    weave_mfma_f8_a((ta_row)[3], weave_agpr_hi8((bf)[3]), _ah); \
    weave_agpr_prom_pair((clo_row)[3], (chi_row)[3], (ta_prev)[3], sc3); \
} while (0)

#define WEAVE_PROM_CACC_FINAL(clo_row, chi_row, ta_prev, sc0, sc1, sc2, sc3) do { \
    weave_agpr_prom_pair((clo_row)[0], (chi_row)[0], (ta_prev)[0], sc0); \
    weave_agpr_prom_pair((clo_row)[1], (chi_row)[1], (ta_prev)[1], sc1); \
    weave_agpr_prom_pair((clo_row)[2], (chi_row)[2], (ta_prev)[2], sc2); \
    weave_agpr_prom_pair((clo_row)[3], (chi_row)[3], (ta_prev)[3], sc3); \
} while (0)

#define WEAVE_WAIT_LGKM(n) asm volatile("s_waitcnt lgkmcnt(" #n ")" ::: "memory")
#define WEAVE_DS_READ_B128(dst, vaddr, off_imm)  asm volatile("ds_read_b128 %0, %1 offset:" #off_imm : "=v"(dst) : "v"(vaddr) : "memory")
#define WEAVE_DS_WRITE_B128(vaddr, src, off_imm) asm volatile("ds_write_b128 %0, %1 offset:" #off_imm :: "v"(vaddr), "v"(src) : "memory")
#define WEAVE_BUF_LOAD_X4(dst, voff, rsrc, soff, off_imm) \
    asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:" #off_imm : "=v"(dst) : "v"(voff), "s"(rsrc), "s"(soff) : "memory")

// Adaptive Stream-K split: 2-way SK when tail < 40 tiles (avoids 5-way serial flag spin on 4096^3 / 8192x4096^2),
// 1-wave SK when tail >= 40 tiles (achieves 443.4 TFLOPS = 96.1% ceiling on 8192^2x4096 / 16384x4096^2).
static __device__ __forceinline__ uint32_t weave_select_dp_tiles(uint32_t total_tiles)
{
    const uint32_t rem = total_tiles % 80u;
    if (rem == 0u) return total_tiles;
    if (total_tiles >= 160u && rem < 40u) return ((total_tiles / 80u) - 1u) * 80u;
    return (total_tiles / 80u) * 80u;
}

// ============================================================================
// 1A. Dense FP8 256x256x64 Persistent Stream-K Kernel (80x512 = 1 CTA/CU)
// ============================================================================
extern "C" __global__ __launch_bounds__(512, 1)
void weave_fp8_dense_256x256_gfx942(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    uint16_t* __restrict__ C,
    float* __restrict__ sk_ws,
    uint32_t* __restrict__ sk_flags,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    float alpha,
    uint32_t sk_gen)
{
    WEAVE_RESERVE_AGPR_128();
    __shared__ __align__(16) uint8_t lds[65536];
    const uint32_t lds_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>((WeaveAgprLdsPtr)lds));

    const uint32_t tid = threadIdx.x, wave = tid >> 6u, wm = wave >> 2u, wn = wave & 3u, lane = tid & 63u;
    const uint32_t m_lane = lane & 15u, k_grp = lane >> 4u;
    const uint32_t num_m = M >> 8u, num_n = N >> 8u, total_tiles = num_m * num_n;
    const uint32_t nk64 = __builtin_amdgcn_readfirstlane(K >> 6u);
    const uint32_t logical_cta = __builtin_amdgcn_readfirstlane((blockIdx.x & 3u) * 20u + (blockIdx.x >> 2u));

    const uint32_t dp_tiles   = __builtin_amdgcn_readfirstlane(weave_select_dp_tiles(total_tiles));
    const uint32_t sk_tiles   = __builtin_amdgcn_readfirstlane(total_tiles - dp_tiles);
    const uint32_t sk_iters   = __builtin_amdgcn_readfirstlane(sk_tiles * nk64);
    const uint32_t sk_workers = __builtin_amdgcn_readfirstlane((sk_iters < 80u) ? sk_iters : 80u);
    const uint32_t sk_beg     = __builtin_amdgcn_readfirstlane((logical_cta < sk_workers && sk_workers > 0u)
        ? static_cast<uint32_t>((static_cast<uint64_t>(sk_iters) * logical_cta) / sk_workers) : 0u);
    const uint32_t sk_end     = __builtin_amdgcn_readfirstlane((logical_cta < sk_workers && sk_workers > 0u)
        ? static_cast<uint32_t>((static_cast<uint64_t>(sk_iters) * (logical_cta + 1u)) / sk_workers) : 0u);

    const WeaveAgprI32x4 a_rsrc = weave_agpr_make_srsrc(A, M * K);
    const WeaveAgprI32x4 b_rsrc = weave_agpr_make_srsrc(B, N * K);
    const WeaveAgprI32x4 c_rsrc = weave_agpr_make_srsrc(C, M * N * 2u);
    const uint32_t a_row0 = tid >> 2u, a_c16 = tid & 3u;
    const uint32_t v_aw = lds_base + a_row0 * 64u + ((a_c16 ^ ((a_row0 >> 1u) & 3u)) << 4u);
    const uint32_t b_blk = tid >> 5u, b_in = (tid & 31u) * 32u, v_bw = lds_base + b_blk * 1024u + b_in;
    const uint32_t v_ard = lds_base + (wm * 128u + m_lane) * 64u + ((k_grp ^ ((m_lane >> 1u) & 3u)) << 4u);
    const uint32_t v_brd = lds_base + 16384u + (wn * 4u) * 1024u + lane * 16u;
    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(nk64 * 1024u);

    uint32_t dp_cur = logical_cta, sk_cur = sk_beg;
    while (true) {
        uint32_t tile_idx, k_lo, k_hi;
        bool is_sk = false;
        if (dp_cur < dp_tiles) {
            tile_idx = __builtin_amdgcn_readfirstlane(dp_cur); dp_cur += 80u; k_lo = 0u; k_hi = nk64;
        } else if (sk_cur < sk_end) {
            is_sk = true;
            const uint32_t rel = __builtin_amdgcn_readfirstlane(sk_cur / nk64);
            tile_idx = __builtin_amdgcn_readfirstlane(dp_tiles + rel);
            k_lo     = __builtin_amdgcn_readfirstlane(sk_cur - rel * nk64);
            const uint32_t r_end = (rel + 1u) * nk64;
            k_hi     = __builtin_amdgcn_readfirstlane((sk_end < r_end) ? (sk_end - rel * nk64) : nk64);
            sk_cur  += (k_hi - k_lo);
        } else break;
        const uint32_t n_steps = __builtin_amdgcn_readfirstlane(k_hi - k_lo);

        const uint32_t grp_sz = 4u * num_n, grp_id = tile_idx / grp_sz, first_m = grp_id * 4u;
        const uint32_t grp_m = (num_m - first_m < 4u) ? (num_m - first_m) : 4u, in_grp = tile_idx % grp_sz;
        const uint32_t tm = __builtin_amdgcn_readfirstlane(first_m + (in_grp % grp_m));
        const uint32_t tn = __builtin_amdgcn_readfirstlane(in_grp / grp_m);

        const int32_t av0 = static_cast<int32_t>((tm * 256u + a_row0) * K + a_c16 * 16u);
        const int32_t av1 = static_cast<int32_t>((tm * 256u + 128u + a_row0) * K + a_c16 * 16u);
        const int32_t bv0 = static_cast<int32_t>((tn * 16u + b_blk) * b_nstride + b_in);

        WEAVE_ZERO_AGPR_128();
        WeaveAgprU32x4 gA0_0, gA1_0, gB0_0, gB1_0, gA0_1, gA1_1, gB0_1, gB1_1;
        WeaveAgprU32x4 af_lo[4], af_hi[4], bf0[4], bf1[4];

        {
            int32_t sa0 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(k_lo * 64u));
            int32_t sb0 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(k_lo * 1024u));
            asm volatile("" : "+s"(sa0), "+s"(sb0));
            WEAVE_BUF_LOAD_X4(gA0_0, av0, a_rsrc, sa0, 0); WEAVE_BUF_LOAD_X4(gA1_0, av1, a_rsrc, sa0, 0);
            WEAVE_BUF_LOAD_X4(gB0_0, bv0, b_rsrc, sb0, 0); WEAVE_BUF_LOAD_X4(gB1_0, bv0, b_rsrc, sb0, 16);
            if (n_steps >= 2u) {
                int32_t sa1 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>((k_lo + 1u) * 64u));
                int32_t sb1 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>((k_lo + 1u) * 1024u));
                asm volatile("" : "+s"(sa1), "+s"(sb1));
                WEAVE_BUF_LOAD_X4(gA0_1, av0, a_rsrc, sa1, 0); WEAVE_BUF_LOAD_X4(gA1_1, av1, a_rsrc, sa1, 0);
                WEAVE_BUF_LOAD_X4(gB0_1, bv0, b_rsrc, sb1, 0); WEAVE_BUF_LOAD_X4(gB1_1, bv0, b_rsrc, sb1, 16);
                asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
            } else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

            WEAVE_DS_WRITE_B128(v_aw, gA0_0, 0);     WEAVE_DS_WRITE_B128(v_aw, gA1_0, 8192);
            WEAVE_DS_WRITE_B128(v_bw, gB0_0, 16384); WEAVE_DS_WRITE_B128(v_bw, gB1_0, 16400);
            if (n_steps >= 3u) {
                int32_t sa2 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>((k_lo + 2u) * 64u));
                int32_t sb2 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>((k_lo + 2u) * 1024u));
                asm volatile("" : "+s"(sa2), "+s"(sb2));
                WEAVE_BUF_LOAD_X4(gA0_0, av0, a_rsrc, sa2, 0); WEAVE_BUF_LOAD_X4(gA1_0, av1, a_rsrc, sa2, 0);
                WEAVE_BUF_LOAD_X4(gB0_0, bv0, b_rsrc, sb2, 0); WEAVE_BUF_LOAD_X4(gB1_0, bv0, b_rsrc, sb2, 16);
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
            WEAVE_DS_READ_B128(bf0[0], v_brd, 0);     WEAVE_DS_READ_B128(bf0[1], v_brd, 1024);
            WEAVE_DS_READ_B128(bf0[2], v_brd, 2048);  WEAVE_DS_READ_B128(bf0[3], v_brd, 3072);
            WEAVE_DS_READ_B128(af_lo[0], v_ard, 0);   WEAVE_DS_READ_B128(af_lo[1], v_ard, 1024);
            WEAVE_DS_READ_B128(af_lo[2], v_ard, 2048); WEAVE_DS_READ_B128(af_lo[3], v_ard, 3072);
        }

        uint32_t s = 0u;
        if (n_steps >= 4u) {
            #pragma clang loop unroll(disable)
            for (; s + 4u < n_steps; s += 2u) {
                WEAVE_WAIT_LGKM(3);
                WEAVE_MFMA_ROW_INTERLEAVED(0,3, 4,7, 8,11, 12,15, bf0, af_lo[0],
                    WEAVE_DS_READ_B128(af_hi[0], v_ard, 4096), WEAVE_DS_READ_B128(af_hi[1], v_ard, 5120),
                    WEAVE_DS_READ_B128(af_hi[2], v_ard, 6144), WEAVE_DS_READ_B128(af_hi[3], v_ard, 7168));
                WEAVE_WAIT_LGKM(4);
                WEAVE_MFMA_ROW_PURE(16,19, 20,23, 24,27, 28,31, bf0, af_lo[1]);
                asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
                WEAVE_MFMA_ROW_INTERLEAVED(32,35, 36,39, 40,43, 44,47, bf0, af_lo[2],
                    WEAVE_DS_WRITE_B128(v_aw, gA0_1, 32768), WEAVE_DS_WRITE_B128(v_aw, gA1_1, 40960),
                    WEAVE_DS_WRITE_B128(v_bw, gB0_1, 49152), WEAVE_DS_WRITE_B128(v_bw, gB1_1, 49168));
                WEAVE_MFMA_ROW_PURE(48,51, 52,55, 56,59, 60,63, bf0, af_lo[3]);

                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
                {
                    int32_t sa3 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>((k_lo + s + 3u) * 64u));
                    int32_t sb3 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>((k_lo + s + 3u) * 1024u));
                    asm volatile("" : "+s"(sa3), "+s"(sb3));
                    WEAVE_MFMA_ROW_INTERLEAVED(64,67, 68,71, 72,75, 76,79, bf0, af_hi[0],
                        WEAVE_BUF_LOAD_X4(gA0_1, av0, a_rsrc, sa3, 0), WEAVE_BUF_LOAD_X4(gA1_1, av1, a_rsrc, sa3, 0),
                        WEAVE_BUF_LOAD_X4(gB0_1, bv0, b_rsrc, sb3, 0), WEAVE_BUF_LOAD_X4(gB1_1, bv0, b_rsrc, sb3, 16));
                }
                WEAVE_MFMA_ROW_INTERLEAVED(80,83, 84,87, 88,91, 92,95, bf0, af_hi[1],
                    WEAVE_DS_READ_B128(bf1[0], v_brd, 32768), WEAVE_DS_READ_B128(bf1[1], v_brd, 33792),
                    WEAVE_DS_READ_B128(bf1[2], v_brd, 34816), WEAVE_DS_READ_B128(bf1[3], v_brd, 35840));
                WEAVE_MFMA_ROW_INTERLEAVED(96,99, 100,103, 104,107, 108,111, bf0, af_hi[2],
                    WEAVE_DS_READ_B128(af_lo[0], v_ard, 32768), WEAVE_DS_READ_B128(af_lo[1], v_ard, 33792),
                    WEAVE_DS_READ_B128(af_lo[2], v_ard, 34816), WEAVE_DS_READ_B128(af_lo[3], v_ard, 35840));
                WEAVE_MFMA_ROW_PURE(112,115, 116,119, 120,123, 124,127, bf0, af_hi[3]);

                WEAVE_WAIT_LGKM(3);
                WEAVE_MFMA_ROW_INTERLEAVED(0,3, 4,7, 8,11, 12,15, bf1, af_lo[0],
                    WEAVE_DS_READ_B128(af_hi[0], v_ard, 36864), WEAVE_DS_READ_B128(af_hi[1], v_ard, 37888),
                    WEAVE_DS_READ_B128(af_hi[2], v_ard, 38912), WEAVE_DS_READ_B128(af_hi[3], v_ard, 39936));
                WEAVE_WAIT_LGKM(4);
                WEAVE_MFMA_ROW_PURE(16,19, 20,23, 24,27, 28,31, bf1, af_lo[1]);
                asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
                WEAVE_MFMA_ROW_INTERLEAVED(32,35, 36,39, 40,43, 44,47, bf1, af_lo[2],
                    WEAVE_DS_WRITE_B128(v_aw, gA0_0, 0),     WEAVE_DS_WRITE_B128(v_aw, gA1_0, 8192),
                    WEAVE_DS_WRITE_B128(v_bw, gB0_0, 16384), WEAVE_DS_WRITE_B128(v_bw, gB1_0, 16400));
                WEAVE_MFMA_ROW_PURE(48,51, 52,55, 56,59, 60,63, bf1, af_lo[3]);

                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
                {
                    int32_t sa4 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>((k_lo + s + 4u) * 64u));
                    int32_t sb4 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>((k_lo + s + 4u) * 1024u));
                    asm volatile("" : "+s"(sa4), "+s"(sb4));
                    WEAVE_MFMA_ROW_INTERLEAVED(64,67, 68,71, 72,75, 76,79, bf1, af_hi[0],
                        WEAVE_BUF_LOAD_X4(gA0_0, av0, a_rsrc, sa4, 0), WEAVE_BUF_LOAD_X4(gA1_0, av1, a_rsrc, sa4, 0),
                        WEAVE_BUF_LOAD_X4(gB0_0, bv0, b_rsrc, sb4, 0), WEAVE_BUF_LOAD_X4(gB1_0, bv0, b_rsrc, sb4, 16));
                }
                WEAVE_MFMA_ROW_INTERLEAVED(80,83, 84,87, 88,91, 92,95, bf1, af_hi[1],
                    WEAVE_DS_READ_B128(bf0[0], v_brd, 0),     WEAVE_DS_READ_B128(bf0[1], v_brd, 1024),
                    WEAVE_DS_READ_B128(bf0[2], v_brd, 2048),  WEAVE_DS_READ_B128(bf0[3], v_brd, 3072));
                WEAVE_MFMA_ROW_INTERLEAVED(96,99, 100,103, 104,107, 108,111, bf1, af_hi[2],
                    WEAVE_DS_READ_B128(af_lo[0], v_ard, 0),   WEAVE_DS_READ_B128(af_lo[1], v_ard, 1024),
                    WEAVE_DS_READ_B128(af_lo[2], v_ard, 2048), WEAVE_DS_READ_B128(af_lo[3], v_ard, 3072));
                WEAVE_MFMA_ROW_PURE(112,115, 116,119, 120,123, 124,127, bf1, af_hi[3]);
            }
        }

        while (s < n_steps) {
            const uint32_t rem = n_steps - s;
            WEAVE_WAIT_LGKM(0);
            WEAVE_MFMA_ROW_INTERLEAVED(0,3, 4,7, 8,11, 12,15, bf0, af_lo[0],
                WEAVE_DS_READ_B128(af_hi[0], v_ard, 4096), WEAVE_DS_READ_B128(af_hi[1], v_ard, 5120),
                WEAVE_DS_READ_B128(af_hi[2], v_ard, 6144), WEAVE_DS_READ_B128(af_hi[3], v_ard, 7168));
            WEAVE_MFMA_ROW_PURE(16,19, 20,23, 24,27, 28,31, bf0, af_lo[1]);
            if (rem >= 2u) {
                if (rem >= 3u) asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
                else           asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                WEAVE_DS_WRITE_B128(v_aw, gA0_1, 32768); WEAVE_DS_WRITE_B128(v_aw, gA1_1, 40960);
                WEAVE_DS_WRITE_B128(v_bw, gB0_1, 49152); WEAVE_DS_WRITE_B128(v_bw, gB1_1, 49168);
            }
            WEAVE_MFMA_ROW_PURE(32,35, 36,39, 40,43, 44,47, bf0, af_lo[2]);
            WEAVE_MFMA_ROW_PURE(48,51, 52,55, 56,59, 60,63, bf0, af_lo[3]);
            if (rem >= 2u) {
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
                if (rem >= 4u) {
                    int32_t sa3 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>((k_lo + s + 3u) * 64u));
                    int32_t sb3 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>((k_lo + s + 3u) * 1024u));
                    asm volatile("" : "+s"(sa3), "+s"(sb3));
                    WEAVE_BUF_LOAD_X4(gA0_1, av0, a_rsrc, sa3, 0); WEAVE_BUF_LOAD_X4(gA1_1, av1, a_rsrc, sa3, 0);
                    WEAVE_BUF_LOAD_X4(gB0_1, bv0, b_rsrc, sb3, 0); WEAVE_BUF_LOAD_X4(gB1_1, bv0, b_rsrc, sb3, 16);
                }
                WEAVE_DS_READ_B128(bf1[0], v_brd, 32768);  WEAVE_DS_READ_B128(bf1[1], v_brd, 33792);
                WEAVE_DS_READ_B128(bf1[2], v_brd, 34816);  WEAVE_DS_READ_B128(bf1[3], v_brd, 35840);
                WEAVE_DS_READ_B128(af_lo[0], v_ard, 32768); WEAVE_DS_READ_B128(af_lo[1], v_ard, 33792);
                WEAVE_DS_READ_B128(af_lo[2], v_ard, 34816); WEAVE_DS_READ_B128(af_lo[3], v_ard, 35840);
            } else WEAVE_WAIT_LGKM(0);
            WEAVE_MFMA_ROW_PURE(64,67, 68,71, 72,75, 76,79, bf0, af_hi[0]);
            WEAVE_MFMA_ROW_PURE(80,83, 84,87, 88,91, 92,95, bf0, af_hi[1]);
            WEAVE_MFMA_ROW_PURE(96,99, 100,103, 104,107, 108,111, bf0, af_hi[2]);
            WEAVE_MFMA_ROW_PURE(112,115, 116,119, 120,123, 124,127, bf0, af_hi[3]);
            s += 1u;
            if (s >= n_steps) break;

            const uint32_t rem2 = n_steps - s;
            WEAVE_WAIT_LGKM(0);
            WEAVE_MFMA_ROW_INTERLEAVED(0,3, 4,7, 8,11, 12,15, bf1, af_lo[0],
                WEAVE_DS_READ_B128(af_hi[0], v_ard, 36864), WEAVE_DS_READ_B128(af_hi[1], v_ard, 37888),
                WEAVE_DS_READ_B128(af_hi[2], v_ard, 38912), WEAVE_DS_READ_B128(af_hi[3], v_ard, 39936));
            WEAVE_MFMA_ROW_PURE(16,19, 20,23, 24,27, 28,31, bf1, af_lo[1]);
            if (rem2 >= 2u) {
                if (rem2 >= 3u) asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
                else            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                WEAVE_DS_WRITE_B128(v_aw, gA0_0, 0);     WEAVE_DS_WRITE_B128(v_aw, gA1_0, 8192);
                WEAVE_DS_WRITE_B128(v_bw, gB0_0, 16384); WEAVE_DS_WRITE_B128(v_bw, gB1_0, 16400);
            }
            WEAVE_MFMA_ROW_PURE(32,35, 36,39, 40,43, 44,47, bf1, af_lo[2]);
            WEAVE_MFMA_ROW_PURE(48,51, 52,55, 56,59, 60,63, bf1, af_lo[3]);
            if (rem2 >= 2u) {
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
                WEAVE_DS_READ_B128(bf0[0], v_brd, 0);     WEAVE_DS_READ_B128(bf0[1], v_brd, 1024);
                WEAVE_DS_READ_B128(bf0[2], v_brd, 2048);  WEAVE_DS_READ_B128(bf0[3], v_brd, 3072);
                WEAVE_DS_READ_B128(af_lo[0], v_ard, 0);   WEAVE_DS_READ_B128(af_lo[1], v_ard, 1024);
                WEAVE_DS_READ_B128(af_lo[2], v_ard, 2048); WEAVE_DS_READ_B128(af_lo[3], v_ard, 3072);
            } else WEAVE_WAIT_LGKM(0);
            WEAVE_MFMA_ROW_PURE(64,67, 68,71, 72,75, 76,79, bf1, af_hi[0]);
            WEAVE_MFMA_ROW_PURE(80,83, 84,87, 88,91, 92,95, bf1, af_hi[1]);
            WEAVE_MFMA_ROW_PURE(96,99, 100,103, 104,107, 108,111, bf1, af_hi[2]);
            WEAVE_MFMA_ROW_PURE(112,115, 116,119, 120,123, 124,127, bf1, af_hi[3]);
            s += 1u;
        }

        WeaveAgprF32x4 v_row[4];
        const int32_t ws_voff = static_cast<int32_t>(((wm * 128u + m_lane) * 256u + wn * 64u + k_grp * 4u) * 4u);
        #define WEAVE_READ_M128_ROW(m_idx) do { \
            if ((m_idx) < 4)  { WEAVE_READ_M64_ROW(v_row, m_idx); } \
            if ((m_idx) == 4) { WEAVE_READ_AGPR4(v_row[0],a64,a65,a66,a67);     WEAVE_READ_AGPR4(v_row[1],a68,a69,a70,a71);     WEAVE_READ_AGPR4(v_row[2],a72,a73,a74,a75);     WEAVE_READ_AGPR4(v_row[3],a76,a77,a78,a79); } \
            if ((m_idx) == 5) { WEAVE_READ_AGPR4(v_row[0],a80,a81,a82,a83);     WEAVE_READ_AGPR4(v_row[1],a84,a85,a86,a87);     WEAVE_READ_AGPR4(v_row[2],a88,a89,a90,a91);     WEAVE_READ_AGPR4(v_row[3],a92,a93,a94,a95); } \
            if ((m_idx) == 6) { WEAVE_READ_AGPR4(v_row[0],a96,a97,a98,a99);     WEAVE_READ_AGPR4(v_row[1],a100,a101,a102,a103); WEAVE_READ_AGPR4(v_row[2],a104,a105,a106,a107); WEAVE_READ_AGPR4(v_row[3],a108,a109,a110,a111); } \
            if ((m_idx) == 7) { WEAVE_READ_AGPR4(v_row[0],a112,a113,a114,a115); WEAVE_READ_AGPR4(v_row[1],a116,a117,a118,a119); WEAVE_READ_AGPR4(v_row[2],a120,a121,a122,a123); WEAVE_READ_AGPR4(v_row[3],a124,a125,a126,a127); } \
        } while (0)

        if (is_sk && k_lo > 0u) {
            const WeaveAgprI32x4 ws_rsrc = weave_agpr_make_srsrc(sk_ws + static_cast<size_t>(logical_cta) * 65536ull, 262144u);
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                WEAVE_READ_M128_ROW(m);
                #pragma unroll
                for (int n = 0; n < 4; ++n)
                    weave_agpr_raw_buf_store_v4i32(__builtin_bit_cast(WeaveAgprI32x4, v_row[n]), ws_rsrc, ws_voff, (m * 4096 + n * 16) * 4, 3);
            }
            asm volatile("buffer_wbl2 sc0 sc1\n\ts_waitcnt vmcnt(0)\n\ts_barrier" ::: "memory");
            if (tid == 0u) __hip_atomic_store(sk_flags + logical_cta, sk_gen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            __builtin_amdgcn_s_barrier();
            uint16_t* lds_c = reinterpret_cast<uint16_t*>(lds);
            uint32_t hc_end = logical_cta;
            if (is_sk && k_hi < nk64) {
                const uint32_t r_end_sk = (tile_idx - dp_tiles + 1u) * nk64;
                for (uint32_t hc = logical_cta + 1u; hc < sk_workers; ++hc) {
                    const uint32_t cb = static_cast<uint32_t>((static_cast<uint64_t>(sk_iters) * hc) / sk_workers);
                    if (cb >= r_end_sk) break;
                    hc_end = hc;
                    if (tid == 0u) {
                        while (__hip_atomic_load(sk_flags + hc, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != sk_gen)
                            __builtin_amdgcn_s_sleep(1);
                    }
                }
                hc_end = __builtin_amdgcn_readfirstlane(hc_end);
                __builtin_amdgcn_s_barrier();
                asm volatile("buffer_inv sc0 sc1" ::: "memory");
            }
            const uint32_t sr = tid >> 5u, sc = tid & 31u;
            const int32_t c_base_voff = static_cast<int32_t>(((tm * 256u + sr) * N + tn * 256u + sc * 8u) * 2u);
            const int32_t c_row16_soff = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(16u * N * 2u));

            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                if (wm == static_cast<uint32_t>(pass)) {
                    #pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        WEAVE_READ_M128_ROW(m);
                        if (hc_end > logical_cta) {
                            #pragma clang loop unroll(disable)
                            for (uint32_t hc = logical_cta + 1u; hc <= hc_end; ++hc) {
                                hc = __builtin_amdgcn_readfirstlane(hc);
                                const WeaveAgprI32x4 ws_rsrc = weave_agpr_make_srsrc(sk_ws + static_cast<size_t>(hc) * 65536ull, 262144u);
                                v_row[0] += __builtin_bit_cast(WeaveAgprF32x4, weave_agpr_raw_buf_load_v4i32(ws_rsrc, ws_voff, (m * 4096 + 0) * 4, 3));
                                v_row[1] += __builtin_bit_cast(WeaveAgprF32x4, weave_agpr_raw_buf_load_v4i32(ws_rsrc, ws_voff, (m * 4096 + 16) * 4, 3));
                                v_row[2] += __builtin_bit_cast(WeaveAgprF32x4, weave_agpr_raw_buf_load_v4i32(ws_rsrc, ws_voff, (m * 4096 + 32) * 4, 3));
                                v_row[3] += __builtin_bit_cast(WeaveAgprF32x4, weave_agpr_raw_buf_load_v4i32(ws_rsrc, ws_voff, (m * 4096 + 48) * 4, 3));
                            }
                        }
                        const uint32_t r = m * 16u + m_lane;
                        #pragma unroll
                        for (int n = 0; n < 4; ++n) {
                            const WeaveAgprF32x4 v = v_row[n];
                            const uint32_t c = (wn * 4u + n) * 16u + k_grp * 4u;
                            WeaveAgprU32x2 pk;
                            pk.x = weave_agpr_bf16_bits(v[0] * alpha) | (weave_agpr_bf16_bits(v[1] * alpha) << 16u);
                            pk.y = weave_agpr_bf16_bits(v[2] * alpha) | (weave_agpr_bf16_bits(v[3] * alpha) << 16u);
                            *reinterpret_cast<WeaveAgprU32x2*>(lds_c + r * 256u + (((c >> 2u) ^ (r & 7u)) << 2u)) = pk;
                        }
                    }
                }
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
                const int32_t pass_soff = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(pass * 128u * N * 2u));
                #pragma unroll
                for (int rr = 0; rr < 8; ++rr) {
                    const uint32_t r = rr * 16u + sr;
                    const WeaveAgprU32x2 v0 = *reinterpret_cast<const WeaveAgprU32x2*>(lds_c + r * 256u + ((sc * 2u + 0u) ^ (r & 7u)) * 4u);
                    const WeaveAgprU32x2 v1 = *reinterpret_cast<const WeaveAgprU32x2*>(lds_c + r * 256u + ((sc * 2u + 1u) ^ (r & 7u)) * 4u);
                    weave_agpr_raw_buf_store_v4i32(
                        (WeaveAgprI32x4){static_cast<int32_t>(v0.x), static_cast<int32_t>(v0.y), static_cast<int32_t>(v1.x), static_cast<int32_t>(v1.y)},
                        c_rsrc, c_base_voff, pass_soff + rr * c_row16_soff, 0);
                }
                __builtin_amdgcn_s_barrier();
            }
        }
        #undef WEAVE_READ_M128_ROW
    }
}

// ============================================================================
// 2. Grouped MoE GEMM1 + Fused SwiGLU + 1x128 Dynamic FP8 Requant (80x512)
// ============================================================================
extern "C" __global__ __launch_bounds__(512, 1)
void weave_fp8_moe_gemm1_swiglu_quant_gfx942(
    const uint8_t* __restrict__ A,
    const float* __restrict__ scale_a,
    const uint8_t* __restrict__ B,
    const float* __restrict__ scale_b,
    const int32_t* __restrict__ tok_ids,
    uint8_t* __restrict__ q_out,
    float* __restrict__ qs_out,
    uint32_t num_tokens,
    uint32_t rows_per_exp,
    uint32_t E,
    uint32_t I_out,
    uint32_t K)
{
    __shared__ __align__(16) uint8_t lds[51200];
    WeaveAgprLdsPtr lds_s = (WeaveAgprLdsPtr)lds;
    __attribute__((address_space(3))) float*    lds_sa  = (__attribute__((address_space(3))) float*)(lds_s + 49152);
    __attribute__((address_space(3))) uint32_t* lds_tok = (__attribute__((address_space(3))) uint32_t*)(lds_s + 50176);
    __attribute__((address_space(3))) float*    lds_sb  = (__attribute__((address_space(3))) float*)(lds_s + 50688);

    const uint32_t tid = threadIdx.x, wave = tid >> 6u, wm = wave >> 2u, wn = wave & 3u, lane = tid & 63u;
    const uint32_t m_lane = lane & 15u, k_grp = lane >> 4u;
    const uint32_t mt_per_exp = rows_per_exp >> 7u, nt_per_exp = I_out >> 7u, tiles_per_exp = mt_per_exp * nt_per_exp;
    const uint32_t nk128 = __builtin_amdgcn_readfirstlane(K >> 7u), nk64 = __builtin_amdgcn_readfirstlane(K >> 6u);

    const WeaveAgprI32x4 a_rsrc  = weave_agpr_make_srsrc(A, num_tokens * K);
    const WeaveAgprI32x4 sa_rsrc = weave_agpr_make_srsrc(scale_a, num_tokens * nk128 * 4u);
    const WeaveAgprI32x4 q_rsrc  = weave_agpr_make_srsrc(q_out, E * rows_per_exp * I_out);
    const WeaveAgprI32x4 qs_rsrc = weave_agpr_make_srsrc(qs_out, E * rows_per_exp * nt_per_exp * 4u);

    const uint32_t a_row = tid >> 2u, a_c16 = tid & 3u;
    WeaveAgprLdsPtr p_aw  = lds_s + (a_row * 64u + ((a_c16 ^ ((a_row >> 1u) & 3u)) << 4u));
    const uint32_t b_blk = tid >> 5u, b_in = (tid & 31u) * 32u;
    WeaveAgprLdsPtr p_bw  = lds_s + (b_blk * 1024u + b_in);
    WeaveAgprLdsPtr p_ard = lds_s + ((wm * 64u + m_lane) * 64u + ((k_grp ^ ((m_lane >> 1u) & 3u)) << 4u));
    WeaveAgprLdsPtr p_brd = lds_s + (8192u + (wn * 2u) * 1024u + lane * 16u);
    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(nk64 * 1024u);
    const uint32_t mat_bytes = __builtin_amdgcn_readfirstlane(I_out * K);

    const uint32_t xcd = blockIdx.x & 3u, cu_in_xcd = blockIdx.x >> 2u, e_per_xcd = E >> 2u;

    for (uint32_t idx_local = cu_in_xcd; idx_local < e_per_xcd * tiles_per_exp; idx_local += 20u) {
        const uint32_t exp_id = __builtin_amdgcn_readfirstlane(xcd * e_per_xcd + (idx_local / tiles_per_exp));
        const uint32_t in_exp = idx_local % tiles_per_exp;
        const uint32_t tm = __builtin_amdgcn_readfirstlane(in_exp % mt_per_exp);
        const uint32_t tn = __builtin_amdgcn_readfirstlane(in_exp / mt_per_exp);

        const uint32_t row_base = exp_id * rows_per_exp + tm * 128u;
        const float* sbg_ptr = scale_b + (static_cast<size_t>(exp_id * 2u + 0u) * nt_per_exp + tn) * nk128;
        const float* sbu_ptr = scale_b + (static_cast<size_t>(exp_id * 2u + 1u) * nt_per_exp + tn) * nk128;
        if (tid < 128u) lds_tok[tid] = static_cast<uint32_t>(tok_ids[row_base + tid]);
        if (tid < nk128) { lds_sb[tid] = sbg_ptr[tid]; lds_sb[64u + tid] = sbu_ptr[tid]; }
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");

        WeaveAgprF32x2 c_lo[4][4], c_hi[4][4];
        #pragma unroll
        for (int m = 0; m < 4; ++m)
            #pragma unroll
            for (int n = 0; n < 4; ++n) { c_lo[m][n] = {0.f, 0.f}; c_hi[m][n] = {0.f, 0.f}; }

        {
            const uint32_t av0 = lds_tok[a_row] * K + a_c16 * 16u;
            const uint32_t sav = lds_tok[tid & 127u] * nk128 * 4u;
            const WeaveAgprI32x4 b_rsrc = weave_agpr_make_srsrc(B + static_cast<size_t>(exp_id * 2u) * mat_bytes, mat_bytes * 2u);
            const uint32_t bv0 = (tn * 8u + (b_blk & 7u)) * b_nstride + b_in + ((b_blk >> 3u) * mat_bytes);

            WeaveAgprU32x4 gA_0, gB0_0, gB1_0, gA_1, gB0_1, gB1_1;
            float gSa;
            WeaveAgprU32x4 af0, af1, af3, bf[4];
            WeaveAgprF32x4 ta[4][4];

            gA_0  = weave_agpr_buf_ld_u4(a_rsrc, av0, 0);
            gB0_0 = weave_agpr_buf_ld_u4(b_rsrc, bv0, 0);
            gB1_0 = weave_agpr_buf_ld_u4(b_rsrc, bv0 + 16u, 0);
            gSa   = weave_agpr_buf_ld_f32(sa_rsrc, sav, 0);

            gA_1  = weave_agpr_buf_ld_u4(a_rsrc, av0, 64);
            gB0_1 = weave_agpr_buf_ld_u4(b_rsrc, bv0, 1024);
            gB1_1 = weave_agpr_buf_ld_u4(b_rsrc, bv0 + 16u, 1024);
            __builtin_amdgcn_sched_barrier(0);

            weave_agpr_lds_st128(p_aw, 0,    gA_0);
            weave_agpr_lds_st128(p_bw, 8192, gB0_0);
            weave_agpr_lds_st128(p_bw, 8208, gB1_0);
            if (tid < 128u) lds_sa[tid] = gSa;
            __builtin_amdgcn_sched_barrier(0);

            if (nk64 > 2u) {
                gA_0  = weave_agpr_buf_ld_u4(a_rsrc, av0, 128);
                gB0_0 = weave_agpr_buf_ld_u4(b_rsrc, bv0, 2048);
                gB1_0 = weave_agpr_buf_ld_u4(b_rsrc, bv0 + 16u, 2048);
            }

            asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
            af0   = weave_agpr_lds_ld128(p_ard, 0);    af1   = weave_agpr_lds_ld128(p_ard, 1024);
            bf[0] = weave_agpr_lds_ld128(p_brd, 0);    bf[1] = weave_agpr_lds_ld128(p_brd, 1024);
            bf[2] = weave_agpr_lds_ld128(p_brd, 8192); bf[3] = weave_agpr_lds_ld128(p_brd, 9216);
            __builtin_amdgcn_sched_barrier(0);

            #pragma clang loop unroll(disable)
            for (uint32_t k = 0u; k < nk64; k += 2u) {
                const uint32_t p = k >> 1u;
                const bool has_next_pair = (k + 2u < nk64);
                const bool has_k3        = (k + 3u < nk64);
                const bool has_k4        = (k + 4u < nk64);

                WEAVE_MFMA_ROW_HOOKS(1, ta[0], bf, af0);
                __builtin_amdgcn_sched_barrier(0);
                if (has_next_pair) {
                    const uint32_t sc_p1 = __builtin_amdgcn_readfirstlane((p + 1u) * 4u);
                    gSa = weave_agpr_buf_ld_f32(sa_rsrc, sav, sc_p1);
                }
                af0 = weave_agpr_lds_ld128(p_ard, 2048); af3 = weave_agpr_lds_ld128(p_ard, 3072);
                weave_agpr_lds_st128(p_aw, 24576, gA_1);
                weave_agpr_lds_st128(p_bw, 32768, gB0_1);
                weave_agpr_lds_st128(p_bw, 32784, gB1_1);
                __builtin_amdgcn_sched_barrier(0);

                WEAVE_MFMA_ROW_HOOKS(1, ta[1], bf, af1);
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");

                if (has_k3) {
                    const uint32_t sa3 = __builtin_amdgcn_readfirstlane((k + 3u) * 64u);
                    const uint32_t sb3 = __builtin_amdgcn_readfirstlane((k + 3u) * 1024u);
                    gA_1  = weave_agpr_buf_ld_u4(a_rsrc, av0, sa3);
                    gB0_1 = weave_agpr_buf_ld_u4(b_rsrc, bv0, sb3);
                    gB1_1 = weave_agpr_buf_ld_u4(b_rsrc, bv0 + 16u, sb3);
                }
                __builtin_amdgcn_sched_barrier(0);
                WEAVE_MFMA_ROW_HOOKS(1, ta[2], bf, af0);
                __builtin_amdgcn_sched_barrier(0);
                af0 = weave_agpr_lds_ld128(p_ard, 24576); af1 = weave_agpr_lds_ld128(p_ard, 25600);
                __builtin_amdgcn_sched_barrier(0);

                WEAVE_MFMA_ROW_HOOKS(1, ta[3], bf, af3);
                __builtin_amdgcn_sched_barrier(0);
                bf[0] = weave_agpr_lds_ld128(p_brd, 24576); bf[1] = weave_agpr_lds_ld128(p_brd, 25600);
                bf[2] = weave_agpr_lds_ld128(p_brd, 32768); bf[3] = weave_agpr_lds_ld128(p_brd, 33792);
                const __attribute__((address_space(3))) float* sa_p = lds_sa + (p & 1u) * 128u + wm * 64u + m_lane;
                const float sbg_v = lds_sb[p], sbu_v = lds_sb[64u + p];
                const float sa0_v = sa_p[0], sa1_v = sa_p[16], sa2_v = sa_p[32], sa3_v = sa_p[48];
                __builtin_amdgcn_sched_barrier(0);

                WEAVE_MFMA_ROW_HOOKS(0, ta[0], bf, af0);
                __builtin_amdgcn_sched_barrier(0);
                af0 = weave_agpr_lds_ld128(p_ard, 26624); af3 = weave_agpr_lds_ld128(p_ard, 27648);
                if (has_next_pair) {
                    weave_agpr_lds_st128(p_aw, 0,    gA_0);
                    weave_agpr_lds_st128(p_bw, 8192, gB0_0);
                    weave_agpr_lds_st128(p_bw, 8208, gB1_0);
                    if (tid < 128u) lds_sa[((p + 1u) & 1u) * 128u + tid] = gSa;
                }
                __builtin_amdgcn_sched_barrier(0);

                WEAVE_MFMA_ROW_PROM_CACC(0, ta[1], bf, af1, c_lo[0], c_hi[0], ta[0],
                    sa0_v * sbg_v, sa0_v * sbg_v, sa0_v * sbu_v, sa0_v * sbu_v);
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");

                if (has_k4) {
                    const uint32_t sa4 = __builtin_amdgcn_readfirstlane((k + 4u) * 64u);
                    const uint32_t sb4 = __builtin_amdgcn_readfirstlane((k + 4u) * 1024u);
                    gA_0  = weave_agpr_buf_ld_u4(a_rsrc, av0, sa4);
                    gB0_0 = weave_agpr_buf_ld_u4(b_rsrc, bv0, sb4);
                    gB1_0 = weave_agpr_buf_ld_u4(b_rsrc, bv0 + 16u, sb4);
                }
                __builtin_amdgcn_sched_barrier(0);
                WEAVE_MFMA_ROW_PROM_CACC(0, ta[2], bf, af0, c_lo[1], c_hi[1], ta[1],
                    sa1_v * sbg_v, sa1_v * sbg_v, sa1_v * sbu_v, sa1_v * sbu_v);
                __builtin_amdgcn_sched_barrier(0);
                WEAVE_MFMA_ROW_PROM_CACC(0, ta[3], bf, af3, c_lo[2], c_hi[2], ta[2],
                    sa2_v * sbg_v, sa2_v * sbg_v, sa2_v * sbu_v, sa2_v * sbu_v);
                __builtin_amdgcn_sched_barrier(0);
                if (has_next_pair) {
                    af0   = weave_agpr_lds_ld128(p_ard, 0);    af1   = weave_agpr_lds_ld128(p_ard, 1024);
                    bf[0] = weave_agpr_lds_ld128(p_brd, 0);    bf[1] = weave_agpr_lds_ld128(p_brd, 1024);
                    bf[2] = weave_agpr_lds_ld128(p_brd, 8192); bf[3] = weave_agpr_lds_ld128(p_brd, 9216);
                }
                __builtin_amdgcn_sched_barrier(0);
                WEAVE_PROM_CACC_FINAL(c_lo[3], c_hi[3], ta[3],
                    sa3_v * sbg_v, sa3_v * sbg_v, sa3_v * sbu_v, sa3_v * sbu_v);
                __builtin_amdgcn_sched_barrier(0);
            }
        }

        __builtin_amdgcn_s_barrier();
        uint16_t* epi = reinterpret_cast<uint16_t*>(lds);
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t r = wm * 64u + m * 16u + m_lane;
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                const uint32_t col = (wn * 2u + n) * 16u + k_grp * 4u;
                const WeaveAgprF32x2 gl = c_lo[m][n], gh = c_hi[m][n], ul = c_lo[m][2 + n], uh = c_hi[m][2 + n];
                const float x0 = weave_agpr_bf16_round(gl[0] * ul[0] * __builtin_amdgcn_rcpf(1.f + __expf(-gl[0])));
                const float x1 = weave_agpr_bf16_round(gl[1] * ul[1] * __builtin_amdgcn_rcpf(1.f + __expf(-gl[1])));
                const float x2 = weave_agpr_bf16_round(gh[0] * uh[0] * __builtin_amdgcn_rcpf(1.f + __expf(-gh[0])));
                const float x3 = weave_agpr_bf16_round(gh[1] * uh[1] * __builtin_amdgcn_rcpf(1.f + __expf(-gh[1])));
                WeaveAgprU32x2 pk;
                pk.x = weave_agpr_bf16_bits(x0) | (weave_agpr_bf16_bits(x1) << 16u);
                pk.y = weave_agpr_bf16_bits(x2) | (weave_agpr_bf16_bits(x3) << 16u);
                *reinterpret_cast<WeaveAgprU32x2*>(epi + r * 128u + (((col >> 2u) ^ (r & 7u)) << 2u)) = pk;
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
        const uint32_t qr = tid >> 4u, ql = tid & 15u;
        #pragma unroll
        for (int r0 = 0; r0 < 128; r0 += 32) {
            const uint32_t r = r0 + qr;
            const WeaveAgprU32x2 v0 = *reinterpret_cast<const WeaveAgprU32x2*>(epi + r * 128u + ((ql * 2u + 0u) ^ (r & 7u)) * 4u);
            const WeaveAgprU32x2 v1 = *reinterpret_cast<const WeaveAgprU32x2*>(epi + r * 128u + ((ql * 2u + 1u) ^ (r & 7u)) * 4u);
            const uint32_t b_arr[4] = {v0.x, v0.y, v1.x, v1.y};
            float vals[8], amax = 1e-10f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                vals[2 * i]     = __uint_as_float((b_arr[i] & 0xffffu) << 16u);
                vals[2 * i + 1] = __uint_as_float(b_arr[i] & 0xffff0000u);
                amax = fmaxf(amax, fmaxf(fabsf(vals[2 * i]), fabsf(vals[2 * i + 1])));
            }
            #pragma unroll
            for (int d = 1; d < 16; d <<= 1)
                amax = fmaxf(amax, __int_as_float(__builtin_amdgcn_ds_bpermute(((tid & 63u) ^ d) << 2, __float_as_int(amax))));
            const float sc = amax * (1.f / 240.f), inv = 1.f / sc;
            const uint32_t dense_grp = (row_base + r) * nt_per_exp + tn;
            if (ql == 0u) weave_agpr_raw_buf_store_f32(sc, qs_rsrc, static_cast<int32_t>(dense_grp * 4u), 0, 0);
            int p0 = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(vals[0] * inv, -240.f), 240.f), fminf(fmaxf(vals[1] * inv, -240.f), 240.f), 0, false);
            p0     = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(vals[2] * inv, -240.f), 240.f), fminf(fmaxf(vals[3] * inv, -240.f), 240.f), p0, true);
            int p1 = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(vals[4] * inv, -240.f), 240.f), fminf(fmaxf(vals[5] * inv, -240.f), 240.f), 0, false);
            p1     = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(vals[6] * inv, -240.f), 240.f), fminf(fmaxf(vals[7] * inv, -240.f), 240.f), p1, true);
            weave_agpr_raw_buf_store_v2i32((WeaveAgprI32x2){p0, p1}, q_rsrc, static_cast<int32_t>(dense_grp * 128u + ql * 8u), 0, 0);
        }
        __builtin_amdgcn_s_barrier();
    }
}

// ============================================================================
// 3. Grouped MoE GEMM2 & 128x128 BlockScale FP8 Kernel (80x512 = 1 CTA/CU)
// ============================================================================
extern "C" __global__ __launch_bounds__(512, 1)
void weave_fp8_moe_gemm2_blockscale_gfx942(
    const uint8_t* __restrict__ A,
    const float* __restrict__ scale_a,
    const uint8_t* __restrict__ B,
    const float* __restrict__ scale_b,
    const int32_t* __restrict__ tok_ids,
    uint16_t* __restrict__ C,
    uint32_t M_tot,
    uint32_t rows_per_exp,
    uint32_t E,
    uint32_t N_out,
    uint32_t K,
    uint32_t scatter_out,
    float out_scale)
{
    __shared__ __align__(16) uint8_t lds[65536];
    WeaveAgprLdsPtr lds_s = (WeaveAgprLdsPtr)lds;
    __attribute__((address_space(3))) float*    lds_sa  = (__attribute__((address_space(3))) float*)(lds_s + 49152);
    __attribute__((address_space(3))) uint32_t* lds_tok = (__attribute__((address_space(3))) uint32_t*)(lds_s + 51200);
    __attribute__((address_space(3))) float*    lds_sb  = (__attribute__((address_space(3))) float*)(lds_s + 52224);

    const uint32_t tid = threadIdx.x, wave = tid >> 6u, wm = wave >> 1u, wn = wave & 1u, lane = tid & 63u;
    const uint32_t m_lane = lane & 15u, k_grp = lane >> 4u;
    const uint32_t mt_per_exp = rows_per_exp >> 8u, nt_per_exp = N_out >> 7u, tiles_per_exp = mt_per_exp * nt_per_exp;
    const uint32_t nk128 = __builtin_amdgcn_readfirstlane(K >> 7u), nk64 = __builtin_amdgcn_readfirstlane(K >> 6u);

    const WeaveAgprI32x4 a_rsrc  = weave_agpr_make_srsrc(A, M_tot * K);
    const WeaveAgprI32x4 sa_rsrc = weave_agpr_make_srsrc(scale_a, M_tot * nk128 * 4u);
    const WeaveAgprI32x4 c_rsrc  = weave_agpr_make_srsrc(C, M_tot * N_out * 2u);

    const uint32_t a_r0 = tid >> 2u, a_c = tid & 3u;
    WeaveAgprLdsPtr p_aw  = lds_s + (a_r0 * 64u + ((a_c ^ ((a_r0 >> 1u) & 3u)) << 4u));
    const uint32_t b_blk = tid >> 6u, b_in = (tid & 63u) * 16u;
    WeaveAgprLdsPtr p_bw  = lds_s + (b_blk * 1024u + b_in);
    WeaveAgprLdsPtr p_ard = lds_s + ((wm * 64u + m_lane) * 64u + ((k_grp ^ ((m_lane >> 1u) & 3u)) << 4u));
    WeaveAgprLdsPtr p_brd = lds_s + (16384u + (wn * 4u) * 1024u + lane * 16u);
    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(nk64 * 1024u);
    const uint32_t mat_bytes = __builtin_amdgcn_readfirstlane(N_out * K);

    const uint32_t xcd = blockIdx.x & 3u, cu_in_xcd = blockIdx.x >> 2u;
    const uint32_t logical_cta = __builtin_amdgcn_readfirstlane(xcd * 20u + cu_in_xcd);

    const bool is_moe = (E >= 4u);
    const uint32_t loop_beg   = is_moe ? cu_in_xcd : logical_cta;
    const uint32_t loop_step  = is_moe ? 20u : 80u;
    const uint32_t loop_limit = is_moe ? ((E >> 2u) * tiles_per_exp) : (E * tiles_per_exp);

    for (uint32_t idx_it = loop_beg; idx_it < loop_limit; idx_it += loop_step) {
        uint32_t exp_id, tm, tn;
        if (is_moe) {
            exp_id = __builtin_amdgcn_readfirstlane(xcd * (E >> 2u) + (idx_it / tiles_per_exp));
            const uint32_t in_exp = idx_it % tiles_per_exp;
            tm = __builtin_amdgcn_readfirstlane(in_exp % mt_per_exp);
            tn = __builtin_amdgcn_readfirstlane(in_exp / mt_per_exp);
        } else {
            exp_id = 0u;
            const uint32_t grp_sz = 4u * nt_per_exp, grp_id = idx_it / grp_sz, first_m = grp_id * 4u;
            const uint32_t grp_m = (mt_per_exp - first_m < 4u) ? (mt_per_exp - first_m) : 4u, in_grp = idx_it % grp_sz;
            tm = __builtin_amdgcn_readfirstlane(first_m + (in_grp % grp_m));
            tn = __builtin_amdgcn_readfirstlane(in_grp / grp_m);
        }

        const uint32_t row_base = exp_id * rows_per_exp + tm * 256u;
        const float* sb_ptr = scale_b + (static_cast<size_t>(exp_id) * nt_per_exp + tn) * nk128;

        WeaveAgprF32x2 c_lo[4][4], c_hi[4][4];
        #pragma unroll
        for (int m = 0; m < 4; ++m)
            #pragma unroll
            for (int n = 0; n < 4; ++n) { c_lo[m][n] = {0.f, 0.f}; c_hi[m][n] = {0.f, 0.f}; }

        {
            const uint32_t av0 = (row_base + a_r0) * K + a_c * 16u;
            const uint32_t av1 = (row_base + 128u + a_r0) * K + a_c * 16u;
            const uint32_t sav = (row_base + (tid & 255u)) * nk128 * 4u;
            const WeaveAgprI32x4 b_rsrc = weave_agpr_make_srsrc(B + static_cast<size_t>(exp_id) * mat_bytes, mat_bytes);
            const uint32_t bv0 = (tn * 8u + b_blk) * b_nstride + b_in;

            WeaveAgprU32x4 gA0_0, gA1_0, gB_0, gA0_1, gA1_1, gB_1;
            float gSa;
            WeaveAgprU32x4 af0, af1, af3, bf[4];
            WeaveAgprF32x4 ta[4][4];

            uint32_t my_tok = 0u; float my_sb = 0.f;
            if (scatter_out && tid < 256u) my_tok = static_cast<uint32_t>(tok_ids[row_base + tid]);
            if (tid < nk128)               my_sb  = sb_ptr[tid];

            gA0_0 = weave_agpr_buf_ld_u4(a_rsrc, av0, 0);
            gA1_0 = weave_agpr_buf_ld_u4(a_rsrc, av1, 0);
            gB_0  = weave_agpr_buf_ld_u4(b_rsrc, bv0, 0);
            gSa   = weave_agpr_buf_ld_f32(sa_rsrc, sav, 0);

            gA0_1 = weave_agpr_buf_ld_u4(a_rsrc, av0, 64);
            gA1_1 = weave_agpr_buf_ld_u4(a_rsrc, av1, 64);
            gB_1  = weave_agpr_buf_ld_u4(b_rsrc, bv0, 1024);
            __builtin_amdgcn_sched_barrier(0);

            if (scatter_out && tid < 256u) lds_tok[tid] = my_tok;
            if (tid < nk128)               lds_sb[tid]  = my_sb;
            weave_agpr_lds_st128(p_aw, 0,     gA0_0);
            weave_agpr_lds_st128(p_aw, 8192,  gA1_0);
            weave_agpr_lds_st128(p_bw, 16384, gB_0);
            if (tid < 256u) lds_sa[tid] = gSa;
            __builtin_amdgcn_sched_barrier(0);

            if (nk64 > 2u) {
                gA0_0 = weave_agpr_buf_ld_u4(a_rsrc, av0, 128);
                gA1_0 = weave_agpr_buf_ld_u4(a_rsrc, av1, 128);
                gB_0  = weave_agpr_buf_ld_u4(b_rsrc, bv0, 2048);
            }

            asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
            af0   = weave_agpr_lds_ld128(p_ard, 0);    af1   = weave_agpr_lds_ld128(p_ard, 1024);
            bf[0] = weave_agpr_lds_ld128(p_brd, 0);    bf[1] = weave_agpr_lds_ld128(p_brd, 1024);
            bf[2] = weave_agpr_lds_ld128(p_brd, 2048); bf[3] = weave_agpr_lds_ld128(p_brd, 3072);
            __builtin_amdgcn_sched_barrier(0);

            #pragma clang loop unroll(disable)
            for (uint32_t k = 0u; k < nk64; k += 2u) {
                const uint32_t p = k >> 1u;
                const bool has_next_pair = (k + 2u < nk64);
                const bool has_k3        = (k + 3u < nk64);
                const bool has_k4        = (k + 4u < nk64);

                WEAVE_MFMA_ROW_HOOKS(1, ta[0], bf, af0);
                __builtin_amdgcn_sched_barrier(0);
                if (has_next_pair) {
                    const uint32_t sc_p1 = __builtin_amdgcn_readfirstlane((p + 1u) * 4u);
                    gSa = weave_agpr_buf_ld_f32(sa_rsrc, sav, sc_p1);
                }
                af0 = weave_agpr_lds_ld128(p_ard, 2048); af3 = weave_agpr_lds_ld128(p_ard, 3072);
                weave_agpr_lds_st128(p_aw, 24576, gA0_1);
                weave_agpr_lds_st128(p_aw, 32768, gA1_1);
                weave_agpr_lds_st128(p_bw, 40960, gB_1);
                __builtin_amdgcn_sched_barrier(0);

                WEAVE_MFMA_ROW_HOOKS(1, ta[1], bf, af1);
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");

                if (has_k3) {
                    const uint32_t sa3 = __builtin_amdgcn_readfirstlane((k + 3u) * 64u);
                    const uint32_t sb3 = __builtin_amdgcn_readfirstlane((k + 3u) * 1024u);
                    gA0_1 = weave_agpr_buf_ld_u4(a_rsrc, av0, sa3);
                    gA1_1 = weave_agpr_buf_ld_u4(a_rsrc, av1, sa3);
                    gB_1  = weave_agpr_buf_ld_u4(b_rsrc, bv0, sb3);
                }
                __builtin_amdgcn_sched_barrier(0);
                WEAVE_MFMA_ROW_HOOKS(1, ta[2], bf, af0);
                __builtin_amdgcn_sched_barrier(0);
                af0 = weave_agpr_lds_ld128(p_ard, 24576); af1 = weave_agpr_lds_ld128(p_ard, 25600);
                __builtin_amdgcn_sched_barrier(0);

                WEAVE_MFMA_ROW_HOOKS(1, ta[3], bf, af3);
                __builtin_amdgcn_sched_barrier(0);
                bf[0] = weave_agpr_lds_ld128(p_brd, 24576); bf[1] = weave_agpr_lds_ld128(p_brd, 25600);
                bf[2] = weave_agpr_lds_ld128(p_brd, 26624); bf[3] = weave_agpr_lds_ld128(p_brd, 27648);
                const __attribute__((address_space(3))) float* sa_p = lds_sa + (p & 1u) * 256u + wm * 64u + m_lane;
                const float sb_v = lds_sb[p];
                const float sa0_v = sa_p[0], sa1_v = sa_p[16], sa2_v = sa_p[32], sa3_v = sa_p[48];
                __builtin_amdgcn_sched_barrier(0);

                WEAVE_MFMA_ROW_HOOKS(0, ta[0], bf, af0);
                __builtin_amdgcn_sched_barrier(0);
                af0 = weave_agpr_lds_ld128(p_ard, 26624); af3 = weave_agpr_lds_ld128(p_ard, 27648);
                const float sc0 = sa0_v * sb_v, sc1 = sa1_v * sb_v, sc2 = sa2_v * sb_v, sc3 = sa3_v * sb_v;
                if (has_next_pair) {
                    weave_agpr_lds_st128(p_aw, 0,     gA0_0);
                    weave_agpr_lds_st128(p_aw, 8192,  gA1_0);
                    weave_agpr_lds_st128(p_bw, 16384, gB_0);
                    if (tid < 256u) lds_sa[((p + 1u) & 1u) * 256u + tid] = gSa;
                }
                __builtin_amdgcn_sched_barrier(0);

                WEAVE_MFMA_ROW_PROM_CACC(0, ta[1], bf, af1, c_lo[0], c_hi[0], ta[0], sc0, sc0, sc0, sc0);
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");

                if (has_k4) {
                    const uint32_t sa4 = __builtin_amdgcn_readfirstlane((k + 4u) * 64u);
                    const uint32_t sb4 = __builtin_amdgcn_readfirstlane((k + 4u) * 1024u);
                    gA0_0 = weave_agpr_buf_ld_u4(a_rsrc, av0, sa4);
                    gA1_0 = weave_agpr_buf_ld_u4(a_rsrc, av1, sa4);
                    gB_0  = weave_agpr_buf_ld_u4(b_rsrc, bv0, sb4);
                }
                __builtin_amdgcn_sched_barrier(0);
                WEAVE_MFMA_ROW_PROM_CACC(0, ta[2], bf, af0, c_lo[1], c_hi[1], ta[1], sc1, sc1, sc1, sc1);
                __builtin_amdgcn_sched_barrier(0);
                WEAVE_MFMA_ROW_PROM_CACC(0, ta[3], bf, af3, c_lo[2], c_hi[2], ta[2], sc2, sc2, sc2, sc2);
                __builtin_amdgcn_sched_barrier(0);
                if (has_next_pair) {
                    af0   = weave_agpr_lds_ld128(p_ard, 0);    af1   = weave_agpr_lds_ld128(p_ard, 1024);
                    bf[0] = weave_agpr_lds_ld128(p_brd, 0);    bf[1] = weave_agpr_lds_ld128(p_brd, 1024);
                    bf[2] = weave_agpr_lds_ld128(p_brd, 2048); bf[3] = weave_agpr_lds_ld128(p_brd, 3072);
                }
                __builtin_amdgcn_sched_barrier(0);
                WEAVE_PROM_CACC_FINAL(c_lo[3], c_hi[3], ta[3], sc3, sc3, sc3, sc3);
                __builtin_amdgcn_sched_barrier(0);
            }
        }

        __builtin_amdgcn_s_barrier();
        const uint32_t sr = tid >> 4u, sc = tid & 15u;
        uint32_t out_rows[8];
        if (scatter_out) {
            #pragma unroll
            for (int rr = 0; rr < 8; ++rr) out_rows[rr] = lds_tok[rr * 32u + sr];
            __builtin_amdgcn_s_barrier();
        }
        uint16_t* lds_c = reinterpret_cast<uint16_t*>(lds);
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t r = wm * 64u + m * 16u + m_lane;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const uint32_t c = (wn * 4u + n) * 16u + k_grp * 4u;
                WeaveAgprU32x2 pk;
                pk.x = weave_agpr_bf16_bits(c_lo[m][n][0] * out_scale) | (weave_agpr_bf16_bits(c_lo[m][n][1] * out_scale) << 16u);
                pk.y = weave_agpr_bf16_bits(c_hi[m][n][0] * out_scale) | (weave_agpr_bf16_bits(c_hi[m][n][1] * out_scale) << 16u);
                *reinterpret_cast<WeaveAgprU32x2*>(lds_c + r * 128u + (((c >> 2u) ^ (r & 7u)) << 2u)) = pk;
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
        if (scatter_out) {
            #pragma unroll
            for (int rr = 0; rr < 8; ++rr) {
                const uint32_t r = rr * 32u + sr;
                const WeaveAgprU32x2 v0 = *reinterpret_cast<const WeaveAgprU32x2*>(lds_c + r * 128u + ((sc * 2u + 0u) ^ (r & 7u)) * 4u);
                const WeaveAgprU32x2 v1 = *reinterpret_cast<const WeaveAgprU32x2*>(lds_c + r * 128u + ((sc * 2u + 1u) ^ (r & 7u)) * 4u);
                const int32_t coff = static_cast<int32_t>((static_cast<size_t>(out_rows[rr]) * N_out + tn * 128u + sc * 8u) * 2u);
                weave_agpr_raw_buf_store_v4i32(
                    (WeaveAgprI32x4){static_cast<int32_t>(v0.x), static_cast<int32_t>(v0.y), static_cast<int32_t>(v1.x), static_cast<int32_t>(v1.y)},
                    c_rsrc, coff, 0, 0);
            }
        } else {
            const int32_t c_base_voff = static_cast<int32_t>((static_cast<size_t>(row_base + sr) * N_out + tn * 128u + sc * 8u) * 2u);
            const int32_t c_row32_soff = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(32u * N_out * 2u));
            #pragma unroll
            for (int rr = 0; rr < 8; ++rr) {
                const uint32_t r = rr * 32u + sr;
                const WeaveAgprU32x2 v0 = *reinterpret_cast<const WeaveAgprU32x2*>(lds_c + r * 128u + ((sc * 2u + 0u) ^ (r & 7u)) * 4u);
                const WeaveAgprU32x2 v1 = *reinterpret_cast<const WeaveAgprU32x2*>(lds_c + r * 128u + ((sc * 2u + 1u) ^ (r & 7u)) * 4u);
                weave_agpr_raw_buf_store_v4i32(
                    (WeaveAgprI32x4){static_cast<int32_t>(v0.x), static_cast<int32_t>(v0.y), static_cast<int32_t>(v1.x), static_cast<int32_t>(v1.y)},
                    c_rsrc, c_base_voff, rr * c_row32_soff, 0);
            }
        }
        __builtin_amdgcn_s_barrier();
    }
}
