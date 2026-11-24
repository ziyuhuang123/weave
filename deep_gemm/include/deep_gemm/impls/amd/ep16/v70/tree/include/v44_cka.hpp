// v44_cka.hpp -- opus55 v44 G2: cached, row-sorted A shadow for GEMM1 (accessors only).
// Layout of scratch region EP16_SC_CK_A_SCALE when the host .so is built with -DV44_CKA_SHADOW=1.
// NOTE (r87 root cause): in the kernel ABI (megamoe_amd.hip V968CkExt) ck_a_scale is a UNION with
// ck_sorted_token_ids, i.e. the first (RR+64)*4 bytes of this region ARE the Stage-1 sorted ids (s1).
// r82..r87 placed the A rows at offset 0 and overwrote s1 (r87: re-read s1[r] = FP8 payload bytes).
// r88 layout keeps s1 intact:
//   [0, al4k((RR+64)*4))                   s1 sorted token ids (unchanged, owned by the planner)
//   [.., + al4k(RR*PB))                    A rows   (FP8, sorted-row order, row r = token of s1_ids[r])
//   [.., + al4k(RR*SG*4))                  A scales (f32, SG = hidden/128 per row)
//   [.., + (RR+64)*4)                      ids[r] = r for valid rows, RR for padding rows
// GEMM1 then gathers A through the ordinary cached path with identity ids (num_tokens := RR).
#pragma once
#ifndef V44_CKA
#define V44_CKA 0
#endif
#ifndef V44_CKA_READ
#define V44_CKA_READ 1   // 0 = diagnostic: run the gather, but GEMM1 still reads the original UC arena
#endif
#ifndef V44_CKA_VERIFY
#define V44_CKA_VERIFY 0  // 1 = r85 diagnostic self-check of the published shadow (error 961/962/963)
#endif
#if V44_CKA
__device__ __forceinline__ uint64_t v44_al4k(uint64_t v) { return (v + 4095ull) & ~4095ull; }
__device__ __forceinline__ uint64_t v44_cka_base(const V968DispatchExt& env) {
  return env.ck.ck_a_scale + v44_al4k((static_cast<uint64_t>(env.ck.ck_route_rows) + 64ull) * 4ull);  // skip s1
}
__device__ __forceinline__ unsigned char* v44_cka_a(const V968DispatchExt& env) {
  return reinterpret_cast<unsigned char*>(v44_cka_base(env));
}
__device__ __forceinline__ float* v44_cka_s(const V968DispatchExt& env) {
  return reinterpret_cast<float*>(v44_cka_base(env) +
                                  v44_al4k(static_cast<uint64_t>(env.ck.ck_route_rows) * env.base.payload_bytes));
}
__device__ __forceinline__ int32_t* v44_cka_ids(const V968DispatchExt& env) {
  return reinterpret_cast<int32_t*>(v44_cka_base(env) +
                                    v44_al4k(static_cast<uint64_t>(env.ck.ck_route_rows) * env.base.payload_bytes) +
                                    v44_al4k(static_cast<uint64_t>(env.ck.ck_route_rows) * env.base.scale_groups * 4ull));
}
#endif
