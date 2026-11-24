#pragma once
#include "v43_pf.hpp"
// V43 shared helpers (included before the GEMM drivers).

__device__ __forceinline__ unsigned v43_phys_mt(unsigned mt, const unsigned* pf) {
  const unsigned t0 = __builtin_amdgcn_readfirstlane(pf[V43_PF_T0]);
  const unsigned t1 = __builtin_amdgcn_readfirstlane(pf[V43_PF_T1]);
  const unsigned pb = __builtin_amdgcn_readfirstlane(pf[V43_PF_P2BASE]);
  return (mt < t0 + t1) ? mt : (pb + (mt - (t0 + t1)));
}

// Bounded work claim: unlike a plain fetch_add, a group that races past the
// current pass bound gets 0xffffffff and consumes nothing.  This is required
// because the visible tile count grows when the Pass-1 plan lands; a discarded
// overshoot must never skip a tile index that becomes valid a moment later.
__device__ __forceinline__ unsigned v43_claim_bounded(
    unsigned long long* cursor, unsigned bound, unsigned char* arena) {
  const unsigned grp_sh = (threadIdx.x >= 256u) ? 32u : 0u;
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(
      reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65472u + grp_sh));
  weave_4wave_barrier_65440(arena);
  if ((threadIdx.x & 255u) == 0u) {
    unsigned long long cur = __hip_atomic_load(cursor, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    for (;;) {
      if ((unsigned)cur >= bound) { sh[0] = 0xffffffffu; break; }
      if (__hip_atomic_compare_exchange_weak(cursor, &cur, cur + 1ull,
                                             __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT)) {
        sh[0] = static_cast<unsigned>(cur);
        break;
      }
    }
  }
  weave_4wave_barrier_65440(arena);
  return __builtin_amdgcn_readfirstlane(sh[0]);
}

// Cached-A shadow lives inside the CK_A scratch block, right after the tile
// descriptors and the 12 KB small-tile scratch.
__device__ __forceinline__ unsigned char* v43_xshadow(const V968CkExt& ck) {
  const unsigned RR = ck.ck_route_rows;
  const unsigned desc_bytes = (16u * (RR / 16u) + 255u) & ~255u;
  return reinterpret_cast<unsigned char*>(ck.ck_a + 64u + desc_bytes + 2u * 1024u * 4u + 4096u);
}