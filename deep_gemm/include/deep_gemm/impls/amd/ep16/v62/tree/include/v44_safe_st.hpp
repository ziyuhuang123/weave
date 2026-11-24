// v44_safe_st.hpp -- opus55 v44 S1: compiler-visible system-scope stores (CDNA3 inline-asm hazard cleanup).
// The inherited V42G helpers emit `global_store_dwordx4 ... sc0 sc1` from inline asm.  LLVM's
// GCNHazardRecognizer does not see the store-data VGPR reads inside the asm, so a following VALU that
// rewrites those VGPRs must be separated manually (`s_nop 1` = 2 wait states in V42G; nothing at all in
// ep16_fic2_post_reg).  With V44_SAFE_ST=1 the same bytes are written with relaxed system-scope atomic
// stores (global_store_dwordx2 ... sc0 sc1, i.e. the same write-through cache policy), which the compiler
// tracks, so no manual wait states are needed.  Ordering against later flag stores is unchanged: callers
// still issue s_waitcnt vmcnt(0) (asm with "memory" clobber) before publishing.
#pragma once
#ifndef V44_SAFE_ST
#define V44_SAFE_ST 0
#endif
// V44_SAFE_ST==2 (r86): inline-asm sc0 sc1 stores kept, every asm store block ends with s_waitcnt vmcnt(0)
// (store data VGPRs are provably consumed before any later VALU can overwrite them).  Default keeps the
// inherited V42G 2-wait-state `s_nop 1` so that a no-flag build stays byte-identical to r78.
#ifndef V44_ASM_TAIL
#if V44_SAFE_ST == 2
#define V44_ASM_TAIL "s_waitcnt vmcnt(0)\n"
#else
#define V44_ASM_TAIL "s_nop 1\n"
#endif
#endif
#if V44_SAFE_ST
// address_space(1) (global) pointers so the compiler emits global_store_* (not flat_store_*, which would
// also occupy lgkmcnt).  The C-style cast performs the generic -> global addrspacecast.
#define V44_GPTR(T, p) ((__attribute__((address_space(1))) T*)(p))
__device__ __forceinline__ void v44_st_sys_u4(void* ptr, unsigned x, unsigned y, unsigned z, unsigned w) {
  auto* p = V44_GPTR(uint64_t, ptr);
  __hip_atomic_store(p, static_cast<uint64_t>(x) | (static_cast<uint64_t>(y) << 32u), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(p + 1, static_cast<uint64_t>(z) | (static_cast<uint64_t>(w) << 32u), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void v44_st_sys_f32(void* ptr, float v) {
  __hip_atomic_store(V44_GPTR(float, ptr), v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void v44_st_sys_u64(void* ptr, uint64_t v) {
  __hip_atomic_store(V44_GPTR(uint64_t, ptr), v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
#endif
