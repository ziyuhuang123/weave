#pragma once
// v1265: the historical official M16 CK body is executed as an exact ISA
// excerpt.  It consumes a by-value CK Argument from global memory, so keep a
// byte-exact 168-byte packet at the tail of the existing expert-id allocation.
// Layout is asserted below; the first 16 bytes are the CK BaseArgument slots.
struct alignas(8) __attribute__((packed)) V1265M16Packet {
  uint64_t base_reserved[2];
  int32_t NumTokens;
  int32_t TopK;
  int32_t M;
  int32_t N;
  int32_t K;
  int32_t StrideA;
  int32_t StrideB;
  int32_t StrideDs;
  int32_t StrideC;
  int32_t KBatch;
  int32_t MPadded;
  int32_t NPadded;
  int32_t KRead;
  int32_t KPadded;
  int32_t AK0;
  int32_t BK0;
  int32_t MBlock;
  int32_t NBlock;
  uint64_t p_sorted_token_ids;
  uint64_t p_sorted_expert_ids;
  uint64_t p_max_token_id;
  uint64_t p_a_grid;
  uint64_t p_b_grid;
  uint64_t p_ds_grid;
  uint64_t p_c_grid;
  uint64_t p_a_scale_grid;
  uint64_t p_b_scale_grid;
  uint8_t a_element_op;
  uint8_t b_element_op;
  uint8_t c_element_op;
  uint8_t tail_padding[5];
};
static_assert(sizeof(V1265M16Packet) == 168, "M16 packet must match CK Argument");
static_assert(offsetof(V1265M16Packet, NumTokens) == 16, "M16 packet NumTokens offset");
static_assert(offsetof(V1265M16Packet, TopK) == 20, "M16 packet TopK offset");
static_assert(offsetof(V1265M16Packet, M) == 24, "M16 packet M offset");
static_assert(offsetof(V1265M16Packet, N) == 28, "M16 packet N offset");
static_assert(offsetof(V1265M16Packet, K) == 32, "M16 packet K offset");
static_assert(offsetof(V1265M16Packet, StrideA) == 36, "M16 packet StrideA offset");
static_assert(offsetof(V1265M16Packet, StrideB) == 40, "M16 packet StrideB offset");
static_assert(offsetof(V1265M16Packet, StrideDs) == 44, "M16 packet StrideDs offset");
static_assert(offsetof(V1265M16Packet, StrideC) == 48, "M16 packet StrideC offset");
static_assert(offsetof(V1265M16Packet, KBatch) == 52, "M16 packet KBatch offset");
static_assert(offsetof(V1265M16Packet, MPadded) == 56, "M16 packet MPadded offset");
static_assert(offsetof(V1265M16Packet, NPadded) == 60, "M16 packet NPadded offset");
static_assert(offsetof(V1265M16Packet, KRead) == 64, "M16 packet KRead offset");
static_assert(offsetof(V1265M16Packet, KPadded) == 68, "M16 packet KPadded offset");
static_assert(offsetof(V1265M16Packet, AK0) == 72, "M16 packet AK0 offset");
static_assert(offsetof(V1265M16Packet, BK0) == 76, "M16 packet BK0 offset");
static_assert(offsetof(V1265M16Packet, MBlock) == 80, "M16 packet MBlock offset");
static_assert(offsetof(V1265M16Packet, NBlock) == 84, "M16 packet NBlock offset");
static_assert(offsetof(V1265M16Packet, p_sorted_token_ids) == 88, "M16 packet p_sorted_token_ids offset");
static_assert(offsetof(V1265M16Packet, p_sorted_expert_ids) == 96, "M16 packet p_sorted_expert_ids offset");
static_assert(offsetof(V1265M16Packet, p_max_token_id) == 104, "M16 packet p_max_token_id offset");
static_assert(offsetof(V1265M16Packet, p_a_grid) == 112, "M16 packet p_a_grid offset");
static_assert(offsetof(V1265M16Packet, p_b_grid) == 120, "M16 packet p_b_grid offset");
static_assert(offsetof(V1265M16Packet, p_ds_grid) == 128, "M16 packet p_ds_grid offset");
static_assert(offsetof(V1265M16Packet, p_c_grid) == 136, "M16 packet p_c_grid offset");
static_assert(offsetof(V1265M16Packet, p_a_scale_grid) == 144, "M16 packet p_a_scale_grid offset");
static_assert(offsetof(V1265M16Packet, p_b_scale_grid) == 152, "M16 packet p_b_scale_grid offset");
static_assert(offsetof(V1265M16Packet, a_element_op) == 160, "M16 packet a_element_op offset");
static_assert(offsetof(V1265M16Packet, b_element_op) == 161, "M16 packet b_element_op offset");
static_assert(offsetof(V1265M16Packet, c_element_op) == 162, "M16 packet c_element_op offset");
static_assert(offsetof(V1265M16Packet, tail_padding) == 163, "M16 packet tail_padding offset");
constexpr unsigned V1265_M16_PACKET_WORDS = 168u / sizeof(uint32_t);

// v3210: the M16 packet no longer shares the expert-id allocation.  The host
// reserves a dedicated, 8-byte-aligned packet/diagnostic buffer behind
// ck_diagnostics.  Packet publication and every read-back use explicit
// system-scope operations so the raw CK body never observes a stale zero
// pointer from the producer's cache.
constexpr unsigned V3210_M16_PACKET_BYTES = 168u;
constexpr unsigned V3210_M16_DIAG_BASE_U64 = 168u / sizeof(uint64_t);
constexpr unsigned V3210_M16_DIAG_U64S = 8u;
constexpr unsigned V3210_CK_DIAGNOSTIC_BUFFER_U64S = 64u;
constexpr uint64_t V3210_M16_PACKET_DIAG_MAGIC = 0x544b503031323356ull;
constexpr unsigned V3210_ERROR_PACKET_PLAN = 219u;
constexpr unsigned V3210_ERROR_PACKET_RAW = 220u;
constexpr unsigned V3210_PACKET_ERROR_CLAIM = 0x504b5445u;  // PKTE
static_assert(V3210_M16_DIAG_BASE_U64 * sizeof(uint64_t) ==
              V3210_M16_PACKET_BYTES,
              "M16 packet must end on an 8-byte diagnostic boundary");
static_assert(V3210_CK_DIAGNOSTIC_BUFFER_U64S >=
              V3210_M16_DIAG_BASE_U64 + V3210_M16_DIAG_U64S,
              "ck_diagnostics must carry the packet and diagnostic tail");

__device__ __forceinline__ V1265M16Packet* v3210_m16_packet(
    const V968CkExt& ck) {
  return reinterpret_cast<V1265M16Packet*>(
      static_cast<uintptr_t>(ck.ck_diagnostics));
}

__device__ __forceinline__ uint64_t* v3210_m16_diagnostics(
    const V968CkExt& ck) {
  return reinterpret_cast<uint64_t*>(
      static_cast<uintptr_t>(ck.ck_diagnostics)) + V3210_M16_DIAG_BASE_U64;
}

__device__ __forceinline__ void v3210_store_system_u32(
    void* destination, unsigned value) {
  asm volatile("flat_store_dword %0, %1 sc0 sc1"
               :: "v"(destination), "v"(value) : "memory");
}
__device__ __forceinline__ void v3210_store_system_u64(
    uint64_t* destination, uint64_t value) {
  asm volatile("flat_store_dwordx2 %0, %1 sc0 sc1"
               :: "v"(destination), "v"(value) : "memory");
}
__device__ __forceinline__ unsigned v3210_load_system_u32(
    const void* source) {
  asm volatile("buffer_inv sc0 sc1" ::: "memory");
  unsigned value;
  asm volatile("flat_load_dword %0, %1 sc0 sc1\n\ts_waitcnt vmcnt(0)"
               : "=v"(value) : "v"(source) : "memory");
  asm volatile("buffer_inv sc0 sc1" ::: "memory");
  return value;
}
__device__ __forceinline__ uint64_t v3210_load_system_u64(
    const uint64_t* source) {
  asm volatile("buffer_inv sc0 sc1" ::: "memory");
  uint64_t value;
  asm volatile("flat_load_dwordx2 %0, %1 sc0 sc1\n\ts_waitcnt vmcnt(0)"
               : "=v"(value) : "v"(source) : "memory");
  asm volatile("buffer_inv sc0 sc1" ::: "memory");
  return value;
}
// Scalar SMRD loads are the path used by the frozen CK body.  A successful
// vector read-back does not warm that path on gfx942, so the consumer gate
// also performs bounded family-0 loads after invalidating scalar L1.  These
// loads repeat the compiler-checked packet comparison before raw CK entry and
// leave the exact packet lines resident for the raw body's GLC loads.
__device__ __forceinline__ unsigned v3210_load_scalar_packet_u32(
    const void* source) {
  const unsigned address_lo =
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(source));
  const unsigned address_hi =
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(source) >> 32);
  unsigned value;
  asm volatile(
      "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)\n\t"
      "v_mov_b32_e32 v0, %[address_lo]\n\t"
      "v_readfirstlane_b32 s0, v0\n\t"
      "v_mov_b32_e32 v1, %[address_hi]\n\t"
      "v_readfirstlane_b32 s1, v1\n\t"
      "buffer_inv sc0 sc1\n\t"
      "s_dcache_inv\n\t"
      "s_load_dword s2, s[0:1], 0 glc\n\t"
      "s_waitcnt lgkmcnt(0)\n\t"
      "s_mov_b32 %[value], s2"
      : [value] "=s"(value)
      : [address_lo] "s"(address_lo), [address_hi] "s"(address_hi)
      : "s0", "s1", "s2", "v0", "v1", "memory");
  return value;
}
__device__ __forceinline__ uint64_t v3210_load_scalar_packet_u64(
    const uint64_t* source) {
  const unsigned address_lo =
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(source));
  const unsigned address_hi =
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(source) >> 32);
  unsigned value_lo, value_hi;
  asm volatile(
      "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)\n\t"
      "v_mov_b32_e32 v0, %[address_lo]\n\t"
      "v_readfirstlane_b32 s0, v0\n\t"
      "v_mov_b32_e32 v1, %[address_hi]\n\t"
      "v_readfirstlane_b32 s1, v1\n\t"
      "buffer_inv sc0 sc1\n\t"
      "s_dcache_inv\n\t"
      "s_load_dwordx2 s[2:3], s[0:1], 0 glc\n\t"
      "s_waitcnt lgkmcnt(0)\n\t"
      "s_mov_b32 %[value_lo], s2\n\t"
      "s_mov_b32 %[value_hi], s3"
      : [value_lo] "=s"(value_lo), [value_hi] "=s"(value_hi)
      : [address_lo] "s"(address_lo), [address_hi] "s"(address_hi)
      : "s0", "s1", "s2", "s3", "v0", "v1", "memory");
  return static_cast<uint64_t>(value_lo) |
         (static_cast<uint64_t>(value_hi) << 32);
}

__device__ __forceinline__ void v3210_system_release() {
  asm volatile(
      "s_waitcnt vmcnt(0)\n\t"
      "buffer_wbl2 sc0 sc1\n\t"
      "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
      "buffer_inv sc0 sc1" ::: "memory");
}

// gfx942 may retain producer-published packet fields in the per-CU scalar L1
// even after the vector/L2 invalidation performed by the plan wait.  Clear that
// scalar path once at the generation boundary before dependent scalar reads.
__device__ __forceinline__ void v3210_invalidate_scalar_l1() {
  asm volatile(
      "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)\n\t"
      "buffer_inv sc0 sc1\n\t"
      "s_dcache_inv\n\t"
      "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" ::: "memory");
}

__device__ __forceinline__ bool v3210_valid_pointer(
    uint64_t pointer, unsigned alignment) {
  return pointer != 0ull && (pointer & (uint64_t)(alignment - 1u)) == 0ull;
}

// Validate every CK pointer dereferenced by the M16 path before the first
// packet-dependent access.  Entry failures become structured packet errors.
__device__ __forceinline__ unsigned v3210_m16_pointer_mismatch(
    const V968CkExt& ck) {
  // ck_diagnostics is validated by the entry fail-closed gate before this call.
  if(!v3210_valid_pointer(ck.ck_max_token_id, 4u)) return 2u;
  if(!v3210_valid_pointer(ck.ck_m64_expert_ids, 4u)) return 3u;
  if(!v3210_valid_pointer(ck.ck_m64_sorted_ids, 4u)) return 4u;
  if(!v3210_valid_pointer(ck.ck_m64_offsets, 4u)) return 5u;
  if(!v3210_valid_pointer(ck.ck_m64_cursors, 4u)) return 6u;
  if(!v3210_valid_pointer(ck.ck_a, 4u)) return 7u;
  if(!v3210_valid_pointer(ck.ck_b, 4u)) return 8u;
  if(!v3210_valid_pointer(ck.ck_stage1_output, 4u)) return 9u;
  if(!v3210_valid_pointer(ck.ck_a_scale, 4u)) return 10u;
  if(!v3210_valid_pointer(ck.ck_b_scale, 4u)) return 11u;
  if(!v3210_valid_pointer(ck.ck_quant_output, 4u)) return 12u;
  if(!v3210_valid_pointer(ck.ck_quant_scale, 4u)) return 13u;
  if(!v3210_valid_pointer(ck.ck_dense_a, 4u)) return 14u;
  if(!v3210_valid_pointer(ck.ck_dense_a_scale, 4u)) return 15u;
  return 0u;
}

__device__ __forceinline__ void v3210_record_packet_error(
    unsigned* error, uint64_t* diagnostics, unsigned code, unsigned rank,
    unsigned generation, unsigned phase, unsigned detail, uint64_t expected,
    uint64_t actual) {
  if(error == nullptr) return;
  const unsigned old_claim = atomicCAS(
      error + 15u, 0u, V3210_PACKET_ERROR_CLAIM);
  if(old_claim != 0u) return;
  const uintptr_t packet_address = diagnostics == nullptr
      ? 0u : reinterpret_cast<uintptr_t>(diagnostics);
  v3210_store_system_u32(error + 1u, rank);
  v3210_store_system_u32(error + 2u, phase);
  v3210_store_system_u32(error + 3u, blockIdx.x);
  v3210_store_system_u32(error + 4u, threadIdx.x >> 6u);
  v3210_store_system_u32(error + 5u, (unsigned)expected);
  v3210_store_system_u32(error + 6u, (unsigned)actual);
  v3210_store_system_u32(error + 7u, generation);
  v3210_store_system_u32(error + 8u, detail);
  v3210_store_system_u32(error + 9u, (unsigned)packet_address);
  v3210_store_system_u32(error + 10u, (unsigned)(packet_address >> 32));
  v3210_store_system_u32(error + 11u, (unsigned)expected);
  v3210_store_system_u32(error + 12u, (unsigned)actual);
  v3210_store_system_u32(error + 13u, (unsigned)(actual >> 32));
  v3210_store_system_u32(error + 14u, V3210_PACKET_ERROR_CLAIM);
  v3210_system_release();
  (void)atomicCAS(error, 0u, code);
  v3210_system_release();
}

__device__ __forceinline__ void v1265_m16_publish_packet(
    const V968CkExt& ck, V1265M16Packet* packet,
    const int32_t* stage1_experts, unsigned rows) {
  v3210_store_system_u64(packet->base_reserved, 0ull);
  v3210_store_system_u64(packet->base_reserved + 1u, 0ull);
  v3210_store_system_u32(&packet->NumTokens, (unsigned)rows);
  v3210_store_system_u32(&packet->TopK, 1u);
  v3210_store_system_u32(&packet->M, (unsigned)rows);
  v3210_store_system_u32(&packet->N, (unsigned)ck.ck_intermediate);
  v3210_store_system_u32(&packet->K, (unsigned)ck.ck_hidden);
  v3210_store_system_u32(&packet->StrideA, (unsigned)ck.ck_hidden);
  v3210_store_system_u32(&packet->StrideB, (unsigned)ck.ck_hidden);
  v3210_store_system_u32(&packet->StrideDs, 0u);
  v3210_store_system_u32(&packet->StrideC, (unsigned)ck.ck_intermediate);
  v3210_store_system_u32(&packet->KBatch, 1u);
  v3210_store_system_u32(&packet->MPadded, (rows + 15u) & ~15u);
  v3210_store_system_u32(
      &packet->NPadded, (unsigned)((ck.ck_intermediate + 127u) & ~127u));
  v3210_store_system_u32(&packet->KRead, (unsigned)ck.ck_hidden);
  v3210_store_system_u32(&packet->KPadded, (unsigned)ck.ck_hidden);
  v3210_store_system_u32(&packet->AK0, (unsigned)ck.ck_hidden / 16u);
  v3210_store_system_u32(&packet->BK0, (unsigned)ck.ck_hidden / 16u);
  v3210_store_system_u32(&packet->MBlock, (rows + 15u) / 16u);
  v3210_store_system_u32(
      &packet->NBlock, (unsigned)ck.ck_intermediate / 128u);
  v3210_store_system_u64(&packet->p_sorted_token_ids, ck.ck_m64_sorted_ids);
  v3210_store_system_u64(&packet->p_sorted_expert_ids,
                         reinterpret_cast<uint64_t>(stage1_experts));
  v3210_store_system_u64(&packet->p_max_token_id, ck.ck_max_token_id);
  v3210_store_system_u64(&packet->p_a_grid, ck.ck_a);
  v3210_store_system_u64(&packet->p_b_grid, ck.ck_b);
  v3210_store_system_u64(&packet->p_ds_grid, 0ull);
  v3210_store_system_u64(&packet->p_c_grid, ck.ck_stage1_output);
  v3210_store_system_u64(&packet->p_a_scale_grid, ck.ck_a_scale);
  v3210_store_system_u64(&packet->p_b_scale_grid, ck.ck_b_scale);
  // Bytes 160..167 contain the three element-op selectors and five padding
  // bytes.  Clear the complete final 64-bit word so a stale diagnostic from a
  // prior replay can never leak into the CK argument.
  v3210_store_system_u64(
      reinterpret_cast<uint64_t*>(reinterpret_cast<unsigned char*>(packet) +
                                  160u),
      0ull);
  v3210_system_release();
}

// Return 0 on success, or a one-based field id naming the first mismatch.
__device__ __forceinline__ unsigned v1265_m16_validate_packet(
    const V968CkExt& ck, const V1265M16Packet* packet,
    const int32_t* stage1_experts, unsigned rows, uint64_t* diagnostics,
    bool scalar_readback = false) {
  if(diagnostics != nullptr) {
    const uint64_t observed_max = scalar_readback
        ? v3210_load_scalar_packet_u64(&packet->p_max_token_id)
        : v3210_load_system_u64(&packet->p_max_token_id);
    v3210_store_system_u64(diagnostics + 2u, observed_max);
  }
#define V3210_CHECK_U32(ID, WORD, EXPECTED) do { \
    const unsigned observed = scalar_readback \
        ? v3210_load_scalar_packet_u32( \
              reinterpret_cast<const unsigned char*>(packet) + (WORD) * 4u) \
        : v3210_load_system_u32( \
              reinterpret_cast<const unsigned char*>(packet) + (WORD) * 4u); \
    if(observed != (unsigned)(EXPECTED)) { \
      if(diagnostics != nullptr) { \
        v3210_store_system_u64(diagnostics + 3u, observed); \
        v3210_store_system_u64(diagnostics + 4u, (ID)); \
      } \
      return (ID); \
    } \
  } while(0)
#define V3210_CHECK_U64(ID, FIELD, EXPECTED) do { \
    const uint64_t observed = scalar_readback \
        ? v3210_load_scalar_packet_u64(&packet->FIELD) \
        : v3210_load_system_u64(&packet->FIELD); \
    if(observed != (uint64_t)(EXPECTED)) { \
      if(diagnostics != nullptr) { \
        v3210_store_system_u64(diagnostics + 3u, observed); \
        v3210_store_system_u64(diagnostics + 4u, (ID)); \
      } \
      return (ID); \
    } \
  } while(0)
  // BaseArgument occupies four words (two u64 values), so NumTokens begins at
  // word 4.  The pointer fields below are loaded by their compiler-checked
  // struct members; keeping both representations in the table catches a
  // hand-written scalar offset drift before the raw body consumes the packet.
  V3210_CHECK_U32(1u, 0u, 0u);
  V3210_CHECK_U32(2u, 1u, 0u);
  V3210_CHECK_U32(3u, 2u, 0u);
  V3210_CHECK_U32(4u, 3u, 0u);
  V3210_CHECK_U32(5u, 4u, rows);
  V3210_CHECK_U32(6u, 5u, 1u);
  V3210_CHECK_U32(7u, 6u, rows);
  V3210_CHECK_U32(8u, 7u, ck.ck_intermediate);
  V3210_CHECK_U32(9u, 8u, ck.ck_hidden);
  V3210_CHECK_U32(10u, 9u, ck.ck_hidden);
  V3210_CHECK_U32(11u, 10u, ck.ck_hidden);
  V3210_CHECK_U32(12u, 11u, 0u);
  V3210_CHECK_U32(13u, 12u, ck.ck_intermediate);
  V3210_CHECK_U32(14u, 13u, 1u);
  V3210_CHECK_U32(15u, 14u, (rows + 15u) & ~15u);
  V3210_CHECK_U32(16u, 15u, (ck.ck_intermediate + 127u) & ~127u);
  V3210_CHECK_U32(17u, 16u, ck.ck_hidden);
  V3210_CHECK_U32(18u, 17u, ck.ck_hidden);
  V3210_CHECK_U32(19u, 18u, ck.ck_hidden / 16u);
  V3210_CHECK_U32(20u, 19u, ck.ck_hidden / 16u);
  V3210_CHECK_U32(21u, 20u, (rows + 15u) / 16u);
  V3210_CHECK_U32(22u, 21u, ck.ck_intermediate / 128u);
  V3210_CHECK_U64(23u, p_sorted_token_ids, ck.ck_m64_sorted_ids);
  V3210_CHECK_U64(24u, p_sorted_expert_ids,
                  reinterpret_cast<uint64_t>(stage1_experts));
  V3210_CHECK_U64(25u, p_max_token_id, ck.ck_max_token_id);
  V3210_CHECK_U64(26u, p_a_grid, ck.ck_a);
  V3210_CHECK_U64(27u, p_b_grid, ck.ck_b);
  V3210_CHECK_U64(28u, p_ds_grid, 0ull);
  V3210_CHECK_U64(29u, p_c_grid, ck.ck_stage1_output);
  V3210_CHECK_U64(30u, p_a_scale_grid, ck.ck_a_scale);
  V3210_CHECK_U64(31u, p_b_scale_grid, ck.ck_b_scale);
  V3210_CHECK_U32(32u, 40u, 0u);
  V3210_CHECK_U32(33u, 41u, 0u);
  if(diagnostics != nullptr) {
    v3210_store_system_u64(diagnostics + 3u, 0ull);
    v3210_store_system_u64(diagnostics + 4u, 0ull);
  }
#undef V3210_CHECK_U32
#undef V3210_CHECK_U64
  return 0u;
}
