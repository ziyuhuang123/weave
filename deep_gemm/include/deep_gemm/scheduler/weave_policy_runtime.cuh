#pragma once
// =============================================================================
// CTA-0 policy publication / all-CTA consumption rail.
//
// The control object is a 64-byte aligned header in device global memory.  CTA
// 0 publishes a complete PolicyDecision, then release-stores the epoch.  Every
// other CTA acquire-loads the epoch and copies the decision.  No host round
// trip and no dynamic parallel launch are involved.
// =============================================================================

#include <stdint.h>
#include <string.h>

#include <deep_gemm/scheduler/weave_cost_model.cuh>

#if defined(__HIPCC__) && !defined(__CUDACC__)
#include <hip/hip_runtime.h>
#endif

// Host-side lock-free atomics for the SM0 ready-token queue on the MSVC host
// pass (nvcc/cl).  Device passes use atomicAdd/ld.acquire/st.release instead;
// g++/clang++ host builds use the __atomic builtins.
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#include <intrin.h>
#endif

// Intra-cluster DSMEM push reuses the existing ptx::mapa_shared +
// ptx::st_async_cluster primitives.  The include is CUDA-only: HIP translation
// units parse this header before any HIP runtime header, and the gfx942 path
// uses the simulation implementation below.
#if defined(__CUDACC__) && !defined(__HIPCC__)
#define WEAVE_POLICY_DSMEM_CUDA 1
#include <deep_gemm/ptx/ld_st.cuh>
#endif

namespace weave {

static constexpr uint32_t kPolicyStatusOk = 1u;
static constexpr uint32_t kPolicyStatusInvalid = 2u;
static constexpr uint32_t kPolicyStatusTimeout = 3u;

WEAVE_HOST_DEVICE void reset_policy_control(PolicyControl* control) {
  if (control == nullptr) {
    return;
  }
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  memset(control, 0, sizeof(PolicyControl));
  control->magic = kPolicyMagic;
  control->version = kPolicyVersion;
#endif
}

WEAVE_HOST_DEVICE bool policy_is_valid(const PolicyDecision& d) {
  return d.magic == kPolicyMagic && d.version == kPolicyVersion && d.valid != 0u;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
WEAVE_DEVICE uint32_t policy_load_acquire(const uint32_t* ptr) {
#if defined(__HIPCC__) && !defined(__CUDACC__)
  return __hip_atomic_load(const_cast<uint32_t*>(ptr), __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  uint32_t val;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(val) : "l"(ptr) : "memory");
  return val;
#else
  return atomicAdd(const_cast<uint32_t*>(ptr), 0u);
#endif
}

WEAVE_DEVICE void policy_store_release(uint32_t* ptr, uint32_t value) {
#if defined(__HIPCC__) && !defined(__CUDACC__)
  __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
#else
  atomicExch(ptr, value);
#endif
}

WEAVE_DEVICE uint64_t policy_clock64() {
#if defined(__HIPCC__) && !defined(__CUDACC__)
  return static_cast<uint64_t>(__builtin_amdgcn_s_memrealtime());
#else
  return clock64();
#endif
}

// Called by one thread in CTA 0.  Returns the published decision so CTA 0 can
// reuse it without a second global-memory read.  The Tier-3 OnlineRoutingPolicy
// is computed from the same winning breakdown and published in the global
// control block (never SMEM); unique_tokens optionally supplies the live
// unique-token count for the rank-hit multiplicity mu.
WEAVE_DEVICE PolicyDecision cta0_select_and_publish(
    PolicyControl* control, HardwareArch arch, const WorkloadSpec& wl,
    const WeaveKnobs& knobs, const uint32_t* m_e,
    uint32_t dedup_pulls, uint32_t scatter_count, uint32_t epoch,
    uint32_t unique_tokens = 0u) {
  const uint64_t t0 = policy_clock64();
  // The online policy (including the per-expert profile LUT) is filled IN
  // PLACE in the global control block -- no 2 KB value local on the device
  // stack, zero SMEM.  Readers still gate on the release-published epoch.
  const PolicyDecision decision = cta0_select_policy(
      arch, wl, knobs, m_e, dedup_pulls, scatter_count,
      control != nullptr ? &control->online : nullptr, unique_tokens);
  const uint64_t t1 = policy_clock64();
  if (control == nullptr) {
    return decision;
  }
  control->magic = kPolicyMagic;
  control->version = kPolicyVersion;
  control->status = decision.valid ? kPolicyStatusOk : kPolicyStatusInvalid;
  control->decision_epoch = epoch;
  control->decision = decision;
  control->decision.selection_cycles = static_cast<uint32_t>(t1 - t0);
  control->epoch = epoch;
  policy_store_release(&control->published_epoch, epoch);
  return decision;
}

// B300/H20 publish 64-bit receive counters whose low 32 bits are the token
// count and whose high bits are publication metadata.  CTA 0 snapshots those
// counters into PolicyControl::histogram before selecting.
WEAVE_DEVICE PolicyDecision cta0_select_from_packed_hist(
    PolicyControl* control, HardwareArch arch, const WorkloadSpec& wl,
    const WeaveKnobs& knobs, const uint64_t* packed_counts,
    uint32_t dedup_pulls, uint32_t scatter_count, uint32_t epoch,
    uint32_t publication_retries = 8u, uint32_t unique_tokens = 0u) {
  (void)publication_retries;
  PolicyDecision invalid{};
  invalid.magic = kPolicyMagic;
  invalid.version = kPolicyVersion;
  if (control == nullptr) {
    return invalid;
  }
  if (packed_counts == nullptr || wl.local_experts > kMaxRuntimeExperts) {
    // The histogram cannot be snapshotted.  Publish the invalid decision under
    // this epoch anyway: every CTA spinning in wait_and_copy_policy() then sees
    // valid == 0 at once and keeps its compiled defaults, instead of burning
    // max_spins (about 2M x 32 ns per launch) on an epoch that never arrives.
    control->magic = kPolicyMagic;
    control->version = kPolicyVersion;
    control->status = kPolicyStatusInvalid;
    control->decision_epoch = epoch;
    control->decision = invalid;
    // Invalid decision: readers keep their compiled defaults; the zeroed
    // online policy is never consumed (kernels gate on policy_is_valid).
    zero_online_policy(&control->online);
    control->epoch = epoch;
    policy_store_release(&control->published_epoch, epoch);
    return invalid;
  }
  uint32_t total = 0u;
  for (uint32_t e = 0u; e < wl.local_experts; ++e) {
    const uint32_t count = static_cast<uint32_t>(packed_counts[e]);
    control->histogram[e] = count;
    total += count;
  }
  control->stats_ready_epoch = epoch;
  return cta0_select_and_publish(
      control, arch, wl, knobs, control->histogram,
      dedup_pulls == 0u ? total : dedup_pulls,
      scatter_count == 0u ? total : scatter_count, epoch, unique_tokens);
}

// Each CTA calls this from thread 0, then broadcasts the value with shared
// memory or __shfl_sync.  A bounded spin keeps a missing producer visible.
WEAVE_DEVICE bool wait_and_copy_policy(
    const PolicyControl* control, uint32_t epoch,
    PolicyDecision* out_decision, uint32_t max_spins = 2000000u) {
  if (control == nullptr || out_decision == nullptr) {
    return false;
  }
  uint32_t spins = 0u;
  while (policy_load_acquire(&control->published_epoch) != epoch) {
    if (++spins > max_spins) {
      return false;
    }
#if defined(__HIPCC__) && !defined(__CUDACC__)
    if ((spins & 63u) == 0u) {
      __builtin_amdgcn_s_sleep(1);
    }
#else
    __nanosleep(32u);
#endif
  }
  *out_decision = control->decision;
  return out_decision->magic == kPolicyMagic &&
         out_decision->version == kPolicyVersion &&
         out_decision->valid != 0u;
}
#endif  // CUDA / HIP device

template <typename WorkspaceT>
WEAVE_DEVICE uint32_t count_remote_expert_sends(
    const WorkspaceT& workspace, uint32_t num_experts,
    uint32_t num_experts_per_rank, uint32_t rank_idx) {
  uint32_t total = 0u;
  if (num_experts_per_rank == 0u) {
    return 0u;
  }
  for (uint32_t e = 0u; e < num_experts; ++e) {
    if (e / num_experts_per_rank == rank_idx) {
      continue;
    }
    total += static_cast<uint32_t>(
        *workspace.get_expert_send_count_ptr(e));
  }
  return total;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
WEAVE_DEVICE PolicyDecision load_published_policy(
    const PolicyControl* control) {
  PolicyDecision d{};
  if (control == nullptr) {
    return d;
  }
  (void)policy_load_acquire(&control->published_epoch);
  d = control->decision;
  return d;
}

// Tier-3 consumption: acquire the publication epoch, then read the online
// policy from the global control block.  One 4-byte field load per consumer;
// nothing is copied into SMEM (zero SMEM increase, zero extra registers).
WEAVE_DEVICE OnlineRoutingPolicy load_published_online(
    const PolicyControl* control) {
  OnlineRoutingPolicy o{};
  if (control == nullptr) {
    return o;
  }
  (void)policy_load_acquire(&control->published_epoch);
  o = control->online;
  return o;
}

WEAVE_DEVICE uint32_t load_active_reduce_warps(const PolicyControl* control) {
  if (control == nullptr) {
    return 0u;
  }
  (void)policy_load_acquire(&control->published_epoch);
  return control->online.active_reduce_warps;
}

WEAVE_DEVICE uint32_t load_active_v_head(const PolicyControl* control) {
  if (control == nullptr) {
    return 0u;
  }
  (void)policy_load_acquire(&control->published_epoch);
  return control->online.active_v_head;
}

// Streaming-reduce work stride: with the Reduce lanes gated on, every
// epilogue + dedicated reduce warp participates (with_reduce); gated off, the
// epilogue warps alone cover all (token, chunk) items (epi_only).  An invalid
// policy keeps the compiled default so pinned builds stay deterministic.
WEAVE_DEVICE uint32_t stream_reduce_stride(
    const PolicyControl* control, uint32_t with_reduce, uint32_t epi_only) {
  const PolicyDecision d = load_published_policy(control);
  if (!policy_is_valid(d)) {
    return with_reduce;
  }
  return load_active_reduce_warps(control) > 0u ? with_reduce : epi_only;
}
#endif

// ---------------------------------------------------------------------------
// 3-Tier upgrade, Tier 3a: SM0 dedup-aware co-scheduling + hot/cold wave
// pairing.  Deterministic, integer-only, O(E + 512) counting-sort + union-find
// + capacity-aware dealing (no general comparison sort, no allocation), so one
// CTA-0 thread builds the permutation well inside the 2 us budget at
// E_local <= 256.  Bit-exact Python mirror: calibrate.py
// build_sm0_expert_order (tests enforce permutation equality + span/variance).
//
// Input per expert: token_count, multi_hit_tokens, primary_partner (expert id
// sharing the most multi-hit tokens; >= E means "none").  Output: a
// permutation order_out[E]:
//   * dedup_active: multi-hit partner edges form clusters (union-find, root =
//     min member id); cluster members stay ADJACENT so a token's completion
//     span max(pos)-min(pos) collapses and next_cnt == hit_count fires
//     continuously from wave 1 (Scatter warps never wait on an empty queue,
//     partial-sum scratch rows stay L2-resident).
//   * hot/cold: hot_thresh = max(13, 2*floor(M/active)); clusters are dealt
//     hot-first round-robin into the ceil(active/wave_size) waves with enough
//     remaining capacity (the last wave is partial), then cold clusters over
//     the same capacities; inside a wave units emit [Hot, Cold, Hot, Cold...]
//     so no wave is packed purely with M <= 12 cold experts.
//   * zero-token experts are appended last in id order.
// ---------------------------------------------------------------------------

static constexpr uint32_t kSm0OrderSentinel = 0xffffffffu;

// Ordering-mode verdict (reported through mode_out, mirrored in calibrate.py).
// The hot/cold pairing boundary is EXPLICIT, not implicit:
//   * dedup active AND real co-occurrence clusters -> kSm0ModeDedupCluster:
//     adjacency of a token's producer set wins, no [Hot,Cold] interleave.
//   * dedup active, per-expert multi_hit only (no partner/anchor edges) ->
//     kSm0ModePendulum: most-shared expert at the centre, symmetric fan-out.
//   * dedup inactive AND experts_per_wave >= 2 -> kSm0ModeHotCold: the only
//     mode that emits [Hot, Cold, Hot, Cold, ...] inside a wave.
//   * experts_per_wave < 2 (one expert fills the whole card, W = 1) ->
//     kSm0ModeNoPair: pairing is impossible, plain load-descending order.
static constexpr uint32_t kSm0ModeHotCold = 0u;
static constexpr uint32_t kSm0ModeDedupCluster = 1u;
static constexpr uint32_t kSm0ModePendulum = 2u;
static constexpr uint32_t kSm0ModeNoPair = 3u;

// Caller-provided scratch (global memory or SMEM on device; heap on host).
// 16 x 256 x 4 B = 16 KB; nothing is allocated inside the helper.
struct Sm0OrderScratch {
  uint32_t sorted[kMaxRuntimeExperts];
  uint32_t tmp[kMaxRuntimeExperts];
  uint32_t parent[kMaxRuntimeExperts];
  uint32_t unit_first[kMaxRuntimeExperts];
  uint32_t unit_last[kMaxRuntimeExperts];
  uint32_t next_member[kMaxRuntimeExperts];
  uint32_t hot_units[kMaxRuntimeExperts];
  uint32_t cold_units[kMaxRuntimeExperts];
  uint32_t bucket[256];
  uint32_t remaining[kMaxRuntimeExperts];
  uint32_t hot_head[kMaxRuntimeExperts];
  uint32_t hot_tail[kMaxRuntimeExperts];
  uint32_t hot_next[kMaxRuntimeExperts];
  uint32_t cold_head[kMaxRuntimeExperts];
  uint32_t cold_tail[kMaxRuntimeExperts];
  uint32_t cold_next[kMaxRuntimeExperts];
};

WEAVE_HOST_DEVICE uint32_t sm0_find(uint32_t* parent, uint32_t x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

// Parallel-safe passes are factored out with a (begin, stride) walk so the
// single-thread entry point uses (0, 1) and the warp-cooperative entry point
// uses (lane, 32) over the IDENTICAL code.  Integer add/min/max are
// associative, so a warp reduction of the per-lane partials is bit-identical
// to the serial accumulation.
WEAVE_HOST_DEVICE void sm0_stats_pass(
    const uint32_t* token_counts, uint32_t num_experts,
    uint32_t begin, uint32_t stride, uint32_t* total, uint32_t* active,
    uint32_t* cmin, uint32_t* cmax) {
  uint32_t t = 0u, a = 0u, lo = 0xffffffffu, hi = 0u;
  for (uint32_t e = begin; e < num_experts; e += stride) {
    const uint32_t c = token_counts[e];
    t += c;
    if (c > 0u) {
      ++a;
      if (c < lo) lo = c;
      if (c > hi) hi = c;
    }
  }
  *total = t;
  *active = a;
  *cmin = lo;
  *cmax = hi;
}

WEAVE_HOST_DEVICE void sm0_unit_init_pass(Sm0OrderScratch* s,
                                          uint32_t num_experts,
                                          uint32_t begin, uint32_t stride) {
  for (uint32_t e = begin; e < num_experts; e += stride) {
    s->unit_first[e] = kSm0OrderSentinel;
    s->unit_last[e] = kSm0OrderSentinel;
    s->next_member[e] = kSm0OrderSentinel;
    s->remaining[e] = 0u;
  }
}

WEAVE_HOST_DEVICE void sm0_parent_init_pass(Sm0OrderScratch* s,
                                            uint32_t num_experts,
                                            uint32_t begin, uint32_t stride) {
  for (uint32_t e = begin; e < num_experts; e += stride) s->parent[e] = e;
}

// Cluster size of one unit root, walked from its member chain.
WEAVE_HOST_DEVICE uint32_t sm0_unit_size(const Sm0OrderScratch* s,
                                         uint32_t root) {
  uint32_t sz = 0u;
  for (uint32_t m = s->unit_first[root]; m != kSm0OrderSentinel;
       m = s->next_member[m]) ++sz;
  return sz;
}

// Composite pendulum key: multi-hit weight first, token count second, both
// count-descending via 16-bit complement so one stable LSD radix reproduces
// the Python mirror's sorted(key=(-multi_hit, -count, id)) contract.
WEAVE_HOST_DEVICE constexpr uint32_t sm0_pendulum_key(uint32_t hits,
                                                      uint32_t count) {
  const uint32_t h = hits < 65535u ? hits : 65535u;
  const uint32_t c = count < 65535u ? count : 65535u;
  return ((65535u - h) << 16) | (65535u - c);
}

// 256-bucket sweeps shared by both radix paths.  Every loop stays ROLLED:
// unrolling a 256-entry sweep (or the 4-way prefix chains) inflates the live
// register set past the 64-register primitive budget on sm_103a while the
// work is memory-bound on the scratch anyway.
WEAVE_HOST_DEVICE void sm0_bucket_clear(uint32_t* bucket) {
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll 1
#endif
  for (uint32_t i = 0u; i < 256u; ++i) bucket[i] = 0u;
}

// Exclusive prefix over the 256 buckets.  DEVICE: one accumulator, no local
// array -- a rolled 4-way split would index a quarter[4] array dynamically and
// ptxas puts it on the device stack (16 B frame), breaking the 0-stack budget.
// HOST: the 4-way split (four independent 64-entry chains + rebase) shortens
// the dependency chain on in-order cores, where stack space is free.  Both
// compute the identical exclusive prefix, so the permutation is unchanged.
WEAVE_HOST_DEVICE void sm0_bucket_prefix(uint32_t* bucket) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  uint32_t acc = 0u;
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll 1
#endif
  for (uint32_t i = 0u; i < 256u; ++i) {
    const uint32_t c = bucket[i];
    bucket[i] = acc;
    acc += c;
  }
#else
  uint32_t quarter[4];
  for (uint32_t q = 0u; q < 4u; ++q) {
    uint32_t acc = 0u;
    for (uint32_t i = q * 64u; i < q * 64u + 64u; ++i) {
      const uint32_t c = bucket[i];
      bucket[i] = acc;
      acc += c;
    }
    quarter[q] = acc;
  }
  uint32_t base = 0u;
  for (uint32_t q = 0u; q < 4u; ++q) {
    const uint32_t next = base + quarter[q];
    for (uint32_t i = q * 64u; i < q * 64u + 64u; ++i) bucket[i] += base;
    base = next;
  }
#endif
}

// Pendulum (centre-out) placement of the active experts, ranked by multi-hit
// weight then load.  Four stable 8-bit LSD radix passes on the composite key
// (a pass whose byte is identical across every key is skipped), then the
// most-shared expert takes the centre slot and the rest fan out symmetrically
// right/left.  All state lives in the caller's scratch -- zero stack arrays.
WEAVE_HOST_DEVICE uint32_t sm0_pendulum_order(
    const uint32_t* token_counts, const uint32_t* multi_hit,
    uint32_t num_experts, uint32_t n, uint32_t* order_out,
    Sm0OrderScratch* s) {
  // Keep the pass loop rolled: unrolling all four radix passes into one live
  // range triples the register footprint on sm_103a for zero throughput gain
  // (each pass is a full 256-bucket sweep over the same scratch).
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll 1
#endif
  for (uint32_t pass = 0u; pass < 4u; ++pass) {
    const uint32_t shift = pass * 8u;
    uint32_t or_all = 0u, and_all = 0xffu;
    for (uint32_t i = 0u; i < n; ++i) {
      const uint32_t e = s->sorted[i];
      const uint32_t b =
          (sm0_pendulum_key(multi_hit[e], token_counts[e]) >> shift) & 0xffu;
      or_all |= b;
      and_all &= b;
    }
    if (or_all == and_all) continue;   // uniform byte: pass is a no-op
    sm0_bucket_clear(s->bucket);
    for (uint32_t i = 0u; i < n; ++i) {
      const uint32_t e = s->sorted[i];
      ++s->bucket[(sm0_pendulum_key(multi_hit[e], token_counts[e]) >> shift) &
                  0xffu];
    }
    sm0_bucket_prefix(s->bucket);
    for (uint32_t i = 0u; i < n; ++i) {
      const uint32_t e = s->sorted[i];
      const uint32_t b =
          (sm0_pendulum_key(multi_hit[e], token_counts[e]) >> shift) & 0xffu;
      s->tmp[s->bucket[b]++] = e;
    }
    for (uint32_t i = 0u; i < n; ++i) s->sorted[i] = s->tmp[i];
  }
  const uint32_t center = (n - 1u) / 2u;
  uint32_t pos = center;
  uint32_t right = center + 1u;
  int32_t left = static_cast<int32_t>(center) - 1;
  bool go_right = true;
  for (uint32_t i = 0u; i < n; ++i) {
    order_out[pos] = s->sorted[i];
    if (go_right && right < n) {
      pos = right++;
    } else if (left >= 0) {
      pos = static_cast<uint32_t>(left--);
    } else if (right < n) {
      pos = right++;
    }
    go_right = !go_right;
  }
  uint32_t idx = n;
  for (uint32_t e = 0u; e < num_experts; ++e) {
    if (token_counts[e] == 0u) order_out[idx++] = e;
  }
  return idx;
}

WEAVE_HOST_DEVICE uint32_t build_sm0_expert_order(
    const uint32_t* token_counts, const uint32_t* multi_hit,
    const uint32_t* primary_partner, uint32_t num_experts,
    uint32_t wave_size, uint32_t dedup_active,
    uint32_t* order_out, Sm0OrderScratch* s,
    uint32_t* mode_out = nullptr) {
  if (mode_out != nullptr) *mode_out = kSm0ModeNoPair;
  if (num_experts == 0u) return 0u;
  if (num_experts > kMaxRuntimeExperts || token_counts == nullptr ||
      order_out == nullptr || s == nullptr) {
    return 0u;
  }
  if (wave_size == 0u) wave_size = 1u;
  uint32_t total = 0u, active = 0u, cmin = 0u, cmax = 0u;
  sm0_stats_pass(token_counts, num_experts, 0u, 1u,
                 &total, &active, &cmin, &cmax);
  if (active == 0u) {
    // No routed work: identity order (every expert is inactive).
    for (uint32_t e = 0u; e < num_experts; ++e) order_out[e] = e;
    return num_experts;
  }
  const uint32_t hot_thresh =
      2u * (total / active) > 13u ? 2u * (total / active) : 13u;

  // (1) Stable count-descending / id-ascending order.  Uniform counts need
  // no sort (id order IS the tie rule); small active sets use insertion sort
  // (cheaper than the 2 x 256 bucket passes); otherwise a 16-bit LSD radix
  // on key = 65535 - min(count, 65535).  All three produce the identical
  // permutation (the Python mirror's sorted() contract).
  uint32_t n = 0u;
  for (uint32_t e = 0u; e < num_experts; ++e) {
    if (token_counts[e] > 0u) s->sorted[n++] = e;
  }
  if (cmin != cmax) {
    if (n <= 32u) {
      for (uint32_t i = 1u; i < n; ++i) {
        const uint32_t e = s->sorted[i];
        const uint32_t ce = token_counts[e] < 65535u ? token_counts[e] : 65535u;
        uint32_t j = i;
        while (j > 0u) {
          const uint32_t cp = token_counts[s->sorted[j - 1u]];
          const uint32_t kp = cp < 65535u ? cp : 65535u;
          if (kp >= ce) break;   // stable: equal keys keep id-asc order
          s->sorted[j] = s->sorted[j - 1u];
          --j;
        }
        s->sorted[j] = e;
      }
    } else {
      // When every key shares one high-byte band (all counts inside a 256
      // window -- every decode / small-count histogram), the low-byte pass
      // alone is the full stable order and pass 2 is skipped.
      const uint32_t key_lo = 65535u - (cmax < 65535u ? cmax : 65535u);
      const uint32_t key_hi = 65535u - (cmin < 65535u ? cmin : 65535u);
      const uint32_t passes = (key_lo >> 8) == (key_hi >> 8) ? 1u : 2u;
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll 1
#endif
      for (uint32_t pass = 0u; pass < passes; ++pass) {
        const uint32_t shift = pass * 8u;
        sm0_bucket_clear(s->bucket);
        for (uint32_t i = 0u; i < n; ++i) {
          const uint32_t c = token_counts[s->sorted[i]];
          const uint32_t key = 65535u - (c < 65535u ? c : 65535u);
          ++s->bucket[(key >> shift) & 0xffu];
        }
        sm0_bucket_prefix(s->bucket);
        for (uint32_t i = 0u; i < n; ++i) {
          const uint32_t e = s->sorted[i];
          const uint32_t c = token_counts[e];
          const uint32_t key = 65535u - (c < 65535u ? c : 65535u);
          s->tmp[s->bucket[(key >> shift) & 0xffu]++] = e;
        }
        for (uint32_t i = 0u; i < n; ++i) s->sorted[i] = s->tmp[i];
      }
    }
  }

  // (2) Union-find over multi-hit partner edges (root = min member id).
  // The path-compressed root cache and the unit-array init share one pass
  // (tmp is free after the sort).
  const bool dedup = dedup_active != 0u && multi_hit != nullptr;
  const bool have_edges = dedup && primary_partner != nullptr;
  if (have_edges) {
    sm0_parent_init_pass(s, num_experts, 0u, 1u);
    for (uint32_t e = 0u; e < num_experts; ++e) {
      const uint32_t p = primary_partner[e];
      if (multi_hit[e] > 0u && p < num_experts && p != e &&
          token_counts[p] > 0u) {
        const uint32_t ra = sm0_find(s->parent, e);
        const uint32_t rb = sm0_find(s->parent, p);
        if (ra != rb) s->parent[ra > rb ? ra : rb] = ra < rb ? ra : rb;
      }
    }
  }
  sm0_unit_init_pass(s, num_experts, 0u, 1u);
  for (uint32_t e = 0u; e < num_experts; ++e) {
    s->tmp[e] = have_edges ? sm0_find(s->parent, e) : e;
  }
  // Cluster census + multi-hit census decide the mode BEFORE any placement.
  bool has_cluster = false;
  uint32_t max_hits = 0u;
  for (uint32_t e = 0u; e < num_experts; ++e) {
    if (token_counts[e] == 0u) continue;
    if (++s->remaining[s->tmp[e]] >= 2u) has_cluster = true;
    if (dedup && multi_hit[e] > max_hits) max_hits = multi_hit[e];
  }
  if (dedup && !has_cluster && max_hits > 0u) {
    // No co-occurrence edges to cluster, but per-expert multi-hit weights are
    // known: centre the most-shared experts (pendulum ordering) so a multi-hit
    // token's producers land in one compact central window.
    if (mode_out != nullptr) *mode_out = kSm0ModePendulum;
    return sm0_pendulum_order(token_counts, multi_hit, num_experts, n,
                              order_out, s);
  }
  const uint32_t mode = (dedup && has_cluster) ? kSm0ModeDedupCluster
      : (wave_size >= 2u ? kSm0ModeHotCold : kSm0ModeNoPair);
  if (mode_out != nullptr) *mode_out = mode;
  if (mode == kSm0ModeNoPair) {
    // W = 1: one expert fills the whole card, so there is no second slot to
    // pair with.  Load-descending order, zero-token experts last.
    uint32_t idx = 0u;
    for (uint32_t i = 0u; i < n; ++i) order_out[idx++] = s->sorted[i];
    for (uint32_t e = 0u; e < num_experts; ++e) {
      if (token_counts[e] == 0u) order_out[idx++] = e;
    }
    return idx;
  }

  // (3) Units (clusters / singletons) in first-appearance order = hottest
  // member descending; members keep count-desc / id-asc order.
  uint32_t n_hot = 0u, n_cold = 0u;
  for (uint32_t i = 0u; i < n; ++i) {
    const uint32_t e = s->sorted[i];
    const uint32_t r = s->tmp[e];
    if (s->unit_last[r] == kSm0OrderSentinel) {
      s->unit_first[r] = e;
      s->unit_last[r] = e;
      if (token_counts[e] >= hot_thresh) s->hot_units[n_hot++] = r;
      else s->cold_units[n_cold++] = r;
    } else {
      s->next_member[s->unit_last[r]] = e;
      s->unit_last[r] = e;
    }
  }
  // Cluster sizes (unit_last is free after the chain build): dealing reads
  // them in O(1) instead of walking every member chain again.
  for (uint32_t i = 0u; i < n_hot; ++i) {
    s->unit_last[s->hot_units[i]] = sm0_unit_size(s, s->hot_units[i]);
  }
  for (uint32_t i = 0u; i < n_cold; ++i) {
    s->unit_last[s->cold_units[i]] = sm0_unit_size(s, s->cold_units[i]);
  }

  // (4) Capacity-aware dealing into ceil(active/wave_size) waves.  Hot units
  // first, then cold units over the remaining capacities; whole clusters
  // never straddle a wave (overflow goes to the emptiest wave, lowest index
  // on ties, preserving adjacency).
  const uint32_t num_waves = ceil_div_u32(active, wave_size);
  for (uint32_t w = 0u; w < num_waves; ++w) {
    s->remaining[w] = wave_size;
    s->hot_head[w] = s->hot_tail[w] = kSm0OrderSentinel;
    s->cold_head[w] = s->cold_tail[w] = kSm0OrderSentinel;
  }
  s->remaining[num_waves - 1u] = active - wave_size * (num_waves - 1u);
  for (uint32_t kind = 0u; kind < 2u; ++kind) {
    const uint32_t* units = kind == 0u ? s->hot_units : s->cold_units;
    const uint32_t n_units = kind == 0u ? n_hot : n_cold;
    uint32_t* head = kind == 0u ? s->hot_head : s->cold_head;
    uint32_t* tail = kind == 0u ? s->hot_tail : s->cold_tail;
    uint32_t* next = kind == 0u ? s->hot_next : s->cold_next;
    uint32_t start = 0u;
    for (uint32_t i = 0u; i < n_units; ++i) {
      const uint32_t u = units[i];
      const uint32_t sz = s->unit_last[u];   // precomputed cluster size
      uint32_t found = kSm0OrderSentinel;
      for (uint32_t k = 0u; k < num_waves; ++k) {
        const uint32_t w = (start + k) % num_waves;
        if (s->remaining[w] >= sz) { found = w; break; }
      }
      if (found == kSm0OrderSentinel) {
        found = 0u;
        for (uint32_t w = 1u; w < num_waves; ++w) {
          if (s->remaining[w] > s->remaining[found]) found = w;
        }
      }
      if (head[found] == kSm0OrderSentinel) head[found] = u;
      else next[tail[found]] = u;
      tail[found] = u;
      next[u] = kSm0OrderSentinel;
      // Saturate at 0: an overflow cluster (larger than every wave's
      // remaining capacity) keeps adjacency and empties its wave; the
      // Python mirror clamps identically (bit-exact on the overflow path).
      s->remaining[found] =
          s->remaining[found] > sz ? s->remaining[found] - sz : 0u;
      start = (found + 1u) % num_waves;
    }
  }

  // (5) Emit waves.  ONLY kSm0ModeHotCold interleaves [Hot, Cold, Hot, Cold];
  // in kSm0ModeDedupCluster the co-occurrence adjacency wins and whole units
  // are emitted hot-first (the dealing above already spreads hot units over
  // every wave, so no wave is packed purely with cold experts either way).
  uint32_t idx = 0u;
  for (uint32_t w = 0u; w < num_waves; ++w) {
    uint32_t hu = s->hot_head[w];
    uint32_t cu = s->cold_head[w];
    while (mode == kSm0ModeHotCold && hu != kSm0OrderSentinel &&
           cu != kSm0OrderSentinel) {
      for (uint32_t m = s->unit_first[hu]; m != kSm0OrderSentinel;
           m = s->next_member[m]) order_out[idx++] = m;
      hu = s->hot_next[hu];
      for (uint32_t m = s->unit_first[cu]; m != kSm0OrderSentinel;
           m = s->next_member[m]) order_out[idx++] = m;
      cu = s->cold_next[cu];
    }
    while (hu != kSm0OrderSentinel) {
      for (uint32_t m = s->unit_first[hu]; m != kSm0OrderSentinel;
           m = s->next_member[m]) order_out[idx++] = m;
      hu = s->hot_next[hu];
    }
    while (cu != kSm0OrderSentinel) {
      for (uint32_t m = s->unit_first[cu]; m != kSm0OrderSentinel;
           m = s->next_member[m]) order_out[idx++] = m;
      cu = s->cold_next[cu];
    }
  }
  // (6) Zero-token experts last, id ascending.
  for (uint32_t e = 0u; e < num_experts; ++e) {
    if (token_counts[e] == 0u) order_out[idx++] = e;
  }
  return idx;
}

// ---------------------------------------------------------------------------
// 3-Tier upgrade, Tier 3b: SM0 centralized ready-token queue.  A lock-free
// MPSC ring that replaces blind `.sys` polling and pending-bitmask register
// bloat with guaranteed-ready pops:
//   * Sender-side local pre-reduce: each (dst_rank, token) starts with
//     remaining_hits = hit_count; every local expert completion calls
//     rtq_notify_local_prereduce (atomicSub); the LAST hit pushes the encoded
//     row, so Scatter warps pop only fully pre-reduced deduplicated rows.
//   * Receiver-side top-k combine: each local token starts with
//     arrived_count = 0; every remote arrival calls rtq_notify_remote_arrival
//     (atomicAdd); the expected-th arrival pushes the token, so Reduce warps
//     pop only tokens whose full top-k (or active-rank set) has arrived.
//
// Protocol (no CAS, no wrap within an epoch): producers reserve a slot with
// atomicAdd(tail), write the token, then release-store the cell's ready flag
// to the current epoch.  The single consumer acquire-loads the head cell's
// ready flag; only a flag equal to the epoch guarantees the token write is
// visible.  rtq_begin_epoch resets the cursors and bumps the epoch, so stale
// cells from prior steps can never be popped (epoch starts >= 1; zeroed cells
// carry ready == 0).  Pushes beyond capacity set a sticky overflow flag and
// return false (the caller falls back to its legacy path); the consumer stops
// at the first unpublished cell.
// ---------------------------------------------------------------------------

struct Sm0ReadyCell {
  uint32_t token;
  uint32_t ready;   // epoch value published by the producer
};

struct alignas(64) Sm0ReadyTokenQueue {
  uint32_t tail;        // producer reservation cursor (atomicAdd)
  uint32_t head;        // single-consumer pop cursor
  uint32_t epoch;       // current step epoch (>= 1)
  uint32_t capacity;    // max pushes per epoch (power-of-two ring sizing)
  uint32_t overflow;    // sticky: a push exceeded capacity
  uint32_t reserved[11];
};

WEAVE_HOST_DEVICE uint32_t rtq_atomic_add(uint32_t* ptr, uint32_t value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return atomicAdd(ptr, value);
#elif defined(_MSC_VER)
  return static_cast<uint32_t>(_InterlockedExchangeAdd(
      reinterpret_cast<volatile long*>(ptr), static_cast<long>(value)));
#else
  return __atomic_fetch_add(ptr, value, __ATOMIC_ACQ_REL);
#endif
}

WEAVE_HOST_DEVICE uint32_t rtq_atomic_sub(uint32_t* ptr, uint32_t value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return atomicSub(ptr, value);
#elif defined(_MSC_VER)
  return static_cast<uint32_t>(_InterlockedExchangeAdd(
      reinterpret_cast<volatile long*>(ptr), -static_cast<long>(value)));
#else
  return __atomic_fetch_sub(ptr, value, __ATOMIC_ACQ_REL);
#endif
}

WEAVE_HOST_DEVICE uint32_t rtq_load_acquire(const uint32_t* ptr) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  return __hip_atomic_load(const_cast<uint32_t*>(ptr), __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__)
  uint32_t val;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(val) : "l"(ptr) : "memory");
  return val;
#elif defined(_MSC_VER)
  return static_cast<uint32_t>(_InterlockedCompareExchange(
      reinterpret_cast<volatile long*>(const_cast<uint32_t*>(ptr)), 0, 0));
#else
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#endif
}

WEAVE_HOST_DEVICE void rtq_store_release(uint32_t* ptr, uint32_t value) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__)
  asm volatile("st.release.gpu.global.u32 [%0], %1;"
               :: "l"(ptr), "r"(value) : "memory");
#elif defined(_MSC_VER)
  _InterlockedExchange(reinterpret_cast<volatile long*>(ptr),
                       static_cast<long>(value));
#else
  __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
#endif
}

WEAVE_HOST_DEVICE void rtq_init(Sm0ReadyTokenQueue* q, uint32_t capacity,
                                uint32_t epoch) {
  q->tail = 0u;
  q->head = 0u;
  q->epoch = epoch;
  q->capacity = capacity;
  q->overflow = 0u;
}

// Next step: reset cursors and publish a FRESH epoch (monotone, never reused)
// so prior-epoch cells can never be popped.
WEAVE_HOST_DEVICE void rtq_begin_epoch(Sm0ReadyTokenQueue* q, uint32_t epoch) {
  q->tail = 0u;
  q->head = 0u;
  q->overflow = 0u;
  q->epoch = epoch;
}

WEAVE_HOST_DEVICE bool rtq_push(Sm0ReadyTokenQueue* q, Sm0ReadyCell* cells,
                                uint32_t token) {
  const uint32_t slot = rtq_atomic_add(&q->tail, 1u);
  if (slot >= q->capacity) {
    q->overflow = 1u;
    return false;
  }
  cells[slot].token = token;
  rtq_store_release(&cells[slot].ready, q->epoch);
  return true;
}

WEAVE_HOST_DEVICE bool rtq_pop(Sm0ReadyTokenQueue* q, const Sm0ReadyCell* cells,
                               uint32_t* token_out) {
  const uint32_t h = q->head;
  if (h >= q->capacity) return false;
  if (rtq_load_acquire(&cells[h].ready) != q->epoch) return false;
  *token_out = cells[h].token;
  q->head = h + 1u;
  return true;
}

// (dst_rank, token) row encoding: rank in the high byte, token < 2^24.
WEAVE_HOST_DEVICE constexpr uint32_t rtq_encode_row(uint32_t dst_rank,
                                                    uint32_t token) {
  return ((dst_rank & 0xffu) << 24) | (token & 0x00ffffffu);
}
WEAVE_HOST_DEVICE constexpr uint32_t rtq_decode_rank(uint32_t row) {
  return row >> 24;
}
WEAVE_HOST_DEVICE constexpr uint32_t rtq_decode_token(uint32_t row) {
  return row & 0x00ffffffu;
}

// Sender-side: remaining_hits starts at hit_count; the last local pre-reduce
// completion pushes the guaranteed-ready deduplicated row.
WEAVE_HOST_DEVICE bool rtq_notify_local_prereduce(
    Sm0ReadyTokenQueue* q, Sm0ReadyCell* cells, uint32_t* remaining_hits,
    uint32_t dst_rank, uint32_t token) {
  const uint32_t prev = rtq_atomic_sub(remaining_hits, 1u);
  if (prev == 1u) {
    return rtq_push(q, cells, rtq_encode_row(dst_rank, token));
  }
  return false;
}

// Receiver-side: arrived_count starts at 0; the expected-th arrival (full
// top-k or the active remote-rank set) pushes the 100%-ready local token.
WEAVE_HOST_DEVICE bool rtq_notify_remote_arrival(
    Sm0ReadyTokenQueue* q, Sm0ReadyCell* cells, uint32_t* arrived_count,
    uint32_t expected_hits, uint32_t token) {
  const uint32_t prev = rtq_atomic_add(arrived_count, 1u);
  if (prev + 1u == expected_hits) {
    return rtq_push(q, cells, token);
  }
  return false;
}

// ===========================================================================
// GPU-scope memory-order primitives shared by the lock-free pre-reduce and
// the sharded ready-token rings.  Device: explicit PTX (CUDA) / __hip_atomic
// (HIP).  Host: the single-threaded simulation the unit tests drive.
// ===========================================================================
WEAVE_HOST_DEVICE uint32_t weave_ld_acq_u32(const uint32_t* ptr) {
  return rtq_load_acquire(ptr);
}
WEAVE_HOST_DEVICE void weave_st_rel_u32(uint32_t* ptr, uint32_t value) {
  rtq_store_release(ptr, value);
}

// fence.acq_rel.gpu: orders EVERY prior global write of this thread (including
// fire-and-forget red.global.add) before every later one, at GPU scope.
WEAVE_HOST_DEVICE void weave_fence_acq_rel_gpu() {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  __hip_atomic_thread_fence(__ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__)
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
#elif defined(_MSC_VER)
  _ReadWriteBarrier();
#else
  __atomic_thread_fence(__ATOMIC_ACQ_REL);
#endif
}

// fence.acquire.gpu: the last claimer's read-side barrier -- no accumulated
// fragment may be read before every contributor's reduction is visible.
WEAVE_HOST_DEVICE void weave_fence_acquire_gpu() {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  __hip_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__)
  asm volatile("fence.acquire.gpu;" ::: "memory");
#elif defined(_MSC_VER)
  _ReadWriteBarrier();
#else
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
#endif
}

// atom.acq_rel.gpu.global.add.u32: the completion counter / claim primitive.
WEAVE_HOST_DEVICE uint32_t weave_atom_add_acq_rel_u32(uint32_t* ptr,
                                                      uint32_t value) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  return __hip_atomic_fetch_add(ptr, value, __ATOMIC_ACQ_REL,
                                __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old) : "l"(ptr), "r"(value) : "memory");
  return old;
#else
  return rtq_atomic_add(ptr, value);
#endif
}

// atom.acq_rel.gpu.global.cas.b32: ONE bounded claim attempt, never a retry
// spin.  A failed claim means another consumer progressed (lock-free).
WEAVE_HOST_DEVICE bool weave_cas_acq_rel_u32(uint32_t* ptr, uint32_t expected,
                                             uint32_t desired) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  return __hip_atomic_compare_exchange_strong(
      ptr, &expected, desired, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
      __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.cas.b32 %0, [%1], %2, %3;"
               : "=r"(old) : "l"(ptr), "r"(expected), "r"(desired) : "memory");
  return old == expected;
#elif defined(_MSC_VER)
  return _InterlockedCompareExchange(
             reinterpret_cast<volatile long*>(ptr),
             static_cast<long>(desired),
             static_cast<long>(expected)) == static_cast<long>(expected);
#else
  // Failure order must not be stronger than acquire (C++ CAS contract).
  return __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#endif
}

// Fire-and-forget global reduction: no return dependency, no lock register.
WEAVE_HOST_DEVICE void weave_red_add_f32(float* dst, float value) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  atomicAdd(dst, value);
#elif defined(__CUDA_ARCH__)
  asm volatile("red.global.add.f32 [%0], %1;" :: "l"(dst), "f"(value)
               : "memory");
#else
  *dst += value;
#endif
}

// Vectorized 16-byte reduction (one instruction per 4 floats).  dst must be
// 16-byte aligned; the caller checks alignment before taking this path.
WEAVE_HOST_DEVICE void weave_red_add_v4_f32(float* dst, float a, float b,
                                            float c, float d) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  atomicAdd(dst, a);
  atomicAdd(dst + 1, b);
  atomicAdd(dst + 2, c);
  atomicAdd(dst + 3, d);
#elif defined(__CUDA_ARCH__)
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
               :: "l"(dst), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
#else
  dst[0] += a;
  dst[1] += b;
  dst[2] += c;
  dst[3] += d;
#endif
}

// ---------------------------------------------------------------------------
// Lock-free local pre-reduce (replaces the `while (true) atom.cas` spinlock).
//
// Every contributor red.global.add's its fragment into the shared accumulator
// and then bumps the completion counter; the contributor observing
// old == expected - 1 is the unique last one and owns the forward.  No loop
// over a lock => no livelock by construction.
//
// MEMORY-ORDER HARDENING (Hopper/Blackwell weak ordering): red.global.add is
// fire-and-forget and is NOT ordered by a later relaxed atomic on a DIFFERENT
// address.  Without a fence the last contributor can observe done_cnt ==
// expected while another contributor's reductions are still in flight, and
// read a half-accumulated row.  Therefore:
//   (1) an explicit fence.acq_rel.gpu AFTER the reduction loop and BEFORE the
//       counter increment, and
//   (2) the counter increment itself is atom.acq_rel.gpu.global.add.u32, and
//   (3) the last claimer issues fence.acquire.gpu before reading the row
//       (lockfree_local_prereduce_load).
// ---------------------------------------------------------------------------
WEAVE_HOST_DEVICE bool lockfree_local_prereduce_accumulate(
    float* scratch, const float* fragment, uint32_t elems,
    uint32_t* done_cnt, uint32_t expected_contributors) {
  if (scratch == nullptr || fragment == nullptr || done_cnt == nullptr ||
      expected_contributors == 0u) {
    return false;
  }
  uint32_t i = 0u;
  // Vector path only when BOTH buffers are 16 B aligned (one red.v4 per 4).
  const bool aligned =
      ((reinterpret_cast<uintptr_t>(scratch) |
        reinterpret_cast<uintptr_t>(fragment)) & 15u) == 0u;
  // Both reduction loops stay rolled: unrolling them keeps 4 floats per
  // unrolled body live and pushes the primitive past its 28-register budget.
  if (aligned) {
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll 1
#endif
    for (; i + 4u <= elems; i += 4u) {
      weave_red_add_v4_f32(scratch + i, fragment[i], fragment[i + 1u],
                           fragment[i + 2u], fragment[i + 3u]);
    }
  }
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll 1
#endif
  for (; i < elems; ++i) weave_red_add_f32(scratch + i, fragment[i]);
  weave_fence_acq_rel_gpu();
  const uint32_t old = weave_atom_add_acq_rel_u32(done_cnt, 1u);
  return old + 1u == expected_contributors;
}

// Last-claimer read-back: acquire fence FIRST, then copy the accumulated row.
WEAVE_HOST_DEVICE void lockfree_local_prereduce_load(
    const float* scratch, uint32_t elems, float* out) {
  if (scratch == nullptr || out == nullptr) return;
  weave_fence_acquire_gpu();
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll 1
#endif
  for (uint32_t i = 0u; i < elems; ++i) out[i] = scratch[i];
}

#if defined(__CUDACC__) || defined(__HIPCC__)
// ---------------------------------------------------------------------------
// Warp-cooperative SM0 ordering (low-register / zero-big-stack design).
//
// ALL 32 lanes of the scheduler warp call this with their lane index.  Every
// array lives in the caller's external Sm0OrderScratch (SMEM or global) -- no
// per-thread array is ever materialized, so there is no 1.5 KB stack frame.
// The parallel-safe passes (histogram stats, union-find parent init, unit
// init, cluster sizes) run strided across the lanes over the SAME helper code
// the serial entry point uses; the order-dependent passes (stable compaction,
// radix scatter, capacity dealing, emission) stay on lane 0 so the emitted
// permutation is BIT-IDENTICAL to build_sm0_expert_order.  Integer add/min/max
// are associative, so the shfl reduction of the per-lane partials cannot
// change the result.
// ---------------------------------------------------------------------------
WEAVE_DEVICE uint32_t sm0_warp_reduce_u32(uint32_t v, uint32_t op) {
  // op: 0 = add, 1 = min, 2 = max.  Full-warp butterfly, 5 rounds.
#pragma unroll
  for (uint32_t off = 16u; off > 0u; off >>= 1u) {
#if defined(__HIPCC__) && !defined(__CUDACC__)
    const uint32_t o = __shfl_xor(static_cast<int>(v), static_cast<int>(off));
#else
    const uint32_t o = __shfl_xor_sync(0xffffffffu, v, off);
#endif
    if (op == 0u) v += o;
    else if (op == 1u) v = v < o ? v : o;
    else v = v > o ? v : o;
  }
  return v;
}

WEAVE_DEVICE uint32_t build_sm0_expert_order_warp(
    const uint32_t* token_counts, const uint32_t* multi_hit,
    const uint32_t* primary_partner, uint32_t num_experts,
    uint32_t wave_size, uint32_t dedup_active,
    uint32_t* order_out, Sm0OrderScratch* s, uint32_t lane,
    uint32_t* mode_out = nullptr) {
  if (num_experts == 0u || num_experts > kMaxRuntimeExperts ||
      token_counts == nullptr || order_out == nullptr || s == nullptr) {
    if (lane == 0u && mode_out != nullptr) *mode_out = kSm0ModeNoPair;
    return 0u;
  }
  // (a) Warp-parallel histogram stats + associative reduction.
  uint32_t total = 0u, active = 0u, cmin = 0u, cmax = 0u;
  sm0_stats_pass(token_counts, num_experts, lane, 32u,
                 &total, &active, &cmin, &cmax);
  total = sm0_warp_reduce_u32(total, 0u);
  active = sm0_warp_reduce_u32(active, 0u);
  cmin = sm0_warp_reduce_u32(cmin, 1u);
  cmax = sm0_warp_reduce_u32(cmax, 2u);
  const bool have_edges = dedup_active != 0u && multi_hit != nullptr &&
                          primary_partner != nullptr;
  // (b) Warp-parallel scratch initialization.
  if (have_edges) sm0_parent_init_pass(s, num_experts, lane, 32u);
  sm0_unit_init_pass(s, num_experts, lane, 32u);
  __syncwarp(0xffffffffu);
  // (c) Order-dependent passes on lane 0, over the identical serial code.
  uint32_t emitted = 0u;
  if (lane == 0u) {
    emitted = build_sm0_expert_order(
        token_counts, multi_hit, primary_partner, num_experts, wave_size,
        dedup_active, order_out, s, mode_out);
  }
  __syncwarp(0xffffffffu);
#if defined(__HIPCC__) && !defined(__CUDACC__)
  return static_cast<uint32_t>(__shfl(static_cast<int>(emitted), 0));
#else
  return __shfl_sync(0xffffffffu, emitted, 0);
#endif
}
#endif  // CUDA / HIP device

// ===========================================================================
// Two-level scheduling, INTER-cluster layer: sharded L2-resident ready-token
// rings.  D in {1,2,4,8} shards, ~2 KiB each (256 slots x 8 B + one 64 B
// header line = 2112 B), so the whole 8-shard set is ~16.5 KiB and stays
// L2-resident -- zero HBM traffic.
//
// THREE explicitly separated modes.  They are NEVER mixed, which is what
// removes the duplicate-consumption bug of a "plain store tail+1 on the own
// SM + neighbour CAS-steals the same tail" design:
//
//   kRingModeClusterSpsc (DEFAULT, cleanest): one shard is bound to ONE
//     cluster's leader SM.  That leader is the ONLY producer AND the ONLY
//     consumer of the shard, so both cursors are private: zero CAS, zero
//     cross-SM tail competition, zero atomics beyond acquire/release.  A pop
//     from a non-owner SM is REJECTED.  The leader fans the token out to its
//     sibling CTAs over DSMEM (cluster_sched_fanout below), which is the
//     intra-cluster layer.
//
//   kRingModeMpscEpoch: no-CAS MPSC for multi-producer enqueue (sender
//     local pre-reduce last hit, receiver top-k arrival).  Producers reserve a
//     slot with ONE atomicAdd(tail) and publish an epoch ready flag; the bound
//     leader is the single consumer.  No wrap within an epoch: capacity is the
//     per-epoch push budget, overflow is sticky and returns false so the
//     caller keeps its legacy path.  Work stealing is NOT offered here.
//
//   kRingModeMpmcSteal: optional work-steal interface.  Vyukov bounded MPMC
//     with per-slot sequence numbers; producers AND consumers claim with ONE
//     bounded atom.acq_rel CAS.  ready_ring_pop_preferred_or_steal routes the
//     home shard and the neighbour shard through the SAME claim_pop, i.e. a
//     UNIFIED atomic claim -- a slot is handed to exactly one consumer.
// ===========================================================================
static constexpr uint32_t kReadyRingMaxShards = 8u;
static constexpr uint32_t kReadyRingSlots = 256u;   // 256 * 8 B = 2048 B
static constexpr uint32_t kRingModeClusterSpsc = 0u;
static constexpr uint32_t kRingModeMpscEpoch = 1u;
static constexpr uint32_t kRingModeMpmcSteal = 2u;

struct ReadyRingSlot {
  uint32_t token;
  uint32_t seq;    // SPSC/MPMC: Vyukov sequence; MPSC: publication epoch
};
static_assert(sizeof(ReadyRingSlot) == 8u, "ready ring slot must stay 8 B");

struct alignas(64) ReadyRingShard {
  uint32_t head;          // consumer cursor
  uint32_t tail;          // producer cursor
  uint32_t owner_sm;      // cluster leader SM exclusively bound to this shard
  uint32_t overflow;      // sticky: an MPSC push exceeded the epoch budget
  uint32_t reserved[12];  // cursors own their own 64 B line
  ReadyRingSlot slots[kReadyRingSlots];
};
static_assert(sizeof(ReadyRingShard) == 64u + 8u * kReadyRingSlots,
              "ready ring shard = one header line + slot array");
static_assert(sizeof(ReadyRingShard) >= 2048u && sizeof(ReadyRingShard) <= 4096u,
              "each ready-token shard must occupy ~2 KiB");

struct alignas(64) ShardedReadyTokenRing {
  uint32_t num_shards;
  uint32_t mode;
  uint32_t epoch;
  uint32_t reserved[13];
  ReadyRingShard shards[kReadyRingMaxShards];
};

WEAVE_HOST_DEVICE constexpr bool ready_ring_shard_count_ok(uint32_t d) {
  return d == 1u || d == 2u || d == 4u || d == 8u;
}

// leader_sms[d] is the SM index of cluster d's leader CTA (nullptr -> d).
WEAVE_HOST_DEVICE void ready_ring_init(ShardedReadyTokenRing* rings,
                                       uint32_t num_shards, uint32_t mode,
                                       const uint32_t* leader_sms = nullptr) {
  if (rings == nullptr) return;
  rings->num_shards = ready_ring_shard_count_ok(num_shards) ? num_shards : 1u;
  rings->mode = mode;
  rings->epoch = 1u;
  for (uint32_t d = 0u; d < kReadyRingMaxShards; ++d) {
    ReadyRingShard* s = &rings->shards[d];
    s->head = 0u;
    s->tail = 0u;
    s->owner_sm = leader_sms != nullptr ? leader_sms[d] : d;
    s->overflow = 0u;
    for (uint32_t i = 0u; i < kReadyRingSlots; ++i) {
      s->slots[i].token = 0u;
      s->slots[i].seq = mode == kRingModeMpscEpoch ? 0u : i;
    }
  }
}

WEAVE_HOST_DEVICE constexpr uint32_t ready_ring_shard_of(
    uint32_t token, uint32_t num_shards) {
  return num_shards <= 1u ? 0u : token % num_shards;
}

// --- Mode 0: cluster-exclusive SPSC (zero CAS, owner-only) -----------------
WEAVE_HOST_DEVICE bool ready_ring_spsc_push(ReadyRingShard* s, uint32_t sm,
                                            uint32_t token) {
  if (s == nullptr || sm != s->owner_sm) return false;
  const uint32_t pos = s->tail;            // private: this SM is the only writer
  ReadyRingSlot* slot = &s->slots[pos & (kReadyRingSlots - 1u)];
  if (weave_ld_acq_u32(&slot->seq) != pos) return false;   // full
  slot->token = token;
  weave_st_rel_u32(&slot->seq, pos + 1u);
  s->tail = pos + 1u;
  return true;
}

WEAVE_HOST_DEVICE bool ready_ring_spsc_pop(ReadyRingShard* s, uint32_t sm,
                                           uint32_t* token_out) {
  if (s == nullptr || token_out == nullptr || sm != s->owner_sm) return false;
  const uint32_t pos = s->head;            // private: this SM is the only reader
  ReadyRingSlot* slot = &s->slots[pos & (kReadyRingSlots - 1u)];
  if (weave_ld_acq_u32(&slot->seq) != pos + 1u) return false;   // empty
  *token_out = slot->token;
  s->head = pos + 1u;
  weave_st_rel_u32(&slot->seq, pos + kReadyRingSlots);
  return true;
}

// --- Mode 1: no-CAS MPSC enqueue + single-consumer epoch pop ---------------
WEAVE_HOST_DEVICE void ready_ring_begin_epoch(ShardedReadyTokenRing* rings,
                                              uint32_t epoch) {
  if (rings == nullptr) return;
  rings->epoch = epoch;
  for (uint32_t d = 0u; d < rings->num_shards; ++d) {
    ReadyRingShard* s = &rings->shards[d];
    s->head = 0u;
    s->tail = 0u;
    s->overflow = 0u;
  }
}

WEAVE_HOST_DEVICE bool ready_ring_mpsc_push(ShardedReadyTokenRing* rings,
                                            ReadyRingShard* s,
                                            uint32_t token) {
  if (rings == nullptr || s == nullptr) return false;
  const uint32_t pos = rtq_atomic_add(&s->tail, 1u);   // ONE atomic, no CAS
  if (pos >= kReadyRingSlots) {
    s->overflow = 1u;
    return false;
  }
  s->slots[pos].token = token;
  weave_st_rel_u32(&s->slots[pos].seq, rings->epoch);
  return true;
}

WEAVE_HOST_DEVICE bool ready_ring_mpsc_pop(ShardedReadyTokenRing* rings,
                                           ReadyRingShard* s, uint32_t sm,
                                           uint32_t* token_out) {
  if (rings == nullptr || s == nullptr || token_out == nullptr ||
      sm != s->owner_sm) {
    return false;
  }
  const uint32_t pos = s->head;
  if (pos >= kReadyRingSlots) return false;
  if (weave_ld_acq_u32(&s->slots[pos].seq) != rings->epoch) return false;
  *token_out = s->slots[pos].token;
  s->head = pos + 1u;
  return true;
}

// --- Mode 2: Vyukov bounded MPMC, UNIFIED atomic claim on both paths -------
WEAVE_HOST_DEVICE bool ready_ring_claim_push(ReadyRingShard* s,
                                             uint32_t token) {
  if (s == nullptr) return false;
  const uint32_t pos = weave_ld_acq_u32(&s->tail);
  ReadyRingSlot* slot = &s->slots[pos & (kReadyRingSlots - 1u)];
  const int32_t diff = static_cast<int32_t>(weave_ld_acq_u32(&slot->seq)) -
                       static_cast<int32_t>(pos);
  if (diff != 0) return false;             // full or stale cursor: bounded fail
  if (!weave_cas_acq_rel_u32(&s->tail, pos, pos + 1u)) return false;
  slot->token = token;
  weave_st_rel_u32(&slot->seq, pos + 1u);
  return true;
}

// The ONE claim path.  Both the home-shard pop and the steal pop call this,
// so a slot can never be handed to two consumers.
WEAVE_HOST_DEVICE bool ready_ring_claim_pop(ReadyRingShard* s,
                                            uint32_t* token_out) {
  if (s == nullptr || token_out == nullptr) return false;
  const uint32_t pos = weave_ld_acq_u32(&s->head);
  ReadyRingSlot* slot = &s->slots[pos & (kReadyRingSlots - 1u)];
  const int32_t diff = static_cast<int32_t>(weave_ld_acq_u32(&slot->seq)) -
                       static_cast<int32_t>(pos + 1u);
  if (diff != 0) return false;             // empty or stale: bounded fail
  if (!weave_cas_acq_rel_u32(&s->head, pos, pos + 1u)) return false;
  *token_out = slot->token;
  weave_st_rel_u32(&slot->seq, pos + kReadyRingSlots);
  return true;
}

// Mode-dispatched pop.  In the cluster-exclusive and MPSC modes the owner
// pops with its private cursor and NO steal is attempted (there is no shared
// tail to race on).  Only kRingModeMpmcSteal offers stealing, and it uses the
// unified claim for BOTH shards.
WEAVE_HOST_DEVICE bool ready_ring_pop_preferred_or_steal(
    ShardedReadyTokenRing* rings, uint32_t sm, uint32_t* token_out) {
  if (rings == nullptr || token_out == nullptr) return false;
  const uint32_t d = rings->num_shards == 0u ? 1u : rings->num_shards;
  if (rings->mode == kRingModeClusterSpsc) {
    return ready_ring_spsc_pop(&rings->shards[sm % d], sm, token_out);
  }
  if (rings->mode == kRingModeMpscEpoch) {
    return ready_ring_mpsc_pop(rings, &rings->shards[sm % d], sm, token_out);
  }
  if (ready_ring_claim_pop(&rings->shards[sm % d], token_out)) return true;
  return ready_ring_claim_pop(&rings->shards[(sm + 1u) % d], token_out);
}

WEAVE_HOST_DEVICE bool ready_ring_push(ShardedReadyTokenRing* rings,
                                       uint32_t shard, uint32_t sm,
                                       uint32_t token) {
  if (rings == nullptr || shard >= rings->num_shards) return false;
  ReadyRingShard* s = &rings->shards[shard];
  if (rings->mode == kRingModeClusterSpsc) {
    return ready_ring_spsc_push(s, sm, token);
  }
  if (rings->mode == kRingModeMpscEpoch) {
    return ready_ring_mpsc_push(rings, s, token);
  }
  return ready_ring_claim_push(s, token);
}

// Scheduler-SM-local readiness tracker: the shard's per-token hit counters
// live in the owning scheduler's SMEM, never touched by another SM.
static constexpr uint32_t kShardTrackerTokens = 64u;

struct ShardTokenTracker {
  uint32_t num_tracked;
  uint32_t expected[kShardTrackerTokens];
  uint32_t received[kShardTrackerTokens];
};

WEAVE_HOST_DEVICE void shard_tracker_reset(ShardTokenTracker* t,
                                           const uint32_t* expected,
                                           uint32_t num_tokens) {
  if (t == nullptr) return;
  t->num_tracked = num_tokens < kShardTrackerTokens ? num_tokens
                                                    : kShardTrackerTokens;
  for (uint32_t i = 0u; i < kShardTrackerTokens; ++i) {
    t->expected[i] = (expected != nullptr && i < t->num_tracked)
        ? expected[i] : 0u;
    t->received[i] = 0u;
  }
}

// True iff this hit completed the token (all local hit_count experts
// pre-reduced / all remote top-k ranks arrived) -> publish into ready_ring[d].
WEAVE_HOST_DEVICE bool shard_tracker_note_hit(ShardTokenTracker* t,
                                              uint32_t slot, uint32_t hits) {
  if (t == nullptr || slot >= t->num_tracked) return false;
  t->received[slot] += hits;
  return t->received[slot] >= t->expected[slot];
}

// ===========================================================================
// Two-level scheduling, INTRA-cluster layer: Cluster DSMEM direct push.
//
// The cluster leader packs a 16-byte-aligned ClusterSchedPacket and pushes it
// straight into a peer CTA's SMEM with the repo's existing primitives:
//   ptx::mapa_shared        -> mapa.shared::cluster
//   ptx::st_async_cluster   -> st.async.shared::cluster.mbarrier::complete_tx
// plus a remote mbarrier.arrive.expect_tx arming the peer's barrier.  The
// transfer bypasses L2 and HBM entirely (~30-40 cycles).  A host-testable
// simulation with identical observable semantics (packet lands in the peer
// slot, completion parity flips once per packet) is provided for the parity
// gate; tests/compile_sm100_dsmem_sched.cu compiles the real PTX path.
// ===========================================================================
static constexpr uint32_t kClusterSizeMax = 8u;

struct alignas(16) ClusterSchedPacket {
  uint32_t epoch;        // publication epoch / parity guard
  uint32_t token_id;     // ready token (or packed task descriptor)
  uint32_t expert_id;
  uint32_t payload;      // block_m | l2_group<<8 | scatter_path<<16 | phase<<18
};
static_assert(sizeof(ClusterSchedPacket) == 16u,
              "ClusterSchedPacket is exactly one 16 B DSMEM transaction");
static_assert(alignof(ClusterSchedPacket) == 16u,
              "st.async.shared::cluster requires 16 B alignment");

WEAVE_HOST_DEVICE constexpr ClusterSchedPacket make_cluster_sched_packet(
    uint32_t epoch, uint32_t token_id, uint32_t expert_id, uint32_t block_m,
    uint32_t l2_swizzle_group, uint32_t scatter_path, uint32_t phase) {
  ClusterSchedPacket p{};
  p.epoch = epoch;
  p.token_id = token_id;
  p.expert_id = expert_id;
  p.payload = (block_m & 0xffu) | ((l2_swizzle_group & 0xffu) << 8) |
              ((scatter_path & 0x3u) << 16) | ((phase & 0x1u) << 18);
  return p;
}

struct ClusterSchedMbar {
  uint64_t word;   // raw mbarrier state word; its address is what mapa remaps
};

struct alignas(16) ClusterSchedChannel {
  ClusterSchedMbar mbar;    // transaction barrier: 1 arrival + 16 expect_tx B
  uint32_t phase;           // consumer-side expected parity phase
  uint32_t sim_completed;   // simulation bookkeeping: completed packet count
  ClusterSchedPacket slot;  // st.async destination (peer writes, owner reads)
};

// Every CTA initializes its OWN channel; the channel must sit at the SAME
// SMEM offset in every CTA of the cluster.
WEAVE_HOST_DEVICE void cluster_sched_barrier_init(ClusterSchedChannel* ch) {
  if (ch == nullptr) return;
  ch->phase = 0u;
  ch->sim_completed = 0u;
  ch->mbar.word = 0ull;
  ch->slot = ClusterSchedPacket{};
#if defined(WEAVE_POLICY_DSMEM_CUDA) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const uint32_t addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&ch->mbar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :: "r"(addr), "r"(1u) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

// Leader push into peer CTA dst_cta.  Real path: remote expect_tx + st.async.
// Simulation path: dst_ch IS the peer's channel object.
WEAVE_HOST_DEVICE void cluster_sched_push(ClusterSchedChannel* dst_ch,
                                          const ClusterSchedPacket& pkt,
                                          uint32_t dst_cta) {
  if (dst_ch == nullptr) return;
#if defined(WEAVE_POLICY_DSMEM_CUDA) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const uint32_t mbar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&dst_ch->mbar));
  const uint32_t remote_bar = deep_gemm::ptx::mapa_shared(mbar_addr, dst_cta);
  asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
               :: "r"(remote_bar),
                  "r"(static_cast<uint32_t>(sizeof(ClusterSchedPacket)))
               : "memory");
  deep_gemm::ptx::st_async_cluster(&dst_ch->slot, pkt, dst_cta, dst_ch->mbar);
#else
  (void)dst_cta;
  dst_ch->slot = pkt;
  dst_ch->sim_completed += 1u;   // 1 arrival + 16 tx bytes -> one parity flip
#endif
}

// Leader fan-out: the SPSC ring's single consumer hands the token to every
// sibling CTA of its cluster over DSMEM (no L2/HBM round trip).
WEAVE_HOST_DEVICE uint32_t cluster_sched_fanout(
    ClusterSchedChannel* channels, uint32_t leader_cta, uint32_t cluster_size,
    const ClusterSchedPacket& pkt) {
  if (channels == nullptr || cluster_size == 0u ||
      cluster_size > kClusterSizeMax || leader_cta >= cluster_size) {
    return 0u;
  }
  uint32_t pushed = 0u;
  for (uint32_t peer = 0u; peer < cluster_size; ++peer) {
    if (peer == leader_cta) continue;
    cluster_sched_push(&channels[peer], pkt, peer);
    ++pushed;
  }
  return pushed;
}

WEAVE_HOST_DEVICE bool cluster_sched_poll(const ClusterSchedChannel* ch,
                                          uint32_t phase) {
  if (ch == nullptr) return false;
#if defined(WEAVE_POLICY_DSMEM_CUDA) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const uint32_t addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&ch->mbar));
  uint32_t completed = 0u;
  asm volatile("{\n\t.reg .pred P1;\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, P1;\n\t}"
               : "=r"(completed) : "r"(addr), "r"(phase) : "memory");
  return completed != 0u;
#else
  return (ch->sim_completed & 1u) != (phase & 1u);
#endif
}

WEAVE_HOST_DEVICE void cluster_sched_wait(ClusterSchedChannel* ch,
                                          uint32_t phase) {
  if (ch == nullptr) return;
#if defined(WEAVE_POLICY_DSMEM_CUDA) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const uint32_t addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&ch->mbar));
  asm volatile("{\n\t.reg .pred P1;\n\t"
               "WEAVE_CS_WAIT:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
               "@P1 bra WEAVE_CS_DONE;\n\t"
               "bra WEAVE_CS_WAIT;\n\t"
               "WEAVE_CS_DONE:\n\t}"
               :: "r"(addr), "r"(phase), "r"(0x989680u) : "memory");
#else
  while (!cluster_sched_poll(ch, phase)) {}
#endif
}

WEAVE_HOST_DEVICE ClusterSchedPacket cluster_sched_load(
    const ClusterSchedChannel* ch) {
  return ch != nullptr ? ch->slot : ClusterSchedPacket{};
}

// Host-testable FIFO inbox model of one peer CTA (delivery order, wrap-around
// and mbarrier transaction accounting), used by the cross-language gate.
static constexpr uint32_t kClusterSchedSlots = 32u;

struct alignas(16) ClusterDsmemScheduleSim {
  uint32_t cluster_size;
  uint32_t leader_cta;
  uint32_t reserved[2];
  uint32_t head[kClusterSizeMax];
  uint32_t tail[kClusterSizeMax];
  uint32_t expected_tx[kClusterSizeMax];
  uint32_t completed_tx[kClusterSizeMax];
  ClusterSchedPacket packets[kClusterSizeMax][kClusterSchedSlots];
};

WEAVE_HOST_DEVICE void cluster_sched_sim_init(ClusterDsmemScheduleSim* sim,
                                              uint32_t cluster_size,
                                              uint32_t leader_cta = 0u) {
  if (sim == nullptr) return;
  sim->cluster_size = cluster_size == 0u ? 1u :
      (cluster_size > kClusterSizeMax ? kClusterSizeMax : cluster_size);
  sim->leader_cta = leader_cta < sim->cluster_size ? leader_cta : 0u;
  for (uint32_t c = 0u; c < kClusterSizeMax; ++c) {
    sim->head[c] = 0u;
    sim->tail[c] = 0u;
    sim->expected_tx[c] = 0u;
    sim->completed_tx[c] = 0u;
  }
}

WEAVE_HOST_DEVICE bool cluster_sched_sim_fanout(
    ClusterDsmemScheduleSim* sim, const ClusterSchedPacket& pkt) {
  if (sim == nullptr) return false;
  // Preflight every peer so a full inbox never causes partial delivery.
  for (uint32_t c = 0u; c < sim->cluster_size; ++c) {
    if (c == sim->leader_cta) continue;
    if (sim->head[c] - sim->tail[c] >= kClusterSchedSlots) return false;
  }
  for (uint32_t c = 0u; c < sim->cluster_size; ++c) {
    if (c == sim->leader_cta) continue;
    sim->expected_tx[c] += sizeof(ClusterSchedPacket);
    sim->packets[c][sim->head[c] % kClusterSchedSlots] = pkt;
    sim->head[c] += 1u;
    sim->completed_tx[c] += sizeof(ClusterSchedPacket);
  }
  return true;
}

WEAVE_HOST_DEVICE bool cluster_sched_sim_pop(ClusterDsmemScheduleSim* sim,
                                             uint32_t cta,
                                             ClusterSchedPacket* out) {
  if (sim == nullptr || out == nullptr || cta >= sim->cluster_size) return false;
  if (sim->head[cta] == sim->tail[cta]) return false;
  *out = sim->packets[cta][sim->tail[cta] % kClusterSchedSlots];
  sim->tail[cta] += 1u;
  return true;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
// Per-expert Tier-1 inner-profile consumption: ONE acquire + ONE 4-byte code
// load per consumer (no full-LUT copy, zero SMEM).  phase: 1 = GEMM1, 2 =
// GEMM2.  An invalid/unpublished policy returns 0 (compiled defaults).
WEAVE_DEVICE uint32_t load_expert_profile_code(
    const PolicyControl* control, uint32_t phase, uint32_t expert_idx) {
  if (control == nullptr || expert_idx >= kMaxRuntimeExperts) {
    return 0u;
  }
  (void)policy_load_acquire(&control->published_epoch);
  if (control->online.phase_profile_active == 0u) {
    return 0u;
  }
  return phase == 1u ? control->online.l1_profile_code[expert_idx]
                     : control->online.l2_profile_code[expert_idx];
}
#endif

}  // namespace weave
