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

// Under HIP, __threadfence() used below is an inline function declared in
// hip/hip_runtime.h (amd_detail/amd_device_functions.h wraps
// __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent")), NOT a compiler keyword.
// megamoe_amd.hip includes this header BEFORE its own hip/hip_runtime.h include,
// so the declaration is missing (gfx942 EP16 build: "use of undeclared
// identifier '__threadfence'").  Pull the HIP runtime declaration here, guarded
// HIP-only so the pure-C++ host tests (g++, no __HIPCC__) are unaffected.
#if defined(__HIPCC__) && !defined(__CUDACC__)
#include <hip/hip_runtime.h>
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

#if defined(__CUDACC__) || defined(__HIPCC__)
WEAVE_DEVICE uint32_t policy_load_acquire(const uint32_t* ptr) {
#if defined(__HIPCC__) && !defined(__CUDACC__)
  return __hip_atomic_load(const_cast<uint32_t*>(ptr), __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_AGENT);
#else
  return atomicAdd(const_cast<uint32_t*>(ptr), 0u);
#endif
}

WEAVE_DEVICE void policy_store_release(uint32_t* ptr, uint32_t value) {
#if defined(__HIPCC__) && !defined(__CUDACC__)
  __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
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
// reuse it without a second global-memory read.
WEAVE_DEVICE PolicyDecision cta0_select_and_publish(
    PolicyControl* control, HardwareArch arch, const WorkloadSpec& wl,
    const WeaveKnobs& knobs, const uint32_t* m_e,
    uint32_t dedup_pulls, uint32_t scatter_count, uint32_t epoch) {
  const uint64_t t0 = policy_clock64();
  const PolicyDecision decision = cta0_select_policy(
      arch, wl, knobs, m_e, dedup_pulls, scatter_count);
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
  __threadfence();
  policy_store_release(&control->published_epoch, epoch);
  control->epoch = epoch;
  __threadfence();
  return control->decision;
}

// B300/H20 publish 64-bit receive counters whose low 32 bits are the token
// count and whose high bits are publication metadata.  CTA 0 snapshots those
// counters into PolicyControl::histogram before selecting.
WEAVE_DEVICE PolicyDecision cta0_select_from_packed_hist(
    PolicyControl* control, HardwareArch arch, const WorkloadSpec& wl,
    const WeaveKnobs& knobs, const uint64_t* packed_counts,
    uint32_t dedup_pulls, uint32_t scatter_count, uint32_t epoch,
    uint32_t publication_retries = 8u) {
  if (control == nullptr || packed_counts == nullptr ||
      wl.local_experts > kMaxRuntimeExperts) {
    PolicyDecision invalid{};
    invalid.magic = kPolicyMagic;
    invalid.version = kPolicyVersion;
    return invalid;
  }
  uint32_t total = 0u;
  for (uint32_t e = 0u; e < wl.local_experts; ++e) {
    const uint32_t count = static_cast<uint32_t>(packed_counts[e]);
    control->histogram[e] = count;
    total += count;
  }
  control->stats_ready_epoch = epoch;
  __threadfence();
  return cta0_select_and_publish(
      control, arch, wl, knobs, control->histogram,
      dedup_pulls == 0u ? total : dedup_pulls,
      scatter_count == 0u ? total : scatter_count, epoch);
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
  __threadfence();
  *out_decision = control->decision;
  return out_decision->magic == kPolicyMagic &&
         out_decision->version == kPolicyVersion &&
         out_decision->valid != 0u;
}
#endif  // CUDA / HIP device

// Count local tokens routed to a remote rank.  DeepGEMM's dispatch counters
// pack (number-of-SMs, local_count) into a uint64_t; the low word is the live
// post-dispatch remote scatter row count M_out used by Eq. (3).
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

// Acquire-load the published epoch, then copy the decision.  Used by kernels
// that keep the policy in global workspace instead of CTA-local shared memory
// (the H20 pre-barrier SMEM region is recycled by scatter/combine).
#if defined(__CUDACC__) || defined(__HIPCC__)
WEAVE_DEVICE PolicyDecision load_published_policy(
    const PolicyControl* control) {
  PolicyDecision d{};
  if (control == nullptr) {
    return d;
  }
  (void)policy_load_acquire(&control->published_epoch);
  __threadfence();
  d = control->decision;
  return d;
}
#endif

// Host helpers for tests and launch wrappers.
WEAVE_HOST_DEVICE bool policy_is_valid(const PolicyDecision& d) {
  return d.magic == kPolicyMagic && d.version == kPolicyVersion && d.valid != 0u;
}

}  // namespace weave


