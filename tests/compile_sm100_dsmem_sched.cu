// Standalone nvcc compile gate for the fused two-level scheduling rail.
//
// Every kernel below instantiates a REAL device path of the synthesis:
//   * cluster DSMEM push / fan-out (mapa.shared::cluster +
//     st.async.shared::cluster.mbarrier::complete_tx::bytes +
//     mbarrier.arrive.expect_tx, via deep_gemm/ptx/ld_st.cuh),
//   * cluster-exclusive SPSC ready-token shard (zero CAS),
//   * no-CAS MPSC epoch enqueue (single atomicAdd slot reservation),
//   * Vyukov MPMC unified-claim pop + work steal,
//   * lock-free local pre-reduce (red.global.add.f32 / red.global.add.v4.f32
//     + fence.acq_rel.gpu + atom.acq_rel.gpu.global.add.u32 + acquire fence),
//   * warp-cooperative and serial SM0 expert ordering (external scratch only),
//   * device-side 3-tier GEMM envelope / per-M profile selection.
//
// Gates (checked by tests/check_dsmem_sched_compile.py):
//   nvcc -std=c++17 -cubin -arch=sm_103a -Xptxas -v \
//        -I deep_gemm/include -I <cutlass>/include \
//        tests/compile_sm100_dsmem_sched.cu -o dsmem_sched_sm103.cubin
//   nvcc -std=c++17 -cubin -arch=sm_90a  -Xptxas -v ... -o dsmem_sched_sm90.cubin
//
// Budgets: 0 errors, 0 bytes spill stores/loads and 0 bytes stack frame on
// every kernel; <= 24 registers for the DSMEM push kernel, <= 28 for the
// lock-free pre-reduce kernel, <= 64 for every other primitive kernel.
//
// The cluster kernels are annotated __cluster_dims__(2,1,1): the DSMEM path is
// only meaningful inside a 2/4/8-CTA Thread Block Cluster on Hopper (sm_90a)
// or Blackwell (sm_103a).  Execution needs a target GPU; this gate proves the
// PTX/ptxas lowering, and tests/test_cost_model_eq123.cpp proves the semantics
// through the isomorphic host simulation.
#include <stdint.h>

#include <deep_gemm/scheduler/weave_policy_runtime.cuh>

// ---------------------------------------------------------------------------
// Intra-cluster DSMEM: leader pushes one 16 B packet into the peer channel.
// Kept minimal so the register budget of the push primitive itself is visible.
// ---------------------------------------------------------------------------
__cluster_dims__(2, 1, 1)
__global__ void dsmem_push_kernel(weave::ClusterSchedChannel* out,
                                  uint32_t epoch, uint32_t token) {
  __shared__ weave::ClusterSchedChannel ch;
  if (threadIdx.x == 0u) {
    weave::cluster_sched_barrier_init(&ch);
    const weave::ClusterSchedPacket pkt =
        weave::make_cluster_sched_packet(epoch, token, 3u, 128u, 2u, 2u, 2u);
    weave::cluster_sched_push(&ch, pkt, (blockIdx.x + 1u) & 1u);
    (void)weave::cluster_sched_poll(&ch, 0u);
    *out = ch;
  }
}

// Leader fan-out of one popped ready token to every sibling CTA.
__cluster_dims__(4, 1, 1)
__global__ void dsmem_fanout_kernel(weave::ClusterSchedChannel* channels,
                                    uint32_t epoch, uint32_t token,
                                    uint32_t* pushed) {
  __shared__ weave::ClusterSchedChannel smem_ch[4];
  if (threadIdx.x == 0u) {
    for (uint32_t c = 0u; c < 4u; ++c) {
      weave::cluster_sched_barrier_init(&smem_ch[c]);
    }
    const weave::ClusterSchedPacket pkt =
        weave::make_cluster_sched_packet(epoch, token, 7u, 64u, 4u, 0u, 1u);
    *pushed = weave::cluster_sched_fanout(smem_ch, 0u, 4u, pkt);
    for (uint32_t c = 0u; c < 4u; ++c) channels[c] = smem_ch[c];
  }
}

// Peer-side parity wait + slot consumption.
__cluster_dims__(2, 1, 1)
__global__ void dsmem_wait_kernel(weave::ClusterSchedPacket* out) {
  __shared__ weave::ClusterSchedChannel ch;
  if (threadIdx.x == 0u) {
    weave::cluster_sched_barrier_init(&ch);
    weave::cluster_sched_wait(&ch, 0u);
    *out = weave::cluster_sched_load(&ch);
  }
}

// ---------------------------------------------------------------------------
// Inter-cluster rings.  Mode 0: cluster-exclusive SPSC (owner-only, zero CAS).
// ---------------------------------------------------------------------------
__global__ void ring_cluster_spsc_kernel(weave::ShardedReadyTokenRing* rings,
                                         uint32_t token, uint32_t* popped) {
  const uint32_t sm = blockIdx.x;
  if (threadIdx.x == 0u) {
    const uint32_t d = weave::ready_ring_shard_of(token, rings->num_shards);
    (void)weave::ready_ring_push(rings, d, sm, token);
    uint32_t got = 0u;
    if (weave::ready_ring_pop_preferred_or_steal(rings, sm, &got)) {
      atomicAdd(popped, got);
    }
  }
}

// Mode 1: many producers reserve slots with one atomicAdd each; the bound
// leader is the single consumer.  32 concurrent lanes exercise the race.
__global__ void ring_mpsc_epoch_kernel(weave::ShardedReadyTokenRing* rings,
                                       uint32_t base, uint32_t* popped) {
  const uint32_t sm = blockIdx.x;
  const uint32_t d = sm % rings->num_shards;
  (void)weave::ready_ring_mpsc_push(rings, &rings->shards[d],
                                    base + threadIdx.x);
  __syncthreads();
  if (threadIdx.x == 0u) {
    uint32_t got = 0u;
    while (weave::ready_ring_pop_preferred_or_steal(rings, sm, &got)) {
      atomicAdd(popped, got);
    }
  }
}

// Mode 2: unified atomic claim on the home shard AND the steal shard.
__global__ void ring_mpmc_steal_kernel(weave::ShardedReadyTokenRing* rings,
                                       uint32_t token, uint32_t* popped) {
  const uint32_t sm = blockIdx.x;
  if (threadIdx.x == 0u) {
    (void)weave::ready_ring_claim_push(&rings->shards[sm % rings->num_shards],
                                       token);
  }
  __syncthreads();
  if (threadIdx.x < 32u) {
    uint32_t got = 0u;
    if (weave::ready_ring_pop_preferred_or_steal(rings, sm, &got)) {
      atomicAdd(popped, got);
    }
  }
}

// Scheduler-local readiness tracker feeding the ring (SMEM only).
__global__ void shard_tracker_kernel(const uint32_t* expected,
                                     uint32_t num_tokens,
                                     weave::ShardedReadyTokenRing* rings,
                                     uint32_t* published) {
  __shared__ weave::ShardTokenTracker tracker;
  if (threadIdx.x == 0u) {
    weave::shard_tracker_reset(&tracker, expected, num_tokens);
    for (uint32_t i = 0u; i < tracker.num_tracked; ++i) {
      if (weave::shard_tracker_note_hit(&tracker, i, tracker.expected[i]) &&
          weave::ready_ring_push(rings, 0u, blockIdx.x, i)) {
        atomicAdd(published, 1u);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Lock-free local pre-reduce: red.global.add(.v4).f32 fragment loop, then
// fence.acq_rel.gpu, then atom.acq_rel.gpu.global.add.u32 on the completion
// counter; the unique last contributor acquire-fences and reads the row back.
// ---------------------------------------------------------------------------
__global__ void lockfree_prereduce_kernel(float* scratch, const float* frag,
                                          uint32_t elems, uint32_t* done_cnt,
                                          uint32_t expected,
                                          uint32_t* last_count) {
  const bool is_last = weave::lockfree_local_prereduce_accumulate(
      scratch, frag, elems, done_cnt, expected);
  if (is_last) atomicAdd(last_count, 1u);
}

// Last-claimer read-back: fence.acquire.gpu BEFORE the accumulated row is
// read, so no in-flight contribution can be observed as a partial sum.
__global__ void lockfree_prereduce_readback_kernel(const float* scratch,
                                                   uint32_t elems,
                                                   float* out) {
  weave::lockfree_local_prereduce_load(scratch, elems, out);
}

// Scalar-only variant (unaligned tail path) to pin the scalar reduction PTX.
__global__ void lockfree_prereduce_scalar_kernel(float* scratch, float value,
                                                 uint32_t* done_cnt,
                                                 uint32_t expected,
                                                 uint32_t* last_count) {
  const bool is_last = weave::lockfree_local_prereduce_accumulate(
      scratch, &value, 1u, done_cnt, expected);
  if (is_last) atomicAdd(last_count, 1u);
}

// ---------------------------------------------------------------------------
// SM0 expert ordering: warp-cooperative (all 32 lanes) and serial entry points
// over the SAME external 16 KB scratch -- no per-thread array, no stack frame.
// ---------------------------------------------------------------------------
__global__ void sm0_order_warp_kernel(const uint32_t* counts,
                                      const uint32_t* multi_hit,
                                      const uint32_t* partner,
                                      uint32_t num_experts, uint32_t wave_size,
                                      uint32_t dedup, uint32_t* order,
                                      uint32_t* mode_out) {
  __shared__ weave::Sm0OrderScratch scratch;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t n = weave::build_sm0_expert_order_warp(
      counts, multi_hit, partner, num_experts, wave_size, dedup, order,
      &scratch, lane, mode_out);
  if (lane == 0u) order[num_experts] = n;
}

__global__ void sm0_order_serial_kernel(const uint32_t* counts,
                                        const uint32_t* multi_hit,
                                        const uint32_t* partner,
                                        uint32_t num_experts,
                                        uint32_t wave_size, uint32_t dedup,
                                        uint32_t* order, uint32_t* mode_out) {
  __shared__ weave::Sm0OrderScratch scratch;
  if (threadIdx.x == 0u) {
    (void)weave::build_sm0_expert_order(counts, multi_hit, partner, num_experts,
                                        wave_size, dedup, order, &scratch,
                                        mode_out);
  }
}

// ---------------------------------------------------------------------------
// Device-side 3-tier selection + per-expert profile-code consumption.
// ---------------------------------------------------------------------------
__global__ void gemm_profile_device_kernel(uint32_t phase, uint32_t n,
                                           uint32_t k, uint32_t m_tokens,
                                           uint32_t* out) {
  const weave::PhaseEnvelope env = weave::select_phase_envelope(
      weave::HardwareArch::B300_SM100, weave::PrecisionMode::FP8xFP4,
      phase, n, k);
  const weave::GemmInnerProfile p = weave::select_inner_from_envelope(
      weave::HardwareArch::B300_SM100, weave::PrecisionMode::FP8xFP4,
      phase, n, k, env, m_tokens, 1u, 0u);
  if (threadIdx.x == 0u) {
    out[0] = weave::encode_phase_envelope(env);
    out[1] = weave::encode_inner_profile(weave::HardwareArch::B300_SM100, p);
    out[2] = env.penalty;
    out[3] = env.score_ns;
    out[4] = env.epw;
    out[5] = env.wg_layout_mode;
  }
}

__global__ void profile_code_load_kernel(const weave::PolicyControl* control,
                                         uint32_t e, uint32_t* out) {
  if (threadIdx.x == 0u) {
    out[0] = weave::load_expert_profile_code(control, 1u, e);
    out[1] = weave::load_expert_profile_code(control, 2u, e);
  }
}
