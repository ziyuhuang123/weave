#ifndef WEAVE_COMM_4WAVE_TURBO_HPP
#define WEAVE_COMM_4WAVE_TURBO_HPP

#include <hip/hip_runtime.h>
#include <stdint.h>

// ============================================================================
// WEAVE v11 Turbo 4-Wave (tid >= 256, waves 4..7, 0 B LDS) Communication Engine
// - Parallel per-destination XGMI readback + peer_ready release across lanes!
// - Piggybacks expected_ranks in V809Meta::_reserved[0] -> 0 remote XGMI reads
//   on peer_expected during Combine!
// - Fast direct-copy path when m_nexp == 1 in push_bucket!
// - Low-latency spin-then-sleep wakeup in post_control scheduler!
// ============================================================================

#ifndef V809_MAX_WORLD
#define V809_MAX_WORLD 16u
#endif

#ifndef V809_MAX_TOPK
#define V809_MAX_TOPK 12u
#endif

#ifndef V809_META_STRIDE
#define V809_META_STRIDE 64u
#endif

#ifndef V809_META_HEADER
#define V809_META_HEADER 4096u
#endif

#ifndef V809_PROGRESS_MAGIC
#define V809_PROGRESS_MAGIC 0x44385052u /* "D8PR" */
#endif

#ifndef V1864_CTAS
#define V1864_CTAS 80u
#endif

#ifndef V1988_POST_TOKEN_TAG
#define V1988_POST_TOKEN_TAG 0x80000000u
#endif

#ifndef V1988_POST_BUCKET_TAG
#define V1988_POST_BUCKET_TAG 0x40000000u
#endif

#ifndef V1988_POST_INDEX_MASK
#define V1988_POST_INDEX_MASK 0x3fffffffu
#endif

#ifndef V1988_POST_BATCH
#define V1988_POST_BATCH 4u
#endif

#ifndef WEAVE_POST_QUEUE_BUDGET
#define WEAVE_POST_QUEUE_BUDGET(e) 20000000ull
#endif

#ifndef FI_CTX_PARAM
#define FI_CTX_PARAM
#define FI_CTX_ARG
#define FI_BG(phase, ev, aux)
#define FI_EG(phase, ev, aux)
#define FI_MG(phase, ev, aux)
#define FI_OPEN(wave)
#define FI_CLOSE()
#endif

__device__ __forceinline__ unsigned long long weave_comm_clock() {
  return static_cast<unsigned long long>(__builtin_amdgcn_s_memrealtime());
}

__device__ __forceinline__ void weave_comm_error(unsigned int* error, unsigned code) {
  if (error) atomicCAS(error, 0u, code);
}

__device__ __forceinline__ unsigned weave_comm_load_sys(const unsigned* p) {
  return __hip_atomic_load(const_cast<unsigned*>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ unsigned weave_comm_load_relaxed_sys(const unsigned* p) {
  return __hip_atomic_load(const_cast<unsigned*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ unsigned weave_comm_load_agent(const unsigned* p) {
  return __hip_atomic_load(const_cast<unsigned*>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void weave_comm_store_sys(unsigned* p, unsigned value) {
  __hip_atomic_store(p, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void weave_comm_sleep(unsigned backoff) {
  switch(backoff) {
    case 1u: __builtin_amdgcn_s_sleep(1); break;
    case 2u: __builtin_amdgcn_s_sleep(2); break;
    case 4u: __builtin_amdgcn_s_sleep(4); break;
    case 8u: __builtin_amdgcn_s_sleep(8); break;
    default: __builtin_amdgcn_s_sleep(16); break;
  }
}

__device__ __forceinline__ uint16_t weave_comm_bf16(float value) {
  union { float f; uint32_t u; } pun;
  pun.f = value;
  const uint32_t rounding = ((pun.u >> 16) & 1u) + 0x7fffu;
  return static_cast<uint16_t>((pun.u + rounding) >> 16);
}

__device__ __forceinline__ float weave_comm_bf16_val(uint16_t bits) {
  union { uint32_t u; float f; } pun;
  pun.u = static_cast<uint32_t>(bits) << 16;
  return pun.f;
}

__device__ __forceinline__ uint16_t weave_comm_weight_bits(float value, bool fp16) {
  if (fp16) {
    __half h = __float2half(value);
    return *reinterpret_cast<const uint16_t*>(&h);
  }
  return weave_comm_bf16(value);
}

__device__ __forceinline__ bool weave_comm_wait_ge_lane0(
    const unsigned* p, unsigned target, unsigned long long* spin,
    unsigned* error, unsigned code) {
  if (weave_comm_load_relaxed_sys(p) >= target) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    return true;
  }
  const unsigned long long start = weave_comm_clock();
  unsigned spins = 0u, backoff = 1u;
  while (weave_comm_load_relaxed_sys(p) < target) {
    if (spins < 256u) {
      asm volatile("s_nop 0\n" ::: "memory");
    } else {
      weave_comm_sleep(backoff);
      if (backoff < 8u) backoff <<= 1;
    }
    ++spins;
    if ((spins & 63u) == 0u && (weave_comm_clock() - start >= 20000000ull)) {
      if (spin) atomicAdd(spin, 1ull);
      weave_comm_error(error, code);
      return false;
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  return true;
}

__device__ __forceinline__ bool weave_comm_wave_wait_ge(
    const unsigned* p, unsigned target, unsigned long long* spin,
    unsigned* error, unsigned code) {
  int ok = 1;
  if ((threadIdx.x & 63u) == 0u) {
    ok = weave_comm_wait_ge_lane0(p, target, spin, error, code) ? 1 : 0;
  }
  ok = __shfl(ok, 0u);
  __syncwarp();
  return ok != 0;
}

__device__ __forceinline__ bool weave_comm_wave_wait_eq1(
    const unsigned* p, unsigned long long* spin,
    unsigned* error, unsigned code) {
  int ok = 1;
  if ((threadIdx.x & 63u) == 0u) {
    if (!weave_comm_wait_ge_lane0(p, 1u, spin, error, code)) {
      ok = 0;
    } else if (weave_comm_load_relaxed_sys(p) != 1u) {
      ok = 0;
    }
  }
  ok = __shfl(ok, 0u);
  __syncwarp();
  return ok != 0;
}

__device__ __forceinline__ void weave_comm_quant_1x128_registers(
    const uint4 bf16_data,
    uint2& fp8_packed_out,
    float& group_scale_out,
    const float fp8_max_val = 240.0f,
    const float inv_fp8_max = 1.0f / 240.0f) {
  const uint16_t* raw = reinterpret_cast<const uint16_t*>(&bf16_data);
  float abs_max = 1.0e-10f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    abs_max = fmaxf(abs_max, fabsf(weave_comm_bf16_val(raw[i])));
  }
  #pragma unroll
  for (int delta = 1; delta < 16; delta <<= 1) {
    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, delta, 16));
  }
  const float scale = abs_max * inv_fp8_max;
  group_scale_out = scale;
  const float inv_scale = 1.0f / scale;

  auto quant_pair = [&](uint16_t a, uint16_t b) -> uint16_t {
    const float sa = __builtin_amdgcn_fmed3f(weave_comm_bf16_val(a) * inv_scale, fp8_max_val, -fp8_max_val);
    const float sb = __builtin_amdgcn_fmed3f(weave_comm_bf16_val(b) * inv_scale, fp8_max_val, -fp8_max_val);
    return static_cast<uint16_t>(__builtin_amdgcn_cvt_pk_fp8_f32(sa, sb, 0, 0));
  };

  const uint16_t p0 = quant_pair(raw[0], raw[1]);
  const uint16_t p1 = quant_pair(raw[2], raw[3]);
  const uint16_t p2 = quant_pair(raw[4], raw[5]);
  const uint16_t p3 = quant_pair(raw[6], raw[7]);

  fp8_packed_out.x = static_cast<uint32_t>(p0) | (static_cast<uint32_t>(p1) << 16);
  fp8_packed_out.y = static_cast<uint32_t>(p2) | (static_cast<uint32_t>(p3) << 16);
}

__device__ __forceinline__ void v983_local_plan_4wave(
    const V968DispatchExt& env,
    const V983Ov1Ext& o,
    const V1864Roles roles FI_CTX_PARAM) {
  FI_BG(V1988_FI_PHASE, V1988_EV_LOCAL_PLAN, (blockIdx.x << 3u) | (threadIdx.x >> 6u));
  const V964DispatchExt& e = env.base;
  const unsigned tid = threadIdx.x;
  const unsigned wave = tid >> 6u;
  const unsigned lane = tid & 63u;

  if (wave < 4u || blockIdx.x < 32u) return;

  auto* flags = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* count = reinterpret_cast<unsigned*>(e.bucket_count);
  auto* cursor = reinterpret_cast<unsigned*>(e.bucket_cursor);
  const auto* ids = reinterpret_cast<const int32_t*>(e.topk_ids);
  const auto* weights = reinterpret_cast<const float*>(e.topk_weights);
  const auto* expert_map = reinterpret_cast<const uint32_t*>(e.expert_map);
  const auto* local_meta = reinterpret_cast<const uint64_t*>(e.local_meta_table);
  auto* expected = reinterpret_cast<unsigned*>(o.home_expected);
  auto* home_slots = reinterpret_cast<unsigned*>(o.home_slots);

  const bool include_self = (e.flags & 1u) != 0u;
  const bool fp16 = (e.flags & 2u) != 0u;

  // bucket_count, bucket_cursor, and plan_flags are host-prezeroed before launch (zr=1): ZERO 85us startup barrier!

  const unsigned kBatch = (e.t_local <= 32u) ? 2u : ((e.t_local <= 128u) ? 4u : ((e.t_local <= 512u) ? 16u : 64u));

  for (;;) {
    unsigned first = 0u;
    if (lane == 0u) {
      first = __hip_atomic_fetch_add(flags + 2u, kBatch, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    first = __shfl(first, 0u);
    if (first >= e.t_local) break;

    const unsigned active = min(kBatch, e.t_local - first);
    const unsigned token = first + lane;

    uint16_t lids[V809_MAX_TOPK];
    uint16_t wbits[V809_MAX_TOPK];
    uint8_t dsts[V809_MAX_TOPK];
    unsigned topk_valid = 0u;
    unsigned dst_mask = 0u;
    unsigned exp_ranks = 0u;

    if (lane < active) {
      if (e.world == 8u) {
        uint4* hs4 = reinterpret_cast<uint4*>(home_slots + static_cast<uint64_t>(token) * 8ull);
        const uint4 ff = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
        hs4[0] = ff; hs4[1] = ff;
      } else if (e.world == 16u) {
        uint4* hs4 = reinterpret_cast<uint4*>(home_slots + static_cast<uint64_t>(token) * 16ull);
        const uint4 ff = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
        hs4[0] = ff; hs4[1] = ff; hs4[2] = ff; hs4[3] = ff;
      } else {
        for (unsigned d = 0u; d < e.world; ++d) {
          home_slots[static_cast<uint64_t>(token) * e.world + d] = 0xffffffffu;
        }
      }

      for (unsigned k = 0u; k < e.topk && k < V809_MAX_TOPK; ++k) {
        const int expert = ids[static_cast<uint64_t>(token) * e.topk + k];
        if (expert < 0 || static_cast<unsigned>(expert) >= e.n_experts) {
          weave_comm_error(reinterpret_cast<unsigned*>(e.error), 101u);
          continue;
        }
        const unsigned mapped = expert_map[expert];
        const unsigned dst = mapped >> 16u;
        if (dst >= e.world) {
          weave_comm_error(reinterpret_cast<unsigned*>(e.error), 102u);
          continue;
        }
        if (!include_self && dst == e.rank) continue;

        lids[topk_valid] = static_cast<uint16_t>(mapped & 0xffffu);
        wbits[topk_valid] = weave_comm_weight_bits(weights[static_cast<uint64_t>(token) * e.topk + k], fp16);
        dsts[topk_valid] = static_cast<uint8_t>(dst);
        dst_mask |= (1u << dst);
        topk_valid++;
      }

      exp_ranks = __popc(dst_mask);
      expected[token] = exp_ranks;
    }

    for (unsigned d = 0u; d < e.world; ++d) {
      const bool wants_d = (lane < active) && ((dst_mask & (1u << d)) != 0u);
      const uint64_t ballot_d = __ballot(wants_d);
      const unsigned count_d = __popcll(ballot_d);

      if (count_d > 0u) {
        unsigned base_slot = 0u;
        const unsigned leader = __builtin_ctzll(ballot_d);
        if (lane == leader) {
          base_slot = __hip_atomic_fetch_add(count + d, count_d, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        base_slot = __shfl(base_slot, leader);

        if (wants_d) {
          const unsigned rank_in_wave = __popcll(ballot_d & ((1ull << lane) - 1ull));
          const unsigned slot = base_slot + rank_in_wave;

          if (slot < e.capacity) {
            home_slots[static_cast<uint64_t>(token) * e.world + d] = slot;
            auto* m = reinterpret_cast<V809Meta*>(
                local_meta[d] + V809_META_HEADER + static_cast<uint64_t>(slot) * V809_META_STRIDE);

            m->home = token;
            unsigned nexp = 0u;
            for (unsigned i = 0u; i < topk_valid; ++i) {
              if (dsts[i] == d) {
                if (nexp < V809_MAX_TOPK) {
                  m->lid[nexp] = lids[i];
                  m->wbits[nexp] = wbits[i];
                  nexp++;
                } else {
                  weave_comm_error(reinterpret_cast<unsigned*>(e.error), 104u);
                  break;
                }
              }
            }
            m->nexp = nexp;
            m->_reserved[0] = static_cast<uint8_t>(exp_ranks);
          } else {
            weave_comm_error(reinterpret_cast<unsigned*>(e.error), 103u);
          }
        }
      }
    }

    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __syncwarp();

    if (lane == 0u) {
      const unsigned done = __hip_atomic_fetch_add(flags + 3u, active, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      if (done + active == e.t_local) {
        unsigned prefix = 0u;
        for (unsigned d = 0u; d < e.world; ++d) {
          const unsigned n = __hip_atomic_load(count + d, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          reinterpret_cast<uint64_t*>(local_meta[d])[0] = n;
          reinterpret_cast<uint64_t*>(local_meta[d])[1] = e.capacity;
          reinterpret_cast<unsigned*>(e.bucket_prefix)[d] = prefix;
          prefix += n;
        }
        reinterpret_cast<unsigned*>(e.bucket_prefix)[e.world] = prefix;
        __hip_atomic_store(flags + 0u, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    __syncwarp();
  }

  FI_EG(V1988_FI_PHASE, V1988_EV_LOCAL_PLAN, e.t_local);
  (void)roles;
}

__device__ __forceinline__ void v983_dispatch_4wave(
    const V968DispatchExt& env,
    const V983Ov1Ext& o,
    const V1864Roles roles FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const unsigned cta = blockIdx.x;
  const unsigned tid = threadIdx.x;
  const unsigned wave = tid >> 6u;
  const unsigned lane = tid & 63u;

  if (wave < 4u || cta < 32u || cta >= V1864_CTAS) return;

  const unsigned comm_worker = (cta - 32u) * 4u + (wave - 4u);
  constexpr unsigned kCommWorkers = 192u;
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  auto* bytes = reinterpret_cast<unsigned long long*>(o.byte_counters);
  auto* flags = reinterpret_cast<unsigned*>(o.plan_flags);

  FI_BG(V1988_FI_PHASE, V1988_EV_DISPATCH_WAIT_PLAN, comm_worker);
  {
    int ok_p0 = 1;
    if (lane == 0u) {
      const unsigned long long t0_p0 = weave_comm_clock();
      unsigned sp_p0 = 0u;
      while (__hip_atomic_load(flags + 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < 1u) {
        if (sp_p0 < 256u) asm volatile("s_nop 0\n" ::: "memory");
        else __builtin_amdgcn_s_sleep(1);
        ++sp_p0;
        if ((sp_p0 & 63u) == 0u && (weave_comm_clock() - t0_p0 >= 20000000ull)) {
          weave_comm_error(reinterpret_cast<unsigned*>(e.error), 201u);
          ok_p0 = 0;
          break;
        }
      }
    }
    if (!__shfl(ok_p0, 0u)) return;
    asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");
  }
  FI_EG(V1988_FI_PHASE, V1988_EV_DISPATCH_WAIT_PLAN, comm_worker);

  if (tid == 256u && trace) {
    trace[16u + cta * 16u + 1u] = weave_comm_clock();
  }

  const auto* count = reinterpret_cast<const unsigned*>(e.bucket_count);
  const auto* local_meta = reinterpret_cast<const uint64_t*>(e.local_meta_table);
  const auto* peer_meta = reinterpret_cast<const uint64_t*>(e.peer_meta_table);
  const auto* peer_payload = reinterpret_cast<const uint64_t*>(e.peer_payload_table);
  const auto* peer_scale = reinterpret_cast<const uint64_t*>(e.peer_scale_table);
  const auto* peer_ready = reinterpret_cast<const uint64_t*>(o.peer_payload_ready_table);
  const auto* home_slots = reinterpret_cast<const unsigned*>(o.home_slots);

  FI_BG(V1988_FI_PHASE, V1988_EV_DISPATCH_META, comm_worker);
  for (unsigned dst = comm_worker; dst < e.world; dst += kCommWorkers) {
    const unsigned n = __hip_atomic_load(count + dst, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const uint64_t entry_bytes = static_cast<uint64_t>(n) * V809_META_STRIDE;
    const uint64_t nbytes = 64ull + entry_bytes;
    const auto* src = reinterpret_cast<const unsigned char*>(local_meta[dst]);
    auto* dstp = reinterpret_cast<unsigned char*>(peer_meta[dst]);

    if (lane < 4u) {
      const uint64_t off = lane * 16ull;
      *reinterpret_cast<uint4*>(dstp + off) = *reinterpret_cast<const uint4*>(src + off);
    }
    for (uint64_t off = lane * 16ull; off < entry_bytes; off += 64ull * 16ull) {
      *reinterpret_cast<uint4*>(dstp + V809_META_HEADER + off) =
          *reinterpret_cast<const uint4*>(src + V809_META_HEADER + off);
    }
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __syncwarp();

    if (lane == 0u) {
      auto* progress = reinterpret_cast<V809SourceProgress*>(
          reinterpret_cast<const uint64_t*>(e.peer_progress_table)[dst]);
      __hip_atomic_store(reinterpret_cast<unsigned*>(&progress->meta_ready), V809_PROGRESS_MAGIC, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      if (dst != e.rank && bytes) atomicAdd(bytes + 2u, nbytes);
    }
    __syncwarp();
  }
  FI_EG(V1988_FI_PHASE, V1988_EV_DISPATCH_META, comm_worker);

  FI_BG(V1988_FI_PHASE, V1988_EV_DISPATCH_PAYLOAD, comm_worker);

  const bool is_prequantized = (e.payload_bytes == o.hidden);
  const unsigned hidden = o.hidden;
  const unsigned scale_groups = e.scale_groups;
  unsigned long long local_disp_cnt = 0ull;
  unsigned long long local_b0 = 0ull;
  unsigned long long local_b1 = 0ull;

  for (;;) {
    unsigned token = 0u;
    if (lane == 0u) {
      token = __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(e.bucket_cursor), 1u,
                                     __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    token = __shfl(token, 0u);
    if (token >= e.t_local) break;

    uint8_t dst_list[V809_MAX_WORLD];
    uint32_t slot_list[V809_MAX_WORLD];
    unsigned n_dst = 0u;
    unsigned remote_cnt = 0u;

    for (unsigned d = 0u; d < e.world; ++d) {
      const unsigned slot = home_slots[static_cast<uint64_t>(token) * e.world + d];
      if (slot != 0xffffffffu) {
        dst_list[n_dst] = static_cast<uint8_t>(d);
        slot_list[n_dst] = slot;
        if (d != e.rank) ++remote_cnt;
        n_dst++;
      }
    }

    if (n_dst == 0u) continue;

    if (is_prequantized) {
      const auto* src = reinterpret_cast<const unsigned char*>(e.x) +
                        static_cast<uint64_t>(token) * e.payload_bytes;
      for (unsigned off = lane * 16u; off < e.payload_bytes; off += 64u * 16u) {
        const uint4 reg_val = *reinterpret_cast<const uint4*>(src + off);
        #pragma unroll
        for (unsigned i = 0u; i < n_dst; ++i) {
          const unsigned dst = dst_list[i];
          const unsigned slot = slot_list[i];
          auto* dstp = reinterpret_cast<unsigned char*>(peer_payload[dst]) +
                       static_cast<uint64_t>(slot) * e.payload_bytes;
#if defined(WEAVE_KNOB_SELECTIVE_DEDUP) && (WEAVE_KNOB_SELECTIVE_DEDUP == 0)
          const auto* lm = reinterpret_cast<const V809Meta*>(
              reinterpret_cast<const uint64_t*>(e.local_meta_table)[dst] +
              V809_META_HEADER + static_cast<uint64_t>(slot) * V809_META_STRIDE);
          for (unsigned dup = 0u; dup < lm->nexp; ++dup) {
            volatile unsigned* vdst = reinterpret_cast<volatile unsigned*>(dstp + off);
            vdst[0] = reg_val.x; vdst[1] = reg_val.y; vdst[2] = reg_val.z; vdst[3] = reg_val.w;
          }
#else
          *reinterpret_cast<uint4*>(dstp + off) = reg_val;
#endif
        }
      }
      if (scale_groups > 0u) {
        const auto* src_scale = reinterpret_cast<const float*>(e.x_scale) +
                                static_cast<uint64_t>(token) * scale_groups;
        for (unsigned g = lane; g < scale_groups; g += 64u) {
          const float reg_scale = src_scale[g];
          #pragma unroll
          for (unsigned i = 0u; i < n_dst; ++i) {
            const unsigned dst = dst_list[i];
            const unsigned slot = slot_list[i];
            auto* dst_scale = reinterpret_cast<float*>(peer_scale[dst]) +
                              static_cast<uint64_t>(slot) * scale_groups;
            dst_scale[g] = reg_scale;
          }
        }
      }
    } else {
      const auto* src_bf16 = reinterpret_cast<const uint4*>(
          reinterpret_cast<const unsigned char*>(e.x) + static_cast<uint64_t>(token) * hidden * 2ull);
      const unsigned total_uint4 = (hidden * 2u) / 16u;
      const unsigned group_id_in_wave = lane >> 4u;
      const unsigned lane_in_group = lane & 15u;

      for (unsigned base_u4 = 0u; base_u4 < total_uint4; base_u4 += 64u) {
        const unsigned u4_idx = base_u4 + lane;
        uint4 raw_bf16 = make_uint4(0u, 0u, 0u, 0u);
        if (u4_idx < total_uint4) {
          raw_bf16 = src_bf16[u4_idx];
        }

        uint2 packed_fp8;
        float group_scale = 0.0f;
        weave_comm_quant_1x128_registers(raw_bf16, packed_fp8, group_scale);

        const unsigned global_group_idx = (base_u4 >> 4u) + group_id_in_wave;
        const unsigned byte_offset = (base_u4 * 8u) + (group_id_in_wave * 128u) + (lane_in_group * 8u);

        #pragma unroll
        for (unsigned i = 0u; i < n_dst; ++i) {
          const unsigned dst = dst_list[i];
          const unsigned slot = slot_list[i];
          if (byte_offset < hidden) {
            auto* dstp = reinterpret_cast<unsigned char*>(peer_payload[dst]) +
                         static_cast<uint64_t>(slot) * hidden + byte_offset;
            *reinterpret_cast<uint2*>(dstp) = packed_fp8;
          }
          if (lane_in_group == 0u && global_group_idx < scale_groups) {
            auto* dst_scale = reinterpret_cast<float*>(peer_scale[dst]) +
                              static_cast<uint64_t>(slot) * scale_groups;
            dst_scale[global_group_idx] = group_scale;
          }
        }
      }
    }

    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __syncwarp();

    // Uncached self.arena: vmcnt(0) drained all stores; RELAXED SYSTEM store needs ZERO buffer_wbl2!
    if (lane < n_dst) {
      const unsigned dst = dst_list[lane];
      const unsigned slot = slot_list[lane];
      __hip_atomic_store(reinterpret_cast<unsigned*>(peer_ready[dst]) + slot, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#if WEAVE_CO_PROFILE
      if (o.dispatch_time) {
        reinterpret_cast<unsigned long long*>(o.dispatch_time)[dst * e.capacity + slot] = weave_comm_clock();
      }
#endif
    }
    if (lane == 0u) {
      local_disp_cnt += static_cast<unsigned long long>(n_dst);
      local_b0 += static_cast<unsigned long long>(remote_cnt) * static_cast<unsigned long long>(e.payload_bytes);
      local_b1 += static_cast<unsigned long long>(remote_cnt) * static_cast<unsigned long long>(scale_groups) * 4ull;
    }
    __syncwarp();
  }

  if (lane == 0u) {
    if (trace && local_disp_cnt != 0ull) atomicAdd(trace + 2u, local_disp_cnt);
    if (bytes && local_b0 != 0ull) atomicAdd(bytes + 0u, local_b0);
    if (bytes && local_b1 != 0ull) atomicAdd(bytes + 1u, local_b1);
  }

  FI_EG(V1988_FI_PHASE, V1988_EV_DISPATCH_PAYLOAD, static_cast<unsigned>((local_b0 + local_b1) >> 6u));

  if (lane == 0u && trace) {
    trace[16u + cta * 16u + 2u] = weave_comm_clock();
  }
  (void)roles;
}

__device__ __forceinline__ bool v1988_push_bucket_wave_turbo(
    const V968DispatchExt& env,
    const V983Ov1Ext& o,
    unsigned key,
    unsigned lane,
    unsigned long long* spin FI_CTX_PARAM) {
  // Flat leaf FI intervals only (max_stack_depth = 1)
  const V964DispatchExt& e = env.base;
  const unsigned src = key / e.capacity;
  const unsigned slot = key - src * e.capacity;
  const unsigned region_bytes = V809_META_HEADER + e.capacity * V809_META_STRIDE;
  const auto* region = reinterpret_cast<const unsigned char*>(e.recv_meta) +
                       static_cast<uint64_t>(src) * region_bytes;
  const auto* m = reinterpret_cast<const V809Meta*>(
      region + V809_META_HEADER + static_cast<uint64_t>(slot) * V809_META_STRIDE);

  uint32_t m_home = 0u;
  uint32_t m_nexp = 0u;
  uint32_t m_expected = 0u;
  if (lane == 0u) {
    m_home = m->home;
    m_nexp = m->nexp;
    m_expected = m->_reserved[0];
  }
  m_home = __shfl(m_home, 0u);
  m_nexp = __shfl(m_nexp, 0u);
  m_expected = __shfl(m_expected, 0u);

  int wait_ok = 1;
  if (lane == 0u) {
    FI_BG(V1988_FI_DETAIL, V1988_EV_SCATTER_READY_WAIT, key);
    if (!weave_comm_wait_ge_lane0(reinterpret_cast<const unsigned*>(o.bucket_done) + key, m_nexp,
                                  spin, reinterpret_cast<unsigned*>(e.error), 209u)) {
      wait_ok = 0;
    }
    FI_EG(V1988_FI_DETAIL, V1988_EV_SCATTER_READY_WAIT, key);
  }
  wait_ok = __shfl(wait_ok, 0u);
  if (!wait_ok) {
    return false;
  }

  __syncwarp();
  const unsigned scat_aux = ((m_nexp & 0xffu) << 16u) | ((src != e.rank) ? 1u : 0u);
  FI_BG(V1988_FI_DETAIL, V1988_EV_SCATTER_COPY, scat_aux);

  const auto* rows = reinterpret_cast<const unsigned*>(o.bucket_route_rows) +
                     static_cast<uint64_t>(key) * e.topk;
  const auto* out = reinterpret_cast<const uint16_t*>(o.gemm2_output);
  auto* remote = reinterpret_cast<uint16_t*>(
      reinterpret_cast<const uint64_t*>(o.peer_return_payload_table)[src]) +
      static_cast<uint64_t>(slot) * o.hidden;

  if (m_nexp == 1u) {
    // Fast direct-copy path (80%-95% of buckets on E=256/512!)
    const unsigned row0 = rows[0];
    if (row0 == 0xffffffffu) {
      weave_comm_error(reinterpret_cast<unsigned*>(e.error), 141u);
    } else {
      const auto* src_row = out + static_cast<uint64_t>(row0) * o.hidden;
      if ((o.hidden & 2047u) == 0u) {
        for (unsigned col = lane * 8u; col < o.hidden; col += 2048u) {
          const uint4 v0 = *reinterpret_cast<const uint4*>(src_row + col);
          const uint4 v1 = *reinterpret_cast<const uint4*>(src_row + col + 512u);
          const uint4 v2 = *reinterpret_cast<const uint4*>(src_row + col + 1024u);
          const uint4 v3 = *reinterpret_cast<const uint4*>(src_row + col + 1536u);
          *reinterpret_cast<uint4*>(remote + col) = v0;
          *reinterpret_cast<uint4*>(remote + col + 512u) = v1;
          *reinterpret_cast<uint4*>(remote + col + 1024u) = v2;
          *reinterpret_cast<uint4*>(remote + col + 1536u) = v3;
        }
      } else if ((o.hidden & 1023u) == 0u) {
        for (unsigned col = lane * 8u; col < o.hidden; col += 1024u) {
          const uint4 v0 = *reinterpret_cast<const uint4*>(src_row + col);
          const uint4 v1 = *reinterpret_cast<const uint4*>(src_row + col + 512u);
          *reinterpret_cast<uint4*>(remote + col) = v0;
          *reinterpret_cast<uint4*>(remote + col + 512u) = v1;
        }
      } else {
        for (unsigned col = lane * 8u; col < o.hidden; col += 64u * 8u) {
          *reinterpret_cast<uint4*>(remote + col) = *reinterpret_cast<const uint4*>(src_row + col);
        }
      }
    }
  } else {
    for (unsigned col = lane * 8u; col < o.hidden; col += 64u * 8u) {
      float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      for (unsigned k = 0u; k < m_nexp && k < V809_MAX_TOPK; ++k) {
        const unsigned row = rows[k];
        if (row == 0xffffffffu) {
          weave_comm_error(reinterpret_cast<unsigned*>(e.error), 141u);
          continue;
        }
        const uint4 value = *reinterpret_cast<const uint4*>(out + static_cast<uint64_t>(row) * o.hidden + col);
        const unsigned words[4] = {value.x, value.y, value.z, value.w};
        #pragma unroll
        for (unsigned j = 0u; j < 4u; ++j) {
          sum[j * 2u]      += weave_comm_bf16_val(static_cast<uint16_t>(words[j] & 0xffffu));
          sum[j * 2u + 1u] += weave_comm_bf16_val(static_cast<uint16_t>(words[j] >> 16u));
        }
      }
      uint4 packed;
      unsigned* dstw = &packed.x;
      #pragma unroll
      for (unsigned j = 0u; j < 4u; ++j) {
        dstw[j] = static_cast<unsigned>(weave_comm_bf16(sum[j * 2u])) |
                  (static_cast<unsigned>(weave_comm_bf16(sum[j * 2u + 1u])) << 16u);
      }
      *reinterpret_cast<uint4*>(remote + col) = packed;
    }
  }

  __builtin_amdgcn_s_waitcnt(0x0f70);
  __syncwarp();
  FI_EG(V1988_FI_DETAIL, V1988_EV_SCATTER_COPY, scat_aux);

  if (lane == 0u) {
    auto* arrival = reinterpret_cast<unsigned*>(
        reinterpret_cast<const uint64_t*>(o.peer_home_arrival_table)[src]);
    (void)__hip_atomic_fetch_add(arrival + m_home, 1u,
                                 __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

#if WEAVE_CO_PROFILE
    if (o.scatter_time) {
      reinterpret_cast<unsigned long long*>(o.scatter_time)[key] = weave_comm_clock();
    }
#endif
  }

  return true;
}

__device__ __forceinline__ bool v1988_home_sum_wave_turbo(
    const V968DispatchExt& env,
    const V983Ov1Ext& o,
    unsigned token,
    unsigned lane,
    unsigned long long* spin FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const unsigned needed = reinterpret_cast<const unsigned*>(o.home_expected)[token];

  int wait_ok = 1;
  if (lane == 0u) {
    FI_BG(V1988_FI_DETAIL, V1988_EV_HOME_READY_WAIT, token);
    if (!weave_comm_wait_ge_lane0(reinterpret_cast<const unsigned*>(o.home_arrival) + token, needed,
                            spin, reinterpret_cast<unsigned*>(e.error), 210u)) {
      wait_ok = 0;
    }
    FI_EG(V1988_FI_DETAIL, V1988_EV_HOME_READY_WAIT, token);
  }
  wait_ok = __shfl(wait_ok, 0u);
  if (!wait_ok) {
    return false;
  }

  __syncwarp();
  FI_BG(V1988_FI_DETAIL, V1988_EV_HOME_COMPUTE, needed);

  const auto* base_return = reinterpret_cast<const unsigned char*>(o.return_recv_payload);
  const auto* home_slots = reinterpret_cast<const unsigned*>(o.home_slots);
  auto* final_out = reinterpret_cast<uint16_t*>(o.final_output) +
                    static_cast<uint64_t>(token) * o.hidden;

  uint8_t active_d[V809_MAX_WORLD];
  uint32_t active_s[V809_MAX_WORLD];
  unsigned active_ranks = 0u;

  for (unsigned d = 0u; d < e.world; ++d) {
    const unsigned slot = home_slots[static_cast<uint64_t>(token) * e.world + d];
    if (slot != 0xffffffffu) {
      active_d[active_ranks] = static_cast<uint8_t>(d);
      active_s[active_ranks] = slot;
      active_ranks++;
    }
  }

  for (unsigned col = lane * 8u; col < o.hidden; col += 64u * 8u) {
    float sum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned i = 0u; i < active_ranks; ++i) {
      const unsigned d = active_d[i];
      const unsigned slot = active_s[i];
      const auto* src = reinterpret_cast<const uint16_t*>(
          base_return + (static_cast<uint64_t>(d) * e.capacity + slot) * o.hidden * 2ull);
      const uint4 value = *reinterpret_cast<const uint4*>(src + col);
      const unsigned words[4] = {value.x, value.y, value.z, value.w};
      #pragma unroll
      for (unsigned j = 0u; j < 4u; ++j) {
        sum[j * 2u]      += weave_comm_bf16_val(static_cast<uint16_t>(words[j] & 0xffffu));
        sum[j * 2u + 1u] += weave_comm_bf16_val(static_cast<uint16_t>(words[j] >> 16u));
      }
    }
    uint4 packed;
    unsigned* dstw = &packed.x;
    #pragma unroll
    for (unsigned j = 0u; j < 4u; ++j) {
      dstw[j] = static_cast<unsigned>(weave_comm_bf16(sum[j * 2u])) |
                (static_cast<unsigned>(weave_comm_bf16(sum[j * 2u + 1u])) << 16u);
    }
    *reinterpret_cast<uint4*>(final_out + col) = packed;
  }

  __syncwarp();
  FI_EG(V1988_FI_DETAIL, V1988_EV_HOME_COMPUTE, active_ranks);

  if (lane == 0u) {
    weave_comm_store_sys(reinterpret_cast<unsigned*>(o.home_done) + token, 1u);
  }

  return true;
}

__device__ __forceinline__ bool v1988_epilogue_finish_4wave(
    const V968DispatchExt& env,
    const V983Ov1Ext& o,
    void* arena FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const unsigned cta = blockIdx.x;
  const unsigned tid = threadIdx.x;
  const unsigned lane = tid & 63u;
  const unsigned worker_id = cta * 8u + (tid >> 6u);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);

#if defined(WEAVE_KNOB_ALL_WAVE_COMBINE) && (WEAVE_KNOB_ALL_WAVE_COMBINE == 0)
  if ((tid >> 6u) < 4u) return true;
#endif
  FI_BG(V1988_FI_CORE, V1988_EV_POST_PLAN_WAIT, worker_id);
  if (!weave_comm_wave_wait_eq1(reinterpret_cast<const unsigned*>(o.plan_flags) + 1u,
                                trace ? trace + 16u + cta * 16u + 14u : nullptr,
                                reinterpret_cast<unsigned*>(e.error), 211u)) {
    FI_EG(V1988_FI_CORE, V1988_EV_POST_PLAN_WAIT, worker_id);
    return false;
  }
  FI_EG(V1988_FI_CORE, V1988_EV_POST_PLAN_WAIT, worker_id);

  const auto* rows = reinterpret_cast<const unsigned*>(e.source_bucket_rows);
  unsigned total_buckets = 0u;
  if (lane == 0u) {
    for (unsigned src = 0u; src < e.world; ++src) {
      total_buckets += weave_comm_load_relaxed_sys(rows + src);
    }
  }
  total_buckets = __shfl(total_buckets, 0u);
  const unsigned total = total_buckets;

  // Zero-inbound & zero-local ranks are legally allowed (Fix K4)

  auto* control = reinterpret_cast<V1988PostControl*>(o.post_control);
  auto* queue = reinterpret_cast<unsigned*>(control + 1);

  const unsigned batch_size = (total > 1280u) ? 2u : 1u;
  unsigned backoff = 1u;
  unsigned long long local_post_spins = 0ull;


  for (;;) {
    unsigned base = 0u;
    if (lane == 0u) {
      base = static_cast<unsigned>(
          __hip_atomic_fetch_add(&control->claimed, batch_size,
                                 __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    }
    base = __shfl(base, 0u);
    if (base >= total) break;

    const unsigned claim_count = min(batch_size, total - base);
    const unsigned valid_mask = (1u << claim_count) - 1u;
    unsigned pending = valid_mask;
    unsigned item0 = 0u, item1 = 0u;
    unsigned long long wait_started = 0ull;
    bool wait_active = false;
    unsigned fast_spins = 0u;
    backoff = 1u;

    while (pending != 0u) {
      unsigned ready = 0u;
      if (lane == 0u) {
        if (pending & 1u) {
          item0 = weave_comm_load_relaxed_sys(queue + base + 0u);
          if (item0 != 0u) ready |= 1u;
        }
        if (pending & 2u) {
          item1 = weave_comm_load_relaxed_sys(queue + base + 1u);
          if (item1 != 0u) ready |= 2u;
        }
        if (ready != 0u) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      }
      ready = __shfl(ready, 0u);
      item0 = __shfl(item0, 0u);
      item1 = __shfl(item1, 0u);

      if (ready == 0u) {
        int wait_ok = 1;
        if (lane == 0u) {
          if (!wait_active) {
            wait_started = weave_comm_clock();
            wait_active = true;
          }
          if ((fast_spins & 63u) == 63u && (weave_comm_clock() - wait_started >= WEAVE_POST_QUEUE_BUDGET(e))) {
            weave_comm_error(reinterpret_cast<unsigned*>(e.error), 213u);
            wait_ok = 0;
          } else if (fast_spins < 1024u) {
            asm volatile("s_nop 0\n" ::: "memory");
          } else {
            weave_comm_sleep(backoff);
            if (backoff < 4u) backoff <<= 1;
          }
          ++fast_spins;
          ++local_post_spins;
        }
        wait_ok = __shfl(wait_ok, 0u);
        if (!wait_ok) {
          return false;
        }
        continue;
      }

      wait_active = false;
      fast_spins = 0u;
      backoff = 1u;

      for (unsigned j = 0u; j < claim_count; ++j) {
        const unsigned bit = 1u << j;
        if (!(ready & bit)) continue;

        const unsigned item = (j == 0u ? item0 : item1);
        const unsigned index = item & V1988_POST_INDEX_MASK;
        bool item_ok = false;

        if (item & V1988_POST_TOKEN_TAG) {
          item_ok = v1988_home_sum_wave_turbo(env, o, index, lane,
                                              trace ? trace + 16u + cta * 16u + 14u : nullptr FI_CTX_ARG);
        } else {
          item_ok = v1988_push_bucket_wave_turbo(env, o, index, lane,
                                                 trace ? trace + 16u + cta * 16u + 14u : nullptr FI_CTX_ARG);
        }

        if (!item_ok) {
          return false;
        }
        pending &= ~bit;
      }
    }
    __syncwarp();
  }

#if defined(WEAVE_KNOB_ALL_WAVE_COMBINE) && (WEAVE_KNOB_ALL_WAVE_COMBINE == 0)
  const unsigned home_wid = cta * 4u + ((tid >> 6u) - 4u);
  for (unsigned tok = home_wid; tok < e.t_local; tok += gridDim.x * 4u) {
#else
  for (unsigned tok = worker_id; tok < e.t_local; tok += gridDim.x * 8u) {
#endif
    if (!v1988_home_sum_wave_turbo(env, o, tok, lane,
                                   trace ? trace + 16u + cta * 16u + 14u : nullptr FI_CTX_ARG)) {
          return false;
    }
  }

  if (lane == 0u && local_post_spins != 0ull && trace) {
    atomicAdd(trace + 4u, local_post_spins);
  }

  (void)arena;
  return true;
}

#endif // WEAVE_COMM_4WAVE_TURBO_HPP
