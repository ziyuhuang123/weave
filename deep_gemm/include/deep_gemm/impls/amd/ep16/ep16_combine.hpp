#pragma once
// v25 Production EP16 combine/return (hardware-uncached IPC/RDMA arena):
//   * self.arena is allocated with hipDeviceMallocUncached (MTYPE_UC), so all
//     stores/loads to node_partial, peer_return_payload, node_output, node_recv,
//     return_recv_payload, and NIC flags bypass L1/L2/MALL in hardware!
//   * Eliminates ALL 5,000+ buffer_wbl2 sc0 sc1 and buffer_inv sc0 sc1 stalls
//     across ep16_bucket, ep16_node_reduce, ep16_home, and RxJob_Pub!
//   * Uses plain s_waitcnt vmcnt(0) + RELAXED SYSTEM atomics (ZERO inline load/store asm).

__device__ __forceinline__ unsigned ep16_slices(unsigned n){
  if(n>=240u)return 1u;
  if(n>=120u)return 2u;
  if(n>=60u)return 4u;
  return 8u;
}

using ep16_u32x4 = unsigned __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void ep16_vmcnt0(){
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
}
__device__ __forceinline__ uint4 ep16_ld_agent_u4(const void* ptr){
  ep16_u32x4 r;
  asm volatile("global_load_dwordx4 %0, %1, off sc1\ns_waitcnt vmcnt(0)\n"
               : "=&v"(r) : "v"(ptr) : "memory");
  return make_uint4(r[0], r[1], r[2], r[3]);
}
__device__ __forceinline__ void ep16_ld2_agent_u4(const void* p0, const void* p1, uint4& o0, uint4& o1){
  ep16_u32x4 r0, r1;
  asm volatile("global_load_dwordx4 %0, %2, off sc1\n"
               "global_load_dwordx4 %1, %3, off sc1\n"
               "s_waitcnt vmcnt(0)\n"
               : "=&v"(r0), "=&v"(r1) : "v"(p0), "v"(p1) : "memory");
  o0 = make_uint4(r0[0], r0[1], r0[2], r0[3]);
  o1 = make_uint4(r1[0], r1[1], r1[2], r1[3]);
}
__device__ __forceinline__ void ep16_st_agent_relaxed(unsigned* p, unsigned v){
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void ep16_st_sys_relaxed(unsigned* p, unsigned v){
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void ep16_bucket(const V968DispatchExt& env,const V983Ov1Ext& o,const Ep16TurboContext& x,unsigned key){
 // gemm2_output read via sc1 load directly from 128MB MALL -> ZERO buffer_inv sc1 L2 wipes!
 auto& e=env.base;unsigned lane=threadIdx.x&63u,src=key/e.capacity,slot=key%e.capacity;
 auto* meta=(const V809Meta*)(e.recv_meta+(uint64_t)src*(V809_META_HEADER+(uint64_t)e.capacity*V809_META_STRIDE)+V809_META_HEADER+(uint64_t)slot*V809_META_STRIDE);
 unsigned home=meta->home,nexp=meta->nexp;
 auto* rows=(const unsigned*)o.bucket_route_rows+(uint64_t)key*e.topk;
 bool cross=(src>>3u)!=(e.rank>>3u);
 if(cross && WEAVE_SCATTER_MODE==2 && home>=x.node_tokens){if(lane==0u)v983_error((unsigned*)e.error,905u);return;}
 uint16_t* dst_base =
#if WEAVE_SCATTER_MODE == 2
   cross ? ((uint16_t*)((const uint64_t*)x.node_partial_table)[src&7u]+((uint64_t)(e.rank&7u)*x.node_tokens+home)*o.hidden) :
#endif
   ((uint16_t*)((uint64_t*)o.peer_return_payload_table)[src]+(uint64_t)slot*o.hidden);
 if(nexp==1u){
  const uint16_t* src_row = (const uint16_t*)o.gemm2_output + (uint64_t)rows[0]*o.hidden;
  #pragma clang loop unroll(disable)
  for(unsigned col=lane*8u;col<o.hidden;col+=1024u){
   if(col + 512u < o.hidden){
    uint4 v0, v1;
    ep16_ld2_agent_u4(src_row + col, src_row + col + 512u, v0, v1);
    *(uint4*)(dst_base + col) = v0;
    *(uint4*)(dst_base + col + 512u) = v1;
   } else {
    *(uint4*)(dst_base + col) = ep16_ld_agent_u4(src_row + col);
   }
  }
 } else {
  #pragma clang loop unroll(disable)
  for(unsigned col=lane*8u;col<o.hidden;col+=512u){
   float sum[8]={};
   unsigned k=0u;
   for(;k+1u<nexp;k+=2u){
    uint4 v0, v1;
    ep16_ld2_agent_u4((const uint16_t*)o.gemm2_output + (uint64_t)rows[k]*o.hidden + col,
                      (const uint16_t*)o.gemm2_output + (uint64_t)rows[k+1u]*o.hidden + col, v0, v1);
    const unsigned w0[4]={v0.x,v0.y,v0.z,v0.w};
    const unsigned w1[4]={v1.x,v1.y,v1.z,v1.w};
    #pragma unroll
    for(unsigned j=0u;j<4u;++j){
     sum[2*j]  +=weave_comm_bf16_val(w0[j]&65535u)+weave_comm_bf16_val(w1[j]&65535u);
     sum[2*j+1]+=weave_comm_bf16_val(w0[j]>>16u)  +weave_comm_bf16_val(w1[j]>>16u);
    }
   }
   if(k<nexp){
    const uint4 v0=ep16_ld_agent_u4((const uint16_t*)o.gemm2_output + (uint64_t)rows[k]*o.hidden + col);
    const unsigned w0[4]={v0.x,v0.y,v0.z,v0.w};
    #pragma unroll
    for(unsigned j=0u;j<4u;++j){sum[2*j]+=weave_comm_bf16_val(w0[j]&65535u);sum[2*j+1]+=weave_comm_bf16_val(w0[j]>>16u);}
   }
   uint4 o0; unsigned* p0=&o0.x;
   #pragma unroll
   for(unsigned j=0u;j<4u;++j)p0[j]=weave_comm_bf16(sum[j*2])|((unsigned)weave_comm_bf16(sum[j*2+1])<<16u);
   *(uint4*)(dst_base + col) = o0;
  }
 }
 // Destination buffers live in uncached self.arena -> vmcnt(0) guarantees HBM/xGMI completion!
 ep16_vmcnt0();__syncwarp();
 if(lane==0u){
#if WEAVE_SCATTER_MODE == 2
  if(cross)__hip_atomic_fetch_add((unsigned*)((uint64_t*)x.node_arrival_table)[src&7u]+home,1u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM);
#endif
  if(!cross)ep16_st_sys_relaxed((unsigned*)((uint64_t*)o.peer_home_arrival_table)[src]+slot,1u);
  ep16_st_sys_relaxed((unsigned*)o.bucket_sent+key,1u);
 }
 __syncwarp();
}

__device__ __forceinline__ void ep16_node_reduce(const V968DispatchExt& env,const V983Ov1Ext& o,const Ep16TurboContext& x,unsigned t,unsigned slice,unsigned nsliced){
 // node_partial and node_output BOTH live in uncached self.arena -> ZERO buffer_inv, ZERO buffer_wbl2!
 unsigned lane=threadIdx.x&63u,mask=((unsigned*)x.node_rank_mask)[t];
 const unsigned cw=((o.hidden/8u+nsliced-1u)/nsliced)*8u,c0=slice*cw,c1=(c0+cw<o.hidden)?(c0+cw):o.hidden;
 for(unsigned col=c0+lane*8u;col<c1;col+=512u){
  float sum[8]={};
  for(unsigned r=0u;r<8u;++r)if(mask&(1u<<r)){
   const uint4 pv=*(const uint4*)((const uint16_t*)x.node_partial+((uint64_t)r*x.node_tokens+t)*o.hidden+col);
   const unsigned pw[4]={pv.x,pv.y,pv.z,pv.w};
   #pragma unroll
   for(unsigned j=0u;j<4u;++j){sum[2*j]+=weave_comm_bf16_val(pw[j]&65535u);sum[2*j+1]+=weave_comm_bf16_val(pw[j]>>16u);}
  }
  uint4 v;unsigned* w=&v.x;
  #pragma unroll
  for(unsigned j=0u;j<4u;++j)w[j]=weave_comm_bf16(sum[2*j])|((unsigned)weave_comm_bf16(sum[2*j+1])<<16u);
  *(uint4*)((uint16_t*)x.node_output+(uint64_t)t*o.hidden+col)=v;
 }
 ep16_vmcnt0();__syncwarp();
 if(lane==0u){
  if(nsliced==1u)ep16_st_sys_relaxed((unsigned*)x.node_ready+t,1u);
  else{
   auto* parts=(unsigned*)x.node_ready+x.node_tokens;
   const unsigned bit=1u<<slice,full=(1u<<nsliced)-1u;
   const unsigned prev=__hip_atomic_fetch_or(parts+t,bit,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
   if((prev|bit)==full)ep16_st_sys_relaxed((unsigned*)x.node_ready+t,1u);
  }
 }
 __syncwarp();(void)env;
}

__device__ __forceinline__ bool ep16_home_ready(const V968DispatchExt& env,const V983Ov1Ext& o,const Ep16TurboContext& x,unsigned t){
 auto& e=env.base;bool cross=false;
 for(unsigned r=0u;r<16u;++r){unsigned slot=((unsigned*)o.home_slots)[(uint64_t)t*16u+r];if(slot==0xffffffffu)continue;
#if WEAVE_SCATTER_MODE == 2
  if((r>>3u)!=(e.rank>>3u)){cross=true;continue;}
#endif
  if(!ep16_load_relaxed((unsigned*)x.return_ready+r*e.capacity+slot))return false;
 }
 return !cross || ep16_load_relaxed((unsigned*)x.node_recv_ready+t)>=1u;
}

__device__ __forceinline__ void ep16_home(const V968DispatchExt& env,const V983Ov1Ext& o,const Ep16TurboContext& x,unsigned t,unsigned slice,unsigned nsliced){
 // return_recv_payload and node_recv live in uncached self.arena; final_output is host-read after kernel exit!
 auto& e=env.base;unsigned lane=threadIdx.x&63u;
 const unsigned cw=((o.hidden/8u+nsliced-1u)/nsliced)*8u,c0=slice*cw,c1=(c0+cw<o.hidden)?(c0+cw):o.hidden;
 for(unsigned col=c0+lane*8u;col<c1;col+=512u){
  float sum[8]={};bool cross=false;
  for(unsigned r=0u;r<16u;++r){unsigned slot=((unsigned*)o.home_slots)[(uint64_t)t*16u+r];if(slot==0xffffffffu)continue;
#if WEAVE_SCATTER_MODE == 2
   if((r>>3u)!=(e.rank>>3u)){cross=true;continue;}
#endif
   const uint4 v=*(const uint4*)((const uint16_t*)o.return_recv_payload+((uint64_t)r*e.capacity+slot)*o.hidden+col);
   const unsigned w[4]={v.x,v.y,v.z,v.w};
   #pragma unroll
   for(unsigned j=0u;j<4u;++j){sum[2*j]+=weave_comm_bf16_val(w[j]&65535u);sum[2*j+1]+=weave_comm_bf16_val(w[j]>>16u);}
  }
#if WEAVE_SCATTER_MODE == 2
  if(cross){
   const uint4 v=*(const uint4*)((const uint16_t*)x.node_recv+(uint64_t)t*o.hidden+col);
   const unsigned w[4]={v.x,v.y,v.z,v.w};
   #pragma unroll
   for(unsigned j=0u;j<4u;++j){sum[2*j]+=weave_comm_bf16_val(w[j]&65535u);sum[2*j+1]+=weave_comm_bf16_val(w[j]>>16u);}
  }
#endif
  uint4 v;unsigned* w=&v.x;
  #pragma unroll
  for(unsigned j=0u;j<4u;++j)w[j]=weave_comm_bf16(sum[2*j])|((unsigned)weave_comm_bf16(sum[2*j+1])<<16u);
  *(uint4*)((uint16_t*)o.final_output+(uint64_t)t*o.hidden+col)=v;
 }
 ep16_vmcnt0();__syncwarp();
 if(lane==0u){
  if(nsliced==1u)ep16_st_agent_relaxed((unsigned*)o.home_done+t,1u);
  else{
   auto* parts=(unsigned*)o.home_done+(unsigned)e.t_local;
   const unsigned bit=1u<<slice,full=(1u<<nsliced)-1u;
   const unsigned prev=__hip_atomic_fetch_or(parts+t,bit,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
   if((prev|bit)==full)ep16_st_agent_relaxed((unsigned*)o.home_done+t,1u);
  }
 }
 __syncwarp();
}

__device__ __forceinline__ void ep16_combine(const V968DispatchExt& env,const V983Ov1Ext& o,const Ep16TurboContext& x FI_CTX_PARAM){
 auto& e=env.base;unsigned lane=threadIdx.x&63u;
#if defined(WEAVE_KNOB_ALL_WAVE_COMBINE) && (WEAVE_KNOB_ALL_WAVE_COMBINE == 0)
 if((threadIdx.x>>6u)==7u)return;
 const unsigned worker=blockIdx.x*2u+(threadIdx.x>>6u)-5u;
 constexpr unsigned kCombineWorkers=64u;
#else
 const unsigned worker=blockIdx.x*3u+(threadIdx.x>>6u)-5u;
 constexpr unsigned kCombineWorkers=96u;
#endif
 auto start=__builtin_amdgcn_s_memrealtime(); unsigned idle_sleep=1u;
 const bool tight_idle=(e.t_local<=128u); unsigned idle_spins=0u;
 bool seen_plan=false, seen_local=false, seen_rx_acq=false;
 auto* ready_base=(unsigned*)o.recv_payload_ready;
 auto* bmp=(unsigned long long*)(ready_base+(uint64_t)e.world*(uint64_t)e.capacity);
 for(;;){
  unsigned did_work=0u;
  unsigned pending=0u;
  // ---- RxJob drain (tiers 0,1,2 in table order; ZERO buffer_inv / buffer_wbl2) ----
  for(unsigned i=worker;i<x.rx_count;i+=kCombineWorkers){auto& r=((const Ep16RxJob*)x.rx_jobs)[i];
   unsigned ready=0u;
   if(lane==0u){
    bool counted=!r.valid_count || ep16_load_relaxed((unsigned*)r.count_ready)>=r.count_target;
    if(counted&&r.valid_count&&!seen_rx_acq){asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");seen_rx_acq=true;}
    unsigned vcnt=counted&&r.valid_count?ep16_load_relaxed((unsigned*)r.valid_count):0xffffffffu;
    bool unused=counted && r.valid_count && r.index>=vcnt;
    unsigned exp_cnt=r.reserved&EP16_RXJ_COUNT_MASK;if(!exp_cnt)exp_cnt=1u;
    if(counted && r.valid_count && !unused){unsigned rm=vcnt-r.index;if(rm<exp_cnt)exp_cnt=rm;}
    bool already=counted && !unused && exp_cnt>0u && ep16_load_relaxed((unsigned*)r.publish+(exp_cnt-1u))>=r.value;
    ready=unused || already || (counted && __hip_atomic_load((const uint64_t*)r.flag,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM)>=x.generation);
    if(ready && !unused && !already){
     const unsigned rx_tier=(r.reserved>>16)&0xffu;
     FI_BG(V1988_FI_CORE, 45u, (rx_tier << 16u) | (r.index & 0xffffu));
     unsigned cnt=r.reserved&EP16_RXJ_COUNT_MASK;if(!cnt)cnt=1u;
     if(r.valid_count){const unsigned room=ep16_load_relaxed((unsigned*)r.valid_count)-r.index;if(room<cnt)cnt=room;}
     for(unsigned q=0u;q<cnt;++q)ep16_st_sys_relaxed((unsigned*)r.publish+q,r.value);
     ep16_vmcnt0();
     const unsigned bsrc=(r.reserved>>24)&0xffu;
     const unsigned lo=(bsrc*e.capacity+r.index)&63u;
     if(rx_tier==EP16_TIER_REMOTE_PRIMARY&&cnt&&lo+cnt<=64u){
      const unsigned long long bits=((cnt>=64u)?~0ull:((1ull<<cnt)-1ull))<<lo;
      __hip_atomic_fetch_or(bmp+(((uint64_t)bsrc*e.capacity+r.index)>>6),bits,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM);
     }
     if(cnt)did_work=1u;
     FI_EG(V1988_FI_CORE, 45u, (rx_tier << 16u) | (cnt & 0xffffu));
    }
   }
   if(!__shfl(ready,0u))++pending;
  }
  unsigned planned=0u;if(lane==0u)planned=ep16_load_relaxed((unsigned*)o.plan_flags+1u);planned=__shfl(planned,0u);
  if(planned==2u)return;
  if(planned==1u){
   if(!seen_plan){asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");seen_plan=true;}
#if WEAVE_SCATTER_MODE == 0 || (defined(WEAVE_KNOB_OVERLAP_PIPELINE) && WEAVE_KNOB_OVERLAP_PIPELINE == 0)
   unsigned bulk_go=0u;
   if(lane==0u)bulk_go=(ep16_load_agent_relaxed((unsigned*)x.compute_done)>=((WeaveTilePlanTable*)env.ck.ck_a)->total_m_tiles*o.n2_blocks)?1u:0u;
   if(!__shfl(bulk_go,0u)){++pending;}
   else
#endif
   {
    const unsigned total_keys=16u*e.capacity;
    for(unsigned key=worker;key<total_keys;key+=kCombineWorkers){
     unsigned src=key/e.capacity,slot=key%e.capacity;
     if(slot>=((unsigned*)e.source_bucket_rows)[src])continue;
     unsigned ready=0u;
     if(lane==0u && !ep16_load_relaxed((unsigned*)o.bucket_sent+key)){
      auto* m=(const V809Meta*)(e.recv_meta+(uint64_t)src*(V809_META_HEADER+(uint64_t)e.capacity*V809_META_STRIDE)+V809_META_HEADER+(uint64_t)slot*V809_META_STRIDE);
      const unsigned nexp=m->nexp;
      const unsigned bd=ep16_load_agent_relaxed((unsigned*)o.bucket_done+key);
      if(nexp>0u && bd==nexp){
       unsigned exp_bd=nexp;
       if(__hip_atomic_compare_exchange_strong((unsigned*)o.bucket_done+key,&exp_bd,0xffffffffu,__ATOMIC_RELAXED,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)){
        ready=1u;
       }else ready=ep16_load_relaxed((unsigned*)o.bucket_sent+key)?0u:2u;
      }else ready=ep16_load_relaxed((unsigned*)o.bucket_sent+key)?0u:2u;
     }
     ready=__shfl(ready,0u);
     if(ready==1u){FI_BG(V1988_FI_CORE,9u,key);ep16_bucket(env,o,x,key);FI_EG(V1988_FI_CORE,9u,key);did_work=1u;}
     else if(ready==2u)++pending;
    }
   }
  }else ++pending;
#if WEAVE_SCATTER_MODE == 2
  {
   const unsigned ns=ep16_slices(x.node_tokens),ntask=x.node_tokens*ns;
   for(unsigned task=worker;task<ntask;task+=kCombineWorkers){
    const unsigned t=task/ns,sl=task%ns;
    unsigned ready=0u;
    if(lane==0u && !ep16_load_relaxed((unsigned*)x.node_ready+t)){
     auto* parts=(unsigned*)x.node_ready+x.node_tokens;
     const bool mine=(ns==1u)||((ep16_load_agent_relaxed(parts+t)&(1u<<sl))==0u);
     if(mine)ready=ep16_load_relaxed((unsigned*)x.node_arrival+t)>=((unsigned*)x.node_expected)[t]?1u:2u;
    }
    ready=__shfl(ready,0u);
    if(ready==1u){FI_BG(V1988_FI_CORE,41u,t);ep16_node_reduce(env,o,x,t,sl,ns);FI_EG(V1988_FI_CORE,41u,t);did_work=1u;}
    else if(ready==2u)++pending;
   }
  }
#endif
  unsigned local_planned=0u;if(lane==0u)local_planned=ep16_load_relaxed((unsigned*)o.plan_flags);local_planned=__shfl(local_planned,0u);
  if(local_planned==2u)return;
  if(local_planned==1u){
   if(!seen_local){asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");seen_local=true;}
   const unsigned nh=ep16_slices(e.t_local),ntask=e.t_local*nh;
   for(unsigned task=worker;task<ntask;task+=kCombineWorkers){
    const unsigned t=task/nh,sl=task%nh;unsigned ready=0u;
    if(lane==0u && !ep16_load_agent_relaxed((unsigned*)o.home_done+t)){
     auto* parts=(unsigned*)o.home_done+(unsigned)e.t_local;
     const bool mine=(nh==1u)||((ep16_load_agent_relaxed(parts+t)&(1u<<sl))==0u);
     if(mine)ready=ep16_home_ready(env,o,x,t)?1u:2u;
    }
    ready=__shfl(ready,0u);
    if(ready==1u){FI_BG(V1988_FI_CORE,12u,t);ep16_home(env,o,x,t,sl,nh);FI_EG(V1988_FI_CORE,12u,t);did_work=1u;}
    else if(ready==2u)++pending;
   }
  }else ++pending;
  if(!pending)return;
  unsigned stop=0u;if(lane==0u){stop=ep16_load_relaxed((unsigned*)e.error);if(__builtin_amdgcn_s_memrealtime()-start>x.timeout_ticks){v983_error((unsigned*)e.error,904u);stop=1u;}}
  if(__shfl(stop,0u))return;
  if(did_work){idle_sleep=1u;idle_spins=0u;}
  else if(tight_idle){if(idle_spins<128u)asm volatile("s_nop 0\n" ::: "memory");else __builtin_amdgcn_s_sleep(1);++idle_spins;}
  else {if(idle_sleep==1u){__builtin_amdgcn_s_sleep(1);idle_sleep=2u;}else if(idle_sleep==2u){__builtin_amdgcn_s_sleep(2);idle_sleep=4u;}else{__builtin_amdgcn_s_sleep(8);}}
 }
}

__device__ __forceinline__ void ep16_scatter_reclaim(const V968DispatchExt& env,const V983Ov1Ext& o,const Ep16TurboContext& x,bool dual FI_CTX_PARAM){
 auto& e=env.base;unsigned lane=threadIdx.x&63u;
 const unsigned w=threadIdx.x>>6u;
 const unsigned worker=(w<4u)?(blockIdx.x*4u+w):(320u+(blockIdx.x-32u)*4u+(w-4u));
 const unsigned stride=dual?512u:320u;
 __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_AGENT);
 const unsigned total_keys=16u*e.capacity;
 unsigned reclaimed=0u;
 for(unsigned pass=0u;pass<2u;++pass){
  unsigned had_unready=0u;
  for(unsigned key=worker;key<total_keys;key+=stride){
   unsigned src=key/e.capacity,slot=key%e.capacity;
   if(slot>=((unsigned*)e.source_bucket_rows)[src])continue;
   unsigned claimed=0u;
   if(lane==0u && !ep16_load_relaxed((unsigned*)o.bucket_sent+key)){
    auto* m=(const V809Meta*)(e.recv_meta+(uint64_t)src*(V809_META_HEADER+(uint64_t)e.capacity*V809_META_STRIDE)+V809_META_HEADER+(uint64_t)slot*V809_META_STRIDE);
    const unsigned nexp=m->nexp;
    if(nexp>0u){
     const unsigned bd=ep16_load_agent_relaxed((unsigned*)o.bucket_done+key);
     if(bd==nexp){
      unsigned exp_bd=nexp;
      if(__hip_atomic_compare_exchange_strong((unsigned*)o.bucket_done+key,&exp_bd,0xffffffffu,__ATOMIC_RELAXED,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)){
       claimed=1u;
      }
     }else if(bd<nexp)had_unready=1u;
    }
   }
   if(__shfl(claimed,0u)){
    if(!reclaimed){FI_BG(V1988_FI_CORE,9u,key);}
    ep16_bucket(env,o,x,key);
    ++reclaimed;
   }
  }
  if(!__shfl(had_unready,0u))break;
 }
 if(reclaimed){FI_EG(V1988_FI_CORE,9u,reclaimed);}
}
