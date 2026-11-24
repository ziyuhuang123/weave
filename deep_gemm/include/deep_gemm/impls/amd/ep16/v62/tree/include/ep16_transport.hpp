#pragma once
// EP16 ABI v1. All tables are device-resident. No remote-node VA may be
// dereferenced by xGMI waves: remote destinations in the turbo peer tables
// MUST name local registered TX staging. Only GDA descriptors hold remote VAs.
#define EP16_GDA_DEVICE_ONLY 1
#define MN_GDA_POST_NOINLINE 0
#include "gda_mesh.hpp"
#include "ep16_fic2_lean.hpp"
#ifndef WEAVE_SCATTER_MODE
#define WEAVE_SCATTER_MODE 2
#endif
static_assert(WEAVE_SCATTER_MODE >= 0 && WEAVE_SCATTER_MODE <= 2);
struct alignas(16) Ep16TxJob {
  uint64_t src, dst, bytes, remote_flag;
  uint64_t ready; // uint32_t*, local or node-local IPC producer release counter
  uint32_t target, phase; // phase: 0 DISPATCH, 1 RETURN
  uint64_t valid_count, count_ready; // optional u32 count and publication word
  uint32_t index, count_target; // skip unused capacity slots AFTER count publication
};
struct alignas(16) Ep16RxJob {
  uint64_t flag; // separately allocated 8-byte RDMA-written generation word
  uint64_t publish; // local uint32_t readiness flag, never adjacent RDMA overwrite
  uint32_t value, reserved;
  uint64_t valid_count, count_ready;
  uint32_t index, count_target;
};
struct alignas(64) Ep16TurboContext {
  uint64_t magic, generation, runtime;
  uint64_t tx_jobs[32]; uint32_t tx_count[32];
  uint64_t rx_jobs; uint32_t rx_count, node_tokens;
  uint64_t return_ready; // uint32_t [16, capacity], for modes 0/1
  uint64_t node_partial_table; // uint64_t [8], owner HBM: float [8,node_tokens,H]
  uint64_t node_arrival_table; // uint64_t [8], owner HBM: uint32_t [node_tokens]
  uint64_t node_expected; // this owner's uint32_t [node_tokens], exact rank count
  uint64_t node_partial;  // this owner's float [8,node_tokens,H]
  uint64_t node_rank_mask; // this owner's uint32_t [node_tokens], contributing rank bits
  uint64_t node_arrival; // this owner's uint32_t [node_tokens]
  uint64_t node_output; // this owner's registered BF16 [node_tokens,H]
  uint64_t node_ready; // uint32_t [node_tokens], release after reduction
  uint64_t node_recv; // home BF16 [t_local,H], one remote-node partial per token
  uint64_t node_recv_ready; // uint32_t [t_local]
  uint64_t compute_done; // zeroed uint32_t scalar, completed stage-2 N tiles
  uint64_t gda_state; // uint64_t [32,4], private state for each QP owner lane
  uint64_t timeout_ticks; // nonzero s_memrealtime budget (100 MHz)
};
static constexpr uint64_t EP16_TURBO_MAGIC=0x4550313654555231ull;
static_assert(sizeof(Ep16TxJob)==80 && sizeof(Ep16RxJob)==48 && sizeof(Ep16TurboContext)==576,"EP16 turbo host ABI size mismatch");

// ---------------------------------------------------------------------------
// RACE29 TrackA design points implemented in this header
// ---------------------------------------------------------------------------
// D1  Group-ready bitmap.  EP16_AR_RECV_PAYLOAD_READY is enlarged by
//     world*ceil(cap/64) u64 words.  The combine publisher sets one bit per
//     ARRIVED dispatch slot with a single atomicOr (release) AFTER it stored the
//     per-slot readiness words, so "bit set" implies "word set" and the consumer
//     may use the bitmap as a one-cacheline shortcut.  The bitmap base is
//     derived on device as (u64*)((u32*)o.recv_payload_ready + world*cap), which
//     keeps the 576-byte Ep16TurboContext ABI untouched.
// D5  Three-tier ready queue.  Every RxJob carries its tier in the high 16 bits
//     of `reserved`: 0=LOCAL_READY, 1=REMOTE_PRIMARY, 2=LOCAL_COPY.
//     ep16_combine() drains tier 0, then 1, then 2 inside each pass, so the
//     compute half can start on the earliest-ready tiles while the RDMA half
//     keeps streaming.  The low 16 bits of `reserved` keep the legacy "publish
//     one word" meaning; a chunked dispatch slot publishes n words.
// ---------------------------------------------------------------------------
#define EP16_RXJ_TIER_SHIFT 16u
#define EP16_RXJ_TIER_MASK  0xffff0000u
#define EP16_RXJ_COUNT_MASK 0x0000ffffu
enum : uint32_t { EP16_TIER_LOCAL_READY = 0u, EP16_TIER_REMOTE_PRIMARY = 1u, EP16_TIER_LOCAL_COPY = 2u };

// A TxJob with EP16_TXJ_CHUNK_GATE set in `phase` spans `target` slots:
//   index   = first slot covered
//   target  = K (slot capacity, K>=1)
//   bytes   = PER-SLOT bytes (payload row or scale row)
//   ready   = uint32_t* base of K per-slot release words
//   nslots  = min(K, valid_count - index)        (computed on the posting lane)
//   nbytes  = nslots * bytes
// It is posted only when ALL `nslots` release words are staged, so a multi-slot
// descriptor can never race a partially staged bucket.
#define EP16_TXJ_CHUNK_GATE  0x80000000u
// With STATIC_SLOTS the descriptor's covered slot count is exactly `target`
// (baked in by the builder) instead of min(target, valid_count - index).
// Used by the mode-2 node-return fanout, whose readiness gates are a dense
// u32 array with no runtime count word.
#define EP16_TXJ_STATIC_SLOTS 0x40000000u
#define EP16_TXJ_PHASE_MASK  0x00000003u

__device__ __forceinline__ unsigned ep16_load_relaxed(const unsigned* p) {
  return __hip_atomic_load(p,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned ep16_load_agent_relaxed(const unsigned* p) {
  return __hip_atomic_load(p,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ unsigned ep16_gda_load_relaxed(const unsigned* p) {
  return __builtin_amdgcn_readfirstlane(ep16_load_relaxed(p));
}
__device__ __forceinline__ unsigned ep16_load(const unsigned* p) {
  return __hip_atomic_load(p,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void ep16_store(unsigned* p,unsigned v) {
  __hip_atomic_store(p,v,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned ep16_gda_load(const unsigned* p){return __builtin_amdgcn_readfirstlane(ep16_load(p));}
__device__ __forceinline__ uint64_t ep16_gda_uniform64(uint64_t v){return (uint64_t)(uint32_t)__builtin_amdgcn_readfirstlane((unsigned)v)|((uint64_t)(uint32_t)__builtin_amdgcn_readfirstlane((unsigned)(v>>32u))<<32u);}

// All-or-nothing check of the n slots of a chunk descriptor.  Conservative: it
// only ever reports "ready" when every covered slot is staged.
__device__ __forceinline__ bool ep16_gate_slots(const unsigned* g,unsigned n){
  if(!g||!n)return false;
  unsigned i=0u;
  // V43G E2: 8 independent relaxed system-scope 8-byte loads per 16 flags (one latency).
  if((((uintptr_t)g)&7u)==0u){
    #pragma clang loop unroll(disable)
    for(;i+16u<=n;i+=16u){
      const unsigned long long* p=reinterpret_cast<const unsigned long long*>(g+i);
      unsigned long long v[8];
      #pragma unroll
      for(unsigned k=0u;k<8u;++k) v[k]=__hip_atomic_load(p+k,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM);
      unsigned ok=1u;
      #pragma unroll
      for(unsigned k=0u;k<8u;++k) ok&=(((unsigned)v[k]!=0u)&&((unsigned)(v[k]>>32u)!=0u))?1u:0u;
      if(!__builtin_amdgcn_readfirstlane(ok))return false;
    }
  }
  #pragma clang loop unroll(disable)
  for(;i<n;++i) if(!ep16_gda_load_relaxed(g+i)) return false;
  return true;
}
// Exact RDMA_WRITE count for [0,nbytes) once both ends are clipped to their
// 64 MiB ibv_reg_mr chunk (0x4000000 mask) and to MN_GDA_MESH_SLICE.  Keeping
// this identical to the posting loop lets the burst reserve window credit up
// front so a descriptor is never half-posted.
__device__ __forceinline__ unsigned ep16_wr_pieces(uint64_t src,uint64_t dst,uint64_t nbytes){
  unsigned n=0u;uint64_t off=0ull;
  while(off<nbytes){
    const unsigned sr=(unsigned)(0x4000000ull-((src+off)&0x3ffffffull));
    const unsigned dr=(unsigned)(0x4000000ull-((dst+off)&0x3ffffffull));
    uint64_t take=(uint64_t)MN_GDA_MESH_SLICE;const uint64_t rem=nbytes-off;
    if(take>rem)take=rem;
    if(take>(uint64_t)sr)take=(uint64_t)sr;
    if(take>(uint64_t)dr)take=(uint64_t)dr;
    if(!take)return 0xffffffffu;
    off+=take;++n;
  }
  return n;
}// Exactly one lane of wave 4 in each of CTAs 0..31 owns one peer/channel.
// Posting is NON-BLOCKING with respect to descriptor order: a bounded look-ahead
// window skips descriptors whose producers are not staged yet.  RC keeps per-QP
// order and every descriptor posts its own data WRs before its flag WR, so an
// out-of-order dispatch cannot expose a flag that certifies unarrived data.
// A descriptor is posted atomically after window credit for ALL of its WRs has
// been reserved, so `posted - retired` never exceeds kWindow.
#if V46_SP
__device__ __forceinline__ void ep16_gda_wave4(
 const V968DispatchExt& env,const V983Ov1Ext& o_tr,const Ep16TurboContext& x,unsigned v46_qc FI_CTX_PARAM) {
  if((threadIdx.x&63u)!=0u || v46_qc>=4u)return;  // opus55 v46: QP channel = comm-wave index cw (0..3)
#else
__device__ __forceinline__ void ep16_gda_wave4(
 const V968DispatchExt& env,const V983Ov1Ext& o_tr,const Ep16TurboContext& x FI_CTX_PARAM) {
  if(threadIdx.x!=256u || blockIdx.x>=4u)return;
  const unsigned v46_qc=blockIdx.x;
#endif
  auto& rt=*(mn_gda_mesh::Runtime*)x.runtime;
  unsigned id=((env.base.rank&7u)*4u+v46_qc), peer=((env.base.rank>>3u)^1u)*8u+(id>>2u);
  auto ch=(mn_gda_mesh::Channel)(id&3u);auto& q=rt.peer[peer].q[ch];
  if(!rt.enabled||rt.rank!=(int)env.base.rank||rt.channels!=4||!q.sq||!q.cq){v983_error((unsigned*)env.base.error,906u);return;}
  auto* state=(volatile uint64_t*)x.gda_state+id*4u;
  state[0]=0;state[1]=0;state[2]=0;state[3]=__builtin_amdgcn_s_memrealtime();
  const auto* jobs=(const Ep16TxJob*)x.tx_jobs[id];
  const unsigned njobs=x.tx_count[id];
  constexpr unsigned kScan=32u;
  constexpr unsigned kBurst=16u;
  const bool small_t=(env.base.t_local<=128u);
  // ---- V43-C: QP hot state resident in registers for the whole engine lifetime ----
  Ep16QReg r; ep16_qreg_load(q, r);
  const uint32_t qpid=q.qpid, sq_count=q.sq_count, cq_count=q.cq_count;
  volatile uint64_t* const sq_db=q.sq_db; volatile uint32_t* const cq_db=q.cq_db;
  const uint64_t arena_base=rt.arena_base, arena_bytes=rt.arena_bytes, peer_arena=rt.peer_arena[peer];
  auto* gda_tr=reinterpret_cast<unsigned long long*>(o_tr.trace)+16u+v46_qc*16u;
  unsigned long long t_disp_first=0ull,t_disp_last=0ull,t_ret_first=0ull;
  unsigned v42_nwf=0u,v42_nnr=0u,v42_nb=0u,v42_ncqe=0u;unsigned long long v42_tpub=0ull,v42_tpost=0ull;
  unsigned idle_spins=0u;bool seen_count_acq=false;
  unsigned j=0u;uint32_t bm=0u;
  // write back software state + cross-QP gate bitmap (G6 reads QP0 state[0]/state[2])
  auto wb=[&](){ ep16_qreg_store(q,r); state[0]=j; state[2]=bm; };
  for(;;){
    asm volatile("" ::: "memory");
    while((bm&1u)&&j<njobs){bm>>=1u;++j;}
    if(j>=njobs && r.posted==r.retired){
      if(o_tr.trace){gda_tr[4u]=__builtin_amdgcn_s_memrealtime();gda_tr[6u]=t_disp_first;gda_tr[7u]=t_disp_last;gda_tr[8u]=t_ret_first;gda_tr[12u]=((unsigned long long)v42_nwf<<32u)|v42_nnr;gda_tr[13u]=((unsigned long long)v42_nb<<32u)|v42_ncqe;gda_tr[14u]=v42_tpub;gda_tr[15u]=v42_tpost;}
      wb();return;
    }
    if(__builtin_amdgcn_s_memrealtime()-state[3]>x.timeout_ticks){wb();v983_error((unsigned*)env.base.error,901u);return;}
    if(ep16_gda_load_relaxed((unsigned*)env.base.error)){wb();return;}
    bool did_work=false,window_full=false;
    if(r.posted!=r.retired){
      bool polled_any=false;
      for(unsigned c=0;c<32u && r.posted!=r.retired;++c){
        int pr=ep16_fic2_poll_reg(rt,q,r,(int*)env.base.error);
        if(pr<0)return;
        if(pr==0)break;
        polled_any=true;did_work=true;++v42_ncqe;
      }
      if(polled_any){asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");*cq_db=r.cq_tail;__builtin_amdgcn_s_waitcnt(0);}
    }
    bool posted_any=false;unsigned posted=0u,scan=0u;
    // G6: QP1..3 fire as soon as QP0 has POSTED the meta descriptor (bit0), never wait for retirement.
    if((id&3u)!=0u && j==0u && (bm&1u)==0u){
      const uint64_t c0_st=((volatile uint64_t*)x.gda_state+(id&~3u)*4u)[2];
      const uint64_t c0_j=((volatile uint64_t*)x.gda_state+(id&~3u)*4u)[0];
      if(c0_j<1ull && (c0_st&1ull)==0ull){__builtin_amdgcn_s_sleep(1);continue;}
    }
    const unsigned long long _tl0=__builtin_amdgcn_s_memrealtime();
    // V43G E3: batched readiness prefetch for the 32-entry look-ahead window.
    uint32_t rp_ok=0u;
    {
      const unsigned nwin=(njobs-j<kScan)?(njobs-j):kScan;
      const uint64_t rp_dummy=(uint64_t)(uintptr_t)env.base.error;
      #pragma clang loop unroll(disable)
      for(unsigned b=0u;b<nwin;b+=16u){
        uint64_t rp[16];unsigned tg[16];
        #pragma unroll
        for(unsigned k=0u;k<16u;++k){
          const unsigned s=b+k;
          const bool live=(s<nwin)&&(((bm>>s)&1u)==0u);
          const Ep16TxJob& tj=jobs[j+(live?s:0u)];
          const uint64_t rr=tj.ready;
          tg[k]=tj.target;
          rp[k]=(live&&rr&&((tj.phase&EP16_TXJ_CHUNK_GATE)==0u))?rr:rp_dummy;
        }
        unsigned rv[16];
        #pragma unroll
        for(unsigned k=0u;k<16u;++k)
          rv[k]=__hip_atomic_load((const unsigned*)(uintptr_t)rp[k],__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM);
        #pragma unroll
        for(unsigned k=0u;k<16u;++k)
          if(rp[k]!=rp_dummy&&rv[k]>=tg[k])rp_ok|=(1u<<(b+k));
      }
      rp_ok=__builtin_amdgcn_readfirstlane(rp_ok);
    }
    while(scan<kScan && posted<kBurst && j+scan<njobs){
      if(bm&(1u<<scan)){++scan;continue;}
      const auto& t=jobs[j+scan];
      const uint32_t prior_mask=(1u<<scan)-1u;
      const unsigned phase_chk=t.phase&EP16_TXJ_PHASE_MASK;
      if(t.remote_flag && phase_chk==0u && (bm&prior_mask)!=prior_mask){++scan;continue;}
      const bool chunked=(t.phase&EP16_TXJ_CHUNK_GATE)!=0u;
      const unsigned phase=t.phase&EP16_TXJ_PHASE_MASK;
      uint32_t vc=0xffffffffu;bool decidable=true;
      if(t.valid_count&&!(t.phase&EP16_TXJ_STATIC_SLOTS)){
        const unsigned cr=ep16_gda_load_relaxed((const unsigned*)t.count_ready);
        if(cr<t.count_target)decidable=false;
        else{
          if(!seen_count_acq){asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");seen_count_acq=true;}
          vc=ep16_gda_load_relaxed((const unsigned*)t.valid_count);
        }
      }
      if(!decidable){++scan;continue;}
      if(!chunked && t.valid_count && t.index>=vc){bm|=(1u<<scan);++scan;continue;}
      unsigned nslots=1u;
      if(chunked){
        if(t.phase&EP16_TXJ_STATIC_SLOTS){nslots=t.target;}
        else{
          if(t.index>=vc){bm|=(1u<<scan);++scan;continue;}
          const unsigned room=vc-t.index;nslots=(room<t.target)?room:t.target;
        }
      }
      const uint64_t dbytes=chunked?(uint64_t)t.bytes*(uint64_t)nslots:t.bytes;
      const unsigned dpieces=dbytes?ep16_wr_pieces(t.src-arena_base,t.dst-peer_arena,dbytes):0u;
      if(dpieces==0xffffffffu){wb();v983_error((unsigned*)env.base.error,908u);return;}
      const unsigned need=dpieces+(t.remote_flag?1u:0u);
      if(need==0u){bm|=(1u<<scan);++scan;continue;}
      if(r.posted-r.retired+(uint64_t)need>(uint64_t)mn_gda_mesh::kWindow){window_full=true;break;}
      if(chunked){if(!ep16_gate_slots((const unsigned*)t.ready,nslots)){++scan;continue;}}
      else if(t.ready && ((rp_ok>>scan)&1u)==0u){++scan;continue;}   // V43G E3
      if(t.ready)asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
      FI_BG(V1988_FI_CORE,(phase==0u?42u:43u),(peer<<16u)|((id&3u)<<12u)|(dpieces&0xfffu));
      uint64_t off=0ull;
      while(off<dbytes){
        const uint64_t s=t.src+off,d=t.dst+off;
        const unsigned sr=(unsigned)(0x4000000ull-((s-arena_base)&0x3ffffffull));
        const unsigned dr=(unsigned)(0x4000000ull-((d-peer_arena)&0x3ffffffull));
        uint64_t take=(uint64_t)MN_GDA_MESH_SLICE;const uint64_t rem=dbytes-off;
        if(take>rem)take=rem;if(take>(uint64_t)sr)take=(uint64_t)sr;if(take>(uint64_t)dr)take=(uint64_t)dr;
        if(!take){wb();v983_error((unsigned*)env.base.error,908u);return;}
        if(!ep16_fic2_post_reg(rt,peer,ch,s,d,(unsigned)take,(int*)env.base.error,phase,false,r))return;
        off+=take;
      }
      if(t.remote_flag){
        auto* cell=(mn_gda_mesh::U64*)(rt.tx_flags+((uint64_t)(peer*4u+(id&3u))*mn_gda_mesh::kWindow+(r.posted&(mn_gda_mesh::kWindow-1u)))*8u);
        {const uint64_t _g=x.generation;asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1\n" :: "v"(cell),"v"(_g) : "memory");}
        if(!ep16_fic2_post_reg(rt,peer,ch,(uint64_t)cell,t.remote_flag,8u,(int*)env.base.error,phase,true,r))return;
      }
      FI_EG(V1988_FI_CORE,(phase==0u?42u:43u),(peer<<16u)|((id&3u)<<12u)|(dpieces&0xfffu));
      {const unsigned long long tn=__builtin_amdgcn_s_memrealtime();if(phase==0u){if(!t_disp_first)t_disp_first=tn;t_disp_last=tn;}else{if(!t_ret_first)t_ret_first=tn;}}
      bm|=(1u<<scan);posted_any=true;did_work=true;++posted;++scan;
      if((id&3u)==0u && j+scan==1u)break;   // V43G E1: meta WR -> doorbell NOW
    }
    if(posted_any){
      const unsigned long long _tp0=__builtin_amdgcn_s_memrealtime();
      asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory"); // publish_batch: SQE sc0|sc1 stores complete
      q.sq_tail=r.sq_tail; *sq_db=((uint64_t)r.sq_tail<<32u)|(uint64_t)(qpid&0x00ffffffu);
      __builtin_amdgcn_s_waitcnt(0);
      state[0]=j;state[2]=bm; // publish cross-QP gate bitmap promptly (G6)
      ++v42_nb;v42_tpub+=__builtin_amdgcn_s_memrealtime()-_tp0;
    }
    if(posted_any)v42_tpost+=__builtin_amdgcn_s_memrealtime()-_tl0;
    if(!did_work){
      if(window_full)++v42_nwf;else if(j<njobs)++v42_nnr;
      if(window_full)__builtin_amdgcn_s_sleep(1);
      else if(small_t&&idle_spins<128u)asm volatile("s_nop 0\n" ::: "memory");
      else if(small_t)__builtin_amdgcn_s_sleep(1);
      else __builtin_amdgcn_s_sleep(2);
      ++idle_spins;
    }else idle_spins=0;
  }
}
