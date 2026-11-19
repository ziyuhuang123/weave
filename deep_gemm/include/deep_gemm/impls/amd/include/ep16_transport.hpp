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
  #pragma clang loop unroll(disable)
  for(unsigned i=0;i<n;++i) if(!ep16_gda_load_relaxed(g+i)) return false;
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
__device__ __forceinline__ void ep16_gda_wave4(
 const V968DispatchExt& env,const Ep16TurboContext& x FI_CTX_PARAM) {
  if(threadIdx.x!=256u || blockIdx.x>=32u)return;
  auto& rt=*(mn_gda_mesh::Runtime*)x.runtime;
  unsigned id=blockIdx.x, peer=((env.base.rank>>3u)^1u)*8u+(id>>2u);
  auto ch=(mn_gda_mesh::Channel)(id&3u);auto& q=rt.peer[peer].q[ch];
  if(!rt.enabled||rt.rank!=(int)env.base.rank||rt.channels!=4||!q.sq||!q.cq){v983_error((unsigned*)env.base.error,906u);return;}
  auto* state=(volatile uint64_t*)x.gda_state+id*4u;
  state[0]=0;state[1]=0;state[2]=0;state[3]=__builtin_amdgcn_s_memrealtime();
  const auto* jobs=(const Ep16TxJob*)x.tx_jobs[id];
  const unsigned njobs=x.tx_count[id];
  constexpr unsigned kScan=24u;   // descriptor look-ahead window (<=32, bitmap width)
  constexpr unsigned kBurst=8u;   // descriptors posted per burst
  const bool small_t = (env.base.t_local <= 64u);
  unsigned idle_spins = 0u;
  bool seen_count_acq = false;
  for(;;) {
    asm volatile("" ::: "memory");
    unsigned j=(unsigned)state[0];
    uint32_t bm=(uint32_t)state[2];
    while((bm&1u)&&j<njobs){bm>>=1u;++j;}                 // retire finished descriptors
    state[0]=j;state[2]=bm;
    if(j>=njobs && q.posted==q.retired)return;
    if(__builtin_amdgcn_s_memrealtime()-state[3]>x.timeout_ticks){v983_error((unsigned*)env.base.error,901u);return;}
    if(ep16_gda_load_relaxed((unsigned*)env.base.error))return;
    bool did_work=false,window_full=false;
    if(q.posted!=q.retired){
      bool polled_any=false;
      for(unsigned c=0;c<16u && q.posted!=q.retired;++c){
        int pr=ep16_fic2_poll(rt,q,(int*)env.base.error,false);
        if(pr<0)return;
        if(pr==0)break;
        polled_any=true;did_work=true;
      }
      if(polled_any){__threadfence_system();*q.cq_db=q.cq_tail;__builtin_amdgcn_s_waitcnt(0);}
    }
    bool posted_any=false;
    unsigned posted=0u,scan=0u;
    while(scan<kScan && posted<kBurst && j+scan<njobs){
      if(bm&(1u<<scan)){++scan;continue;}
      const auto& t=jobs[j+scan];
      const uint32_t prior_mask=(1u<<scan)-1u;
      if(t.remote_flag && (bm&prior_mask)!=prior_mask){++scan;continue;}
      const bool chunked=(t.phase&EP16_TXJ_CHUNK_GATE)!=0u;
      const unsigned phase=t.phase&EP16_TXJ_PHASE_MASK;
      // ---- 1. sender-side count that defines this descriptor ----------------
      uint32_t vc=0xffffffffu;bool decidable=true;
      if(t.valid_count&&!(t.phase&EP16_TXJ_STATIC_SLOTS)){
        const unsigned cr=ep16_gda_load_relaxed((const unsigned*)t.count_ready);
        if(cr<t.count_target)decidable=false;
        else {
          if(!seen_count_acq){asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");seen_count_acq=true;}
          vc=ep16_gda_load_relaxed((const unsigned*)t.valid_count);
        }
      }
      if(!decidable){++scan;continue;}
      if(!chunked && t.valid_count && t.index>=vc){bm|=(1u<<scan);++scan;continue;}
      unsigned nslots=1u;
      if(chunked){
        if(t.phase&EP16_TXJ_STATIC_SLOTS){
          nslots=t.target;                                       // builder-baked count
        }else{
          if(t.index>=vc){bm|=(1u<<scan);++scan;continue;}        // unused chunk
          const unsigned room=vc-t.index;
          nslots=(room<t.target)?room:t.target;
        }
      }
      // ---- 2. exact byte range and window credit ---------------------------
      const uint64_t dbytes=chunked?(uint64_t)t.bytes*(uint64_t)nslots:t.bytes;
      const unsigned dpieces=dbytes?ep16_wr_pieces(t.src-rt.arena_base,t.dst-rt.peer_arena[peer],dbytes):0u;
      if(dpieces==0xffffffffu){v983_error((unsigned*)env.base.error,908u);return;}
      const unsigned need=dpieces+(t.remote_flag?1u:0u);
      if(need==0u){bm|=(1u<<scan);++scan;continue;}               // descriptor posts nothing
      if(q.posted-q.retired+(uint64_t)need>(uint64_t)mn_gda_mesh::kWindow){window_full=true;break;}
      // ---- 3. per-slot readiness (all-or-nothing for a chunk) --------------
      if(chunked){ if(!ep16_gate_slots((const unsigned*)t.ready,nslots)){++scan;continue;} }
      else if(t.ready && ep16_gda_load_relaxed((const unsigned*)t.ready)<t.target){++scan;continue;}
      if(t.ready)asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
      // ---- 4. data WRs, each strictly inside one 64 MiB lkey/rkey chunk -----
      FI_BG(V1988_FI_CORE, (phase == 0u ? 42u : 43u), (peer << 16u) | ((id & 3u) << 12u) | (dpieces & 0xfffu));
      uint64_t off=0ull;
      while(off<dbytes){
        const uint64_t s=t.src+off,d=t.dst+off;
        const unsigned sr=(unsigned)(0x4000000ull-((s-rt.arena_base)&0x3ffffffull));
        const unsigned dr=(unsigned)(0x4000000ull-((d-rt.peer_arena[peer])&0x3ffffffull));
        uint64_t take=(uint64_t)MN_GDA_MESH_SLICE;const uint64_t rem=dbytes-off;
        if(take>rem)take=rem;
        if(take>(uint64_t)sr)take=(uint64_t)sr;
        if(take>(uint64_t)dr)take=(uint64_t)dr;
        if(!take){v983_error((unsigned*)env.base.error,908u);return;}
        if(!ep16_fic2_post(rt,peer,ch,s,d,(unsigned)take,(int*)env.base.error,phase,false,false,false))return;
        off+=take;
      }
      // ---- 5. the descriptor's own 8-byte flag WR, on the SAME RC QP --------
      if(t.remote_flag){
        auto* cell=(mn_gda_mesh::U64*)(rt.tx_flags+((uint64_t)(peer*4u+(id&3u))*mn_gda_mesh::kWindow+(q.posted&(mn_gda_mesh::kWindow-1u)))*8u);
        *cell=x.generation;
        if(!ep16_fic2_post(rt,peer,ch,(uint64_t)cell,t.remote_flag,8u,(int*)env.base.error,phase,true,false,false))return;
      }
      FI_EG(V1988_FI_CORE, (phase == 0u ? 42u : 43u), (peer << 16u) | ((id & 3u) << 12u) | (dpieces & 0xfffu));
      bm|=(1u<<scan);posted_any=true;did_work=true;++posted;++scan;
      state[0]=j;state[2]=bm;
    }
    if(posted_any){ep16_fic2_publish_batch();mn_gda_mesh::ring_sq_doorbell(q);}
    state[0]=j;state[2]=bm;
    // r30 D1: on a small-S critical path the coarse s_sleep(2) backoff parks
    // the only posting wave past the first producer arrival.  t_local <= 64
    // spins 128 times on s_nop and then parks one cycle only.
    if(!did_work){
      if(window_full)__builtin_amdgcn_s_sleep(1);
      else if(small_t && idle_spins<128u)asm volatile("s_nop 0\n" ::: "memory");
      else if(small_t)__builtin_amdgcn_s_sleep(1);
      else __builtin_amdgcn_s_sleep(2);
      ++idle_spins;
    } else idle_spins=0;
  }
}