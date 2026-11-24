#pragma once
#include "v44_safe_st.hpp"  // opus55 v44 S1
// Same FIC2 SQE/CQE wire protocol as gda_mesh.hpp. Only the owning lane calls
// these routines. Backpressure and timeouts belong to the outer service loop,
// avoiding nested polling/draining loops and duplicated CQ validation bodies.
//
// RACE29 TrackA (ds1017) changes, all byte-compatible with the FIC2 wire format:
//   * ep16_fic2_post() no longer publishes (threadfence_system + buffer_wbl2)
//     after EVERY SQE.  The caller batches N SQEs and calls
//     ep16_fic2_publish_batch() once before the single SQ doorbell.  The
//     ordering requirement is "SQE bytes visible to the NIC before the
//     doorbell", which a single publish point at the end of the burst satisfies
//     exactly; the old per-WR flush was redundant work on the posting lane.
//   * the 32-dword SQE zero-fill is emitted as 8 vectorised 16-byte stores.
__device__ __forceinline__ int ep16_fic2_poll(mn_gda_mesh::Runtime& rt,mn_gda_mesh::Queue& q,int* err,bool ring_db=true){
 using namespace mn_gda_mesh;
 const volatile uint8_t* c=q.cq+(uint64_t)q.cq_head*64u;
 if(__builtin_amdgcn_readfirstlane((unsigned)((c[63]&1u)==q.cq_phase)))return 0;
 asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
 const uint32_t w0=__builtin_amdgcn_readfirstlane(cqe_word(c)),w1=__builtin_amdgcn_readfirstlane(cqe_word(c+4u)),id=w1&65535u;
 int bad=0;
 if((w0&255u)||(w0>>8u)!=q.qpid||((w1>>16u)&255u)!=MN_GDA_MESH_OPCODE)bad=-7101;
 const U64 seq=sequence_for(q,id);
 if(!seq||seq<=q.retired||q.posted-seq>=kWindow)bad=-7102;
 if(bad){
  #pragma clang loop unroll(disable)
  for(unsigned i=0;i<16u;++i)q.last_cqe[i]=cqe_word(c+i*4u);
  set_failure(rt,q,err,bad);return -1;
 }
 int accepted=complete_tag(q,id);
 if(accepted<0){set_failure(rt,q,err,-7102);return -1;}
 if(accepted)++q.unique_cqes;else ++q.duplicate_cqes;
 q.cq_head=(q.cq_head+1u)&(q.cq_count-1u);q.cq_tail=(q.cq_tail+1u)&(q.cq_count-1u);
 if(!q.cq_head)q.cq_phase^=1u;
 if(ring_db){__threadfence_system();*q.cq_db=q.cq_tail;__builtin_amdgcn_s_waitcnt(0);}return 1;
}
// One publish point for a whole burst of freshly written SQEs.  Must be called
// after the last SQE store of the burst and before ring_sq_doorbell().
__device__ __forceinline__ void ep16_fic2_publish_batch(){
 // V42F F2: SQE + flag-source cells are written with global_store sc0 sc1
 // (write-through); completion of those stores is all the doorbell needs.
 asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
}
// Vectorised 128-byte SQE clear: 8 x 16-byte stores instead of 32 dword stores.
// The 23 dwords that the descriptor never uses are cleared here and the 9 used
// dwords are overwritten immediately below, so the two groups are disjoint and
// a single compiler barrier keeps the zero-fill ahead of the field stores.
__device__ __forceinline__ void ep16_fic2_sqe_zero(uint32_t* w){
 uint4* w4=(uint4*)(void*)w;
 #pragma clang loop unroll(full)
 for(unsigned i=0;i<8u;++i)w4[i]=make_uint4(0u,0u,0u,0u);
 asm volatile("" ::: "memory");
}
// fence=false: the caller guarantees it will call ep16_fic2_publish_batch()
// once after the burst and before the SQ doorbell.  Wire bytes are unchanged.
__device__ __forceinline__ bool ep16_fic2_post(mn_gda_mesh::Runtime& rt,unsigned peer,mn_gda_mesh::Channel ch,
 uint64_t src,uint64_t dst,unsigned bytes,int* err,unsigned phase,bool flag=false,bool ring_db=true,bool fence=true){
 using namespace mn_gda_mesh;auto& q=rt.peer[peer].q[ch];
 if(!bytes||bytes>MN_GDA_MESH_SLICE||rt.arena_bytes<bytes||src<rt.arena_base||src-rt.arena_base>rt.arena_bytes-bytes||
    dst<rt.peer_arena[peer]||dst-rt.peer_arena[peer]>rt.arena_bytes-bytes||(src>>48u)||(dst>>48u)){
  set_failure(rt,q,err,-7106);return false;
 }
 // The outer loop calls only when the burst budget covers this WR, and this
 // lane owns the QP.
 const unsigned lc=(unsigned)((src-rt.arena_base)>>26u),rc=(unsigned)((dst-rt.peer_arena[peer])>>26u);
 const uint32_t lk=lc<16u?q.lkey_chunks[lc]:q.lkey,rk=rc<16u?q.rkey_chunks[rc]:q.rkey;
 U64 seq=allocate_sequence(q);if(!seq){set_failure(rt,q,err,-7108);return false;}
 unsigned id=tag_for(seq),slot=q.sq_tail,osn=q.osn;q.osn=(osn+1u)&0x7fffu;
 auto* w=q.sq+(uint64_t)slot*32u;
 {
  // V42F F2: whole 128-byte SQE as 8 x global_store_dwordx4 sc0 sc1 (write-through).
  typedef unsigned v42f_u4 __attribute__((ext_vector_type(4)));
  const v42f_u4 a0={(MN_GDA_MESH_OPCODE&255u)|(1u<<8u)|((id&65535u)<<16u),
                    (q.cid&0xffffffu)<<8u,(uint32_t)dst,
                    (uint32_t)((dst>>32u)&65535u)|((osn&65535u)<<16u)};
  const v42f_u4 a1={(rk&0xffffffu)|(1u<<24u),0u,0u,(slot&65535u)<<16u};
  const v42f_u4 a2={(uint32_t)src,(uint32_t)((src>>32u)&65535u)|((lk&65535u)<<16u),
                    (bytes<<1u)|((lk>>16u)&1u),0u};
  const v42f_u4 z={0u,0u,0u,0u};
  unsigned char* wb=(unsigned char*)(void*)w;
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb),"v"(a0) : "memory");
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+16),"v"(a1) : "memory");
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+32),"v"(a2) : "memory");
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+48),"v"(z) : "memory");
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+64),"v"(z) : "memory");
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+80),"v"(z) : "memory");
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+96),"v"(z) : "memory");
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+112),"v"(z) : "memory");
 }
 if(fence)publish_single_lane();
 q.sq_tail=(slot+1u)&(q.sq_count-1u);if(ring_db)ring_sq_doorbell(q);
 if(flag)q.tx_flag_bytes[phase]+=bytes;else q.tx_payload_bytes[phase]+=bytes;
 return true;
}
__device__ __forceinline__ bool ep16_fic2_flag(mn_gda_mesh::Runtime& rt,unsigned peer,mn_gda_mesh::Channel ch,
 uint64_t dst,uint64_t generation,int* err,unsigned phase){
 using namespace mn_gda_mesh;auto& q=rt.peer[peer].q[ch];
 auto* cell=(U64*)(rt.tx_flags+((uint64_t)(peer*CHANNELS+ch)*kWindow+(q.posted&(kWindow-1u)))*8u);
 *cell=generation;
 return ep16_fic2_post(rt,peer,ch,(uint64_t)cell,dst,8u,err,phase,true);
}
// ---- V43-C register-state QP bookkeeping (single owning lane) ----
struct Ep16QReg {
  uint64_t posted, retired;
  uint32_t sq_tail, cq_head, cq_tail, cq_phase, osn;
  uint32_t cp1, cp2;              // copies[32] as two bitmasks (kCopies==2)
  uint64_t unique_cqes, duplicate_cqes;
  uint64_t pay0, pay1, flg0, flg1;
};
__device__ __forceinline__ uint32_t ep16_tag_for(uint64_t seq) {
  return static_cast<uint32_t>((seq - 1ull) % 65535ull) + 1u;
}
__device__ __forceinline__ uint64_t ep16_seq_for(uint64_t posted, uint32_t id) {
  if (!posted || !id || id > 65535u) return 0;
  const uint32_t back = (ep16_tag_for(posted) + 65535u - id) % 65535u;
  return back < posted ? posted - back : 0;
}
__device__ __forceinline__ void ep16_qreg_load(mn_gda_mesh::Queue& q, Ep16QReg& r) {
  r.posted = q.posted; r.retired = q.retired;
  r.sq_tail = q.sq_tail; r.cq_head = q.cq_head; r.cq_tail = q.cq_tail;
  r.cq_phase = q.cq_phase; r.osn = q.osn;
  r.cp1 = 0u; r.cp2 = 0u;
  #pragma unroll
  for (unsigned i = 0u; i < 32u; ++i) {
    if (q.copies[i] >= 1u) r.cp1 |= (1u << i);
    if (q.copies[i] >= 2u) r.cp2 |= (1u << i);
  }
  r.unique_cqes = q.unique_cqes; r.duplicate_cqes = q.duplicate_cqes;
  r.pay0 = q.tx_payload_bytes[0]; r.pay1 = q.tx_payload_bytes[1];
  r.flg0 = q.tx_flag_bytes[0]; r.flg1 = q.tx_flag_bytes[1];
}
__device__ __forceinline__ void ep16_qreg_store(mn_gda_mesh::Queue& q, const Ep16QReg& r) {
  q.posted = r.posted; q.retired = r.retired;
  q.sq_tail = r.sq_tail; q.cq_head = r.cq_head; q.cq_tail = r.cq_tail;
  q.cq_phase = r.cq_phase; q.osn = r.osn;
  #pragma unroll
  for (unsigned i = 0u; i < 32u; ++i) {
    q.copies[i] = ((r.cp2 >> i) & 1u) ? 2u : (((r.cp1 >> i) & 1u) ? 1u : 0u);
  }
  q.unique_cqes = r.unique_cqes; q.duplicate_cqes = r.duplicate_cqes;
  q.tx_payload_bytes[0] = r.pay0; q.tx_payload_bytes[1] = r.pay1;
  q.tx_flag_bytes[0] = r.flg0; q.tx_flag_bytes[1] = r.flg1;
}
// poll: identical CQE wire reads; bookkeeping on registers. ring_db handled by caller.
__device__ __forceinline__ int ep16_fic2_poll_reg(mn_gda_mesh::Runtime& rt, mn_gda_mesh::Queue& q,
                                                  Ep16QReg& r, int* err) {
  using namespace mn_gda_mesh;
  const volatile uint8_t* c = q.cq + (uint64_t)r.cq_head * 64u;
  if (__builtin_amdgcn_readfirstlane((unsigned)((c[63] & 1u) == r.cq_phase))) return 0;
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  const uint32_t w0 = __builtin_amdgcn_readfirstlane(cqe_word(c));
  const uint32_t w1 = __builtin_amdgcn_readfirstlane(cqe_word(c + 4u));
  const uint32_t id = w1 & 65535u;
  int bad = 0;
  if ((w0 & 255u) || (w0 >> 8u) != q.qpid || ((w1 >> 16u) & 255u) != MN_GDA_MESH_OPCODE) bad = -7101;
  const uint64_t seq = ep16_seq_for(r.posted, id);
  if (!seq || seq <= r.retired || r.posted - seq >= kWindow) bad = -7102;
  if (bad) {
    #pragma clang loop unroll(disable)
    for (unsigned i = 0u; i < 16u; ++i) q.last_cqe[i] = cqe_word(c + i * 4u);
    ep16_qreg_store(q, r); set_failure(rt, q, err, bad); return -1;
  }
  // complete_tag on bitmasks
  const unsigned index = (unsigned)((seq - 1ull) & (kWindow - 1u));
  if ((r.cp2 >> index) & 1u) { ep16_qreg_store(q, r); set_failure(rt, q, err, -7102); return -1; }
  const bool first = ((r.cp1 >> index) & 1u) == 0u;
  r.cp1 |= (1u << index);
  if (!first) r.cp2 |= (1u << index);
  if (first) ++r.unique_cqes; else ++r.duplicate_cqes;
  while (r.retired < r.posted && ((r.cp2 >> (r.retired & (kWindow - 1u))) & 1u)) ++r.retired;
  r.cq_head = (r.cq_head + 1u) & (q.cq_count - 1u);
  r.cq_tail = (r.cq_tail + 1u) & (q.cq_count - 1u);
  if (!r.cq_head) r.cq_phase ^= 1u;
  return 1;
}
// post: identical 128B SQE (8 x sc0|sc1 write-through stores); credit/tag on registers.
__device__ __forceinline__ bool ep16_fic2_post_reg(mn_gda_mesh::Runtime& rt, unsigned peer,
    mn_gda_mesh::Channel ch, uint64_t src, uint64_t dst, unsigned bytes, int* err,
    unsigned phase, bool flag, Ep16QReg& r) {
  using namespace mn_gda_mesh; auto& q = rt.peer[peer].q[ch];
  if (!bytes || bytes > MN_GDA_MESH_SLICE || rt.arena_bytes < bytes || src < rt.arena_base ||
      src - rt.arena_base > rt.arena_bytes - bytes || dst < rt.peer_arena[peer] ||
      dst - rt.peer_arena[peer] > rt.arena_bytes - bytes || (src >> 48u) || (dst >> 48u)) {
    ep16_qreg_store(q, r); set_failure(rt, q, err, -7106); return false;
  }
  if (r.posted - r.retired >= kWindow) { ep16_qreg_store(q, r); set_failure(rt, q, err, -7108); return false; }
  const uint64_t seq = ++r.posted;
  const unsigned cidx = (unsigned)((seq - 1ull) & (kWindow - 1u));
  r.cp1 &= ~(1u << cidx); r.cp2 &= ~(1u << cidx);
  const unsigned id = ep16_tag_for(seq);
  const unsigned slot = r.sq_tail, osn = r.osn; r.osn = (osn + 1u) & 0x7fffu;
  const unsigned lc = (unsigned)((src - rt.arena_base) >> 26u), rc = (unsigned)((dst - rt.peer_arena[peer]) >> 26u);
  const uint32_t lk = lc < 16u ? q.lkey_chunks[lc] : q.lkey, rk = rc < 16u ? q.rkey_chunks[rc] : q.rkey;
  auto* w = q.sq + (uint64_t)slot * 32u;
  {
    typedef unsigned v43_u4 __attribute__((ext_vector_type(4)));
    const v43_u4 a0 = {(MN_GDA_MESH_OPCODE & 255u) | (1u << 8u) | ((id & 65535u) << 16u),
                       (q.cid & 0xffffffu) << 8u, (uint32_t)dst,
                       (uint32_t)((dst >> 32u) & 65535u) | ((osn & 65535u) << 16u)};
    const v43_u4 a1 = {(rk & 0xffffffu) | (1u << 24u), 0u, 0u, (slot & 65535u) << 16u};
    const v43_u4 a2 = {(uint32_t)src, (uint32_t)((src >> 32u) & 65535u) | ((lk & 65535u) << 16u),
                       (bytes << 1u) | ((lk >> 16u) & 1u), 0u};
    const v43_u4 z = {0u, 0u, 0u, 0u};
    unsigned char* wb = (unsigned char*)(void*)w;
#if V44_SAFE_ST == 1
    (void)z;
    v44_st_sys_u4(wb, a0[0], a0[1], a0[2], a0[3]);
    v44_st_sys_u4(wb + 16, a1[0], a1[1], a1[2], a1[3]);
    v44_st_sys_u4(wb + 32, a2[0], a2[1], a2[2], a2[3]);
    for (unsigned _k = 3u; _k < 8u; ++_k) v44_st_sys_u4(wb + 16u * _k, 0u, 0u, 0u, 0u);
#elif V44_SAFE_ST == 2
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n"
                 "global_store_dwordx4 %0, %2, off offset:16 sc0 sc1\n"
                 "global_store_dwordx4 %0, %3, off offset:32 sc0 sc1\n"
                 "global_store_dwordx4 %0, %4, off offset:48 sc0 sc1\n"
                 "global_store_dwordx4 %0, %4, off offset:64 sc0 sc1\n"
                 "global_store_dwordx4 %0, %4, off offset:80 sc0 sc1\n"
                 "global_store_dwordx4 %0, %4, off offset:96 sc0 sc1\n"
                 "global_store_dwordx4 %0, %4, off offset:112 sc0 sc1\n"
                 "s_waitcnt vmcnt(0)\n"
                 :: "v"(wb), "v"(a0), "v"(a1), "v"(a2), "v"(z) : "memory");
#else
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb), "v"(a0) : "memory");
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+16), "v"(a1) : "memory");
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+32), "v"(a2) : "memory");
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+48), "v"(z) : "memory");
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+64), "v"(z) : "memory");
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+80), "v"(z) : "memory");
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+96), "v"(z) : "memory");
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" :: "v"(wb+112), "v"(z) : "memory");
#endif
  }
  r.sq_tail = (slot + 1u) & (q.sq_count - 1u);
  if (flag) { if (phase == 0u) r.flg0 += bytes; else r.flg1 += bytes; }
  else      { if (phase == 0u) r.pay0 += bytes; else r.pay1 += bytes; }
  return true;
}
