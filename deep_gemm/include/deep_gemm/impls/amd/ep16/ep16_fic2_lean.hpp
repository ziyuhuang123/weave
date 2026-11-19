#pragma once
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
 __threadfence_system();
 const uint32_t w0=__builtin_amdgcn_readfirstlane(cqe_word(c)),w1=__builtin_amdgcn_readfirstlane(cqe_word(c+4u)),id=w1&65535u;
 int bad=0;
 if((w0&255u)||(w0>>8u)!=q.qpid||((w1>>16u)&255u)!=MN_GDA_MESH_OPCODE)bad=-7101;
 const U64 seq=sequence_for(q,id);
 if(!seq||seq<=q.retired||q.posted-seq>=kWindow)bad=-7102;
 if(!bad){
  const unsigned ix=(unsigned)((seq-1u)&(kWindow-1u));
  const bool first=q.copies[ix]==0u;
  #pragma clang loop unroll(disable)
  for(unsigned i=0;i<16u;++i){
   uint32_t word=__builtin_amdgcn_readfirstlane(cqe_word(c+i*4u))&(i==15u?~(1u<<24u):~0u);
   if(first)q.first_cqe[ix][i]=word;
   else if(q.first_cqe[ix][i]!=word)bad=-7103;
  }
 }
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
 mn_gda_mesh::publish_single_lane();
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
 ep16_fic2_sqe_zero((uint32_t*)(void*)w);
 w[0]=(MN_GDA_MESH_OPCODE&255u)|(1u<<8u)|((id&65535u)<<16u);
 w[1]=(q.cid&0xffffffu)<<8u;w[2]=(uint32_t)dst;
 w[3]=(uint32_t)((dst>>32u)&65535u)|((osn&65535u)<<16u);
 w[4]=(rk&0xffffffu)|(1u<<24u);w[7]=(slot&65535u)<<16u;
 w[8]=(uint32_t)src;w[9]=(uint32_t)((src>>32u)&65535u)|((lk&65535u)<<16u);
 w[10]=(bytes<<1u)|((lk>>16u)&1u);
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