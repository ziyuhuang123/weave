#pragma once
// =============================================================================================================
// Official Weave 21-knob surface -- EXACTLY 21 compile-time knobs.
//
// There are no public runtime environment knobs in this header.  Every value below is a preprocessor
// constant consumed by device code, so changing one changes the compiled gfx942 ISA and the executed
// schedule.  Defaults reproduce the V62 opus55 baseline layout (4 x 0/8 dispatch CTAs, 8 x 4/4 scatter
// CTAs, 25% head split, 16-token scatter claim, remote-pass priority, SC0|SC1 return stores, 8-token
// pull batches, 1024-column node-reduce chunks, 10 reduce waves).
//
//  1 WEAVE_BLOCK_M_L1            stage-1 tile rows              {16,32,64,128}
//  2 WEAVE_BLOCK_M_L2            stage-2 tile rows              {16,32,64,128}
//  3 WEAVE_ADAPT_BM              per-expert dynamic BM 0/1     {16,32,64,128} when 1
//  4 WEAVE_DISPATCH_ACTIVE_SMS   dispatch CTA budget           1..40
//  5 WEAVE_SCATTER_ACTIVE_SMS    scatter CTA budget            1..40
//  6 WEAVE_DISPATCH_WARPS        dispatch waves per CTA        1..8
//  7 WEAVE_SCATTER_WARPS         scatter waves per CTA         1..8
//  8 WEAVE_CU_WAVE_SPLIT         CTA wave split 0 auto / 1 pure-compute 0-8 / 2 4-4 / 3 comm-heavy 8-0
//  9 WEAVE_SM0_STREAM_SCHED      stream-scheduler bias 0..3
// 10 WEAVE_V_HEAD                q0 head split percent         0..100
// 11 WEAVE_READY_GROUP_SIZE      scatter claim group            1..64
// 12 WEAVE_READY_QUEUE           scatter pass priority          0..3
// 13 WEAVE_SCATTER_OWNER         scatter worker owner rotation  0..7
// 14 WEAVE_SCATTER_STAGING       return store policy            0 SC1 / 1 SC0 / 2 SC0|SC1
// 15 WEAVE_LOCAL_SCATTER_DEDUP   nexp==1 local fast path        0/1
// 16 WEAVE_DEDUP_MODE            dedup variant                  0..3
// 17 WEAVE_REG_PULL_BYTES        staged pull bytes per lane      1024/2048/4096/8192
// 18 WEAVE_COMBINE_SYNC_CHUNK    node-reduce column chunk       512/1024/2048/4096
// 19 WEAVE_REDUCE_WARPS          dedicated reduce waves        1..16
// 20 WEAVE_PRECISION_MODE        body numeric schedule          0 default / 1 lean kD3 / 2 fallback bk64
// 21 WEAVE_REG_BALANCE           register/schedule balance      0 default / 1 lean / 2 aggressive kD2
// =============================================================================================================

#ifndef WEAVE_BLOCK_M_L1
#define WEAVE_BLOCK_M_L1 64
#endif
#ifndef WEAVE_BLOCK_M_L2
#define WEAVE_BLOCK_M_L2 64
#endif
#ifndef WEAVE_ADAPT_BM
#define WEAVE_ADAPT_BM 0
#endif
#ifndef WEAVE_DISPATCH_ACTIVE_SMS
#define WEAVE_DISPATCH_ACTIVE_SMS 4
#endif
#ifndef WEAVE_SCATTER_ACTIVE_SMS
#define WEAVE_SCATTER_ACTIVE_SMS 8
#endif
#ifndef WEAVE_DISPATCH_WARPS
#define WEAVE_DISPATCH_WARPS 8
#endif
#ifndef WEAVE_SCATTER_WARPS
#define WEAVE_SCATTER_WARPS 4
#endif
#ifndef WEAVE_CU_WAVE_SPLIT
#define WEAVE_CU_WAVE_SPLIT 0
#endif
#ifndef WEAVE_SM0_STREAM_SCHED
#define WEAVE_SM0_STREAM_SCHED 1
#endif
#ifndef WEAVE_V_HEAD
#define WEAVE_V_HEAD 25
#endif
#ifndef WEAVE_READY_GROUP_SIZE
#define WEAVE_READY_GROUP_SIZE 166
#endif
#ifndef WEAVE_READY_QUEUE
#define WEAVE_READY_QUEUE 1
#endif
#ifndef WEAVE_SCATTER_OWNER
#define WEAVE_SCATTER_OWNER 1
#endif
#ifndef WEAVE_SCATTER_STAGING
#define WEAVE_SCATTER_STAGING 2
#endif
#ifndef WEAVE_LOCAL_SCATTER_DEDUP
#define WEAVE_LOCAL_SCATTER_DEDUP 1
#endif
#ifndef WEAVE_DEDUP_MODE
#define WEAVE_DEDUP_MODE 1
#endif
#ifndef WEAVE_REG_PULL_BYTES
#define WEAVE_REG_PULL_BYTES 8192
#endif
#ifndef WEAVE_COMBINE_SYNC_CHUNK
#define WEAVE_COMBINE_SYNC_CHUNK 4096
#endif
#ifndef WEAVE_REDUCE_WARPS
#define WEAVE_REDUCE_WARPS 0
#endif
#ifndef WEAVE_PRECISION_MODE
#define WEAVE_PRECISION_MODE 0
#endif
#ifndef WEAVE_REG_BALANCE
#define WEAVE_REG_BALANCE 0
#endif

#define WEAVE_KNOB_COUNT 21
static_assert(WEAVE_KNOB_COUNT == 21, "the official Weave surface is exactly 21 knobs");

// ---- 1/2: independent stage tile rows ---------------------------------------------------------------
static_assert(WEAVE_BLOCK_M_L1 == 16 || WEAVE_BLOCK_M_L1 == 32 || WEAVE_BLOCK_M_L1 == 64 || WEAVE_BLOCK_M_L1 == 128,
              "WEAVE_BLOCK_M_L1 in {16,32,64,128}");
static_assert(WEAVE_BLOCK_M_L2 == 16 || WEAVE_BLOCK_M_L2 == 32 || WEAVE_BLOCK_M_L2 == 64 || WEAVE_BLOCK_M_L2 == 128,
              "WEAVE_BLOCK_M_L2 in {16,32,64,128}");
static_assert((WEAVE_BLOCK_M_L1 & 15) == 0 && (WEAVE_BLOCK_M_L2 & 15) == 0,
              "stage tile rows are multiples of 16");
// ---- 3 ---------------------------------------------------------------------------------------------
static_assert(WEAVE_ADAPT_BM == 0 || WEAVE_ADAPT_BM == 1, "WEAVE_ADAPT_BM 0/1");
// ---- 4/5/6/7 ---------------------------------------------------------------------------------------
static_assert(WEAVE_DISPATCH_ACTIVE_SMS >= 0 && WEAVE_DISPATCH_ACTIVE_SMS <= 80, "dispatch SMS 0..80");
static_assert(WEAVE_SCATTER_ACTIVE_SMS >= 0 && WEAVE_SCATTER_ACTIVE_SMS <= 80, "scatter SMS 0..80");
static_assert(WEAVE_DISPATCH_ACTIVE_SMS + WEAVE_SCATTER_ACTIVE_SMS <= 80,
              "dispatch+scatter CTA budget <= 80");
static_assert(WEAVE_DISPATCH_WARPS >= 0 && WEAVE_DISPATCH_WARPS <= 8, "dispatch warps 0..8");
static_assert(WEAVE_SCATTER_WARPS >= 0 && WEAVE_SCATTER_WARPS <= 8, "scatter warps 0..8");
// ---- 8 ---------------------------------------------------------------------------------------------
static_assert(WEAVE_CU_WAVE_SPLIT >= 0 && WEAVE_CU_WAVE_SPLIT <= 3, "WEAVE_CU_WAVE_SPLIT 0..3");
// ---- 9/10 ------------------------------------------------------------------------------------------
static_assert(WEAVE_SM0_STREAM_SCHED >= 0 && WEAVE_SM0_STREAM_SCHED <= 3, "SM0 stream sched 0..3");
static_assert(WEAVE_V_HEAD >= 0 && WEAVE_V_HEAD <= 100, "WEAVE_V_HEAD 0..100");
// ---- 11/12/13/14 -----------------------------------------------------------------------------------
static_assert(WEAVE_READY_GROUP_SIZE >= 0 && WEAVE_READY_GROUP_SIZE <= 64, "ready group 0..64");
static_assert(WEAVE_READY_QUEUE >= 0 && WEAVE_READY_QUEUE <= 3, "ready queue 0..3");
static_assert(WEAVE_SCATTER_OWNER >= 0 && WEAVE_SCATTER_OWNER <= 7, "scatter owner 0..7");
static_assert(WEAVE_SCATTER_STAGING >= 0 && WEAVE_SCATTER_STAGING <= 2, "scatter staging 0..2");
// ---- 15/16 -----------------------------------------------------------------------------------------
static_assert(WEAVE_LOCAL_SCATTER_DEDUP == 0 || WEAVE_LOCAL_SCATTER_DEDUP == 1, "local dedup 0/1");
static_assert(WEAVE_DEDUP_MODE >= 0 && WEAVE_DEDUP_MODE <= 3, "dedup mode 0..3");
// ---- 17/18/19 --------------------------------------------------------------------------------------
static_assert(WEAVE_REG_PULL_BYTES == 0 || WEAVE_REG_PULL_BYTES == 1024 || WEAVE_REG_PULL_BYTES == 2048 ||
              WEAVE_REG_PULL_BYTES == 4096 || WEAVE_REG_PULL_BYTES == 8192,
              "WEAVE_REG_PULL_BYTES in {0,1024,2048,4096,8192}");
static_assert(WEAVE_COMBINE_SYNC_CHUNK == 0 || WEAVE_COMBINE_SYNC_CHUNK == 16 || WEAVE_COMBINE_SYNC_CHUNK == 32 ||
              WEAVE_COMBINE_SYNC_CHUNK == 64 || WEAVE_COMBINE_SYNC_CHUNK == 128 || WEAVE_COMBINE_SYNC_CHUNK == 512 ||
              WEAVE_COMBINE_SYNC_CHUNK == 1024 || WEAVE_COMBINE_SYNC_CHUNK == 2048 || WEAVE_COMBINE_SYNC_CHUNK == 4096,
              "WEAVE_COMBINE_SYNC_CHUNK in {0,16,32,64,128,512,1024,2048,4096}");
static_assert(WEAVE_REDUCE_WARPS >= 0 && WEAVE_REDUCE_WARPS <= 16, "reduce warps 0..16");
// ---- 20/21 -----------------------------------------------------------------------------------------
static_assert(WEAVE_PRECISION_MODE >= 0 && WEAVE_PRECISION_MODE <= 2, "precision mode 0..2");
static_assert(WEAVE_REG_BALANCE >= 0 && WEAVE_REG_BALANCE <= 2, "reg balance 0..2");

// ---- derived layout: WEAVE_CU_WAVE_SPLIT changes the REAL CTA role assignment ------------------------
// 0 auto      : pure 0-8 layout whenever the fixed-tile hardware contract holds
//               (WEAVE_BLOCK_M_L1==128 && WEAVE_BLOCK_M_L2==128 && WEAVE_ADAPT_BM==0): (D+K) CTAs 0/8 comm,
//               every remaining CTA is a pure 8/0 compute CTA.  Otherwise the V62 mixed baseline
//               (D CTAs 0/8 dispatch + K CTAs 4/4 scatter).
// 1 pure 0-8  : (D+K) CTAs 0/8 comm, every remaining CTA is a pure 8/0 compute CTA
// 2 4-4       : (D+K) CTAs 4/4, every remaining CTA is 8/0
// 3 8-0 comm  : 2*(D+K) CTAs 8/0 comm, remaining CTAs are 8/0 compute
// Auto keeps the measured mixed 4-wave arm for BM=64; the explicit split=1
// arm still selects the pure 0/8 hardware-s_barrier body for BM=64 or 128.
#define WEAVE_AUTO_PURE08 (((WEAVE_BLOCK_M_L1) == 128) && ((WEAVE_BLOCK_M_L2) == 128) && !(WEAVE_ADAPT_BM))
#if WEAVE_CU_WAVE_SPLIT == 0
#if WEAVE_AUTO_PURE08
#define WEAVE_LAYOUT_N08 (WEAVE_DISPATCH_ACTIVE_SMS + WEAVE_SCATTER_ACTIVE_SMS)
#define WEAVE_LAYOUT_N44 0
#else
#define WEAVE_LAYOUT_N08 (WEAVE_DISPATCH_ACTIVE_SMS)
#define WEAVE_LAYOUT_N44 (WEAVE_SCATTER_ACTIVE_SMS)
#endif
#elif WEAVE_CU_WAVE_SPLIT == 1
#define WEAVE_LAYOUT_N08 (WEAVE_DISPATCH_ACTIVE_SMS + WEAVE_SCATTER_ACTIVE_SMS)
#define WEAVE_LAYOUT_N44 0
#elif WEAVE_CU_WAVE_SPLIT == 2
#define WEAVE_LAYOUT_N08 0
#define WEAVE_LAYOUT_N44 (WEAVE_DISPATCH_ACTIVE_SMS + WEAVE_SCATTER_ACTIVE_SMS)
#else
#define WEAVE_LAYOUT_N08 (2 * (WEAVE_DISPATCH_ACTIVE_SMS + WEAVE_SCATTER_ACTIVE_SMS))
#define WEAVE_LAYOUT_N44 0
#endif
#if !V70_PATHB
static_assert(WEAVE_LAYOUT_N08 + WEAVE_LAYOUT_N44 <= 80, "role layout must fit 80 CTAs");
#endif

// ---- derived comm-pool geometry ---------------------------------------------------------------------
// Dispatch warps set the remote-plan wave groups; scatter warps scale the dedicated reduce pool.
#define WEAVE_LAYOUT_WCOMM (8 * WEAVE_LAYOUT_N08 + 4 * WEAVE_LAYOUT_N44)
#if WEAVE_DISPATCH_WARPS >= 4
#define WEAVE_LAYOUT_NRP 4
#elif WEAVE_DISPATCH_WARPS >= 2
#define WEAVE_LAYOUT_NRP 2
#else
#define WEAVE_LAYOUT_NRP 1
#endif
#define WEAVE_LAYOUT_CW_POOL (8 + 4 * WEAVE_LAYOUT_NRP)
#define WEAVE_LAYOUT_NNR_RAW ((WEAVE_LAYOUT_WCOMM - WEAVE_LAYOUT_CW_POOL) * WEAVE_SCATTER_WARPS / 16)
#define WEAVE_LAYOUT_NNR ((WEAVE_LAYOUT_NNR_RAW) < 1 ? 1 : ((WEAVE_LAYOUT_NNR_RAW) > (WEAVE_REDUCE_WARPS) ? (WEAVE_REDUCE_WARPS) : (WEAVE_LAYOUT_NNR_RAW)))
#define WEAVE_LAYOUT_NPULL (WEAVE_LAYOUT_WCOMM - WEAVE_LAYOUT_CW_POOL - WEAVE_LAYOUT_NNR)
#if !V70_PATHB
static_assert(WEAVE_LAYOUT_WCOMM >= 16, "W_comm = 8*N08 + 4*N44 must be >= 16");
#endif
#if !V70_PATHB
static_assert(WEAVE_LAYOUT_WCOMM >= WEAVE_LAYOUT_CW_POOL + 2, "need >= 1 reduce wave and >= 1 pull wave");
#endif
#if !V70_PATHB
static_assert(WEAVE_LAYOUT_NPULL >= 1, "NPULL >= 1");
#endif

// ---- derived pull / scatter constants ---------------------------------------------------------------
#define WEAVE_LAYOUT_PULL_BT ((WEAVE_REG_PULL_BYTES) / 1024)
#if !V70_PATHB
static_assert(WEAVE_LAYOUT_PULL_BT >= 1 && WEAVE_LAYOUT_PULL_BT <= 8, "pull tokens 1..8");
#endif
