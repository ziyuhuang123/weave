// ============================================================================
// ep16_turbo_host.hip -- host launch adapter for P_ep16_turbo_overlap.co
//
// Build:  tools/build_turbo_host.sh   (hipcc, CPU-only: HIP_VISIBLE_DEVICES="")
//
// The library is deliberately split in two halves:
//   * pure integer layout/plan/packet construction (ep16_turbo_*.hpp), which
//     needs no GPU and is unit tested on the CPU, and
//   * a thin HIP shim: hipModuleLoad / hipModuleGetFunction /
//     hipModuleLaunchKernel plus hipMemset-based generation reset.
//
// Allocation, IPC handle exchange and QP setup stay in the existing
// libmega_ipc.so / libep16_gda_host_v2.so providers; this library never
// allocates device memory itself, so the two arms cannot fight over the
// process-global provider state.
// ============================================================================
#include <hip/hip_runtime.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdint>

#include "ep16_turbo_abi.hpp"
#include "ep16_turbo_layout.hpp"
#include "ep16_turbo_plan.hpp"
#include "ep16_turbo_packet.hpp"

#define EP16_TURBO_HOST_ABI 1u

static char g_err[4096] = {0};
static char g_probe[8192] = {0};
static hipModule_t g_module = nullptr;
static hipFunction_t g_function = nullptr;
static char g_module_path[1024] = {0};

static int set_err(int rc, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
  return rc;
}

#define HIP_CK(expr) do {                                                      \
  hipError_t _e = (expr);                                                      \
  if (_e != hipSuccess)                                                        \
    return set_err(1000 + (int)_e, "%s failed: %s (%d)", #expr,                \
                   hipGetErrorString(_e), (int)_e);                            \
} while (0)

extern "C" uint32_t ep16_turbo_abi_version() { return EP16_TURBO_HOST_ABI; }
extern "C" const char* ep16_turbo_last_error() { return g_err; }
extern "C" const char* ep16_turbo_kernel_name() { return EP16T_KERNEL_NAME; }

// ---------------------------------------------------------------------------
// ABI probe: the authoritative sizeof/offsetof table the python ctypes mirror
// is validated against.  Emitted as JSON so the check is one dict comparison.
// ---------------------------------------------------------------------------
extern "C" const char* ep16_turbo_abi_probe() {
  char* p = g_probe;
  size_t n = sizeof(g_probe);
  int w = snprintf(p, n,
    "{\"schema\":\"ep16-turbo-host-abi/1\","
    "\"sizeof\":{"
      "\"V964DispatchExt\":%zu,\"V968CkExt\":%zu,\"V968DispatchExt\":%zu,"
      "\"V983Ov1Ext\":%zu,\"V983Envelope\":%zu,"
      "\"Ep16TxJob\":%zu,\"Ep16RxJob\":%zu,\"Ep16TurboContext\":%zu,"
      "\"V809Meta\":%zu,\"V809SourceProgress\":%zu,\"V1988PostControl\":%zu,"
      "\"WeaveMTileDesc\":%zu,\"WeaveTilePlanTable\":%zu,"
      "\"Ep16TurboShape\":%zu,\"Ep16ArenaLayout\":%zu,\"Ep16ScratchLayout\":%zu,"
      "\"Ep16TurboExternal\":%zu,\"Ep16TurboBases\":%zu,"
      "\"Ep16TurboLaunchSpec\":%zu,\"Ep16PlanAudit\":%zu},"
    "\"offsetof\":{"
      "\"V983Envelope.ov1\":%zu,\"V968DispatchExt.ck\":%zu,"
      "\"V964DispatchExt.t_local\":%zu,\"V964DispatchExt.recv_meta\":%zu,"
      "\"V964DispatchExt.sort_capacity\":%zu,\"V964DispatchExt.ingress_header\":%zu,"
      "\"V968CkExt.ck_route_rows\":%zu,\"V983Ov1Ext.hidden\":%zu,"
      "\"Ep16TxJob.ready\":%zu,\"Ep16TxJob.target\":%zu,\"Ep16TxJob.valid_count\":%zu,"
      "\"Ep16TxJob.index\":%zu,"
      "\"Ep16RxJob.publish\":%zu,\"Ep16RxJob.value\":%zu,\"Ep16RxJob.valid_count\":%zu,"
      "\"Ep16RxJob.index\":%zu,"
      "\"Ep16TurboContext.tx_jobs\":%zu,\"Ep16TurboContext.tx_count\":%zu,"
      "\"Ep16TurboContext.rx_jobs\":%zu,\"Ep16TurboContext.return_ready\":%zu,"
      "\"Ep16TurboContext.node_partial\":%zu,\"Ep16TurboContext.compute_done\":%zu,"
      "\"Ep16TurboContext.gda_state\":%zu,\"Ep16TurboContext.timeout_ticks\":%zu,"
      "\"V809SourceProgress.meta_ready\":%zu,\"V809Meta.lid\":%zu},"
    "\"constants\":{"
      "\"EP16_TURBO_MAGIC\":%llu,\"V809_PROGRESS_MAGIC\":%u,"
      "\"V809_META_HEADER\":%u,\"V809_META_STRIDE\":%u,\"V809_MAX_CHUNKS\":%u,"
      "\"MN_GDA_MESH_SLICE\":%u,\"MN_GDA_MESH_WINDOW\":%u,"
      "\"GRID\":%u,\"BLOCK\":%u,\"QPS\":%u,"
      "\"ARENA_REGIONS\":%d,\"SCRATCH_REGIONS\":%d,\"PT_BYTES\":%llu},"
    "\"kernel\":\"%s\"}",
    sizeof(V964DispatchExt), sizeof(V968CkExt), sizeof(V968DispatchExt),
    sizeof(V983Ov1Ext), sizeof(V983Envelope),
    sizeof(Ep16TxJob), sizeof(Ep16RxJob), sizeof(Ep16TurboContext),
    sizeof(V809Meta), sizeof(V809SourceProgress), sizeof(V1988PostControl),
    sizeof(WeaveMTileDesc), sizeof(WeaveTilePlanTable),
    sizeof(Ep16TurboShape), sizeof(Ep16ArenaLayout), sizeof(Ep16ScratchLayout),
    sizeof(Ep16TurboExternal), sizeof(Ep16TurboBases),
    sizeof(Ep16TurboLaunchSpec), sizeof(Ep16PlanAudit),
    offsetof(V983Envelope, ov1), offsetof(V968DispatchExt, ck),
    offsetof(V964DispatchExt, t_local), offsetof(V964DispatchExt, recv_meta),
    offsetof(V964DispatchExt, sort_capacity), offsetof(V964DispatchExt, ingress_header),
    offsetof(V968CkExt, ck_route_rows), offsetof(V983Ov1Ext, hidden),
    offsetof(Ep16TxJob, ready), offsetof(Ep16TxJob, target),
    offsetof(Ep16TxJob, valid_count), offsetof(Ep16TxJob, index),
    offsetof(Ep16RxJob, publish), offsetof(Ep16RxJob, value),
    offsetof(Ep16RxJob, valid_count), offsetof(Ep16RxJob, index),
    offsetof(Ep16TurboContext, tx_jobs), offsetof(Ep16TurboContext, tx_count),
    offsetof(Ep16TurboContext, rx_jobs), offsetof(Ep16TurboContext, return_ready),
    offsetof(Ep16TurboContext, node_partial), offsetof(Ep16TurboContext, compute_done),
    offsetof(Ep16TurboContext, gda_state), offsetof(Ep16TurboContext, timeout_ticks),
    offsetof(V809SourceProgress, meta_ready), offsetof(V809Meta, lid),
    (unsigned long long)EP16_TURBO_MAGIC, (unsigned)V809_PROGRESS_MAGIC,
    (unsigned)V809_META_HEADER, (unsigned)V809_META_STRIDE, (unsigned)V809_MAX_CHUNKS,
    (unsigned)MN_GDA_MESH_SLICE, (unsigned)MN_GDA_MESH_WINDOW,
    (unsigned)EP16T_GRID, (unsigned)EP16T_BLOCK, (unsigned)EP16T_QPS,
    (int)EP16_AR_COUNT, (int)EP16_SC_COUNT, (unsigned long long)EP16_PT_BYTES,
    EP16T_KERNEL_NAME);
  (void)w;
  return g_probe;
}

// ---------------------------------------------------------------------------
// pure-host planning wrappers
// ---------------------------------------------------------------------------
extern "C" int ep16_turbo_validate(const Ep16TurboShape* shape) {
  const char* why = nullptr;
  int rc = ep16_turbo_validate_shape(*shape, &why);
  if (rc) set_err(rc, "invalid shape (code %d): %s", rc, why ? why : "?");
  return rc;
}

extern "C" int ep16_turbo_plan(const Ep16TurboShape* shape,
                               Ep16ArenaLayout* arena,
                               Ep16ScratchLayout* scratch) {
  int rc = ep16_turbo_plan_arena(*shape, arena);
  if (rc) return set_err(rc, "ep16_turbo_plan_arena rc=%d", rc);
  rc = ep16_turbo_plan_scratch(*shape, scratch);
  if (rc) return set_err(rc, "ep16_turbo_plan_scratch rc=%d", rc);
  return 0;
}

extern "C" int ep16_turbo_peer_tables(const Ep16TurboShape* shape,
                                      const Ep16ArenaLayout* arena,
                                      const Ep16ScratchLayout* scratch,
                                      const Ep16TurboBases* bases,
                                      uint64_t* blob) {
  int rc = ep16_turbo_build_peer_tables(*shape, *arena, *bases, blob);
  if (rc) return set_err(rc, "ep16_turbo_build_peer_tables rc=%d", rc);
  ep16_turbo_fill_local_meta_table(*arena, *scratch, *bases, blob);
  return 0;
}

extern "C" int ep16_turbo_jobs(const Ep16TurboShape* shape,
                               const Ep16ArenaLayout* arena,
                               const Ep16ScratchLayout* scratch,
                               const Ep16TurboBases* bases,
                               Ep16TxJob* tx, uint32_t tx_cap,
                               Ep16RxJob* rx, uint32_t rx_cap) {
  int rc = ep16_turbo_build_tx(*shape, *arena, *scratch, *bases, tx, tx_cap);
  if (rc) return set_err(rc, "ep16_turbo_build_tx rc=%d", rc);
  rc = ep16_turbo_build_rx(*shape, *arena, *scratch, *bases, rx, rx_cap);
  if (rc) return set_err(rc, "ep16_turbo_build_rx rc=%d", rc);
  return 0;
}

extern "C" int ep16_turbo_audit(const Ep16TurboShape* shape,
                                const Ep16ArenaLayout* arena,
                                const Ep16ScratchLayout* scratch,
                                const Ep16TurboBases* bases,
                                const Ep16TxJob* tx, const Ep16RxJob* rx,
                                Ep16PlanAudit* out) {
  int rc = ep16_turbo_audit_plan(*shape, *arena, *scratch, *bases, tx, rx, out);
  if (rc)
    return set_err(rc, "plan audit failed: %u errors, first code %u at job %u",
                   out ? out->errors : 0u, out ? out->first_error_code : 0u,
                   out ? out->first_error_job : 0u);
  return 0;
}

extern "C" int ep16_turbo_packet(const Ep16TurboShape* shape,
                                 const Ep16ArenaLayout* arena,
                                 const Ep16ScratchLayout* scratch,
                                 const Ep16TurboBases* bases,
                                 const Ep16TurboExternal* ext,
                                 const Ep16TurboLaunchSpec* spec,
                                 V983Envelope* env, Ep16TurboContext* ctx) {
  int rc = ep16_turbo_build_packet(*shape, *arena, *scratch, *bases, *ext, *spec, env, ctx);
  if (rc) return set_err(rc, "ep16_turbo_build_packet rc=%d", rc);
  rc = ep16_turbo_check_entry_guard(*env, *ctx);
  if (rc) return set_err(rc, "device entry guard would reject the packet (check %d)", rc);
  return 0;
}

// ---------------------------------------------------------------------------
// HIP shim (caches loaded code objects in-process so hot-swapping 21 arms
// never triggers hipModuleUnload VM unmap races on active ROCr/GDA queues)
// ---------------------------------------------------------------------------
struct Ep16CachedModule {
  char path[1024];
  hipModule_t mod;
  hipFunction_t fn;
};
static Ep16CachedModule g_mod_cache[64];
static int g_mod_cache_cnt = 0;

extern "C" int ep16_turbo_module_load(const char* co_path) {
  if (g_module) return set_err(-1, "a code object is already loaded: %s", g_module_path);
  if (!co_path || !co_path[0]) return set_err(-2, "empty code object path");
  for (int i = 0; i < g_mod_cache_cnt; ++i) {
    if (std::strcmp(g_mod_cache[i].path, co_path) == 0) {
      g_module = g_mod_cache[i].mod;
      g_function = g_mod_cache[i].fn;
      snprintf(g_module_path, sizeof(g_module_path), "%s", co_path);
      return 0;
    }
  }
  HIP_CK(hipModuleLoad(&g_module, co_path));
  hipError_t e = hipModuleGetFunction(&g_function, g_module, EP16T_KERNEL_NAME);
  if (e != hipSuccess) {
    (void)!hipModuleUnload(g_module);   // best effort; report the real failure
    g_module = nullptr;
    return set_err(1000 + (int)e, "hipModuleGetFunction(%s) failed: %s",
                   EP16T_KERNEL_NAME, hipGetErrorString(e));
  }
  snprintf(g_module_path, sizeof(g_module_path), "%s", co_path);
  if (g_mod_cache_cnt < 64) {
    snprintf(g_mod_cache[g_mod_cache_cnt].path, sizeof(g_mod_cache[g_mod_cache_cnt].path), "%s", co_path);
    g_mod_cache[g_mod_cache_cnt].mod = g_module;
    g_mod_cache[g_mod_cache_cnt].fn = g_function;
    ++g_mod_cache_cnt;
  }
  return 0;
}

extern "C" int ep16_turbo_module_unload() {
  if (!g_module) return 0;
  g_module = nullptr;
  g_function = nullptr;
  g_module_path[0] = 0;
  return 0;
}

extern "C" uint64_t ep16_turbo_function_ptr() { return (uint64_t)(uintptr_t)g_function; }

extern "C" int ep16_turbo_kernel_attributes(int32_t* out /* [8] */) {
  if (!g_function) return set_err(-1, "no code object loaded");
  int v = 0;
  static const hipFunction_attribute attrs[8] = {
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_NUM_REGS,
    HIP_FUNC_ATTRIBUTE_PTX_VERSION,
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION,
    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
  };
  for (int i = 0; i < 8; ++i) {
    v = 0;
    hipError_t e = hipFuncGetAttribute(&v, attrs[i], g_function);
    out[i] = (e == hipSuccess) ? v : -1;
  }
  return 0;
}

// grid/block support both legacy 80x512 (8-wave) and V70 Path B 160x256 (2x4-wave/CU).
extern "C" int ep16_turbo_launch_v20(uint64_t stream, uint64_t envelope_host,
                                 uint64_t context_dev, uint32_t grid,
                                 uint32_t block) {
  if (!g_function) return set_err(-1, "no code object loaded");
  uint32_t dyn_smem = 0u;
  if (grid == EP16T_GRID && block == EP16T_BLOCK) {
    dyn_smem = 0u;
  } else if (grid == 160u && block == 256u) {
    int static_lds = 0;
    if (hipFuncGetAttribute(&static_lds, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, g_function) == hipSuccess
        && static_lds >= 0 && static_lds < 32768) {
      // Path B uses 32 KiB per CTA (2 CTAs/CU = 64 KiB/CU); supply dynamic LDS
      // only when the compiled code object leaves LDS dynamic (static_lds < 32 KiB).
      dyn_smem = (static_lds == 0) ? 32768u : 0u;
    }
  } else {
    return set_err(-2, "turbo kernel requires (%u,%u) or (160,256), got %u/%u",
                   EP16T_GRID, EP16T_BLOCK, grid, block);
  }
  if (!envelope_host || !context_dev) return set_err(-3, "null host envelope or device context pointer");
  struct alignas(8) { V983Envelope packet; uint64_t context; } args;
  static_assert(sizeof(args) == 808 && offsetof(decltype(args), context) == 800,
                "v20 kernel explicit argument ABI mismatch");
  memcpy(&args.packet, (const void*)(uintptr_t)envelope_host, sizeof(V983Envelope));
  args.context = context_dev;
  size_t arg_size = sizeof(args);
  void* config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, (void*)&args,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, (void*)&arg_size,
    HIP_LAUNCH_PARAM_END
  };
  HIP_CK(hipModuleLaunchKernel(g_function, grid, 1, 1, block, 1, 1, dyn_smem,
                               (hipStream_t)(uintptr_t)stream, nullptr, config));
  return 0;
}

extern "C" int ep16_turbo_launch(uint64_t stream, uint64_t envelope_host,
                                 uint64_t context_dev, uint32_t grid,
                                 uint32_t block) {
  return ep16_turbo_launch_v20(stream, envelope_host, context_dev, grid, block);
}

// ---------------------------------------------------------------------------
// generation reset.  Zeroes exactly the regions the layout marks as readiness
// / cursor / done state; bulk data regions are skipped because every consumer
// is gated behind one of those words.
// ---------------------------------------------------------------------------
extern "C" int ep16_turbo_reset(uint64_t stream, uint64_t arena, uint64_t scratch,
                               const Ep16ArenaLayout* L, const Ep16ScratchLayout* S,
                               uint32_t mask) {
  hipStream_t st = (hipStream_t)(uintptr_t)stream;
  if (mask & 1u) {
    for (int i = 0; i < EP16_AR_COUNT; ++i) {
      if (!L->zero_on_reset[i]) continue;
      HIP_CK(hipMemsetAsync((void*)(uintptr_t)(arena + L->off[i]), 0, (size_t)L->size[i], st));
    }
  }
  if (mask & 2u) {
    for (int i = 0; i < EP16_SC_COUNT; ++i) {
      if (!S->zero_on_reset[i]) continue;
      HIP_CK(hipMemsetAsync((void*)(uintptr_t)(scratch + S->off[i]), 0, (size_t)S->size[i], st));
    }
  }
  return 0;
}

// Convenience: one memset over a region named by enum, so the python layer
// never has to recompute an offset.
extern "C" int ep16_turbo_memset_arena(uint64_t stream, uint64_t arena,
                                       const Ep16ArenaLayout* L, int region,
                                       int value) {
  if (region < 0 || region >= EP16_AR_COUNT) return set_err(-1, "bad arena region %d", region);
  HIP_CK(hipMemsetAsync((void*)(uintptr_t)(arena + L->off[region]), value,
                        (size_t)L->size[region], (hipStream_t)(uintptr_t)stream));
  return 0;
}

extern "C" int ep16_turbo_memset_scratch(uint64_t stream, uint64_t scratch,
                                         const Ep16ScratchLayout* S, int region,
                                         int value) {
  if (region < 0 || region >= EP16_SC_COUNT) return set_err(-1, "bad scratch region %d", region);
  HIP_CK(hipMemsetAsync((void*)(uintptr_t)(scratch + S->off[region]), value,
                        (size_t)S->size[region], (hipStream_t)(uintptr_t)stream));
  return 0;
}
