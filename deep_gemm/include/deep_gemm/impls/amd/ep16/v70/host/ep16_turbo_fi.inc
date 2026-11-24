// ============================================================================
// fi_ext.inc -- fi_lite (FILITE02) capture extension for ep16_turbo_host.hip
// Appended to a COPY of sub6 ep16_turbo_host.hip (same TU -> access g_module,
// set_err, HIP_CK). Writes the module's `fi_lite_global` FiBuf via
// hipModuleGetGlobal; allocates rec[n_slots*cap] + overflow; dumps FILITE02.
// FiBuf 72 B / FiRec 16 B (static_assert locked). CPU-only compilable.
// ============================================================================
#include <vector>
#include <string>
#include <cerrno>

struct FiRec { uint64_t t; uint32_t tag; uint32_t aux; };
struct FiBuf {
  FiRec* rec; uint32_t* overflow; uint64_t timer_hz;
  uint32_t cap; uint32_t warps_per_cta; uint32_t n_slots; char arch[32];
};
static_assert(sizeof(FiRec)==16, "FiRec must be 16 bytes");
static_assert(sizeof(FiBuf)==72, "FiBuf must be 72 bytes");
static_assert(offsetof(FiBuf,timer_hz)==16, "FiBuf.timer_hz@16");
static_assert(offsetof(FiBuf,cap)==24, "FiBuf.cap@24");
static_assert(offsetof(FiBuf,warps_per_cta)==28, "FiBuf.warps_per_cta@28");
static_assert(offsetof(FiBuf,n_slots)==32, "FiBuf.n_slots@32");
static_assert(offsetof(FiBuf,arch)==36, "FiBuf.arch@36");

static FiBuf g_fibuf;
static void* g_fi_sym = nullptr;
static size_t g_fi_bytes = 0;
static bool g_fi_up = false;

extern "C" int ep16_turbo_fi_upload(uint32_t n_slots, uint32_t cap,
                                    uint32_t warps_per_cta, uint64_t timer_hz,
                                    const char* arch) {
  if (!g_module) return set_err(-1, "fi_upload: no code object loaded");
  if (g_fi_up)   return set_err(-2, "fi_upload: already uploaded");
  if (n_slots==0u || cap==0u || warps_per_cta==0u || timer_hz==0ull)
    return set_err(-4, "fi_upload: bad params");
  hipDeviceptr_t sym; size_t bytes = 0;
  HIP_CK(hipModuleGetGlobal(&sym, &bytes, g_module, "fi_lite_global"));
  if (bytes < sizeof(FiBuf))
    return set_err(-3, "fi_lite_global symbol %zu B < 72", bytes);
  memset(&g_fibuf, 0, sizeof(g_fibuf));
  g_fibuf.cap = cap; g_fibuf.warps_per_cta = warps_per_cta;
  g_fibuf.n_slots = n_slots; g_fibuf.timer_hz = timer_hz;
  snprintf(g_fibuf.arch, sizeof(g_fibuf.arch), "%s", arch ? arch : "gfx942");
  size_t recbytes = (size_t)n_slots * (size_t)cap * sizeof(FiRec);
  HIP_CK(hipMalloc((void**)&g_fibuf.rec, recbytes));
  HIP_CK(hipMalloc((void**)&g_fibuf.overflow, sizeof(uint32_t)));
  HIP_CK(hipMemset(g_fibuf.rec, 0, recbytes));
  HIP_CK(hipMemset(g_fibuf.overflow, 0, sizeof(uint32_t)));
  HIP_CK(hipMemcpy((void*)(uintptr_t)sym, &g_fibuf, sizeof(FiBuf), hipMemcpyHostToDevice));
  g_fi_sym = (void*)(uintptr_t)sym; g_fi_bytes = bytes; g_fi_up = true;
  return 0;
}

extern "C" int ep16_turbo_fi_reset(uint64_t stream) {
  if (!g_fi_up) return set_err(-1, "fi_reset: not uploaded");
  hipStream_t st = (hipStream_t)(uintptr_t)stream;
  size_t recbytes = (size_t)g_fibuf.n_slots * (size_t)g_fibuf.cap * sizeof(FiRec);
  HIP_CK(hipMemsetAsync(g_fibuf.rec, 0, recbytes, st));
  HIP_CK(hipMemsetAsync(g_fibuf.overflow, 0, sizeof(uint32_t), st));
  return 0;
}

static std::string fi_meta_aug(const char* meta_json) {
  std::string m = (meta_json && *meta_json) ? std::string(meta_json) : std::string("{}");
  size_t close = m.find_last_of('}');
  if (close == std::string::npos) return std::string("{}");
  size_t first = m.find_first_not_of(" \t\r\n");
  bool empty_obj = true;
  for (size_t i = first + 1; i < close; ++i)
    if (m[i]!=' '&&m[i]!='\t'&&m[i]!='\r'&&m[i]!='\n') { empty_obj=false; break; }
  char inj[256];
  snprintf(inj, sizeof inj, "\"timer_hz\":%llu,\"gran\":0,\"arch\":\"%s\"",
           (unsigned long long)g_fibuf.timer_hz, g_fibuf.arch);
  m.insert(close, std::string(empty_obj ? "" : ",") + inj);
  return m;
}

extern "C" int ep16_turbo_fi_download(const char* path, const char* meta_json) {
  if (!g_fi_up) return set_err(-1, "fi_download: not uploaded");
  HIP_CK(hipDeviceSynchronize());
  size_t n = (size_t)g_fibuf.n_slots * (size_t)g_fibuf.cap;
  std::vector<FiRec> all(n);
  uint32_t ovf = 0;
  HIP_CK(hipMemcpy(all.data(), g_fibuf.rec, n * sizeof(FiRec), hipMemcpyDeviceToHost));
  HIP_CK(hipMemcpy(&ovf, g_fibuf.overflow, sizeof(uint32_t), hipMemcpyDeviceToHost));
  std::vector<uint32_t> cnt(g_fibuf.n_slots, 0);
  for (uint32_t s = 0; s < g_fibuf.n_slots; ++s)
    while (cnt[s] < g_fibuf.cap && all[(size_t)s*g_fibuf.cap + cnt[s]].tag != 0u) ++cnt[s];
  FILE* f = fopen(path, "wb");
  if (!f) return set_err(-3, "fi_download fopen(%s): %s", path, strerror(errno));
  const std::string meta = fi_meta_aug(meta_json);
  const uint32_t hdr[6] = {g_fibuf.n_slots, g_fibuf.cap, g_fibuf.warps_per_cta,
                           ovf, 0u /*FI_TIMER*/, (uint32_t)meta.size()};
  fwrite("FILITE02", 1, 8, f);
  fwrite(hdr, 4, 6, f);
  fwrite(meta.data(), 1, meta.size(), f);
  fwrite(cnt.data(), 4, cnt.size(), f);
  for (uint32_t s = 0; s < g_fibuf.n_slots; ++s)
    if (cnt[s]) fwrite(&all[(size_t)s*g_fibuf.cap], sizeof(FiRec), cnt[s], f);
  fclose(f);
  return (int)ovf;
}

extern "C" int ep16_turbo_fi_free() {
  if (!g_fi_up) return 0;
  if (g_fibuf.rec) { hipError_t e=hipFree(g_fibuf.rec); (void)e; }
  if (g_fibuf.overflow) { hipError_t e=hipFree(g_fibuf.overflow); (void)e; }
  memset(&g_fibuf, 0, sizeof(g_fibuf));
  g_fi_up = false; g_fi_sym = nullptr; g_fi_bytes = 0;
  return 0;
}

extern "C" const char* ep16_turbo_fi_probe() {
  static char buf[640];
  snprintf(buf, sizeof buf,
    "{\"fi_uploaded\":%s,\"n_slots\":%u,\"cap\":%u,\"warps_per_cta\":%u,"
    "\"timer_hz\":%llu,\"rec_bytes\":%zu,\"symbol_bytes\":%zu,"
    "\"sizeof_FiBuf\":%zu,\"sizeof_FiRec\":%zu,\"arch\":\"%s\"}",
    g_fi_up?"true":"false", g_fibuf.n_slots, g_fibuf.cap, g_fibuf.warps_per_cta,
    (unsigned long long)g_fibuf.timer_hz,
    (size_t)g_fibuf.n_slots*(size_t)g_fibuf.cap*sizeof(FiRec), g_fi_bytes,
    sizeof(FiBuf), sizeof(FiRec), g_fibuf.arch);
  return buf;
}
