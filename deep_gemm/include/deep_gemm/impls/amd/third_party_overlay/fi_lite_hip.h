// fast-instrument / fi_lite_hip.h —— HIP/ROCm (CDNA, wave64) 的 kernel 内打点，宏 API 与 CUDA 版完全一致。
//
// 与既有 CUDA 头文件的关系：同一 FILITE02 转储格式（16 字节记录，既有解码器可读本头文件的转储——只需认识 meta 里的新字段，
// 新字段只有 timer_hz / gran / arch，均为 JSON，不影响二进制布局）。宏 API：FI_OPEN / FI_OPEN_BY / FI_B / FI_E / FI_M /
// FI_CLOSE / FI_BG / FI_EG / FI_MG / FI_CTX_PARAM / FI_CTX_ARG，与 CUDA 版同名同参。
//
// 编译开关：
//   FI_LEVEL  0=全空（正式构建） 1=只记 OPEN/CLOSE  2=完整事件
//   FI_TIMER  0=全卡墙钟计时器 s_memrealtime（固定频率，跨 CU 同一时基；转储 meta 写 timer_hz，解码按它换算成 ns）
//             1=本地周期计数器 readcyclecounter（每泳道 OPEN/CLOSE 各记一对 (s_memrealtime, cycle) 锚点，解码线性换算）
//   FI_GRAN   0=wave 粒度：每个 wave64 由 lane 0 记录（每个 wave 一条泳道）
//             1=CU 粒度：每个 CTA 只由线性线程号 0（一维块即 threadIdx.x==0）记录；OPEN 的 aux 记 HW_ID 解出的
//               XCC/SE/CU（编码见下），解码按物理 CU 归组，每个 CU 一条时间条。主机侧 fi_alloc 的 warps_per_cta 传 1。
//   FI_GROUPS 事件组位掩码（默认全开），被屏蔽的 FI_BG/FI_EG/FI_MG 编译为空。
//
// OPEN 的 aux 编码（gfx942，取自 HW_ID 与 XCC_ID）：bits[3:0]=CU, bits[5:4]=SIMD, bits[9:8]=SE, bits[15:12]=XCC。
//   物理 CU 全局号 = (XCC<<6)|(SE<<4)|CU（SE 2bit、CU 4bit；MI308X 每 XCC 的 SE/CU 组合唯一）。
//
// 热路径开销：每个事件 = 1 次计时器读 + 1 次 16B 流式全局写（global_store_dwordx4，发出即走），无全局读、无 atomic
// （仅泳道写满时一次 atomicOr 置溢出位）。FI_GRAN=1 把记录线程数降到每 CTA 一个，开销再降。
#pragma once
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>

#ifndef FI_LEVEL
#define FI_LEVEL 0
#endif
#ifndef FI_TIMER
#define FI_TIMER 0
#endif
#ifndef FI_GRAN
#define FI_GRAN 0
#endif
#ifndef FI_GROUPS
#define FI_GROUPS 0xffffffffu
#endif

#if defined(__gfx942__)
#define FI_ARCH_STR "gfx942"
#elif defined(__gfx941__)
#define FI_ARCH_STR "gfx941"
#elif defined(__gfx940__)
#define FI_ARCH_STR "gfx940"
#elif defined(__gfx90a__)
#define FI_ARCH_STR "gfx90a"
#elif defined(__gfx90__)
#define FI_ARCH_STR "gfx90"
#elif defined(__gfx950__)
#define FI_ARCH_STR "gfx950"
#elif defined(__gfx1030__)
#define FI_ARCH_STR "gfx1030"
#elif defined(__gfx1100__)
#define FI_ARCH_STR "gfx1100"
#elif defined(__gfx1101__)
#define FI_ARCH_STR "gfx1101"
#elif defined(__gfx1103__)
#define FI_ARCH_STR "gfx1103"
#elif defined(__gfx1200__)
#define FI_ARCH_STR "gfx1200"
#elif defined(__gfx1201__)
#define FI_ARCH_STR "gfx1201"
#else
#define FI_ARCH_STR "gfx-unknown"
#endif

#define FI_TAG_END 0x80000000u     // 区间结束
#define FI_TAG_OPEN 0x40000000u    // FI_OPEN：低 16 位=角色号；aux=HW_ID 编码（见头注）；t=计时器
#define FI_TAG_MARK 0x20000000u    // 瞬时事件
#define FI_TAG_CLK 0x10000000u     // 锚点的 cycle 半边（紧跟 OPEN/CLOSE 之后）；aux=0 开、1 关
#define FI_TAG_CLOSE 0x08000000u   // FI_CLOSE
#define FI_TAG_IDMASK 0x0fffu

struct FiRec {
    uint64_t t;    // FI_TIMER=0：s_memrealtime tick（meta.timer_hz 换算 ns）；=1：cycle（OPEN/CLOSE 仍是 s_memrealtime）
    uint32_t tag;  // 事件号 | 标志位
    uint32_t aux;
};
struct FiBuf {
    FiRec* rec;          // [n_slots][cap]，launch 前清零
    uint32_t* overflow;  // bit0=某泳道写满；bit1=slot 越界
    uint64_t timer_hz;   // s_memrealtime 的固定频率（host 填；device 只读，用于超时换算）
    uint32_t cap;
    uint32_t warps_per_cta;
    uint32_t n_slots;
    char arch[32];       // 运行设备的 arch 名（fi_alloc 从 hipGetDeviceProperties.gcnArchName 取， ':' 前段）
};
struct FiCtx {
    FiRec* p;    // 本泳道基址；非记录线程为 nullptr
    uint32_t n;  // 已写条数（寄存器）
    uint32_t cap;
};

extern "C" __device__ FiBuf fi_lite_global;

__device__ __forceinline__ uint64_t fi_gtimer() {
    return __builtin_amdgcn_s_memrealtime();   // s_memrealtime：固定频率墙钟，全卡同一时基
}
__device__ __forceinline__ uint64_t fi_clk() {
    return __builtin_readcyclecounter();       // CU 本地周期计数器
}
__device__ __forceinline__ uint64_t fi_now() {
#if FI_TIMER == 1
    return fi_clk();
#else
    return fi_gtimer();
#endif
}
__device__ __forceinline__ uint32_t fi_lane() {
    return threadIdx.x & 63u;                  // wave64 内 lane 号
}
__device__ __forceinline__ uint32_t fi_hw_aux() {
    // gfx942：HW_ID 寄存器号 4，CU[11:8](4b)、SIMD[5:4](2b)、SE[14:13](2b)；XCC_ID=20，XCC[3:0](4b)。
    // __builtin_amdgcn_s_getreg((size<<11)|(offset<<6)|hwreg)
    const uint32_t cu = __builtin_amdgcn_s_getreg((3u << 11) | (8u << 6) | 4u);
    const uint32_t simd = __builtin_amdgcn_s_getreg((1u << 11) | (4u << 6) | 4u);
    const uint32_t se = __builtin_amdgcn_s_getreg((1u << 11) | (13u << 6) | 4u);
    const uint32_t xcc = __builtin_amdgcn_s_getreg((3u << 11) | (0u << 6) | 20u);
    return (cu & 15u) | ((simd & 3u) << 4) | ((se & 3u) << 8) | ((xcc & 15u) << 12);
}
__device__ __forceinline__ void fi_store(FiCtx& c, uint64_t t, uint32_t tag, uint32_t aux) {
    if (c.p == nullptr) return;
    if (c.n >= c.cap) {
        if (c.n == c.cap) atomicOr(fi_lite_global.overflow, 1u);
        c.n = c.cap + 1u;  // 只报一次
        return;
    }
    FiRec r;
    r.t = t;
    r.tag = tag;
    r.aux = aux;
    c.p[c.n] = r;          // 16B 一次写齐（编译为 global_store_dwordx4）
    c.n++;
}
__device__ __forceinline__ FiCtx fi_open(bool recorder, uint32_t role) {
    FiCtx c;
    c.p = nullptr;
    c.n = 0;
    c.cap = 0;
    const FiBuf b = fi_lite_global;
    if (!recorder || b.rec == nullptr) return c;
    // OV1 is specialized to a one-dimensional 80 x 512 launch.  Avoid
    // gridDim/blockDim here: referencing them expands the AMD kernarg with
    // hidden block/group fields and breaks the frozen 800-byte ABI.
    const uint32_t cta = blockIdx.x;
    const uint32_t linear = threadIdx.x;
#if FI_GRAN == 1
    const uint32_t w = 0;  // CU 粒度：每 CTA 一条泳道，host 侧 warps_per_cta 传 1
#else
    const uint32_t w = linear >> 6;
#endif
    const uint32_t s = cta * b.warps_per_cta + w;
    if (w >= b.warps_per_cta || s >= b.n_slots) {
        atomicOr(b.overflow, 2u);
        return c;
    }
    c.p = b.rec + (size_t)s * b.cap;
    c.cap = b.cap;
    const uint64_t g = fi_gtimer();
#if FI_TIMER == 1
    const uint64_t k = fi_clk();
#endif
    fi_store(c, g, FI_TAG_OPEN | (role & 0xffffu), fi_hw_aux());
#if FI_TIMER == 1
    fi_store(c, k, FI_TAG_CLK, 0u);
#endif
    return c;
}
__device__ __forceinline__ void fi_close(FiCtx& c) {
    if (c.p == nullptr) return;
    const uint64_t g = fi_gtimer();
#if FI_TIMER == 1
    const uint64_t k = fi_clk();
#endif
    fi_store(c, g, FI_TAG_CLOSE, 0u);
#if FI_TIMER == 1
    fi_store(c, k, FI_TAG_CLK, 1u);
#endif
}

#define FI_GROUP_ON(g) (((FI_GROUPS) >> ((g) & 31)) & 1u)

// FI_GRAN=0：每个 wave64 的 lane 0 记录；FI_GRAN=1：每 CTA 只由线性线程号 0（一维块即 threadIdx.x==0）记录。
#if FI_LEVEL >= 1
#define FI_CTX_PARAM , FiCtx& fi_ctx_
#define FI_CTX_ARG , fi_ctx_
#if FI_GRAN == 1
#define FI_OPEN(role) FiCtx fi_ctx_ = fi_open((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) == 0u, (uint32_t)(role))
#else
#define FI_OPEN(role) FiCtx fi_ctx_ = fi_open(fi_lane() == 0u, (uint32_t)(role))
#endif
#define FI_OPEN_BY(pred, role) FiCtx fi_ctx_ = fi_open((pred), (uint32_t)(role))
#define FI_CLOSE() fi_close(fi_ctx_)
#else
#define FI_CTX_PARAM
#define FI_CTX_ARG
#define FI_OPEN(role) do { } while (0)
#define FI_OPEN_BY(pred, role) do { } while (0)
#define FI_CLOSE() do { } while (0)
#endif

#if FI_LEVEL >= 2
#define FI_BG(g, id, aux) do { if (FI_GROUP_ON(g)) fi_store(fi_ctx_, fi_now(), (uint32_t)(id) & FI_TAG_IDMASK, (uint32_t)(aux)); } while (0)
#define FI_EG(g, id, aux) do { if (FI_GROUP_ON(g)) fi_store(fi_ctx_, fi_now(), ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_END, (uint32_t)(aux)); } while (0)
#define FI_MG(g, id, aux) do { if (FI_GROUP_ON(g)) fi_store(fi_ctx_, fi_now(), ((uint32_t)(id) & FI_TAG_IDMASK) | FI_TAG_MARK, (uint32_t)(aux)); } while (0)
#else
#define FI_BG(g, id, aux) do { } while (0)
#define FI_EG(g, id, aux) do { } while (0)
#define FI_MG(g, id, aux) do { } while (0)
#endif
#define FI_B(id, aux) FI_BG(0, id, aux)
#define FI_E(id, aux) FI_EG(0, id, aux)
#define FI_M(id, aux) FI_MG(0, id, aux)

// ---------------- host 侧 ----------------
// n_ctas：grid 的 CTA 总数；warps_per_cta：FI_GRAN=0 时按 wave64 上取整（threads/64），FI_GRAN=1 时传 1；
// cap：每泳道最大记录数（OPEN/CLOSE 及锚点也占条数）。
static inline hipError_t fi_alloc(FiBuf* b, uint32_t n_ctas, uint32_t warps_per_cta, uint32_t cap) {
    memset(b, 0, sizeof(*b));
    b->cap = cap;
    b->warps_per_cta = warps_per_cta;
    b->n_slots = n_ctas * warps_per_cta;
    int wall_khz = 0;
    hipError_t e = hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0);
    if (e != hipSuccess) return e;
    b->timer_hz = (uint64_t)wall_khz * 1000ull;   // MI308X 实测报 100 MHz
    {   // arch 名在主机侧取不到编译期 __gfx* 宏（主机编译趟不定义），改为运行时查询
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
            snprintf(b->arch, sizeof b->arch, "%s", prop.gcnArchName);
            char* c = strchr(b->arch, ':');
            if (c) *c = '\0';
        }
        if (!b->arch[0]) snprintf(b->arch, sizeof b->arch, "%s", FI_ARCH_STR);
    }
    if (const char* ov = getenv("FI_TIMER_HZ")) b->timer_hz = strtoull(ov, nullptr, 10);
    if (b->timer_hz == 0) return hipErrorInvalidValue;
    e = hipMalloc(&b->rec, (size_t)b->n_slots * cap * sizeof(FiRec));
    if (e == hipSuccess) e = hipMalloc(&b->overflow, sizeof(uint32_t));
    if (e == hipSuccess) e = hipMemset(b->rec, 0, (size_t)b->n_slots * cap * sizeof(FiRec));
    if (e == hipSuccess) e = hipMemset(b->overflow, 0, sizeof(uint32_t));
    return e;
}
static inline hipError_t fi_attach(const FiBuf* b) {
    FiBuf z;
    memset(&z, 0, sizeof(z));
    return hipMemcpyToSymbol(HIP_SYMBOL(fi_lite_global), b ? b : &z, sizeof(FiBuf));
}
// 要采的那次 launch 前调用（异步，排在同一 stream 上）。
static inline hipError_t fi_reset(const FiBuf* b, hipStream_t st = 0) {
    hipError_t e = hipMemsetAsync(b->rec, 0, (size_t)b->n_slots * b->cap * sizeof(FiRec), st);
    if (e == hipSuccess) e = hipMemsetAsync(b->overflow, 0, sizeof(uint32_t), st);
    return e;
}
// 转储：magic "FILITE02" | u32 n_slots, cap, warps_per_cta, overflow, timer, meta_len | meta(JSON) | u32 cnt[n_slots] | 紧凑记录
// meta 在调用方 JSON 上注入三个字段：timer_hz（s_memrealtime 频率）、gran（FI_GRAN）、arch（编译目标架构）。
static inline int fi_dump(const FiBuf* b, const char* path, const char* meta_json) {
    if (hipDeviceSynchronize() != hipSuccess) return -1;
    std::vector<FiRec> all((size_t)b->n_slots * b->cap);
    uint32_t ovf = 0;
    if (hipMemcpy(all.data(), b->rec, all.size() * sizeof(FiRec), hipMemcpyDeviceToHost) != hipSuccess) return -2;
    if (hipMemcpy(&ovf, b->overflow, 4, hipMemcpyDeviceToHost) != hipSuccess) return -2;
    std::vector<uint32_t> cnt(b->n_slots, 0);
    for (uint32_t s = 0; s < b->n_slots; ++s)
        while (cnt[s] < b->cap && all[(size_t)s * b->cap + cnt[s]].tag != 0u) ++cnt[s];
    FILE* f = fopen(path, "wb");
    if (!f) {
        // 同 CUDA 侧：把 errno 原文打出来，别只返回 -3（fi_bugs_h20-sk-qwen B10）。
        fprintf(stderr,
                "fi_dump: fopen(\"%s\", \"wb\") 失败：%s (errno=%d)。"
                "rocprofv3/nsys 下被测进程可能写不了工作目录：把转储路径换成 /tmp 下的文件，"
                "采集结束后再由 shell 拷回；或先把目标目录 chmod 0777。\n",
                path, strerror(errno), errno);
        return -3;
    }
    std::string meta = (meta_json && meta_json[0] == '{') ? meta_json : "{}";
    char inject[160];
    snprintf(inject, sizeof inject, "\"timer_hz\":%llu,\"gran\":%d,\"arch\":\"%s\",",
             (unsigned long long)b->timer_hz, (int)FI_GRAN, b->arch[0] ? b->arch : FI_ARCH_STR);
    meta.insert(1, inject);
    const uint32_t ml = (uint32_t)meta.size();
    const uint32_t hdr[6] = {b->n_slots, b->cap, b->warps_per_cta, ovf, (uint32_t)FI_TIMER, ml};
    fwrite("FILITE02", 1, 8, f);
    fwrite(hdr, 4, 6, f);
    fwrite(meta.data(), 1, ml, f);
    fwrite(cnt.data(), 4, cnt.size(), f);
    for (uint32_t s = 0; s < b->n_slots; ++s)
        if (cnt[s]) fwrite(&all[(size_t)s * b->cap], sizeof(FiRec), cnt[s], f);
    fclose(f);
    return (int)ovf;
}
static inline void fi_free(FiBuf* b) {
    // hipFree 带 nodiscard：直接丢掉返回值会让 hipcc 报 -Wunused-value（-Wno-unused-result 压不住），
    // 而 fi_logcheck 把编译告警当作日志不干净。这里显式接住并丢弃。
    hipError_t e_rec_ = hipFree(b->rec);
    hipError_t e_ovf_ = hipFree(b->overflow);
    (void)e_rec_;
    (void)e_ovf_;
    memset(b, 0, sizeof(*b));
}
