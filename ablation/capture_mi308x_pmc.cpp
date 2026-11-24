// Real intra-kernel MI308X (gfx942) hardware PMC sampling + Inter-Node RDMA NIC
// hardware counters (WeaveProfiler V2).
//
// Loaded by rocprofiler-sdk through ROCP_TOOL_LIBRARIES, so the tool is attached
// before HSA initialisation inside the real 8-rank EP8 / 16-rank EP16 benchmark
// process.  A dedicated host thread calls
// rocprofiler_sample_device_counting_service() in a tight loop while the
// persistent megakernel runs; every sample is a real PMU read stamped with both
// CLOCK_MONOTONIC and the HSA system clock (the same 100 MHz domain the kernel's
// s_memrealtime FI anchors use), so software and hardware share one measured
// absolute time base.
//
// V2 adds the SECOND communication fabric of a 2-node EP16 run: the Inter-Node
// RDMA NIC.  In the SAME sampling loop, pre-opened sysfs counter fds are read
// with pread() (~5 us total, no open/close per sample, no added cadence jitter):
//   /sys/class/infiniband/<dev>/ports/<p>/hw_counters/{rx,tx}_bytes   (preferred)
//   /sys/class/infiniband/<dev>/ports/<p>/counters/port_{xmit,rcv}_data
//       (IB spec: octets / 4, so raw deltas are scaled x4 to bytes)
//   /sys/class/net/<rdma-netdev>/statistics/{rx,tx}_bytes             (RoCE fallback,
//       only for netdevs whose device/infiniband association exists)
// Only ACTIVE ports are used; the exact counter paths, link_layer and link rate
// (Gb/s from ports/<p>/rate, or Mbps from netdev speed) are recorded in the
// <base>.json sidecar, and the raw cumulative reads are appended to every
// <base>.bin record after the PMC doubles (stride = n_pmc + n_nic_raw).  Deltas
// and the %-of-link-rate curve are computed offline by the exporter so the
// per-interval arithmetic stays auditable.  When no RDMA port is visible (e.g.
// a single-node EP8 run or a container without sysfs passthrough) the NIC
// section is empty and the record format is byte-identical to V1.
//
// Env (unchanged from V1; NIC discovery is automatic, no new knob):
//   PMC_GPU_INDEX   which GPU agent to profile (default 0)
//   PMC_COUNTERS    comma-separated real counter names
//   PMC_PERIOD_US   target sampling cadence in us (default 20)
//   PMC_OUT         output base path; writes <base>.bin and <base>.json
//   PMC_MAX_SAMPLES ring capacity (default 200000)
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/fwd.h>
#include <hsa/hsa.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

namespace {

struct Sample
{
    uint64_t mono_ns;
    uint64_t hsa_ns;
    uint64_t seq;
    uint64_t nrec;
    // followed by nrec doubles (aggregated per counter, in PMC_COUNTERS order)
};

std::vector<rocprofiler_agent_id_t> g_gpus;
std::vector<std::string>            g_want;
std::vector<rocprofiler_counter_id_t> g_ids;
std::vector<std::string>            g_got;
rocprofiler_context_id_t            g_ctx{};
rocprofiler_counter_config_id_t     g_cfg{};
std::atomic<bool>                   g_run{false};
std::atomic<bool>                   g_configured{false};
std::atomic<bool>                   g_sampling{false};
std::atomic<uint64_t>               g_seq{0};
std::atomic<uint64_t>               g_err{0};
std::vector<double>                 g_acc;
std::vector<uint8_t>                g_seen;
std::vector<std::string>            g_unmatched;
size_t                              g_last_nrec = 0;
std::thread                         g_thr;
std::string                         g_out;
size_t                              g_period_us = 20;
size_t                              g_nc = 0;

// mark log: (tag, mono_ns, hsa_ns)
struct Mark { uint64_t tag; uint64_t mono_ns; uint64_t hsa_ns; };
std::vector<Mark> g_marks;

// ---------------------------------------------------------------- NIC (RDMA) side
// One pre-opened sysfs counter fd per raw field; raw cumulative reads are
// appended to every PMC record so deltas are computed offline (auditable).
struct NicCounter
{
    std::string path;    // exact sysfs file read every sample
    std::string kind;    // hw_tx_bytes / hw_rx_bytes / port_xmit_data / ...
    double      scale;   // raw delta -> bytes (4.0 for IB port_*_data, else 1.0)
    int         fd{-1};
    double      last{0.0};
};

struct NicPort
{
    std::string device;      // mlx5_0 ... or the RoCE netdev name
    int         port{0};     // IB port number (0 for a netdev)
    std::string link_layer;  // InfiniBand / Ethernet
    std::string state;
    std::string source;      // infiniband_hw_counters / infiniband_port_counters / netdev_statistics
    std::string rate_raw;
    std::string rate_path;
    double      rate_gbps{0.0};
    std::vector<NicCounter> counters;
};

std::vector<NicPort>  g_nic_ports;
size_t                g_nic_raw = 0;   // raw NIC fields appended per sample
std::atomic<uint64_t> g_nic_err{0};

std::string read_sysfs(const std::string& path)
{
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if(fd < 0) return {};
    char buf[256];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if(n <= 0) return {};
    buf[n] = 0;
    std::string s(buf);
    while(!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    return s;
}

bool sysfs_exists(const std::string& path)
{
    return access(path.c_str(), F_OK) == 0;
}

std::vector<std::string> list_dir(const std::string& path)
{
    std::vector<std::string> out;
    DIR* d = opendir(path.c_str());
    if(!d) return out;
    while(dirent* e = readdir(d))
    {
        std::string n = e->d_name;
        if(n != "." && n != "..") out.push_back(n);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

std::string json_escape(const std::string& s)
{
    std::string o;
    for(char c : s)
    {
        if(c == '"' || c == '\\') { o += '\\'; o += c; }
        else if(c == '\n') o += "\\n";
        else if(c == '\r') o += "\\r";
        else if(c == '\t') o += "\\t";
        else o += c;
    }
    return o;
}

void nic_close_fds()
{
    for(auto& p : g_nic_ports)
        for(auto& c : p.counters)
            if(c.fd >= 0) { close(c.fd); c.fd = -1; }
    g_nic_ports.clear();
    g_nic_raw = 0;
}

// Auto-discover ACTIVE RDMA ports and pre-open their hardware counter files.
// Preference: IB hw_counters byte counters, then IB port data counters (x4),
// then RoCE netdev statistics (only netdevs with a device/infiniband link).
// class_root defaults to /sys/class; the test hook may point it at a synthetic
// tree so the discovery / pread logic is CPU-unit-testable without real NICs.
void discover_nic_ports(const std::string& class_root = "/sys/class")
{
    nic_close_fds();
    const std::string ib_root = class_root + "/infiniband";
    for(auto& dev : list_dir(ib_root))
    {
        std::string ports_dir = ib_root + "/" + dev + "/ports";
        for(auto& pn : list_dir(ports_dir))
        {
            std::string pdir = ports_dir + "/" + pn;
            std::string state = read_sysfs(pdir + "/state");
            if(state.find("ACTIVE") == std::string::npos) continue;
            NicPort port;
            port.device = dev;
            port.port = atoi(pn.c_str());
            port.state = state;
            port.link_layer = read_sysfs(pdir + "/link_layer");
            port.rate_path = pdir + "/rate";
            port.rate_raw = read_sysfs(port.rate_path);
            port.rate_gbps = strtod(port.rate_raw.c_str(), nullptr);  // "200 Gb/sec (4X EDR)" -> 200
            std::string hw_tx = pdir + "/hw_counters/tx_bytes", hw_rx = pdir + "/hw_counters/rx_bytes";
            std::string pd_tx = pdir + "/counters/port_xmit_data", pd_rx = pdir + "/counters/port_rcv_data";
            if(sysfs_exists(hw_tx) && sysfs_exists(hw_rx))
            {
                port.source = "infiniband_hw_counters";
                port.counters.push_back(NicCounter{hw_tx, "hw_tx_bytes", 1.0, -1, 0.0});
                port.counters.push_back(NicCounter{hw_rx, "hw_rx_bytes", 1.0, -1, 0.0});
            }
            else if(sysfs_exists(pd_tx) && sysfs_exists(pd_rx))
            {
                // IB spec: port_xmit_data / port_rcv_data count octets divided
                // by 4 (lane-0 octets), so a raw delta is x4 to get bytes.
                port.source = "infiniband_port_counters";
                port.counters.push_back(NicCounter{pd_tx, "port_xmit_data", 4.0, -1, 0.0});
                port.counters.push_back(NicCounter{pd_rx, "port_rcv_data", 4.0, -1, 0.0});
            }
            else continue;
            bool ok = true;
            for(auto& c : port.counters)
            {
                c.fd = open(c.path.c_str(), O_RDONLY | O_CLOEXEC);
                if(c.fd < 0) ok = false;
            }
            if(!ok)
            {
                for(auto& c : port.counters) if(c.fd >= 0) { close(c.fd); c.fd = -1; }
                continue;
            }
            g_nic_ports.push_back(std::move(port));
        }
    }
    if(g_nic_ports.empty())
    {
        // RoCE fallback: a netdev whose PCI device also exposes an infiniband
        // class entry is the RDMA netdev; plain TCP netdevs are NOT used.
        const std::string net_root = class_root + "/net";
        for(auto& n : list_dir(net_root))
        {
            std::string ndir = net_root + "/" + n;
            if(!sysfs_exists(ndir + "/device/infiniband")) continue;
            std::string tx = ndir + "/statistics/tx_bytes", rx = ndir + "/statistics/rx_bytes";
            if(!sysfs_exists(tx) || !sysfs_exists(rx)) continue;
            NicPort port;
            port.device = n;
            port.port = 0;
            port.link_layer = "Ethernet";
            port.state = read_sysfs(ndir + "/operstate");
            port.source = "netdev_statistics";
            port.rate_path = ndir + "/speed";
            port.rate_raw = read_sysfs(port.rate_path);
            double mbps = strtod(port.rate_raw.c_str(), nullptr);
            port.rate_gbps = mbps > 0 ? mbps / 1000.0 : 0.0;
            port.counters.push_back(NicCounter{tx, "netdev_tx_bytes", 1.0, -1, 0.0});
            port.counters.push_back(NicCounter{rx, "netdev_rx_bytes", 1.0, -1, 0.0});
            bool ok = true;
            for(auto& c : port.counters)
            {
                c.fd = open(c.path.c_str(), O_RDONLY | O_CLOEXEC);
                if(c.fd < 0) ok = false;
            }
            if(!ok)
            {
                for(auto& c : port.counters) if(c.fd >= 0) { close(c.fd); c.fd = -1; }
                continue;
            }
            g_nic_ports.push_back(std::move(port));
        }
    }
    for(auto& p : g_nic_ports) g_nic_raw += p.counters.size();
    fprintf(stderr, "[pmc] nic: %zu active RDMA port(s), %zu raw counter field(s)%s\n",
            g_nic_ports.size(), g_nic_raw, g_nic_ports.empty() ? " (no RDMA sysfs visible)" : "");
    for(auto& p : g_nic_ports)
        fprintf(stderr, "[pmc] nic: %s port %d %s rate=%.3f Gb/s (%s) source=%s\n",
                p.device.c_str(), p.port, p.link_layer.c_str(), p.rate_gbps,
                p.rate_raw.c_str(), p.source.c_str());
}

// One pread() per pre-opened counter fd (~5 us total).  A failed read keeps the
// previous raw value (that interval's delta is 0) and is counted in nic_errors.
bool read_nic_raw(std::vector<double>& out)
{
    bool all_ok = true;
    size_t k = 0;
    char buf[64];
    for(auto& p : g_nic_ports)
        for(auto& c : p.counters)
        {
            double v = c.last;
            ssize_t n = pread(c.fd, buf, sizeof(buf) - 1, 0);
            if(n > 0)
            {
                buf[n] = 0;
                v = double(strtoull(buf, nullptr, 10));
            }
            else
            {
                all_ok = false;
                g_nic_err.fetch_add(1);
            }
            c.last = v;
            out[k++] = v;
        }
    return all_ok;
}

std::string nic_json()
{
    std::string s = "\"nic_raw_per_sample\": " + std::to_string(g_nic_raw)
                  + ",\n  \"nic_errors\": " + std::to_string(g_nic_err.load())
                  + ",\n  \"nic_ports\": [";
    for(size_t i = 0; i < g_nic_ports.size(); ++i)
    {
        const auto& p = g_nic_ports[i];
        char rate[64];
        snprintf(rate, sizeof(rate), "%.6f", p.rate_gbps);
        s += i ? ", " : "";
        s += "{\"device\": \"" + json_escape(p.device) + "\", \"port\": " + std::to_string(p.port)
           + ", \"link_layer\": \"" + json_escape(p.link_layer) + "\", \"state\": \"" + json_escape(p.state)
           + "\", \"source\": \"" + json_escape(p.source) + "\", \"rate_gbps\": " + rate
           + ", \"rate_raw\": \"" + json_escape(p.rate_raw) + "\", \"rate_path\": \"" + json_escape(p.rate_path)
           + "\", \"counters\": [";
        for(size_t j = 0; j < p.counters.size(); ++j)
        {
            const auto& c = p.counters[j];
            char sc[32];
            snprintf(sc, sizeof(sc), "%.1f", c.scale);
            s += j ? ", " : "";
            s += "{\"path\": \"" + json_escape(c.path) + "\", \"kind\": \"" + json_escape(c.kind)
               + "\", \"scale\": " + sc + "}";
        }
        s += "]}";
    }
    s += "]";
    return s;
}

uint64_t mono_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

uint64_t hsa_ns()
{
    uint64_t t = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &t);
    return t;
}

rocprofiler_status_t agent_cb(rocprofiler_agent_version_t, const void** agents, size_t n, void*)
{
    for(size_t i = 0; i < n; ++i)
    {
        auto* a = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
        if(a->type == ROCPROFILER_AGENT_TYPE_GPU) g_gpus.push_back(a->id);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t counter_cb(rocprofiler_agent_id_t, rocprofiler_counter_id_t* ids, size_t n, void*)
{
    for(size_t i = 0; i < n; ++i)
    {
        rocprofiler_counter_info_v0_t info{};
        if(rocprofiler_query_counter_info(ids[i], ROCPROFILER_COUNTER_INFO_VERSION_0, &info) !=
           ROCPROFILER_STATUS_SUCCESS)
            continue;
        if(!info.name) continue;
        for(auto& w : g_want)
            if(w == info.name)
            {
                bool dup = false;
                for(auto& got : g_got) if(got == w) dup = true;
                if(!dup) { g_ids.push_back(ids[i]); g_got.push_back(w); }
            }
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

void set_config_cb(rocprofiler_context_id_t ctx, rocprofiler_agent_id_t,
                   rocprofiler_device_counting_agent_cb_t set_config, void*)
{
    if(set_config(ctx, g_cfg) != ROCPROFILER_STATUS_SUCCESS) g_err.fetch_add(1);
}

// One synchronous PMU read.  Records are aggregated per counter id (a counter
// such as TCC_EA0_RDREQ_DRAM has one instance per EA channel / XCC).
bool sample_once(std::vector<double>& vals)
{
    static std::vector<rocprofiler_counter_record_t> recs(4096);
    size_t n = recs.size();
    rocprofiler_user_data_t ud{};
    ud.value = g_seq.load();
    auto s = rocprofiler_sample_device_counting_service(g_ctx, ud, rocprofiler_counter_flag_t(0),
                                                        recs.data(), &n);
    if(s != ROCPROFILER_STATUS_SUCCESS)
    {
        uint64_t e = g_err.fetch_add(1);
        if(e < 3 || e % 8192 == 0)
            fprintf(stderr, "[pmc] sample st=%d n_in=%zu (err#%llu)\n", (int) s, n,
                    (unsigned long long) e);
        return false;
    }
    std::fill(vals.begin(), vals.end(), 0.0);
    // Match by counter NAME, not by id handle: the gfx942 MFMA counters are
    // "emulated", and the id carried by a record can be the underlying real
    // counter rather than the requested emulated one, so handle equality alone
    // silently drops them to zero.
    static std::unordered_map<uint64_t, int> cache;
    g_last_nrec = n;
    for(size_t i = 0; i < n; ++i)
    {
        rocprofiler_counter_id_t cid{};
        if(rocprofiler_query_record_counter_id(recs[i].id, &cid) != ROCPROFILER_STATUS_SUCCESS)
            continue;
        auto it = cache.find(cid.handle);
        int k;
        if(it != cache.end()) k = it->second;
        else
        {
            k = -1;
            for(size_t j = 0; j < g_ids.size(); ++j)
                if(cid.handle == g_ids[j].handle) { k = int(j); break; }
            if(k >= 0) { cache.emplace(cid.handle, k); vals[size_t(k)] += recs[i].counter_value; continue; }
            rocprofiler_counter_info_v0_t info{};
            if(rocprofiler_query_counter_info(cid, ROCPROFILER_COUNTER_INFO_VERSION_0, &info) ==
                   ROCPROFILER_STATUS_SUCCESS && info.name)
            {
                for(size_t j = 0; j < g_got.size(); ++j)
                    if(g_got[j] == info.name) { k = int(j); break; }
            }
            cache.emplace(cid.handle, k);
            if(k < 0 && g_unmatched.size() < 24)
            {
                bool seen = false;
                for(auto& u : g_unmatched) if(u == (info.name ? info.name : "?")) seen = true;
                if(!seen) g_unmatched.emplace_back(info.name ? info.name : "?");
            }
        }
        if(k >= 0) vals[size_t(k)] += recs[i].counter_value;
    }
    return true;
}

// Written periodically as well as at exit: the benchmark process can tear down
// without rocprofiler calling tool_finalize, and the samples must survive.
std::mutex g_flush_mu;
std::atomic<bool> g_flush_req{false};

void flush_locked(const std::vector<char>& buf, uint64_t emitted, size_t stride)
{
    std::string bin = g_out + ".bin";
    if(FILE* f = fopen(bin.c_str(), "wb"))
    {
        uint64_t hdr[4] = {0x504d4353414d504cull, uint64_t(stride), emitted, uint64_t(sizeof(Sample))};
        fwrite(hdr, sizeof(hdr), 1, f);
        fwrite(buf.data(), 1, buf.size(), f);
        fclose(f);
    }
    std::string js = g_out + ".json";
    if(FILE* f = fopen(js.c_str(), "w"))
    {
        fprintf(f, "{\n  \"magic\": \"PMCSAMPL\",\n  \"gpu_agent_index\": %d,\n",
                getenv("PMC_GPU_INDEX") ? atoi(getenv("PMC_GPU_INDEX")) : 0);
        fprintf(f, "  \"period_us_target\": %zu,\n  \"samples\": %llu,\n  \"errors\": %llu,\n",
                g_period_us, (unsigned long long) emitted, (unsigned long long) g_err.load());
        fprintf(f, "  \"record_header_bytes\": %zu,\n  \"counters\": [", sizeof(Sample));
        for(size_t i = 0; i < g_got.size(); ++i)
            fprintf(f, "%s\"%s\"", i ? ", " : "", g_got[i].c_str());
        fprintf(f, "],\n  \"requested\": [");
        for(size_t i = 0; i < g_want.size(); ++i)
            fprintf(f, "%s\"%s\"", i ? ", " : "", g_want[i].c_str());
        fprintf(f, "],\n  \"marks\": [");
        for(size_t i = 0; i < g_marks.size(); ++i)
            fprintf(f, "%s{\"tag\": %llu, \"mono_ns\": %llu, \"hsa_ns\": %llu}", i ? ", " : "",
                    (unsigned long long) g_marks[i].tag, (unsigned long long) g_marks[i].mono_ns,
                    (unsigned long long) g_marks[i].hsa_ns);
        fprintf(f, "],\n  \"last_record_count\": %zu,\n  \"unmatched_record_counters\": [", g_last_nrec);
        for(size_t i = 0; i < g_unmatched.size(); ++i)
            fprintf(f, "%s\"%s\"", i ? ", " : "", g_unmatched[i].c_str());
        fprintf(f, "],\n  %s,\n  \"sampler\": \"rocprofiler-sdk device_counting_service "
                   "(rocprofiler_sample_device_counting_service, synchronous) + Linux RDMA NIC "
                   "sysfs hardware counters (pread on pre-opened fds in the same loop)\"\n}\n",
                nic_json().c_str());
        fclose(f);
    }
}

std::vector<char> g_buf;
uint64_t g_emitted = 0;
size_t g_stride = 0;

void flush_out(const std::vector<char>& buf, uint64_t emitted, size_t stride)
{
    std::lock_guard<std::mutex> lk(g_flush_mu);
    flush_locked(buf, emitted, stride);
}

void flush_atexit()
{
    if(g_stride) flush_out(g_buf, g_emitted, g_stride);
}

void sampler_loop()
{
    // Record layout: Sample header, g_nc PMC doubles, g_nic_raw NIC raw doubles.
    const size_t stride = g_nc + g_nic_raw;
    g_stride = stride;
    std::vector<double> vals(g_nc, 0.0);
    std::vector<double> nic_vals(g_nic_raw, 0.0);
    std::vector<double> prev(g_nc, 0.0);
    bool have_prev = false;
    // rocprofiler-register invokes tool_initialize before HSA is loaded, and
    // device counting cannot start until it is (start_context returns
    // ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED).  Wait for the runtime here.
    bool started = false;
    for(int i = 0; i < 60000 && g_run.load(std::memory_order_relaxed); ++i)
    {
        rocprofiler_status_t st = rocprofiler_start_context(g_ctx);
        if(st == ROCPROFILER_STATUS_SUCCESS) { started = true; break; }
        if(st != ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED)
        {
            fprintf(stderr, "[pmc] start_context failed st=%d\n", (int) st);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    fprintf(stderr, "[pmc] context started=%d, sampling %zu PMC counters + %zu NIC raw fields\n",
            (int) started, g_nc, g_nic_raw);
    // The DCS set_config callback runs asynchronously after start_context.  A
    // short settle is enough here (the host attaches one launch before the
    // measured one); a long sleep would desynchronise the 8 EP8 ranks.
    if(started) std::this_thread::sleep_for(std::chrono::milliseconds(3));
    // Raw cumulative reads are logged; deltas are computed offline so the exact
    // per-interval arithmetic is auditable.
    g_buf.reserve(size_t(1) << 24);
    auto next = std::chrono::steady_clock::now();
    uint64_t iters = 0;
    while(started && g_run.load(std::memory_order_relaxed))
    {
        ++iters;
        if(iters % 8192 == 0)
            fprintf(stderr, "[pmc] iters=%llu emitted=%llu err=%llu\n",
                    (unsigned long long) iters, (unsigned long long) g_emitted,
                    (unsigned long long) g_err.load());
        uint64_t m = mono_ns();
        uint64_t h = hsa_ns();
        bool ok = sample_once(vals);
        {
            // Append and flush under one lock so an external pmc_flush() can
            // never observe a half-appended record or truncate a live buffer.
            std::lock_guard<std::mutex> lk(g_flush_mu);
            if(ok)
            {
                // Same-loop NIC hardware reads on pre-opened fds (pread, no
                // open/close): the raw cumulative values ride along with the
                // PMC sample under the same two clock stamps.
                if(g_nic_raw) read_nic_raw(nic_vals);
                Sample s{m, h, g_seq.fetch_add(1), uint64_t(stride)};
                const char* p = reinterpret_cast<const char*>(&s);
                g_buf.insert(g_buf.end(), p, p + sizeof(Sample));
                p = reinterpret_cast<const char*>(vals.data());
                g_buf.insert(g_buf.end(), p, p + g_nc * sizeof(double));
                if(g_nic_raw)
                {
                    p = reinterpret_cast<const char*>(nic_vals.data());
                    g_buf.insert(g_buf.end(), p, p + g_nic_raw * sizeof(double));
                }
                ++g_emitted;
                have_prev = true;
                prev = vals;
            }
            if(iters % 512 == 0 || g_flush_req.exchange(false))
                flush_locked(g_buf, g_emitted, stride);
        }
        (void) have_prev;
        next += std::chrono::microseconds(g_period_us);
        auto now = std::chrono::steady_clock::now();
        if(next > now) std::this_thread::sleep_for(next - now);
        else next = now;
        if(g_emitted >= size_t(400000)) break;
    }
    flush_out(g_buf, g_emitted, stride);
}

// Attaching at rocprofiler-register time (ROCP_TOOL_LIBRARIES) leaves the SDK
// without a usable HSA intercept table inside a torch process: every
// rocprofiler_sample_device_counting_service then returns HSA_NOT_LOADED.
// PMC_EXPLICIT_START defers the attach to pmc_start(), which the benchmark
// calls after torch.cuda.init(), via rocprofiler_force_configure.
int do_setup()
{
    const char* cn = getenv("PMC_COUNTERS");
    const char* op = getenv("PMC_OUT");
    if(!cn || !op) { fprintf(stderr, "[pmc] PMC_COUNTERS / PMC_OUT unset; sampler idle\n"); return 0; }
    // One counting context per agent: only the designated torchrun rank attaches.
    const char* lr = getenv("LOCAL_RANK");
    if(const char* want = getenv("PMC_ENABLE_RANK"))
    {
        if(!lr || atoi(lr) != atoi(want))
        {
            fprintf(stderr, "[pmc] LOCAL_RANK=%s != PMC_ENABLE_RANK=%s; sampler idle\n",
                    lr ? lr : "?", want);
            return 0;
        }
    }
    g_out = op;
    if(const char* p = getenv("PMC_PERIOD_US")) g_period_us = size_t(atoll(p));
    int gi = getenv("PMC_GPU_INDEX") ? atoi(getenv("PMC_GPU_INDEX")) : (lr ? atoi(lr) : 0);
    {
        std::string s(cn);
        size_t a = 0;
        while(a <= s.size())
        {
            size_t b = s.find(',', a);
            if(b == std::string::npos) b = s.size();
            std::string t = s.substr(a, b - a);
            while(!t.empty() && (t.front() == ' ' || t.front() == '\t')) t.erase(t.begin());
            while(!t.empty() && (t.back() == ' ' || t.back() == '\t' || t.back() == '\r')) t.pop_back();
            if(!t.empty()) g_want.push_back(t);
            if(b == s.size()) break;
            a = b + 1;
        }
    }
    if(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, &agent_cb,
                                          sizeof(rocprofiler_agent_v0_t), nullptr) != ROCPROFILER_STATUS_SUCCESS
       || g_gpus.empty())
    { fprintf(stderr, "[pmc] no GPU agents\n"); return -1; }
    if(gi < 0 || size_t(gi) >= g_gpus.size()) gi = 0;
    auto agent = g_gpus[size_t(gi)];
    rocprofiler_iterate_agent_supported_counters(agent, &counter_cb, nullptr);
    g_nc = g_got.size();
    if(g_nc == 0) { fprintf(stderr, "[pmc] none of the requested counters exist\n"); return -1; }
    for(auto& w : g_want)
    {
        bool ok = false;
        for(auto& got : g_got) if(got == w) ok = true;
        if(!ok) fprintf(stderr, "[pmc] WARNING counter not available: %s\n", w.c_str());
    }
    rocprofiler_status_t st;
    if((st = rocprofiler_create_context(&g_ctx)) != ROCPROFILER_STATUS_SUCCESS)
    { fprintf(stderr, "[pmc] create_context failed st=%d\n", (int) st); return -1; }
    if((st = rocprofiler_create_counter_config(agent, g_ids.data(), g_ids.size(), &g_cfg)) !=
       ROCPROFILER_STATUS_SUCCESS)
    { fprintf(stderr, "[pmc] create_counter_config failed st=%d n=%zu\n", (int) st, g_ids.size()); return -1; }
    rocprofiler_buffer_id_t nobuf{};
    nobuf.handle = 0;
    if((st = rocprofiler_configure_device_counting_service(g_ctx, nobuf, agent, &set_config_cb, nullptr)) !=
       ROCPROFILER_STATUS_SUCCESS)
    { fprintf(stderr, "[pmc] configure_device_counting_service failed st=%d\n", (int) st); return -1; }
    fprintf(stderr, "[pmc] configured %zu counters on GPU agent %d, period %zu us -> %s\n",
            g_nc, gi, g_period_us, g_out.c_str());
    // Inter-Node RDMA NIC discovery is pure sysfs (no GPU / HSA dependency) and
    // pre-opens every counter fd once, so the sampling loop only does pread().
    discover_nic_ports();
    g_configured.store(true);
    return 0;
}

// Context creation / service configuration is only legal while rocprofiler is
// configuring clients (early, via ROCP_TOOL_LIBRARIES).  Starting the context
// and sampling is only legal once HSA is fully up (late, after torch.cuda.init).
// The two phases are therefore split.
void begin_sampling()
{
    if(!g_configured.load() || g_sampling.exchange(true)) return;
    g_run.store(true);
    atexit(&flush_atexit);
    g_thr = std::thread(&sampler_loop);
}

int tool_initialize(rocprofiler_client_finalize_t, void*)
{
    int rc = do_setup();
    if(rc == 0 && !getenv("PMC_EXPLICIT_START")) begin_sampling();
    return rc;
}

void tool_finalize(void*)
{
    if(g_run.exchange(false))
    {
        if(g_thr.joinable()) g_thr.join();
    }
    rocprofiler_stop_context(g_ctx);
}

}  // namespace

extern "C" {

// Host-side marks in the same two clock domains as the samples.
void pmc_mark(uint64_t tag)
{
    g_marks.push_back(Mark{tag, mono_ns(), hsa_ns()});
}

void pmc_now(uint64_t* m, uint64_t* h)
{
    if(m) *m = mono_ns();
    if(h) *h = hsa_ns();
}

uint64_t pmc_sample_count() { return g_seq.load(); }
uint64_t pmc_error_count() { return g_err.load(); }
// Stop sampling, join the thread, then flush.  Called by the host before it
// tears down so the record file is complete and no thread outlives the runtime.
void pmc_stop()
{
    if(g_run.exchange(false))
    {
        if(g_thr.joinable()) g_thr.join();
    }
    if(g_stride) flush_out(g_buf, g_emitted, g_stride);
}

// Request a flush from the sampler thread (it owns g_buf); if sampling has
// already stopped, flush directly.
void pmc_flush()
{
    if(!g_stride) return;
    if(g_run.load()) g_flush_req.store(true);
    else flush_out(g_buf, g_emitted, g_stride);
}

// ---- NIC-only test hooks (CPU unit tests; no GPU / rocprofiler attach needed) ----
// Re-run discovery on the real /sys/class; returns the number of ACTIVE RDMA ports.
int pmc_nic_discover()
{
    discover_nic_ports();
    return int(g_nic_ports.size());
}

// Discovery against an explicit class root (synthetic-tree CPU tests).
int pmc_nic_discover_root(const char* root)
{
    discover_nic_ports(root && *root ? std::string(root) : "/sys/class");
    return int(g_nic_ports.size());
}

size_t pmc_nic_raw_count() { return g_nic_raw; }

// One raw read of every pre-opened NIC counter into out[pmc_nic_raw_count()].
// 0 = all reads ok, -1 = no ports, -2 = at least one pread failed (value held).
int pmc_nic_read(double* out)
{
    if(!g_nic_raw || !out) return -1;
    std::vector<double> v(g_nic_raw, 0.0);
    bool ok = read_nic_raw(v);
    for(size_t i = 0; i < g_nic_raw; ++i) out[i] = v[i];
    return ok ? 0 : -2;
}

// Discovery provenance as a complete JSON object (the .json sidecar embeds the
// same nic_json() fragment inside its outer braces).  Returns the full length;
// copies at most cap-1 bytes + NUL into buf when buf != nullptr.
int pmc_nic_info_json(char* buf, size_t cap)
{
    std::string s = "{" + nic_json() + "}";
    if(buf && cap)
    {
        size_t n = std::min(cap - 1, s.size());
        memcpy(buf, s.data(), n);
        buf[n] = 0;
    }
    return int(s.size());
}

rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t, const char*, uint32_t,
                                                           rocprofiler_client_id_t*);

// Explicit attach after the host runtime is up (torch.cuda.init()).
int pmc_start()
{
    static std::atomic<bool> once{false};
    if(once.exchange(true)) return g_configured.load() ? 0 : -1;
    if(!g_configured.load())
    {
        rocprofiler_status_t st = rocprofiler_force_configure(&rocprofiler_configure);
        fprintf(stderr, "[pmc] force_configure st=%d\n", (int) st);
        if(!g_configured.load()) do_setup();
    }
    begin_sampling();
    fprintf(stderr, "[pmc] pmc_start configured=%d sampling=%d\n",
            (int) g_configured.load(), (int) g_sampling.load());
    return g_configured.load() ? 0 : -1;
}

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    if(id) id->name = "pmc_sampler";
    static rocprofiler_tool_configure_result_t res{};
    res.size       = sizeof(rocprofiler_tool_configure_result_t);
    res.initialize = &tool_initialize;
    res.finalize   = &tool_finalize;
    res.tool_data  = nullptr;
    return &res;
}

}  // extern "C"
