// Host model of the H20 V9 per-token streaming-Combine protocol implemented in
// deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh (kSmemStreamCombineActive).
//
// The device protocol cannot be exercised on this machine (no H20), so this test
// replays its exact integer logic on the host:
//   sender   -- every (dst_rank, token, top-k slot) row is scattered one BLOCK_N
//               hidden-column block at a time; each slice releases one arrival
//               count into arrival[token * kRowBlocks + col_block] on the peer.
//   receiver -- items are (token, column block); an item is ready iff its arrival
//               count >= popc(valid top-k mask); a per-token claim bitmask hands
//               each item to exactly one participant, which FP32-accumulates all
//               valid slots and writes the BF16 result slice of y.
//   schedule -- a bounded middle pass over the dedicated Reduce warps' slice of
//               the item space, then a whole-CTA tail pass that must finish only
//               the leftovers;
//   skew     -- the Reduce warps keep streaming after the local scatter
//               terminate until the before-combine NVLink barrier completes;
//   barriers -- a named-barrier state machine of every role's rendezvous order
//               (deadlock-free, plus the counter-example that must deadlock).
//
// Build/run: python tests/test_h20_v9_stream_combine.py
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const char* label) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("[FAIL] %s\n", label);
    }
}

uint32_t popc(uint32_t v) { return static_cast<uint32_t>(__builtin_popcount(v)); }
uint32_t ffs(uint32_t v) { return v ? static_cast<uint32_t>(__builtin_ctz(v)) : 0u; }

// Mirrors the device constants for the EP4 Qwen3.8-Max work point.
constexpr uint32_t kHidden = 8192;
constexpr uint32_t kBlockN = 256;
constexpr uint32_t kBlockM = 64;
constexpr uint32_t kRowBlocks = kHidden / kBlockN;      // 32 == kStreamSubChunks
constexpr uint32_t kNumRanks = 4;
constexpr uint32_t kNumExperts = 512;
constexpr uint32_t kELocal = kNumExperts / kNumRanks;
constexpr uint32_t kNumTopk = 10;
constexpr uint32_t kVecsPerLane = kBlockN / 256;        // 1
constexpr uint32_t kPairsPerVec = 4;

struct Workload {
    uint32_t num_tokens;
    std::vector<int32_t> topk_idx;   // [token * kNumTopk + slot], <0 == dropped
    std::vector<float> payload;      // [slot][token][hidden]

    int32_t route(uint32_t t, uint32_t s) const { return topk_idx[t * kNumTopk + s]; }
    uint32_t mask(uint32_t t) const {
        uint32_t m = 0;
        for (uint32_t s = 0; s < kNumTopk; ++s)
            if (route(t, s) >= 0) m |= 1u << s;
        return m;
    }
    float at(uint32_t slot, uint32_t t, uint32_t e) const {
        return payload[(static_cast<size_t>(slot) * num_tokens + t) * kHidden + e];
    }
};

Workload make_workload(uint32_t num_tokens, unsigned seed, bool drop_routes) {
    Workload w;
    w.num_tokens = num_tokens;
    w.topk_idx.resize(static_cast<size_t>(num_tokens) * kNumTopk);
    w.payload.resize(static_cast<size_t>(kNumTopk) * num_tokens * kHidden);
    std::mt19937 rng(seed);
    for (uint32_t t = 0; t < num_tokens; ++t)
        for (uint32_t s = 0; s < kNumTopk; ++s) {
            int32_t e = static_cast<int32_t>(rng() % kNumExperts);
            if (drop_routes && (rng() % 7) == 0) e = -1;
            w.topk_idx[t * kNumTopk + s] = e;
        }
    for (size_t i = 0; i < w.payload.size(); ++i)
        w.payload[i] = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 256.0f;
    return w;
}

// ---- sender model ---------------------------------------------------------
struct Sender {
    std::vector<uint32_t> arrival;   // [token * kRowBlocks + col_block]

    explicit Sender(uint32_t num_tokens)
        : arrival(static_cast<size_t>(num_tokens) * kRowBlocks, 0u) {}

    // One (dst_rank, token, slot) row, one BLOCK_N hidden-column block.
    void scatter_slice(uint32_t token, uint32_t col_block) {
        ++arrival[static_cast<size_t>(token) * kRowBlocks + col_block];
    }
};

// ---- receiver model -------------------------------------------------------
struct Receiver {
    const Workload& w;
    std::vector<uint32_t> claim;         // [token] bitmask of claimed column blocks
    std::vector<uint32_t> reduce_count;  // [token * kRowBlocks + chunk]
    std::vector<float> y;                // [token * kHidden + elem]
    uint32_t participants;

    Receiver(const Workload& wl, uint32_t num_participants)
        : w(wl), claim(wl.num_tokens, 0u),
          reduce_count(static_cast<size_t>(wl.num_tokens) * kRowBlocks, 0u),
          y(static_cast<size_t>(wl.num_tokens) * kHidden, 0.0f),
          participants(num_participants) {}

    // Mirrors stream_smem_reduce_items(): returns the number of newly claimed
    // items.  All 32 lanes of the warp are modelled explicitly.
    uint32_t sweep(uint32_t gid, const std::vector<uint32_t>& arrival, bool wait_ready) {
        uint32_t claimed = 0;
        const uint32_t total_items = w.num_tokens * kRowBlocks;
        for (uint32_t work_id = gid; work_id < total_items; work_id += participants) {
            const uint32_t token = work_id / kRowBlocks;
            const uint32_t chunk = work_id % kRowBlocks;
            const uint32_t bit = 1u << chunk;
            if (claim[token] & bit) continue;
            const uint32_t mask = w.mask(token);
            const uint32_t expected = popc(mask);
            const size_t ai = static_cast<size_t>(token) * kRowBlocks + chunk;
            if (arrival[ai] < expected && !wait_ready) continue;
            // atomicOr: exactly one participant observes the bit clear.
            const uint32_t old = claim[token];
            claim[token] = old | bit;
            if (old & bit) continue;
            ++claimed;
            ++reduce_count[ai];
            for (uint32_t lane = 0; lane < 32; ++lane) {
                float reduced[kVecsPerLane * kPairsPerVec][2] = {};
                uint32_t m = mask;
                while (m) {
                    const uint32_t slot = ffs(m);
                    m &= (m - 1u);
                    for (uint32_t j = 0; j < kVecsPerLane; ++j)
                        for (uint32_t l = 0; l < kPairsPerVec; ++l) {
                            const uint32_t e = chunk * kBlockN + (j * 32 + lane) * 8 + l * 2;
                            reduced[j * kPairsPerVec + l][0] += w.at(slot, token, e);
                            reduced[j * kPairsPerVec + l][1] += w.at(slot, token, e + 1);
                        }
                }
                for (uint32_t j = 0; j < kVecsPerLane; ++j)
                    for (uint32_t l = 0; l < kPairsPerVec; ++l) {
                        const uint32_t e = chunk * kBlockN + (j * 32 + lane) * 8 + l * 2;
                        y[static_cast<size_t>(token) * kHidden + e] = reduced[j * kPairsPerVec + l][0];
                        y[static_cast<size_t>(token) * kHidden + e + 1] = reduced[j * kPairsPerVec + l][1];
                    }
            }
        }
        return claimed;
    }
};

// One scatter event = one row slice.  Events are grouped into L2 tiles exactly
// like the device: pool rows are ordered by expert, BLOCK_M rows per m-block,
// and the N-major raster sweeps every column block of an m-block before moving
// to the next one.  A prefix of this schedule therefore completes whole tiles.
struct Slice { uint32_t token; uint32_t col_block; uint32_t tile; };

std::vector<Slice> scatter_schedule(const Workload& w) {
    // Pool rows in expert order, one per valid (token, slot).
    std::vector<std::pair<uint32_t, uint32_t>> rows;  // (expert, token)
    for (uint32_t t = 0; t < w.num_tokens; ++t)
        for (uint32_t s = 0; s < kNumTopk; ++s) {
            const int32_t e = w.route(t, s);
            if (e >= 0) rows.emplace_back(static_cast<uint32_t>(e), t);
        }
    std::sort(rows.begin(), rows.end());
    std::vector<Slice> events;
    const uint32_t num_m_blocks = static_cast<uint32_t>((rows.size() + kBlockM - 1) / kBlockM);
    for (uint32_t mb = 0; mb < num_m_blocks; ++mb)
        for (uint32_t nb = 0; nb < kRowBlocks; ++nb) {
            const uint32_t tile = mb * kRowBlocks + nb;
            for (uint32_t r = mb * kBlockM;
                 r < std::min<uint32_t>((mb + 1) * kBlockM, static_cast<uint32_t>(rows.size())); ++r)
                events.push_back({rows[r].second, nb, tile});
        }
    return events;
}

void run_case(uint32_t num_tokens, uint32_t disp_warps, uint32_t num_sms,
              uint32_t reduce_warps, unsigned seed, bool drop_routes,
              uint32_t mid_percent, const char* label) {
    const uint32_t warps_per_sm = disp_warps + 4 + 16;
    const Workload w = make_workload(num_tokens, seed, drop_routes);
    const auto events = scatter_schedule(w);
    Sender sender(num_tokens);
    const uint32_t participants = num_sms * warps_per_sm;
    Receiver recv(w, participants);

    std::vector<float> ref(static_cast<size_t>(num_tokens) * kHidden, 0.0f);
    for (uint32_t t = 0; t < num_tokens; ++t) {
        uint32_t m = w.mask(t);
        while (m) {
            const uint32_t s = ffs(m);
            m &= (m - 1u);
            for (uint32_t e = 0; e < kHidden; ++e)
                ref[static_cast<size_t>(t) * kHidden + e] += w.at(s, t, e);
        }
    }

    // Middle: a prefix of whole L2 tiles has been scattered; the rest is still in
    // flight on this rank and on the peers.
    const uint32_t total_tiles = events.empty() ? 0 : events.back().tile + 1;
    const uint32_t mid_tiles = total_tiles * mid_percent / 100;
    size_t mid_prefix = 0;
    while (mid_prefix < events.size() && events[mid_prefix].tile < mid_tiles) ++mid_prefix;
    for (size_t i = 0; i < mid_prefix; ++i)
        sender.scatter_slice(events[i].token, events[i].col_block);
    const std::vector<uint32_t> mid_arrival = sender.arrival;

    // Readiness must be exact: item (t, c) is ready iff every valid slot's row
    // has landed column block c.
    std::vector<uint32_t> landed(static_cast<size_t>(num_tokens) * kRowBlocks, 0u);
    for (size_t i = 0; i < mid_prefix; ++i)
        ++landed[static_cast<size_t>(events[i].token) * kRowBlocks + events[i].col_block];
    bool readiness_exact = true;
    uint32_t ready_items = 0;
    std::vector<uint8_t> ready(static_cast<size_t>(num_tokens) * kRowBlocks, 0u);
    for (uint32_t t = 0; t < num_tokens && readiness_exact; ++t) {
        const uint32_t expected = popc(w.mask(t));
        for (uint32_t c = 0; c < kRowBlocks; ++c) {
            const size_t ai = static_cast<size_t>(t) * kRowBlocks + c;
            const bool r = mid_arrival[ai] >= expected;
            if (r != (landed[ai] == expected)) { readiness_exact = false; break; }
            ready[ai] = r ? 1u : 0u;
            if (r) ++ready_items;
        }
    }
    check(readiness_exact, "middle readiness is exact per (token, column block)");

    // Middle participants, in device order: the dedicated Reduce warps (last
    // `reduce_warps` warps of the dispatch group) sweep while the scatter warps
    // still run, then the 16 epilogue warps sweep the same snapshot after their
    // GEMM work ends and before the before-combine NVLink barrier.
    std::vector<uint32_t> mid_gids;
    for (uint32_t sm = 0; sm < num_sms; ++sm)
        for (uint32_t r = 0; r < reduce_warps; ++r)
            mid_gids.push_back(sm * warps_per_sm + (disp_warps - reduce_warps) + r);
    for (uint32_t sm = 0; sm < num_sms; ++sm)
        for (uint32_t e = 0; e < 16; ++e)
            mid_gids.push_back(sm * warps_per_sm + disp_warps + 4 + e);
    uint32_t mid_claimed = 0;
    for (uint32_t gid : mid_gids)
        for (int pass = 0; pass < 4; ++pass) {
            const uint32_t c = recv.sweep(gid, mid_arrival, false);
            mid_claimed += c;
            if (c == 0) break;
        }
    // A middle participant owns exactly the items congruent to its gid.
    uint32_t expect_mid = 0;
    for (size_t item = 0; item < ready.size(); ++item)
        if (ready[item] &&
            std::find(mid_gids.begin(), mid_gids.end(),
                      static_cast<uint32_t>(item % participants)) != mid_gids.end())
            ++expect_mid;

    // Tail: the remaining slices land, then EVERY CTA warp sweeps its own slice.
    for (size_t i = mid_prefix; i < events.size(); ++i)
        sender.scatter_slice(events[i].token, events[i].col_block);
    uint32_t tail_claimed = 0;
    for (uint32_t gid = 0; gid < participants; ++gid)
        tail_claimed += recv.sweep(gid, sender.arrival, true);

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s: every (token, block) fully arrived", label);
    bool all_arrived = true;
    for (uint32_t t = 0; t < num_tokens; ++t)
        for (uint32_t c = 0; c < kRowBlocks; ++c)
            if (sender.arrival[static_cast<size_t>(t) * kRowBlocks + c] != popc(w.mask(t)))
                all_arrived = false;
    check(all_arrived, buf);

    std::snprintf(buf, sizeof(buf), "%s: every item reduced exactly once", label);
    bool exactly_once = true;
    for (uint32_t v : recv.reduce_count) if (v != 1u) exactly_once = false;
    check(exactly_once, buf);

    std::snprintf(buf, sizeof(buf), "%s: middle + tail claims == item count", label);
    check(mid_claimed + tail_claimed == num_tokens * kRowBlocks, buf);

    std::snprintf(buf, sizeof(buf), "%s: tail only finished leftovers", label);
    check(tail_claimed == num_tokens * kRowBlocks - mid_claimed, buf);

    std::snprintf(buf, sizeof(buf), "%s: y matches the FP32 reference", label);
    bool y_ok = true;
    for (size_t i = 0; i < ref.size(); ++i)
        if (recv.y[i] != ref[i]) { y_ok = false; break; }
    check(y_ok, buf);

    std::snprintf(buf, sizeof(buf), "%s: middle pass claimed exactly its ready items", label);
    check(mid_claimed == expect_mid, buf);

    std::printf("[INFO] %s tokens=%u tiles=%u mid_tiles=%u ready_items=%u "
                "mid_items=%u tail_items=%u\n",
                label, num_tokens, total_tiles, mid_tiles, ready_items,
                mid_claimed, tail_claimed);
}

// Rank-skew window.  Phase A: this rank's own L2 tiles are still being
// scattered (local terminate not yet published); phase B: the local scatter has
// terminated (scat_meta.w == 1) but the peers are still landing rows until the
// before-combine NVLink barrier completes (scat_meta.w == 2); phase C: tail.
// `stop_at_local_terminate` replays the pre-fix rule (Reduce warps exit at
// w == 1 after one more sweep); the fixed rule keeps them sweeping through B.
uint32_t run_skew_case(uint32_t num_tokens, uint32_t num_sms, uint32_t reduce_warps,
                       unsigned seed, uint32_t a_percent, uint32_t b_percent,
                       bool stop_at_local_terminate, const char* label) {
    const uint32_t disp_warps = reduce_warps <= 2 ? 4 : 8;
    const uint32_t warps_per_sm = disp_warps + 4 + 16;
    const Workload w = make_workload(num_tokens, seed, false);
    const auto events = scatter_schedule(w);
    Sender sender(num_tokens);
    const uint32_t participants = num_sms * warps_per_sm;
    Receiver recv(w, participants);
    std::vector<uint32_t> reduce_gids, epi_gids;
    for (uint32_t sm = 0; sm < num_sms; ++sm) {
        for (uint32_t r = 0; r < reduce_warps; ++r)
            reduce_gids.push_back(sm * warps_per_sm + (disp_warps - reduce_warps) + r);
        for (uint32_t e = 0; e < 16; ++e)
            epi_gids.push_back(sm * warps_per_sm + disp_warps + 4 + e);
    }
    const size_t a_end = events.size() * a_percent / 100;
    const size_t b_end = events.size() * b_percent / 100;
    auto sweep_all = [&](const std::vector<uint32_t>& gids) {
        uint32_t c = 0;
        for (uint32_t g : gids) c += recv.sweep(g, sender.arrival, false);
        return c;
    };
    // Phase A: the Reduce warps sweep while slices land.
    uint32_t a_claims = 0;
    for (size_t i = 0; i < a_end; ++i) {
        sender.scatter_slice(events[i].token, events[i].col_block);
        if ((i % 4096) == 4095) a_claims += sweep_all(reduce_gids);
    }
    a_claims += sweep_all(reduce_gids);
    // Phase B: the epilogue does its single pre-barrier sweep; the remaining
    // peer slices keep landing.
    const uint32_t b_epi = sweep_all(epi_gids);
    uint32_t b_reduce = 0;
    bool reduce_alive = true;
    if (stop_at_local_terminate) {
        b_reduce += sweep_all(reduce_gids);   // last sweep at w == 1, then exit
        reduce_alive = false;
    }
    for (size_t i = a_end; i < b_end; ++i) {
        sender.scatter_slice(events[i].token, events[i].col_block);
        if (reduce_alive and (i % 2048) == 2047) b_reduce += sweep_all(reduce_gids);
    }
    if (reduce_alive) b_reduce += sweep_all(reduce_gids);
    // Phase C: barrier passed, everything lands, whole-CTA tail.
    for (size_t i = b_end; i < events.size(); ++i)
        sender.scatter_slice(events[i].token, events[i].col_block);
    uint32_t tail = 0;
    for (uint32_t gid = 0; gid < participants; ++gid)
        tail += recv.sweep(gid, sender.arrival, true);

    char buf[256];
    bool once = true;
    for (uint32_t v : recv.reduce_count) if (v != 1u) once = false;
    std::snprintf(buf, sizeof(buf), "%s: every item reduced exactly once", label);
    check(once, buf);
    std::snprintf(buf, sizeof(buf), "%s: A + B + tail claims == item count", label);
    check(a_claims + b_epi + b_reduce + tail == num_tokens * kRowBlocks, buf);
    std::printf("[INFO] %s rule=%s A=%u B_epi=%u B_reduce=%u tail=%u\n", label,
                stop_at_local_terminate ? "stop@local_terminate" : "stream_until_barrier",
                a_claims, b_epi, b_reduce, tail);
    return b_reduce;
}

// ---- named-barrier schedule model ----------------------------------------
// Replays each role's rendezvous order in the fused kernel's Combine section on
// the comm_only SMEM engine and detects deadlock.  Units are warps.  A named
// barrier instance releases when its arrivals reach its count; arriving at a
// pending instance with a different count is reported as a bug.
enum class Op { Bar, SetW, WaitW };
struct Step { Op op; uint32_t id; uint32_t value; };

bool schedule_completes(const std::vector<std::vector<Step>>& roles,
                        const std::vector<uint32_t>& role_warps, const char* label) {
    std::vector<size_t> pc(roles.size(), 0);
    std::vector<bool> waiting(roles.size(), false);
    uint32_t w = 0;
    struct Inst { uint32_t count = 0; uint32_t arrived = 0; std::vector<size_t> who; };
    std::vector<Inst> bars(128);
    bool count_mismatch = false;
    for (int iter = 0; iter < 10000; ++iter) {
        bool progress = false, all_done = true;
        for (size_t r = 0; r < roles.size(); ++r) {
            if (pc[r] >= roles[r].size()) continue;
            all_done = false;
            if (waiting[r]) continue;
            const Step& s = roles[r][pc[r]];
            if (s.op == Op::SetW) {
                w = s.value; ++pc[r]; progress = true;
            } else if (s.op == Op::WaitW) {
                if (w >= s.value) { ++pc[r]; progress = true; }
            } else {
                auto& b = bars[s.id];
                if (b.arrived == 0) b.count = s.value;
                else if (b.count != s.value) count_mismatch = true;
                b.arrived += role_warps[r];
                b.who.push_back(r);
                waiting[r] = true;
                progress = true;
                if (b.arrived >= b.count) {
                    for (size_t q : b.who) { waiting[q] = false; ++pc[q]; }
                    b = Inst{};
                }
            }
        }
        if (all_done) return not count_mismatch;
        if (not progress) {
            std::printf("[INFO] %s: deadlock, role pcs =", label);
            for (size_t r = 0; r < roles.size(); ++r) std::printf(" %zu", pc[r]);
            std::printf("\n");
            return false;
        }
    }
    return false;
}

void check_barrier_schedules() {
    constexpr uint32_t S = 2, R = 2, D = S + R, N = 4, E = 16, ALL = D + N + E;
    constexpr uint32_t kIdx1 = 1, kIdx8 = 8, kNvl = 99;   // 99 models the NVLink barrier (epilogue only)
    const std::vector<uint32_t> warps = {S, R, E, N};
    // Fixed schedule (this change): the Reduce warps skip the post-terminate
    // rendezvous (kScatHandoffSyncThreads = S + E) and stream until w == 2.
    const std::vector<std::vector<Step>> fixed = {
        /*S*/ {{Op::WaitW, 0, 1}, {Op::Bar, kIdx1, S + E}, {Op::Bar, kIdx8, ALL},
               {Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx1, D + E}},
        /*R*/ {{Op::WaitW, 0, 2}, {Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx8, ALL},
               {Op::Bar, kIdx1, D + E}},
        /*E*/ {{Op::SetW, 0, 1}, {Op::Bar, kIdx1, S + E}, {Op::Bar, kNvl, E},
               {Op::SetW, 0, 2}, {Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx8, ALL},
               {Op::Bar, kIdx1, D + E}},
        /*N*/ {{Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx8, ALL}},
    };
    check(schedule_completes(fixed, warps, "fixed"),
          "barrier schedule: stream-until-barrier rule completes without deadlock");
    // Pre-fix schedule: the Reduce warps exit at w == 1 and join the full-width
    // rendezvous -- deadlock-free, but idle through the skew window.
    const std::vector<std::vector<Step>> prefix = {
        /*S*/ {{Op::WaitW, 0, 1}, {Op::Bar, kIdx1, D + E}, {Op::Bar, kIdx8, ALL},
               {Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx1, D + E}},
        /*R*/ {{Op::WaitW, 0, 1}, {Op::Bar, kIdx1, D + E}, {Op::Bar, kIdx8, ALL},
               {Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx1, D + E}},
        /*E*/ {{Op::SetW, 0, 1}, {Op::Bar, kIdx1, D + E}, {Op::Bar, kNvl, E},
               {Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx1, D + E}},
        /*N*/ {{Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx8, ALL}},
    };
    check(schedule_completes(prefix, warps, "pre-fix"),
          "barrier schedule: pre-fix rule completes (its defect is idleness, not deadlock)");
    // Counter-example: streaming until w == 2 while still inside the full-width
    // rendezvous must deadlock -- the reason kScatHandoffSyncThreads exists.
    const std::vector<std::vector<Step>> naive = {
        /*S*/ {{Op::WaitW, 0, 1}, {Op::Bar, kIdx1, D + E}, {Op::Bar, kIdx8, ALL},
               {Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx1, D + E}},
        /*R*/ {{Op::WaitW, 0, 2}, {Op::Bar, kIdx1, D + E}, {Op::Bar, kIdx8, ALL},
               {Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx1, D + E}},
        /*E*/ {{Op::SetW, 0, 1}, {Op::Bar, kIdx1, D + E}, {Op::Bar, kNvl, E},
               {Op::SetW, 0, 2}, {Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx8, ALL},
               {Op::Bar, kIdx1, D + E}},
        /*N*/ {{Op::Bar, kIdx8, ALL}, {Op::Bar, kIdx8, ALL}},
    };
    check(not schedule_completes(naive, warps, "naive"),
          "barrier schedule: streaming to w==2 inside the full-width rendezvous deadlocks (counter-example)");
}

}  // namespace

// Mirrors the device capacity assert: arrival takes the per-slot x per-chunk
// scatter counter array, claim takes the flat per-slot array.  Getting these two
// backwards overflows arrival by kRowBlocks / kNumRanks, which no runtime test
// would notice until it corrupts the neighbouring workspace region.
void check_workspace_capacity() {
    constexpr uint32_t kMaxScatterChunks = kHidden / 64;
    for (uint32_t tokens : {64u, 256u, 1024u, 4096u, 16384u}) {
        const uint32_t slots = kNumRanks * tokens;   // kNumMaxTokensPerRank == tokens
        char buf[160];
        std::snprintf(buf, sizeof(buf), "capacity tokens=%u: arrival fits the chunk-count array", tokens);
        check(static_cast<uint64_t>(tokens) * kRowBlocks <=
              static_cast<uint64_t>(slots) * kMaxScatterChunks, buf);
        std::snprintf(buf, sizeof(buf), "capacity tokens=%u: claim fits the flat slot array", tokens);
        check(tokens <= slots, buf);
        std::snprintf(buf, sizeof(buf), "capacity tokens=%u: flat array is NOT wide enough for arrival", tokens);
        // Documents why the assignment must not be swapped back.
        check(kRowBlocks > kNumRanks ? !(static_cast<uint64_t>(tokens) * kRowBlocks <= slots) : true, buf);
    }
}

int main() {
    check_workspace_capacity();
    // EP4 Qwen3.8-Max: 78 SMs; WEAVE_REDUCE_WARPS=1/2 size a 4-warp dispatch
    // group, 4/8 size an 8-warp one (host four_wg_regs_fit keeps 4 Math WGs).
    run_case(1024, 4, 78, 2, 11, false, 60, "S1024_rw2_mid60");
    run_case(1024, 4, 78, 2, 11, false, 90, "S1024_rw2_mid90");
    run_case(4096, 4, 78, 2, 12, false, 80, "S4096_rw2_mid80");
    run_case(1024, 8, 78, 4, 13, false, 75, "S1024_rw4_mid75");
    run_case(1024, 4, 78, 2, 14, true, 70, "S1024_dropped_routes");
    run_case(256, 4, 78, 2, 15, true, 50, "S256_sparse");
    run_case(1024, 4, 78, 1, 16, false, 85, "S1024_rw1");
    run_case(1024, 8, 78, 8, 17, false, 85, "S1024_rw8");
    // A single SM still has to cover the whole item space at the tail.
    run_case(64, 4, 1, 2, 18, true, 40, "S64_one_sm");
    // Nothing scattered yet: the middle pass must claim nothing and the tail all.
    run_case(128, 4, 78, 2, 19, false, 0, "S128_mid0");
    // Rank-skew window: the fixed rule must keep reclaiming items that land
    // after the local terminate; the pre-fix rule leaves them to the tail.
    const std::tuple<uint32_t, uint32_t, unsigned, uint32_t, uint32_t, const char*> skew_cases[] = {
        {1024u, 2u, 21u, 55u, 90u, "S1024_rw2_skew"},
        {4096u, 2u, 22u, 60u, 95u, "S4096_rw2_skew"},
        {1024u, 1u, 23u, 50u, 85u, "S1024_rw1_skew"},
        {1024u, 4u, 24u, 40u, 80u, "S1024_rw4_skew"},
    };
    for (const auto& [tok, rw, seed, a, b, name] : skew_cases) {
        char lbl[96];
        std::snprintf(lbl, sizeof(lbl), "%s_prefix", name);
        const uint32_t old_b = run_skew_case(tok, 78, rw, seed, a, b, true, lbl);
        std::snprintf(lbl, sizeof(lbl), "%s_fixed", name);
        const uint32_t new_b = run_skew_case(tok, 78, rw, seed, a, b, false, lbl);
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "%s: fixed rule reclaims skew-window items the pre-fix rule left to the tail", name);
        check(new_b > old_b, buf);
    }
    check_barrier_schedules();
    std::printf("CHECKS=%d FAILURES=%d\n", g_checks, g_failures);
    if (g_failures == 0)
        std::printf("ALL H20 V9 STREAM-COMBINE PROTOCOL CHECKS PASS\n");
    return g_failures == 0 ? 0 : 1;
}
