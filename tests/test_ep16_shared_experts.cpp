// tests/test_ep16_shared_experts.cpp -- CPU-only ABI gate for the v59
// num_shared_experts > 0 wiring in deep_gemm/include/deep_gemm/impls/amd/ep16/.
//
// Lossless-by-construction checks:
//   * the ABI structs keep their frozen sizes (Ep16TurboShape 96,
//     Ep16TurboExternal 128) -- the shared-experts fields live in reserved
//     words, so the byte layout of the legacy ABI is untouched;
//   * num_shared_experts == 0: validation, helpers and the built packet are
//     byte-identical to the legacy behavior (ck_flags stays 0, the shared
//     helpers return the legacy values);
//   * num_shared_experts in 1..8: the shape validates, the shared row space
//     is exactly t_local * N_shared, the recommended route_rows is the legacy
//     floor + shared rows, and the packet's V968CkExt::ck_flags carries the
//     count in bits [3:0];
//   * fail-closed: > 8 shared experts, stray reserved words, missing shared
//     weight pointers, stray pointers at 0 shared experts, and a route_rows
//     floor that does not cover the shared rows are all rejected.
#include "ep16_turbo_packet.hpp"

#include <cstdio>
#include <cstring>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (cond) { printf("[PASS] %s\n", msg); } \
                              else { printf("[FAIL] %s\n", msg); g_fail = 1; } } while (0)

static Ep16TurboShape base_shape(uint32_t route_rows) {
  Ep16TurboShape s{};
  s.rank = 3; s.world = 16; s.ranks_per_node = 8; s.channels = 4;
  s.hidden = 1024; s.intermediate = 512; s.topk = 8;
  s.n_experts = 512;               // E_local = 32
  s.capacity = 8;
  s.t_local = 64; s.node_tokens = 64;
  s.mode = 0; s.route_rows = route_rows;
  s.slot_chunk = 8; s.meta_chunk_slots = 8;
  s.flags = 0; s.enable_dispatch_time = 0; s.enable_scatter_time = 0;
  for (auto& r : s.reserved) r = 0;
  return s;
}

static Ep16TurboExternal base_external(uint64_t shared_b, uint64_t shared_b_scale,
                                        uint64_t shared_w2, uint64_t shared_w2_scale) {
  Ep16TurboExternal ext{};
  ext.x = 0x10000000ull; ext.x_scale = 0x10010000ull;
  ext.topk_ids = 0x10020000ull; ext.topk_weights = 0x10030000ull;
  ext.expert_map = 0x10040000ull;
  ext.ck_b = 0x10050000ull; ext.ck_b_scale = 0x10060000ull;
  ext.ck_w2 = 0x10070000ull; ext.ck_w2_scale = 0x10080000ull;
  ext.final_output = 0x10090000ull;
  ext.reserved[0] = shared_b;
  ext.reserved[1] = shared_b_scale;
  ext.reserved[2] = shared_w2;
  ext.reserved[3] = shared_w2_scale;
  ext.reserved[4] = 0;
  ext.reserved[5] = 0;
  return ext;
}

static Ep16TurboBases base_bases() {
  Ep16TurboBases b{};
  b.arena = 0x20000000ull;
  b.scratch = 0x30000000ull;
  return b;
}

int main() {
  CHECK(sizeof(Ep16TurboShape) == 96, "Ep16TurboShape ABI frozen at 96 bytes");
  CHECK(sizeof(Ep16TurboExternal) == 128, "Ep16TurboExternal ABI frozen at 128 bytes");
  CHECK(sizeof(Ep16TurboLaunchSpec) == 64, "Ep16TurboLaunchSpec ABI frozen at 64 bytes");

  const uint32_t legacy_rr = 16u * 8u * 8u + 16u * 32u;   // world*cap*topk + 16*E_local
  // ---- num_shared_experts == 0: lossless legacy behavior -------------------
  {
    Ep16TurboShape s = base_shape(legacy_rr);
    const char* why = nullptr;
    CHECK(ep16_turbo_validate_shape(s, &why) == 0, "default shape (0 shared) validates");
    CHECK(ep16t_shared_experts(s) == 0u, "shared accessor: 0");
    CHECK(ep16t_shared_rows(s) == 0u, "shared rows: 0");
    CHECK(ep16t_route_rows_recommend(s) == legacy_rr,
          "recommended route_rows == legacy formula at 0 shared experts");
    Ep16TurboExternal ext = base_external(0, 0, 0, 0);
    Ep16TurboBases b = base_bases();
    Ep16ArenaLayout L{};
    Ep16ScratchLayout S{};
    Ep16TurboLaunchSpec spec{};
    spec.generation = 7; spec.runtime = 0x40000000ull; spec.timeout_ticks = 1000000ull;
    CHECK(ep16_turbo_plan_arena(s, &L) == 0, "arena plan (0 shared) ok");
    CHECK(ep16_turbo_plan_scratch(s, &S) == 0, "scratch plan (0 shared) ok");
    V983Envelope env{};
    Ep16TurboContext ctx{};
    CHECK(ep16_turbo_build_packet(s, L, S, b, ext, spec, &env, &ctx) == 0,
          "packet build (0 shared) ok");
    CHECK(env.base.ck.ck_flags == 0u, "ck_flags stays 0 at 0 shared experts");
    // missing-pointer symmetry: stray shared pointers at 0 shared fail closed
    Ep16TurboExternal stray = base_external(1, 0, 0, 0);
    CHECK(ep16_turbo_build_packet(s, L, S, b, stray, spec, &env, &ctx) == -38,
          "stray shared pointer at 0 shared experts rejected (-38)");
  }
  // ---- num_shared_experts == 2: the wiring is live -------------------------
  {
    const uint32_t rr = legacy_rr + 2u * 64u;   // + shared rows
    Ep16TurboShape s = base_shape(rr);
    s.reserved[1] = 2u;
    const char* why = nullptr;
    CHECK(ep16_turbo_validate_shape(s, &why) == 0, "shape with 2 shared experts validates");
    CHECK(ep16t_shared_experts(s) == 2u, "shared accessor: 2");
    CHECK(ep16t_shared_rows(s) == 2u * 64u, "shared rows == t_local * N_shared");
    CHECK(ep16t_route_rows_recommend(s) == rr,
          "recommended route_rows == legacy floor + shared rows");
    Ep16TurboExternal ext = base_external(0x50000000ull, 0x50010000ull,
                                          0x50020000ull, 0x50030000ull);
    Ep16TurboBases b = base_bases();
    Ep16ArenaLayout L{};
    Ep16ScratchLayout S{};
    Ep16TurboLaunchSpec spec{};
    spec.generation = 7; spec.runtime = 0x40000000ull; spec.timeout_ticks = 1000000ull;
    CHECK(ep16_turbo_plan_arena(s, &L) == 0, "arena plan (2 shared) ok");
    CHECK(ep16_turbo_plan_scratch(s, &S) == 0, "scratch plan (2 shared) ok");
    V983Envelope env{};
    Ep16TurboContext ctx{};
    CHECK(ep16_turbo_build_packet(s, L, S, b, ext, spec, &env, &ctx) == 0,
          "packet build (2 shared, 4 pointers) ok");
    CHECK(env.base.ck.ck_flags == 2u, "ck_flags carries num_shared_experts in bits [3:0]");
    CHECK(env.base.ck.ck_route_rows == rr, "ck_route_rows covers the shared row space");
    // fail-closed: missing one shared weight pointer
    Ep16TurboExternal missing = base_external(0x50000000ull, 0, 0, 0);
    CHECK(ep16_turbo_build_packet(s, L, S, b, missing, spec, &env, &ctx) == -36,
          "missing shared weight pointers rejected (-36)");
    // fail-closed: stray reserved[4]
    Ep16TurboExternal stray4 = base_external(0x50000000ull, 0x50010000ull,
                                             0x50020000ull, 0x50030000ull);
    stray4.reserved[4] = 1;
    CHECK(ep16_turbo_build_packet(s, L, S, b, stray4, spec, &env, &ctx) == -37,
          "stray external reserved[4] rejected (-37)");
  }
  // ---- fail-closed guards ---------------------------------------------------
  {
    Ep16TurboShape s = base_shape(legacy_rr + 2u * 64u);
    s.reserved[1] = 9u;                       // > 8
    const char* why = nullptr;
    CHECK(ep16_turbo_validate_shape(s, &why) == 21, "> 8 shared experts rejected (code 21)");
    Ep16TurboShape s2 = base_shape(legacy_rr + 2u * 64u);
    s2.reserved[1] = 2u; s2.reserved[2] = 1u;  // stray reserved word
    CHECK(ep16_turbo_validate_shape(s2, &why) == 22, "stray shape reserved[2] rejected (code 22)");
    Ep16TurboShape s3 = base_shape(legacy_rr);  // shared rows do NOT fit
    s3.reserved[1] = 2u;
    CHECK(ep16_turbo_validate_shape(s3, &why) == 23,
          "route_rows floor without shared rows rejected (code 23)");
  }
  if (g_fail) { printf("EP16 SHARED-EXPERTS ABI GATE FAILED\n"); return 1; }
  printf("ALL EP16 SHARED-EXPERTS ABI CHECKS PASS\n");
  return 0;
}
