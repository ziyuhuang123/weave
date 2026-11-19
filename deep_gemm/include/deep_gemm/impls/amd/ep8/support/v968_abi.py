#!/usr/bin/env python3
"""CPU ABI and exact reference for v964 dispatch, receiver sorting, and route directory."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "dacsm-v968-route-ingress-gemm1/1"
MAX_WORLD = 16
MAX_TOPK = 8
SORT_BLOCK = 32
META_STRIDE = 64
META_HEADER = 4096
META_BYTES = 64
CHUNK_ROWS = 8
MAX_CHUNKS = 512
PROGRESS_MAGIC = 0x44385052
PROGRESS_BYTES = 2112
FLAG_INCLUDE_SELF = 1
FLAG_WEIGHT_FP16 = 2
ROUTE_VERSION = 1
ROUTE_HEADER_BYTES = 128
ROUTE_ENTRY_BYTES = 16
ROUTE_BLOCK_VARIANTS = (32, 64, 128)
ROUTE_READY_MAGIC = 0x4438524459523031
ROUTE_ERROR_MAGIC = 0x4438524459453031
ROUTE_BLOCK_PREFIX_VARIANTS = 3
# Frozen CK stage-1 tile metadata: ck_m64_block_rows is an ABI constant.
CK_M64_BLOCK_ROWS = 64
# Maximum runtime route block_m; this controls worst-case route-table capacity.
# The frozen CK M64 metadata and the runtime M128 capacity are independent.
CK_MAX_BLOCK_ROWS = 128
GRID_BARRIER_SLOTS = 16
INGRESS_VERSION = 1
INGRESS_HEADER_BYTES = 128
SOURCE_INGRESS_STATE_BYTES = 48
BLOCK_INGRESS_STATE_BYTES = 32
INGRESS_MAX_BLOCKS = 131072
INGRESS_READY_MAGIC = 0x4438494E47523131
INGRESS_DONE_MAGIC = 0x4438494E47444E31
INGRESS_ERROR_MAGIC = 0x4438494E47455231
INGRESS_BLOCK_READY_MARKER = 0x424C4B52
INGRESS_STATUS_BUILDING = 0
INGRESS_STATUS_READY = 1
INGRESS_STATUS_COMPLETE = 2
INGRESS_STATUS_FAILED = 3

INGRESS_HEADER_DTYPE = np.dtype({
    "names": [
        "magic", "version", "generation", "status", "error_code", "world",
        "capacity", "topk", "nexp", "block_variant", "block_m", "block_count",
        "max_blocks", "route_rows", "source_stride_bytes",
    ] + [f"reserved{i}" for i in range(16)],
    "formats": ["<u8"] + ["<u4"] * 30,
    "offsets": [0] + list(range(8, 128, 4)),
    "itemsize": INGRESS_HEADER_BYTES,
})
SOURCE_INGRESS_DTYPE = np.dtype({
    "names": [
        "generation", "observed_rows", "claim", "sealed_generation",
        "stage_error", "stage_ok", "stage_previous", "stage_current",
        "stage_generation", "stage_bm", "stage_tag",
    ],
    "formats": ["<u4"] * 10 + ["<u8"],
    "offsets": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    "itemsize": SOURCE_INGRESS_STATE_BYTES,
})
BLOCK_INGRESS_DTYPE = np.dtype({
    "names": ["ready_tag", "generation", "seen", "valid_rows", "lid",
              "reserved0", "reserved1"],
    "formats": ["<u8", "<u4", "<u4", "<u4", "<u4", "<u4", "<u4"],
    "offsets": [0, 8, 12, 16, 20, 24, 28],
    "itemsize": BLOCK_INGRESS_STATE_BYTES,
})

_EXT = struct.Struct("<18Q10I17Q2I3Q2I")
EXT_BYTES = _EXT.size
_CK = struct.Struct("<15Q12I")
CK_EXT_BYTES = _CK.size
ABI_BYTES = EXT_BYTES + CK_EXT_BYTES
_META = struct.Struct("<II8H8H24x")
assert _META.size == META_BYTES
assert EXT_BYTES == 360
assert CK_EXT_BYTES == 168 and ABI_BYTES == 528


def route_sort_capacity(*, world: int, capacity: int, topk: int,
                        local_experts: int) -> int:
    """Worst-case sorted rows: all received routes plus one 32-row tail each."""
    return (int(world) * int(capacity) * int(topk) +
            int(local_experts) * SORT_BLOCK)


def ck_route_capacity(*, world: int, capacity: int, topk: int,
                      local_experts: int) -> int:
    """Worst-case runtime-M row table: routes plus one max-M tail per expert.

    ``CK_MAX_BLOCK_ROWS`` is intentionally independent of the frozen CK
    ``ck_m64_block_rows == 64`` ABI field.  Runtime M32/M64/M128 all share
    this allocation bound so one code object can serve every selected shape.
    """
    return (int(world) * int(capacity) * int(topk) +
            int(local_experts) * CK_MAX_BLOCK_ROWS)


def _nonzero_u64(name: str, value: int) -> int:
    value = int(value)
    if value <= 0 or value >= 1 << 64 or value & 7:
        raise ValueError(f"{name} must be a nonzero 8-byte-aligned uint64")
    return value


def chunk_count(rows: int) -> int:
    rows = int(rows)
    if rows < 0:
        raise ValueError("negative row count")
    return (rows + CHUNK_ROWS - 1) // CHUNK_ROWS


def weight_bits(values: np.ndarray, weight_format: str) -> np.ndarray:
    """Convert f32 router weights to the exact 16-bit metadata representation."""
    values = np.asarray(values, dtype=np.float32)
    if weight_format == "fp16":
        return np.asarray(values, dtype=np.float16).view(np.uint16)
    if weight_format == "bf16":
        raw = values.view(np.uint32)
        rounded = (raw + np.uint32(0x7FFF) + ((raw >> 16) & 1)) >> 16
        return rounded.astype(np.uint16)
    raise ValueError("weight_format must be 'bf16' or 'fp16'")


def bits_to_weight(bits: np.ndarray, weight_format: str) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint16)
    if weight_format == "fp16":
        return bits.view(np.float16).astype(np.float32)
    if weight_format == "bf16":
        return (bits.astype(np.uint32) << 16).view(np.float32)
    raise ValueError("weight_format must be 'bf16' or 'fp16'")


@dataclass(frozen=True)
class DispatchExtension:
    x: int
    x_scale: int
    topk_ids: int
    topk_weights: int
    expert_map: int
    local_meta_table: int
    peer_meta_table: int
    peer_payload_table: int
    peer_scale_table: int
    peer_progress_table: int
    bucket_count: int
    bucket_cursor: int
    bucket_prefix: int
    chunk_prefix: int
    total_chunks: int
    error: int
    sync_state: int
    reserved0: int
    t_local: int
    topk: int
    capacity: int
    payload_bytes: int
    scale_groups: int
    world: int
    rank: int
    include_self: int
    n_experts: int
    recv_meta: int
    expert_counts: int
    expert_offsets: int
    expert_cursors: int
    sorted_ids: int
    sorted_weights: int
    sorted_expert_ids: int
    num_valid_ids: int
    sort_stats: int
    route_header: int
    source_bucket_rows: int
    padded_bases: int
    block_prefix: int
    inverse_generation: int
    inverse_row: int
    route_entries: int
    route_generation: int
    sort_capacity: int
    sort_block: int
    ingress_header: int
    ingress_source_states: int
    ingress_block_states: int
    ingress_max_blocks: int
    ingress_block_m: int
    max_chunks: int = MAX_CHUNKS
    weight_format: str = "fp16"
    ck: "CkExtension | None" = None

    def _check(self) -> None:
        for name in (
            "x", "x_scale", "topk_ids", "topk_weights", "expert_map",
            "local_meta_table", "peer_meta_table", "peer_payload_table",
            "peer_scale_table", "peer_progress_table", "bucket_count",
            "bucket_cursor", "bucket_prefix", "chunk_prefix", "total_chunks",
            "error", "sync_state", "recv_meta", "expert_counts",
            "expert_offsets", "expert_cursors", "sorted_ids",
            "sorted_weights", "sorted_expert_ids", "num_valid_ids",
            "sort_stats", "route_header", "source_bucket_rows",
            "padded_bases", "block_prefix", "inverse_generation",
            "inverse_row", "route_entries", "route_generation",
            "ingress_header", "ingress_source_states",
            "ingress_block_states",
        ):
            _nonzero_u64(name, getattr(self, name))
        if int(self.reserved0) != 0:
            raise ValueError("reserved0 must be zero")
        if not 1 <= self.world <= MAX_WORLD:
            raise ValueError(f"world outside [1,{MAX_WORLD}]")
        if not 1 <= self.topk <= MAX_TOPK:
            raise ValueError(f"topk outside [1,{MAX_TOPK}]")
        if self.t_local <= 0 or self.n_experts <= 0:
            raise ValueError("t_local and n_experts must be positive")
        if not self.t_local <= self.capacity <= MAX_CHUNKS * CHUNK_ROWS:
            raise ValueError(
                f"capacity must be in [t_local,{MAX_CHUNKS * CHUNK_ROWS}]")
        if self.payload_bytes <= 0 or self.payload_bytes % 16:
            raise ValueError("payload_bytes must be a positive multiple of 16")
        if self.scale_groups < 0:
            raise ValueError("scale_groups cannot be negative")
        if not 0 <= self.rank < self.world:
            raise ValueError("rank outside world")
        if self.include_self not in (0, 1):
            raise ValueError("include_self must be 0 or 1")
        if self.weight_format not in ("bf16", "fp16"):
            raise ValueError("weight_format must be 'bf16' or 'fp16'")
        if self.max_chunks != MAX_CHUNKS:
            raise ValueError("max_chunks ABI mismatch")
        if self.sort_block != SORT_BLOCK:
            raise ValueError("sort_block ABI mismatch")
        if self.sort_capacity < SORT_BLOCK or self.sort_capacity >= 1 << 32:
            raise ValueError("sort_capacity outside uint32")
        if self.n_experts % self.world:
            raise ValueError("n_experts must be divisible by world")
        local_experts = self.n_experts // self.world
        if not 0 < local_experts <= 65535:
            raise ValueError("local expert count outside lid width")
        if self.sort_capacity < local_experts * SORT_BLOCK:
            raise ValueError("sort_capacity cannot hold one padded block per expert")
        if self.ingress_max_blocks == 0 or self.ingress_max_blocks >= 1 << 31:
            raise ValueError("ingress_max_blocks outside supported range")
        if self.ingress_block_m not in ROUTE_BLOCK_VARIANTS:
            raise ValueError("ingress_block_m must be 32, 64, or 128")
        if self.ingress_max_blocks < (
                self.sort_capacity + SORT_BLOCK - 1) // SORT_BLOCK:
            raise ValueError("ingress_max_blocks cannot cover padded rows")
        if self.world * (self.capacity + 1) >= 1 << 24:
            raise ValueError("v67 g2p route span does not fit CK low-24 field")

    @property
    def flags(self) -> int:
        return (FLAG_INCLUDE_SELF if self.include_self else 0) | (
            FLAG_WEIGHT_FP16 if self.weight_format == "fp16" else 0)

    def pack(self) -> bytes:
        self._check()
        # Keep this order identical to V918DispatchExt in stage_dispatch_sort.hip:
        # 18 Q, 10 I, 9 Q, 2 I.  A flat 27Q/12I layout has the same byte size
        # but places the scalar launch fields at the wrong offsets.
        return _EXT.pack(
            self.x, self.x_scale, self.topk_ids, self.topk_weights,
            self.expert_map, self.local_meta_table, self.peer_meta_table,
            self.peer_payload_table, self.peer_scale_table,
            self.peer_progress_table, self.bucket_count, self.bucket_cursor,
            self.bucket_prefix, self.chunk_prefix, self.total_chunks,
            self.error, self.sync_state, self.reserved0,
            self.t_local, self.topk, self.capacity, self.payload_bytes,
            self.scale_groups, self.world, self.rank, self.flags,
            self.n_experts, self.max_chunks,
            self.recv_meta, self.expert_counts, self.expert_offsets,
            self.expert_cursors, self.sorted_ids, self.sorted_weights,
            self.sorted_expert_ids, self.num_valid_ids, self.sort_stats,
            self.route_header, self.source_bucket_rows, self.padded_bases,
            self.block_prefix, self.inverse_generation, self.inverse_row,
            self.route_entries, self.route_generation,
            self.sort_capacity, self.sort_block,
            self.ingress_header, self.ingress_source_states,
            self.ingress_block_states, self.ingress_max_blocks,
            self.ingress_block_m,
        )

    def pack_with_ck(self) -> bytes:
        """Pack the v968 base ABI plus the fail-closed CK stage1 suffix."""
        if self.ck is None:
            raise ValueError("CK suffix is required for v968 GPU launches")
        local_experts = self.n_experts // self.world
        max_route_rows = ck_route_capacity(
            world=self.world, capacity=self.capacity, topk=self.topk,
            local_experts=local_experts)
        if self.ck.ck_route_rows < max_route_rows:
            raise ValueError("CK route capacity is below the worst-case runtime-M table")
        if self.ck.ck_num_tokens != self.world * self.capacity * self.topk:
            raise ValueError("CK num_tokens must equal world*capacity*topk")
        if self.ck.ck_hidden != self.payload_bytes:
            raise ValueError("CK hidden must equal payload_bytes")
        if self.ck.ck_groups_per_row != self.scale_groups:
            raise ValueError("CK groups_per_row must equal scale_groups")
        if self.ck.ck_local_experts != local_experts:
            raise ValueError("CK local_experts mismatch")
        return self.pack() + self.ck.pack()

    def fields(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["flags"] = self.flags
        return out

    def receipt(self) -> dict[str, Any]:
        packed = self.pack()
        return {
            "schema": SCHEMA,
            "status": "DISPATCH_EXTENSION_VALID",
            "bytes": len(packed),
            "abi_bytes": ABI_BYTES,
            "sha256": hashlib.sha256(packed).hexdigest(),
            "meta_stride": META_STRIDE,
            "meta_header": META_HEADER,
            "meta_bytes": META_BYTES,
            "chunk_rows": CHUNK_ROWS,
            "max_chunks": MAX_CHUNKS,
            "progress_bytes": PROGRESS_BYTES,
            "sort_block": SORT_BLOCK,
            "sort_route_encoding": "g2p | (k << 24); g2p=source*(capacity+1)+slot",
            "sort_padding_encoding": "world*(capacity+1)",
            "route_directory_version": ROUTE_VERSION,
            "route_header_bytes": ROUTE_HEADER_BYTES,
            "route_entry_bytes": ROUTE_ENTRY_BYTES,
            "route_block_variants": list(ROUTE_BLOCK_VARIANTS),
            "route_generation": "host-managed nonzero u32 at route_generation pointer",
            "route_publication": "prepare, grid barrier, build_rows, grid barrier, finalize",
            "ingress_publication": (
                "prepare, directory-ready poll, monotonic payload_rows consume, "
                "per-source seal, per-block ready tags, finish"),
            "ingress_lds": "zero static helper LDS; block staging in exclusive per-source global state",
            "flags": self.flags,
            "weight_format": self.weight_format,
            "active_transport": "intra-node IPC, EP8",
            "ep16": "ABI bound only; separate transport not implemented",
        }


@dataclass(frozen=True)
class CkExtension:
    """Appended fail-closed CK stage1 ABI; all fields are checked before use."""
    ck_a: int
    ck_a_scale: int
    ck_b: int
    ck_b_scale: int
    ck_dense_a: int
    ck_dense_a_scale: int
    ck_stage1_output: int
    ck_quant_output: int
    ck_quant_scale: int
    ck_m64_sorted_ids: int
    ck_m64_expert_ids: int
    ck_m64_offsets: int
    ck_m64_cursors: int
    ck_max_token_id: int
    ck_diagnostics: int
    ck_route_rows: int
    ck_num_tokens: int
    ck_dense_row_capacity: int
    ck_hidden: int
    ck_intermediate: int
    ck_local_experts: int
    ck_topk: int = 1
    ck_groups_per_row: int = 0
    ck_m64_block_rows: int = 64
    ck_n_block_cols: int = 128
    ck_route_sentinel: int = 0
    ck_flags: int = 0

    def _check(self) -> None:
        names = (
            "ck_a", "ck_a_scale", "ck_b", "ck_b_scale", "ck_dense_a",
            "ck_dense_a_scale", "ck_stage1_output", "ck_quant_output",
            "ck_quant_scale", "ck_m64_sorted_ids", "ck_m64_expert_ids",
            "ck_m64_offsets", "ck_m64_cursors", "ck_max_token_id",
            "ck_diagnostics",
        )
        for name in names:
            _nonzero_u64(name, getattr(self, name))
        if self.ck_a != self.ck_dense_a:
            raise ValueError("ck_a must alias ck_dense_a")
        if self.ck_a_scale != self.ck_dense_a_scale:
            raise ValueError("ck_a_scale must alias ck_dense_a_scale")
        if int(self.ck_route_rows) <= 0 or int(self.ck_route_rows) >= 1 << 31:
            raise ValueError("ck_route_rows outside supported range")
        if int(self.ck_num_tokens) <= 0 or int(self.ck_num_tokens) >= 1 << 31:
            raise ValueError("ck_num_tokens outside supported range")
        if int(self.ck_dense_row_capacity) <= int(self.ck_route_rows):
            raise ValueError("dense CK capacity must include the sentinel row")
        if int(self.ck_hidden) <= 0 or int(self.ck_hidden) % 128:
            raise ValueError("ck_hidden must be a positive multiple of 128")
        if int(self.ck_hidden) // 128 % 2:
            raise ValueError("CK stage1 requires an even K-block count")
        if int(self.ck_intermediate) <= 0 or int(self.ck_intermediate) % 128:
            raise ValueError("ck_intermediate must be a positive multiple of 128")
        if int(self.ck_local_experts) <= 0:
            raise ValueError("ck_local_experts must be positive")
        if int(self.ck_topk) != 1:
            raise ValueError("v968 CK route ids require topk=1")
        if int(self.ck_groups_per_row) != int(self.ck_hidden) // 128:
            raise ValueError("ck_groups_per_row must equal hidden/128")
        if int(self.ck_m64_block_rows) != 64 or int(self.ck_n_block_cols) != 128:
            raise ValueError("v968 CK tile geometry must be M64/N128")
        if int(self.ck_route_sentinel) != int(self.ck_route_rows):
            raise ValueError("ck_route_sentinel must equal declared route capacity")
        if int(self.ck_flags) != 0:
            raise ValueError("ck_flags must be zero")

    def pack(self) -> bytes:
        self._check()
        return _CK.pack(
            self.ck_a, self.ck_a_scale, self.ck_b, self.ck_b_scale,
            self.ck_dense_a, self.ck_dense_a_scale, self.ck_stage1_output,
            self.ck_quant_output, self.ck_quant_scale, self.ck_m64_sorted_ids,
            self.ck_m64_expert_ids, self.ck_m64_offsets, self.ck_m64_cursors,
            self.ck_max_token_id, self.ck_diagnostics, self.ck_route_rows,
            self.ck_num_tokens, self.ck_dense_row_capacity, self.ck_hidden,
            self.ck_intermediate, self.ck_local_experts, self.ck_topk,
            self.ck_groups_per_row, self.ck_m64_block_rows,
            self.ck_n_block_cols, self.ck_route_sentinel, self.ck_flags,
        )


def reference_dispatch(*, x: np.ndarray, x_scale: np.ndarray,
                       topk_ids: np.ndarray, topk_weights: np.ndarray,
                       expert_map: Sequence[int], rank: int, world: int,
                       include_self: int = 1,
                       weight_format: str = "fp16") -> dict[str, Any]:
    """Build canonical destination buckets and their exact payloads.

    Device slot order is defined by atomic cursors, so validation keys buckets
    by ``home`` token rather than by physical slot.
    """
    x = np.asarray(x)
    x_scale = np.asarray(x_scale)
    ids = np.asarray(topk_ids, dtype=np.int32)
    weights = np.asarray(topk_weights, dtype=np.float32)
    emap = np.asarray(expert_map, dtype=np.uint32).reshape(-1)
    if x.ndim != 2 or x.shape[0] <= 0:
        raise ValueError("x must be [tokens,row_bytes]")
    if ids.shape != weights.shape or ids.ndim != 2 or ids.shape[0] != x.shape[0]:
        raise ValueError("topk ids/weights shape mismatch")
    topk = int(ids.shape[1])
    if not 1 <= topk <= MAX_TOPK:
        raise ValueError("topk outside ABI")
    if not 1 <= world <= MAX_WORLD or not 0 <= rank < world:
        raise ValueError("bad rank/world")
    if include_self not in (0, 1):
        raise ValueError("bad include_self")
    if weight_format not in ("bf16", "fp16"):
        raise ValueError("bad weight_format")
    if np.any(ids < 0) or np.any(ids >= emap.size):
        raise ValueError("expert id outside expert_map")
    mapped = emap[ids]
    dst_all = mapped >> 16
    lid_all = mapped & 0xFFFF
    if np.any(dst_all >= world):
        raise ValueError("expert_map destination outside world")
    wb_all = weight_bits(weights, weight_format)

    buckets: dict[int, list[dict[str, Any]]] = {d: [] for d in range(world)}
    route_count = [0 for _ in range(world)]
    for tok in range(ids.shape[0]):
        by_dst: dict[int, list[int]] = {}
        for k in range(topk):
            dst = int(dst_all[tok, k])
            if include_self == 0 and dst == rank:
                continue
            by_dst.setdefault(dst, []).append(k)
        for dst, ks in by_dst.items():
            routes = [{
                "k": int(k),
                "global_expert": int(ids[tok, k]),
                "lid": int(lid_all[tok, k]),
                "weight": float(weights[tok, k]),
                "weight_bits": int(wb_all[tok, k]),
            } for k in ks]
            buckets[dst].append({
                "home": int(tok),
                "payload": np.array(x[tok], copy=True),
                "scale": np.array(x_scale[tok], copy=True),
                "routes": routes,
            })
            route_count[dst] += len(routes)

    counts = [len(buckets[d]) for d in range(world)]
    prefixes = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    chunk_counts = [chunk_count(c) for c in counts]
    chunk_prefix = np.concatenate(([0], np.cumsum(chunk_counts))).astype(np.int32)
    if max(chunk_counts, default=0) > MAX_CHUNKS:
        raise ValueError("bucket count exceeds V809_MAX_CHUNKS")
    return {
        "buckets": buckets,
        "counts": counts,
        "prefix": prefixes,
        "chunk_counts": chunk_counts,
        "chunk_prefix": chunk_prefix,
        "total_chunks": int(chunk_prefix[-1]),
        "route_count": route_count,
        "weight_format": weight_format,
    }


def pack_progress(*, meta_ready: int = 0, payload_rows: int = 0,
                  chunk_done: Sequence[int] = ()) -> bytes:
    """Pack the 2,112-byte SourceProgress record used by the device ABI."""
    meta_ready = int(meta_ready)
    payload_rows = int(payload_rows)
    if meta_ready < 0 or meta_ready >= 1 << 64:
        raise ValueError("meta_ready outside uint64")
    if not 0 <= payload_rows <= MAX_CHUNKS * CHUNK_ROWS:
        raise ValueError("payload_rows outside progress capacity")
    if len(chunk_done) > MAX_CHUNKS:
        raise ValueError("chunk_done exceeds MAX_CHUNKS")
    raw = bytearray(PROGRESS_BYTES)
    struct.pack_into("<QQ", raw, 0, meta_ready, payload_rows)
    for index, value in enumerate(chunk_done):
        value = int(value)
        if not 0 <= value < 1 << 32:
            raise ValueError("chunk_done value outside uint32")
        struct.pack_into("<I", raw, 16 + index * 4, value)
    return bytes(raw)


def unpack_progress(raw: bytes | bytearray | memoryview) -> dict[str, Any]:
    if len(raw) != PROGRESS_BYTES:
        raise ValueError("SourceProgress size mismatch")
    meta_ready, payload_rows = struct.unpack_from("<QQ", raw, 0)
    chunks = np.frombuffer(raw, dtype="<u4", count=MAX_CHUNKS, offset=16)
    return {
        "meta_ready": int(meta_ready),
        "payload_rows": int(payload_rows),
        "chunk_done": chunks.copy(),
    }


def pack_meta(home: int, routes: Sequence[Mapping[str, Any]]) -> bytes:
    if not 0 <= len(routes) <= MAX_TOPK:
        raise ValueError("invalid route count")
    lids = [int(r["lid"]) for r in routes]
    wbits = [int(r["weight_bits"]) & 0xFFFF for r in routes]
    lids += [0] * (MAX_TOPK - len(lids))
    wbits += [0] * (MAX_TOPK - len(wbits))
    return _META.pack(int(home), len(routes), *lids, *wbits)


def unpack_meta(raw: bytes, weight_format: str = "fp16") -> dict[str, Any]:
    if len(raw) != META_BYTES:
        raise ValueError("metadata stride mismatch")
    vals = _META.unpack(raw)
    home, nexp = vals[:2]
    lids = list(vals[2:2 + MAX_TOPK])
    bits = np.asarray(vals[2 + MAX_TOPK:2 + 2 * MAX_TOPK], dtype=np.uint16)
    return {
        "home": int(home),
        "nexp": int(nexp),
        "lid": [int(x) for x in lids],
        "weight_bits": [int(x) for x in bits],
        "weights": [float(x) for x in bits_to_weight(bits, weight_format)],
    }


def reference_m64_layout(*, expert_counts: Sequence[int],
                         sorted_ids: Sequence[int],
                         expert_offsets: Sequence[int], capacity: int,
                         topk: int, world: int, local_experts: int) -> dict[str, Any]:
    """Build the exact dense M64 route order used by the v968 device transform.

    Real routes stay in the v964 expert-bucket order.  Their CK id is the new
    dense row index (topk=1), so every valid id is contiguous and padding uses
    one sentinel equal to the padded row count.
    """
    counts = np.asarray(expert_counts, dtype=np.int64).reshape(-1)
    ids = np.asarray(sorted_ids, dtype=np.uint32).reshape(-1)
    offsets = np.asarray(expert_offsets, dtype=np.int64).reshape(-1)
    if counts.size != local_experts or offsets.size != local_experts + 1:
        raise ValueError("M64 expert count/offset shape mismatch")
    if np.any(counts < 0) or capacity <= 0 or not 1 <= topk <= MAX_TOPK:
        raise ValueError("invalid M64 source shape")
    padded = ((counts + 63) // 64) * 64
    m64_offsets = np.concatenate(([0], np.cumsum(padded))).astype(np.int64)
    route_prefix = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    if m64_offsets[-1] >= 1 << 31 or route_prefix[-1] >= 1 << 31:
        raise ValueError("M64 route table exceeds int32")
    if np.any(offsets[1:] < offsets[:-1]) or offsets[0] != 0:
        raise ValueError("invalid v964 expert offsets")
    sentinel = int(m64_offsets[-1])
    rows = np.full(sentinel, sentinel, dtype=np.int32)
    blocks = np.full((sentinel + 63) // 64, -1, dtype=np.int32)
    sources: list[tuple[int, int, int, int, int]] = []
    seen_rows: set[int] = set()
    for lid in range(local_experts):
        count = int(counts[lid])
        if count > offsets[lid + 1] - offsets[lid]:
            raise ValueError("expert count exceeds 32-row source span")
        begin = int(m64_offsets[lid])
        end = int(m64_offsets[lid + 1])
        for r in range(count):
            old = int(ids[int(offsets[lid]) + r])
            g2p, k = old & 0xFFFFFF, old >> 24
            src = g2p // (capacity + 1)
            slot = g2p - src * (capacity + 1)
            if src >= world or slot >= capacity or k >= topk:
                raise ValueError("invalid v964 route identity")
            row = begin + r
            if row in seen_rows or not 0 <= row < sentinel:
                raise ValueError("duplicate or out-of-range M64 dense row")
            seen_rows.add(row)
            rows[row] = row
            sources.append((lid, r, src, slot, k))
        for block in range(begin // 64, end // 64):
            blocks[block] = lid
    if blocks.size and np.any(blocks < 0):
        raise ValueError("M64 block has no unique expert")
    return {
        "m64_offsets": m64_offsets,
        "route_prefix": route_prefix,
        "sorted_ids": rows,
        "expert_ids": blocks,
        "sources": sources,
        "m64_rows": sentinel,
        "total_routes": int(route_prefix[-1]),
        "sentinel": sentinel,
    }


__all__ = [
    "SCHEMA", "MAX_WORLD", "MAX_TOPK", "META_STRIDE", "META_HEADER",
    "META_BYTES", "CHUNK_ROWS", "MAX_CHUNKS", "PROGRESS_MAGIC",
    "PROGRESS_BYTES", "EXT_BYTES", "FLAG_INCLUDE_SELF", "FLAG_WEIGHT_FP16",
    "INGRESS_VERSION", "INGRESS_HEADER_BYTES", "SOURCE_INGRESS_STATE_BYTES",
    "BLOCK_INGRESS_STATE_BYTES", "INGRESS_MAX_BLOCKS", "INGRESS_READY_MAGIC",
    "INGRESS_DONE_MAGIC", "INGRESS_ERROR_MAGIC", "INGRESS_BLOCK_READY_MARKER",
    "INGRESS_STATUS_BUILDING", "INGRESS_STATUS_READY",
    "INGRESS_STATUS_COMPLETE", "INGRESS_STATUS_FAILED",
    "INGRESS_HEADER_DTYPE", "SOURCE_INGRESS_DTYPE", "BLOCK_INGRESS_DTYPE",
    "ROUTE_VERSION", "ROUTE_HEADER_BYTES", "ROUTE_ENTRY_BYTES",
    "ROUTE_BLOCK_VARIANTS", "ROUTE_BLOCK_PREFIX_VARIANTS",
    "ROUTE_READY_MAGIC", "ROUTE_ERROR_MAGIC", "CK_M64_BLOCK_ROWS",
    "CK_MAX_BLOCK_ROWS",
    "ck_route_capacity", "route_sort_capacity", "GRID_BARRIER_SLOTS",
    "DispatchExtension", "CkExtension", "ABI_BYTES", "CK_EXT_BYTES",
    "reference_m64_layout",
    "reference_dispatch", "pack_meta", "unpack_meta",
    "pack_progress", "unpack_progress", "chunk_count", "weight_bits",
    "bits_to_weight",
]
