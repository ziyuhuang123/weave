#!/usr/bin/env python3
"""Packed OV1 suffix for v983; order mirrors V983Ov1Ext exactly."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, fields

_STRUCT = struct.Struct("<30Q8I")
assert _STRUCT.size == 272


RUNTIME_M_POLICY = (
    # v1905 keeps the generated M32 body in the frozen CO but does not route
    # correctness cells into it until that body has a separately proven fix.
    {"max_local_tokens": 128, "block_m": 64, "gemm_width": 256},
    {"max_local_tokens": None, "block_m": 128, "gemm_width": 128},
)


def select_runtime_geometry(local_tokens: int, hidden: int) -> dict[str, int]:
    """Select one same-CO GEMM tile from the per-rank token count.

    The thresholds are the accepted-runtime sweep points for S=128/1024/8192
    (16/128/1024 local tokens). v1905 maps local_tokens<=128 to the accepted
    M64 body; the M32 body remains present but is not selected until separately
    validated. Performance routing can be retuned without changing the CO/ABI.
    """
    if local_tokens <= 0:
        raise ValueError("local_tokens must be positive")
    if hidden <= 0:
        raise ValueError("hidden must be positive")
    for row in RUNTIME_M_POLICY:
        limit = row["max_local_tokens"]
        if limit is None or local_tokens <= limit:
            block_m = int(row["block_m"])
            width = int(row["gemm_width"])
            if hidden % width:
                raise ValueError("hidden must be divisible by gemm_width")
            return {"block_m": block_m, "gemm_width": width,
                    "n2_blocks": hidden // width}
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Ov1Extension:
    peer_payload_ready_table: int
    recv_payload_ready: int
    peer_return_payload_table: int
    return_recv_payload: int
    peer_home_arrival_table: int
    home_arrival: int
    home_expected: int
    home_slots: int
    plan_flags: int
    a_tile_ready: int
    a_tile_rows_done: int
    stage1_tile_done: int
    row_n_done: int
    bucket_done: int
    bucket_route_rows: int
    bucket_sent: int
    row_route_id: int
    stage2_weights: int
    final_output: int
    gemm2_output: int
    ck_w2: int
    ck_w2_scale: int
    trace: int
    byte_counters: int
    scatter_time: int
    dispatch_time: int
    home_done: int
    post_control: int
    gemm_width: int
    peer_post_state_table: int
    hidden: int
    intermediate: int
    n1_blocks: int
    n2_blocks: int
    mode: int
    block_m: int
    comm_chunk_rows: int
    role_config: int

    def _check(self) -> None:
        pointer_fields = [name for name in (
            "peer_payload_ready_table", "recv_payload_ready",
            "peer_return_payload_table", "return_recv_payload",
            "peer_home_arrival_table", "home_arrival", "home_expected",
            "home_slots", "plan_flags", "a_tile_ready", "a_tile_rows_done",
            "stage1_tile_done", "row_n_done", "bucket_done",
            "bucket_route_rows", "bucket_sent", "row_route_id",
            "stage2_weights", "final_output", "gemm2_output", "ck_w2",
            "ck_w2_scale", "trace", "byte_counters", "scatter_time",
            "dispatch_time", "home_done", "post_control",
            "peer_post_state_table")]
        for name in pointer_fields:
            value = int(getattr(self, name))
            if value <= 0 or value & 7:
                raise ValueError(f"{name} must be a nonzero 8-byte-aligned pointer")
        if self.role_config not in (0, 1, 2):
            raise ValueError("role_config must be 0 (64/16), 1 (72/8), or 2 (76/4)")
        if self.hidden != 4096 or self.intermediate != 2048:
            raise ValueError("v983 is pinned to DSK hidden=4096 intermediate=2048")
        if self.n1_blocks != 16:
            raise ValueError("v1868 expects 16 GEMM1 N blocks")
        if self.block_m not in (32, 64, 128):
            raise ValueError("block_m must be one of 32, 64, 128")
        expected_width = 128 if self.block_m == 128 else 256
        if self.gemm_width != expected_width:
            raise ValueError("gemm_width does not match block_m")
        if self.n2_blocks != self.hidden // self.gemm_width:
            raise ValueError("n2_blocks must equal hidden/gemm_width")
        if self.mode not in (0, 1):
            raise ValueError("invalid mode")
        if self.comm_chunk_rows == 0:
            raise ValueError("comm_chunk_rows must be positive")

    def pack(self) -> bytes:
        self._check()
        return _STRUCT.pack(*(int(getattr(self, f.name)) for f in fields(self)))

    def receipt(self) -> dict[str, object]:
        packed = self.pack()
        return {"bytes": len(packed), "sha256": hashlib.sha256(packed).hexdigest(),
                "format": "<30Q8I", "fields": list(self.__dict__)}
