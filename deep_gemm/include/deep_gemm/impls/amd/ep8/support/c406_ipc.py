#!/usr/bin/env python3
"""ctypes binding for the V407 C406-vs-C2-SDMA diagnostic.

Only receive buffers are exported: send buffers are local to each rank.  For a
destination peer, the opened IPC alias plus that peer's base offset gives the
actual remote chunk address.  The C406 direct-store kernel and the C2
``hipMemcpyPeerAsync`` transport consume the same stage geometry.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import threading
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

HANDLE_BYTES = 64
MAX_WORLD = 8
HERE = Path(__file__).resolve().parent


def _prefix_ends(values: list[int]) -> list[int]:
    out = [0]
    running = 0
    for value in values:
        running += int(value)
        out.append(running)
    return out


def stage_geometry(stage: str, all_send_ct: list[list[int]], *, rank: int,
                   world: int, element_bytes: int) -> dict[str, list[int]]:
    """Resolve one V406 stage against the exact RCCL all-to-all manifests.

    The hidden/meta receive buffers are ordered by source.  The combine receive
    buffer on a peer is ordered by that peer's original destination chunks.
    Both directions therefore need the gathered count matrix, not just the
    local rank's send row.
    """
    if stage not in ("hidden", "meta", "combine"):
        raise ValueError(f"unknown communication stage {stage!r}")
    if not (1 <= world <= MAX_WORLD):
        raise ValueError(f"world={world} outside [1, {MAX_WORLD}]")
    if len(all_send_ct) != world or any(len(row) != world for row in all_send_ct):
        raise ValueError("send-count matrix is not world x world")
    if any(int(value) < 0 for row in all_send_ct for value in row):
        raise ValueError("send-count matrix contains a negative count")
    if int(element_bytes) <= 0:
        raise ValueError("element_bytes must be positive")
    matrix = [[int(value) for value in row] for row in all_send_ct]
    if rank < 0 or rank >= world:
        raise ValueError(f"rank={rank} outside world={world}")

    if stage in ("hidden", "meta"):
        local = matrix[rank]
        src_prefix = _prefix_ends(local)
        counts = list(local)
        # Source chunks are concatenated on each destination rank.  Our chunk
        # starts after every lower-numbered source rank's chunk to that peer.
        dst_offsets = [sum(matrix[src][dst] for src in range(rank))
                       for dst in range(world)]
    else:
        # Local partial_bf16 is ordered by destination/source chunk.  Returning
        # chunk d to peer d starts after lower-numbered received-source chunks.
        recv_from_src = [matrix[src][rank] for src in range(world)]
        src_prefix = _prefix_ends(recv_from_src)
        counts = [matrix[dst][rank] for dst in range(world)]
        # Peer d receives our returning chunk at the slot allocated to this
        # rank in peer d's own original send_order.
        dst_offsets = [sum(matrix[dst][src] for src in range(rank))
                       for dst in range(world)]

    scaled_prefix = [value * int(element_bytes) for value in src_prefix]
    scaled_counts = [value * int(element_bytes) for value in counts]
    scaled_offsets = [value * int(element_bytes) for value in dst_offsets]
    for dst in range(world):
        if stage in ("hidden", "meta"):
            capacity = sum(matrix[src][dst] for src in range(world)) * int(element_bytes)
        else:
            capacity = sum(matrix[dst][src] for src in range(world)) * int(element_bytes)
        if scaled_offsets[dst] + scaled_counts[dst] > capacity:
            raise ValueError(
                f"{stage} destination peer {dst} range exceeds receive capacity")
    return {
        "source_prefix_bytes": scaled_prefix,
        "count_bytes": scaled_counts,
        "destination_offset_bytes": scaled_offsets,
    }
DEFAULT_LIB = HERE / "libc406_ipc_driver.so"
LIB_PATH = Path(os.environ.get("C406_IPC_LIB", str(DEFAULT_LIB))).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class C406IpcDriver:
    def __init__(self) -> None:
        if not LIB_PATH.is_file():
            raise FileNotFoundError(
                f"{LIB_PATH}; run v407/audit/build_c406_ipc_driver.sh first")
        self.lib = ctypes.CDLL(str(LIB_PATH))
        self.lib_sha256 = sha256_file(LIB_PATH)
        self.lib.c406_driver_version.restype = ctypes.c_int
        if self.lib.c406_driver_version() != 407:
            raise RuntimeError("unexpected c406_ipc_driver ABI version")
        vp = ctypes.c_void_p
        u64p = ctypes.POINTER(ctypes.c_uint64)
        cp = ctypes.c_char_p
        self.lib.c406_get_view.argtypes = [
            vp, ctypes.c_size_t, vp, ctypes.c_size_t,
            cp, u64p, u64p, u64p, ctypes.POINTER(ctypes.c_int),
            cp, ctypes.c_size_t]
        self.lib.c406_get_view.restype = ctypes.c_int
        self.lib.c406_open_view.argtypes = [cp, ctypes.POINTER(vp), cp, ctypes.c_size_t]
        self.lib.c406_open_view.restype = ctypes.c_int
        self.lib.c406_close_view.argtypes = [vp, cp, ctypes.c_size_t]
        self.lib.c406_close_view.restype = ctypes.c_int
        self.lib.c406_launch.argtypes = [
            u64p, u64p, u64p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            vp, cp, ctypes.c_size_t]
        self.lib.c406_launch.restype = ctypes.c_int
        self.lib.c406_sdma_init.argtypes = [ctypes.c_int, cp, ctypes.c_size_t]
        self.lib.c406_sdma_init.restype = ctypes.c_int
        self.lib.c406_sdma_copy.argtypes = [
            u64p, u64p, u64p, ctypes.c_int, ctypes.c_int, vp, cp,
            ctypes.c_size_t]
        self.lib.c406_sdma_copy.restype = ctypes.c_int
        self.lib.c406_sdma_info.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            cp, ctypes.c_size_t]
        self.lib.c406_sdma_info.restype = ctypes.c_int
        self.lib.c406_sdma_shutdown.argtypes = [cp, ctypes.c_size_t]
        self.lib.c406_sdma_shutdown.restype = ctypes.c_int
        self.lib.c406_kernel_info.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), cp, ctypes.c_size_t]
        self.lib.c406_kernel_info.restype = ctypes.c_int
        self._lock = threading.Lock()

    @staticmethod
    def _check(rc: int, err: ctypes.Array[ctypes.c_char], what: str) -> None:
        if rc:
            msg = ctypes.string_at(err).decode(errors="replace") if err.value else ""
            raise RuntimeError(f"{what} rc={rc}: {msg}")

    def get_view(self, tensor: torch.Tensor) -> dict[str, Any]:
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("IPC view requires a contiguous CUDA tensor")
        storage = tensor.untyped_storage()
        handle = ctypes.create_string_buffer(HANDLE_BYTES)
        base = ctypes.c_uint64()
        off = ctypes.c_uint64()
        span = ctypes.c_uint64()
        source = ctypes.c_int()
        err = ctypes.create_string_buffer(4096)
        rc = self.lib.c406_get_view(
            ctypes.c_void_p(int(tensor.data_ptr())),
            int(tensor.numel() * tensor.element_size()),
            ctypes.c_void_p(int(storage.data_ptr())), int(storage.nbytes()),
            handle, ctypes.byref(base), ctypes.byref(off), ctypes.byref(span),
            ctypes.byref(source), err, ctypes.sizeof(err))
        self._check(int(rc), err, "c406_get_view")
        return {
            "handle": bytes(handle.raw),
            "base_addr": int(base.value),
            "base_offset": int(off.value),
            "range_bytes": int(span.value),
            "tensor_bytes": int(tensor.numel() * tensor.element_size()),
            "storage_bytes": int(storage.nbytes()),
            "handle_source": int(source.value),
        }

    def open_view(self, handle: bytes) -> int:
        if len(handle) != HANDLE_BYTES:
            raise ValueError(f"IPC handle length {len(handle)} != {HANDLE_BYTES}")
        alias = ctypes.c_void_p()
        err = ctypes.create_string_buffer(4096)
        rc = self.lib.c406_open_view(
            ctypes.create_string_buffer(handle, len(handle)),
            ctypes.byref(alias), err, ctypes.sizeof(err))
        self._check(int(rc), err, "c406_open_view")
        return int(alias.value or 0)

    def close_view(self, alias: int) -> None:
        if not alias:
            return
        err = ctypes.create_string_buffer(4096)
        rc = self.lib.c406_close_view(
            ctypes.c_void_p(alias), err, ctypes.sizeof(err))
        self._check(int(rc), err, "c406_close_view")

    def launch(self, src: list[int], dst: list[int], bytes_: list[int], *,
               vector16: bool, blocks: int = 80) -> None:
        world = len(src)
        if not (1 <= world <= MAX_WORLD) or len(dst) != world or len(bytes_) != world:
            raise ValueError("c406 launch vectors must have equal world lengths")
        u64 = ctypes.c_uint64
        s = (u64 * world)(*[int(x) for x in src])
        d = (u64 * world)(*[int(x) for x in dst])
        n = (u64 * world)(*[int(x) for x in bytes_])
        stream = int(torch.cuda.current_stream().cuda_stream)
        err = ctypes.create_string_buffer(4096)
        with self._lock:
            rc = self.lib.c406_launch(
                s, d, n, world, int(blocks), 1 if vector16 else 0,
                ctypes.c_void_p(stream), err, ctypes.sizeof(err))
            self._check(int(rc), err, "c406_launch")

    def init_sdma(self, world: int) -> None:
        err = ctypes.create_string_buffer(4096)
        with self._lock:
            rc = self.lib.c406_sdma_init(int(world), err, ctypes.sizeof(err))
            self._check(int(rc), err, "c406_sdma_init")

    def sdma_copy(self, src: list[int], dst: list[int], bytes_: list[int],
                  *, rank: int, world: int) -> None:
        if not (1 <= world <= MAX_WORLD) or len(src) != world or len(dst) != world:
            raise ValueError("SDMA copy vectors must have equal world lengths")
        if len(bytes_) != world or not (0 <= rank < world):
            raise ValueError("invalid SDMA byte vector or rank")
        u64 = ctypes.c_uint64
        s = (u64 * world)(*[int(x) for x in src])
        d = (u64 * world)(*[int(x) for x in dst])
        n = (u64 * world)(*[int(x) for x in bytes_])
        stream = int(torch.cuda.current_stream().cuda_stream)
        err = ctypes.create_string_buffer(4096)
        with self._lock:
            rc = self.lib.c406_sdma_copy(
                s, d, n, int(world), int(rank), ctypes.c_void_p(stream),
                err, ctypes.sizeof(err))
            self._check(int(rc), err, "c406_sdma_copy")

    def sdma_info(self) -> dict[str, int]:
        world = ctypes.c_int()
        streams = ctypes.c_int()
        err = ctypes.create_string_buffer(4096)
        with self._lock:
            rc = self.lib.c406_sdma_info(
                ctypes.byref(world), ctypes.byref(streams), err,
                ctypes.sizeof(err))
            self._check(int(rc), err, "c406_sdma_info")
        return {"world": int(world.value), "peer_streams": int(streams.value)}

    def shutdown_sdma(self) -> None:
        err = ctypes.create_string_buffer(4096)
        with self._lock:
            rc = self.lib.c406_sdma_shutdown(err, ctypes.sizeof(err))
            self._check(int(rc), err, "c406_sdma_shutdown")

    def kernel_info(self) -> dict[str, Any]:
        ints = [ctypes.c_int() for _ in range(7)]
        err = ctypes.create_string_buffer(4096)
        rc = self.lib.c406_kernel_info(
            *[ctypes.byref(x) for x in ints], err, ctypes.sizeof(err))
        self._check(int(rc), err, "c406_kernel_info")
        keys = ("num_regs", "sgpr_count", "lds_static_bytes",
                "dynamic_lds_bytes", "occupancy_blocks_per_cu",
                "threads", "comm_waves")
        return dict(zip(keys, [int(x.value) for x in ints]))


_DRIVER: C406IpcDriver | None = None


def get_driver() -> C406IpcDriver:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = C406IpcDriver()
    return _DRIVER


class C406Views:
    """One exported base view per receive buffer, opened on every peer."""

    NAMES = ("recv_hidden", "recv_meta", "recv_back")

    def __init__(self, tensors: dict[str, torch.Tensor], *, rank: int,
                 world_size: int) -> None:
        if not (1 <= world_size <= MAX_WORLD):
            raise ValueError(f"world_size={world_size} outside IPC limit")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.driver = get_driver()
        self.local: dict[str, dict[str, Any]] = {}
        for name in self.NAMES:
            tensor = tensors[name]
            view = self.driver.get_view(tensor)
            view.update({"tensor": tensor, "name": name})
            self.local[name] = view
            print(
                f"V407_IPC_VIEW rank={self.rank} name={name} "
                f"source={view['handle_source']} base_offset={view['base_offset']} "
                f"tensor_bytes={view['tensor_bytes']} "
                f"storage_bytes={view['storage_bytes']} "
                f"range_bytes={view['range_bytes']}", flush=True)
        wire_local = {
            name: {
                "handle": bytes(self.local[name]["handle"]),
                "base_addr": int(self.local[name]["base_addr"]),
                "base_offset": int(self.local[name]["base_offset"]),
                "range_bytes": int(self.local[name]["range_bytes"]),
                "tensor_bytes": int(self.local[name]["tensor_bytes"]),
                "storage_bytes": int(self.local[name]["storage_bytes"]),
                "handle_source": int(self.local[name]["handle_source"]),
            }
            for name in self.NAMES
        }
        gathered: list[dict[str, dict[str, Any]] | None] = [None] * world_size
        # Never pickle CUDA tensors into all_gather_object; only IPC metadata
        # crosses the control plane.
        dist.all_gather_object(gathered, wire_local)
        if any(x is None for x in gathered):
            raise RuntimeError("missing IPC metadata from a rank")
        self.meta: list[dict[str, dict[str, Any]]] = [
            {k: dict(v) for k, v in x.items()} for x in gathered if x is not None]
        if len(self.meta) != world_size:
            raise RuntimeError("IPC metadata rank count mismatch")
        self.aliases: dict[str, dict[int, int]] = {name: {} for name in self.NAMES}
        for name in self.NAMES:
            for peer in range(world_size):
                if peer == self.rank:
                    self.aliases[name][peer] = int(self.local[name]["base_addr"])
                else:
                    self.aliases[name][peer] = self.driver.open_view(
                        self.meta[peer][name]["handle"])
        dist.barrier()

    def pointer(self, name: str, peer: int, offset: int = 0) -> int:
        base = self.aliases[name][peer]
        base_off = int(self.meta[peer][name]["base_offset"])
        limit = int(self.meta[peer][name]["tensor_bytes"])
        if offset < 0 or offset > limit:
            raise ValueError(f"{name} peer {peer} offset {offset} outside {limit}")
        return base + base_off + int(offset)

    def close(self) -> None:
        for name in self.NAMES:
            for peer, alias in self.aliases[name].items():
                if peer != self.rank and alias:
                    self.driver.close_view(alias)
            self.aliases[name].clear()
        dist.barrier()

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "dacsm-amdladb-c406-ipc/1",
            "rank": self.rank,
            "world_size": self.world_size,
            "driver_lib": str(LIB_PATH),
            "driver_lib_sha256": self.driver.lib_sha256,
            "buffers": {
                name: {
                    "handle_sha256": hashlib.sha256(
                        self.local[name]["handle"]).hexdigest(),
                    "base_offset": int(self.local[name]["base_offset"]),
                    "range_bytes": int(self.local[name]["range_bytes"]),
                    "tensor_bytes": int(self.local[name]["tensor_bytes"]),
                    "storage_bytes": int(self.local[name]["storage_bytes"]),
                    "handle_source": int(self.local[name]["handle_source"]),
                }
                for name in self.NAMES
            },
            "kernel": self.driver.kernel_info(),
            "sdma_transport": {
                "implementation": "V401/C2 hipMemcpyPeerAsync",
                "runtime": self.driver.sdma_info(),
            },
        }


class C406Executor:
    def __init__(self, ctx: dict, views: C406Views) -> None:
        self.ctx = ctx
        self.views = views
        self.rank = int(ctx["rank"])
        self.world = int(ctx["world_size"])
        self.hidden = int(ctx["hidden"])
        self.hidden_bytes = self.hidden * 2
        self.all_send_ct = [list(map(int, x)) for x in ctx["all_send_ct"]]
        self.all_send_cm = [list(map(int, x)) for x in ctx["all_send_cm"]]
        self.launches = 0
        self.last_stage: dict[str, Any] | None = None

    def _launch_stage(self, stage: str, src_tensor: torch.Tensor,
                      dst_name: str, source_prefix: list[int], bytes_: list[int],
                      dst_offsets: list[int], *, vector16: bool) -> None:
        src_base = int(src_tensor.data_ptr())
        src_ptrs = [src_base + source_prefix[d] for d in range(self.world)]
        dst_ptrs = [self.views.pointer(dst_name, d, dst_offsets[d])
                    for d in range(self.world)]
        self.views.driver.launch(
            src_ptrs, dst_ptrs, bytes_, vector16=vector16,
            blocks=int(self.ctx.get("persistent_blocks", 80)))
        # V406 used host-visible completion plus a rank barrier between stages.
        # Keeping it inside the event window makes the measured communication
        # interval include the synchronization the direct-store path requires.
        torch.cuda.synchronize()
        dist.barrier()
        self.launches += 1
        self.last_stage = {
            "stage": stage,
            "world": self.world,
            "bytes": list(map(int, bytes_)),
            "source_prefix_bytes": list(map(int, source_prefix)),
            "destination_offset_bytes": list(map(int, dst_offsets)),
            "vector16": bool(vector16),
            "launch_index": self.launches,
        }

    def launch_hidden(self) -> None:
        if self.ctx["comm_arm"] != "c406":
            raise RuntimeError("C406 executor called outside c406 arm")
        geometry = stage_geometry(
            "hidden", self.all_send_ct, rank=self.rank, world=self.world,
            element_bytes=self.hidden_bytes)
        self._launch_stage("hidden", self.ctx["send_hidden"], "recv_hidden",
                           geometry["source_prefix_bytes"],
                           geometry["count_bytes"],
                           geometry["destination_offset_bytes"], vector16=True)

    def launch_meta(self) -> None:
        if self.ctx["comm_arm"] != "c406":
            raise RuntimeError("C406 executor called outside c406 arm")
        geometry = stage_geometry(
            "meta", self.all_send_cm, rank=self.rank, world=self.world,
            element_bytes=8)
        self._launch_stage("meta", self.ctx["send_meta"], "recv_meta",
                           geometry["source_prefix_bytes"],
                           geometry["count_bytes"],
                           geometry["destination_offset_bytes"], vector16=False)

    def launch_combine(self) -> None:
        if self.ctx["comm_arm"] != "c406":
            raise RuntimeError("C406 executor called outside c406 arm")
        geometry = stage_geometry(
            "combine", self.all_send_ct, rank=self.rank, world=self.world,
            element_bytes=self.hidden_bytes)
        self._launch_stage("combine", self.ctx["partial_bf16"], "recv_back",
                           geometry["source_prefix_bytes"],
                           geometry["count_bytes"],
                           geometry["destination_offset_bytes"], vector16=True)


class SdmaExecutor:
    """C2 transport on the same receive-buffer IPC views as C406.

    Each destination peer gets one persistent non-blocking stream.  The caller's
    current stream is bridged with an event so the SDMA engine sees all work that
    produced the source buffer.  The transfer is complete before this method
    returns, matching C2's timed-stage synchronization.
    """

    def __init__(self, ctx: dict, views: C406Views) -> None:
        self.ctx = ctx
        self.views = views
        self.rank = int(ctx["rank"])
        self.world = int(ctx["world_size"])
        self.hidden = int(ctx["hidden"])
        self.hidden_bytes = self.hidden * 2
        self.all_send_ct = [list(map(int, x)) for x in ctx["all_send_ct"]]
        self.all_send_cm = [list(map(int, x)) for x in ctx["all_send_cm"]]
        self.launches = 0
        self.last_stage: dict[str, Any] | None = None
        self.views.driver.init_sdma(self.world)

    def _launch_stage(self, stage: str, src_tensor: torch.Tensor,
                      dst_name: str, source_prefix: list[int], bytes_: list[int],
                      dst_offsets: list[int]) -> None:
        if self.ctx["comm_arm"] != "sdma":
            raise RuntimeError("SDMA executor called outside sdma arm")
        src_base = int(src_tensor.data_ptr())
        src_ptrs = [src_base + source_prefix[d] for d in range(self.world)]
        dst_ptrs = [self.views.pointer(dst_name, d, dst_offsets[d])
                    for d in range(self.world)]
        self.views.driver.sdma_copy(
            src_ptrs, dst_ptrs, bytes_, rank=self.rank, world=self.world)
        torch.cuda.synchronize()
        dist.barrier()
        self.launches += 1
        self.last_stage = {
            "stage": stage,
            "world": self.world,
            "transport": "V401/C2 hipMemcpyPeerAsync",
            "bytes": list(map(int, bytes_)),
            "source_prefix_bytes": list(map(int, source_prefix)),
            "destination_offset_bytes": list(map(int, dst_offsets)),
            "launch_index": self.launches,
        }

    def launch_hidden(self) -> None:
        geometry = stage_geometry(
            "hidden", self.all_send_ct, rank=self.rank, world=self.world,
            element_bytes=self.hidden_bytes)
        self._launch_stage(
            "hidden", self.ctx["send_hidden"], "recv_hidden",
            geometry["source_prefix_bytes"], geometry["count_bytes"],
            geometry["destination_offset_bytes"])

    def launch_meta(self) -> None:
        geometry = stage_geometry(
            "meta", self.all_send_cm, rank=self.rank, world=self.world,
            element_bytes=8)
        self._launch_stage(
            "meta", self.ctx["send_meta"], "recv_meta",
            geometry["source_prefix_bytes"], geometry["count_bytes"],
            geometry["destination_offset_bytes"])

    def launch_combine(self) -> None:
        geometry = stage_geometry(
            "combine", self.all_send_ct, rank=self.rank, world=self.world,
            element_bytes=self.hidden_bytes)
        self._launch_stage(
            "combine", self.ctx["partial_bf16"], "recv_back",
            geometry["source_prefix_bytes"], geometry["count_bytes"],
            geometry["destination_offset_bytes"])


def provenance() -> dict[str, Any]:
    d = get_driver()
    return {
        "status": "V407_C406_VS_C2_SDMA_WITH_UNCHANGED_AITER_COMPUTE_CO_DIAGNOSTIC",
        "driver_lib": str(LIB_PATH),
        "driver_lib_sha256": d.lib_sha256,
        "kernel": d.kernel_info(),
        "sdma_transport": d.sdma_info(),
        "source": (
            "C406 communication kernel from v406/audit/c3_wave_driver.cpp; "
            "SDMA arm uses v401/C2 hipMemcpyPeerAsync with one stream per peer"),
    }
