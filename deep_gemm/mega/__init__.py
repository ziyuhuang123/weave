import os
import torch
import types
import warnings
from typing import Tuple, Optional, Union
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C
from . import precision as _precision
from . import autotune as _autotune
from .autotune import autotune_moe, get_runtime_selector as weave_autotune_selector  # noqa: F401


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: Optional[str] = None,
                 activation: str = 'swiglu',
                 base: Optional['SymmBuffer'] = None):
        # Precision: explicit `mma_type` > WEAVE_PRECISION_MODE > 'fp8xfp4' (`None` / 'auto' =
        # not specified), validated against the current GPU before anything is allocated
        mma_type = _precision.resolve_symm_buffer_mma_type(mma_type, _precision.current_gpu_arch())

        # Align token count
        num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe(mma_type))

        # Init
        assert activation in ('swiglu', 'situ'), f'Only `swiglu` and `situ` activations are supported, got `{activation}`'
        assert activation != 'situ' or mma_type == 'fp8xfp4', '`situ` activation requires `fp8xfp4` MMA'
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation
        if mma_type == 'fp4xfp4':
            self._nvfp4_layout_key = _nvfp4_layout_key(self)
        if base is not None:
            if not isinstance(base, SymmBuffer):
                raise ValueError('Main MegaMoE base reuse requires a SymmBuffer')
            if (mma_type == 'fp4xfp4') != (base.mma_type == 'fp4xfp4'):
                raise ValueError('Cannot reuse a symmetric buffer across NVFP4 and main MegaMoE protocols')
            if mma_type == 'fp4xfp4' and self._nvfp4_layout_key != base._nvfp4_layout_key:
                raise ValueError('NVFP4 base reuse requires an identical buffer configuration')

        # Allocate or reuse a symmetric buffer
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            mma_type, activation,
            num_shared_experts
        )
        if base is None:
            allocator = torch if group.size() == 1 else symm_mem
            self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
            self.handle = (
                types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
                if group.size() == 1
                else symm_mem.rendezvous(self.buffer, group=group)
            )
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            assert base.buffer is not None and base.handle is not None and base.group is group, \
                'Cannot reuse an invalid symmetric buffer'
            assert num_bytes <= base.buffer.nbytes, \
                (f'The reused Mega MoE config requires {num_bytes} bytes, '
                 f'but the symmetric buffer only has {base.buffer.nbytes} bytes')
            self.buffer = base.buffer
            self.handle = base.handle

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf,
         self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None


# TODO: remove this function
def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 num_shared_experts: int = 0,
                                 use_fp8_dispatch: Union[bool, None] = None,
                                 mma_type: Optional[str] = None,
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Resolve first so the legacy `use_fp8_dispatch` check sees the effective precision
    mma_type = _precision.resolve_symm_buffer_mma_type(mma_type, _precision.current_gpu_arch())

    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split('x')[0] == 'fp8')
        warnings.warn(
            f'`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`',
            DeprecationWarning, stacklevel=3
        )

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts,
        mma_type=mma_type, activation=activation
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    # Unsqueeze for 2D
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    # Transpose
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(torch.stack([gate, up], dim=2).reshape(g, n, *rest))
    return result.squeeze(0) if squeeze_group_dim else result


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    # Unsqueeze for 2D
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    # Transpose
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = 'swiglu'
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
           Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    assert activation in ('swiglu', 'situ'), f'Only `swiglu` and `situ` activations are supported, got `{activation}`'
    if isinstance(l1_weights, tuple):
        # FP8/FP4: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = _interleave_weights(l1_weights[0])
        l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = _interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed


def _nvfp4_layout_key(buffer):
    return (buffer.group.size(), buffer.num_experts, buffer.num_max_tokens_per_rank,
            buffer.num_topk, buffer.hidden, buffer.intermediate_hidden,
            buffer.num_shared_experts, buffer.mma_type, buffer.activation, _C.get_num_sms())


def _validate_weight_pair(name, pair):
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise TypeError(f'{name} must be a (weight, scale-factor) pair')
    if not all(isinstance(t, torch.Tensor) for t in pair):
        raise TypeError(f'{name} weight and scale factor must both be tensors')
    return pair


def _validate_shared_weights(buffer, l1, l2, scaled=False):
    if (l1 is None) != (l2 is None):
        raise ValueError('Shared L1 and L2 weights must be provided together')
    if l1 is None:
        count = 0
    else:
        if scaled:
            l1, _ = _validate_weight_pair('shared_l1_weights', l1)
            l2, _ = _validate_weight_pair('shared_l2_weights', l2)
        if not isinstance(l1, torch.Tensor) or not isinstance(l2, torch.Tensor):
            raise TypeError('Shared L1 and L2 weights must be tensors')
        if l1.dim() != 2 or l2.dim() != 2:
            raise ValueError('Shared L1 and L2 weights must be rank-2 tensors')
        width = l2.size(1)
        if (width <= 0 or width % buffer.intermediate_hidden != 0 or
            l1.shape != (2 * width, buffer.hidden) or l2.size(0) != buffer.hidden):
            raise ValueError('Shared-expert weights do not match the symmetric buffer layout')
        count = width // buffer.intermediate_hidden
    if buffer.mma_type == 'fp4xfp4' and buffer.num_shared_experts != count:
        raise ValueError(
            f'Shared-expert layout mismatch: buffer={buffer.num_shared_experts}, call={count}')


def _validate_nvfp4_weights(buffer, l1, l2):
    experts = buffer.num_experts // buffer.group.size()
    for name, pair, n, k in (
        ('l1_weights', l1, 2 * buffer.intermediate_hidden, buffer.hidden),
        ('l2_weights', l2, buffer.hidden, buffer.intermediate_hidden),
    ):
        weight, sf = _validate_weight_pair(name, pair)
        if weight.dtype != torch.int8 or sf.dtype != torch.int32:
            raise TypeError(f'{name} requires packed INT8 weights and INT32-packed E4M3 scale factors')
        if weight.shape != (experts, n, k // 2) or sf.shape != (experts, n, k // 64):
            raise ValueError(f'{name} shapes do not match the NVFP4 symmetric buffer layout')
        if not weight.is_contiguous() or sf.stride() != (n * (k // 64), 1, n):
            raise ValueError(f'{name} requires contiguous weights and MN-major packed scale factors')
        if weight.device != buffer.buffer.device or sf.device != buffer.buffer.device:
            raise ValueError(f'{name} must be on the symmetric buffer device')


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     situ_beta: Optional[float] = None,
                     situ_linear_beta: Optional[float] = None):
    if not isinstance(sym_buffer, SymmBuffer):
        raise ValueError('Main MegaMoE requires a SymmBuffer')
    if sym_buffer.mma_type == 'fp4xfp4':
        raise ValueError('FP8/FP4 MegaMoE cannot use an NVFP4 symmetric buffer')
    _validate_shared_weights(sym_buffer, shared_l1_weights, shared_l2_weights, scaled=True)
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math, situ_beta, situ_linear_beta
    )

def bf16_mega_moe(y: torch.Tensor,
                  l1_weights: torch.Tensor,
                  l2_weights: torch.Tensor,
                  sym_buffer: SymmBuffer,
                  shared_l1_weights: Optional[torch.Tensor] = None,
                  shared_l2_weights: Optional[torch.Tensor] = None,
                  cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                  activation: str = 'swiglu',
                  activation_clamp: Optional[float] = None,
                  fast_math: bool = True):
    if not isinstance(sym_buffer, SymmBuffer):
        raise ValueError('Main MegaMoE requires a SymmBuffer')
    if sym_buffer.mma_type == 'fp4xfp4':
        raise ValueError('BF16 MegaMoE cannot use an NVFP4 symmetric buffer')
    _validate_shared_weights(sym_buffer, shared_l1_weights, shared_l2_weights)
    _C.bf16_mega_moe(
        y,
        l1_weights,
        l2_weights,
        shared_l1_weights,
        shared_l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        activation, activation_clamp,
        fast_math
    )


def fp4_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: Optional[torch.Tensor] = None,
                     shared_l2_weights: Optional[torch.Tensor] = None,
                     x_bf16: Optional[torch.Tensor] = None,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 16),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True,
                     l1_alphas: Optional[torch.Tensor] = None,
                     l2_alphas: Optional[torch.Tensor] = None,
                     a2_scales: Optional[torch.Tensor] = None,
                     routed_scaling_factor: float = 1.0):
    """Run NVFP4 routed experts, optionally fused with BF16 shared experts.

    NVFP4 operands use packed E2M1 values and per-16-element E4M3 scales.
    ``l1_alphas`` and ``l2_alphas`` carry optional per-expert model scales.
    The caller must fold the FC1 input global scale into ``l1_alphas``;
    ``a2_scales`` is separate because the FC2 input is produced and requantized
    inside this kernel; when provided, every entry must be finite and strictly
    positive. Shared weights and ``x_bf16`` must be provided together;
    ``routed_scaling_factor`` is applied before adding their BF16 output.

    CUDA Graph replay must reuse the captured ``x_bf16`` allocation and token
    count because both are encoded in the captured shared-input tensor map.
    """
    if not isinstance(sym_buffer, SymmBuffer) or sym_buffer.mma_type != 'fp4xfp4':
        raise ValueError(
            'NVFP4 MegaMoE requires an fp4xfp4 symmetric buffer, '
            f"got {getattr(sym_buffer, 'mma_type', None)!r}")
    if _nvfp4_layout_key(sym_buffer) != sym_buffer._nvfp4_layout_key:
        raise ValueError('NVFP4 buffer configuration or configured SM count has changed')
    if sym_buffer.activation != activation:
        raise ValueError(
            f'Activation mismatch: buffer={sym_buffer.activation!r}, call={activation!r}')
    _validate_nvfp4_weights(sym_buffer, l1_weights, l2_weights)
    if not isinstance(y, torch.Tensor) or y.dtype != torch.bfloat16:
        raise TypeError('NVFP4 output must be a BF16 tensor')
    if (y.dim() != 2 or y.size(1) != sym_buffer.hidden or
        y.size(0) > sym_buffer.num_max_tokens_per_rank or not y.is_contiguous() or
        y.device != sym_buffer.buffer.device):
        raise ValueError('NVFP4 output does not match the symmetric buffer layout')
    if not ((shared_l1_weights is None) == (shared_l2_weights is None) == (x_bf16 is None)):
        raise ValueError('Shared L1 weights, shared L2 weights, and x_bf16 must be provided together')
    _validate_shared_weights(sym_buffer, shared_l1_weights, shared_l2_weights)
    if x_bf16 is not None:
        for name, tensor in (('shared_l1_weights', shared_l1_weights),
                             ('shared_l2_weights', shared_l2_weights), ('x_bf16', x_bf16)):
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16:
                raise TypeError(f'{name} must be a BF16 tensor')
            if not tensor.is_contiguous() or tensor.device != y.device:
                raise ValueError(f'{name} must be contiguous and on the output device')
        if x_bf16.dim() != 2 or x_bf16.size(0) < y.size(0) or x_bf16.size(1) != sym_buffer.hidden:
            raise ValueError('x_bf16 must cover the output tokens and match the buffer hidden size')
    _C.fp4_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        shared_l1_weights, shared_l2_weights, x_bf16,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math,
        l1_alphas, l2_alphas, a2_scales, routed_scaling_factor
    )


class SM90SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu'):
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_sm90_mega_moe(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch, activation,
        )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        for name in (
            'x', 'x_sf', 'topk_idx', 'topk_weights',
            'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf',
        ):
            setattr(self, name, None)
        self.buffer = None
        self.group = None


def get_symm_buffer_for_sm90_mega_moe(group: dist.ProcessGroup,
                                      num_experts: int,
                                      num_max_tokens_per_rank: int, num_topk: int,
                                      hidden: int, intermediate_hidden: int,
                                      use_fp8_dispatch: bool = True,
                                      activation: str = 'swiglu') -> SM90SymmBuffer:
    num_max_tokens_per_rank = align(
        num_max_tokens_per_rank, _C.get_token_alignment_for_sm90_mega_moe())
    return SM90SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation,
    )


def transform_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """SM90 (Hopper) variant of `transform_weights_for_mega_moe`.

    SM90 has no TMEM / UTCCP path, so the SF tensors are consumed directly by
    WGMMA promote and don't need the 4x32 transpose. With block (128, 128)
    weight quantization, weight SFs are read by the math warpgroup directly
    from global memory in their natural ``(E, N/128, K/128)`` MN-major layout
    and require no transformation. Only L1's gate/up FP8 weight interleave is
    preserved.
    """
    def validate_pair(name, pair):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f'{name} must be a (FP8 weight, FP32 scale-factor) pair')
        weight, sf = pair
        if not isinstance(weight, torch.Tensor) or not isinstance(sf, torch.Tensor):
            raise TypeError(f'{name} weight and scale factor must both be tensors')
        if weight.ndim != 3 or sf.ndim != 3:
            raise ValueError(f'{name} weight and scale factor must both be rank-3 tensors')
        if weight.dtype != torch.float8_e4m3fn or sf.dtype != torch.float32:
            raise TypeError(f'{name} requires an FP8 E4M3 weight and an FP32 scale factor')
        if weight.device != sf.device:
            raise ValueError(f'{name} weight and scale factor must be on the same device')
        if not weight.is_contiguous() or not sf.is_contiguous():
            raise ValueError(f'{name} weight and scale factor must use contiguous natural layouts')
        num_experts, n, k = weight.shape
        if num_experts <= 0 or n <= 0 or k <= 0 or n % 128 != 0 or k % 128 != 0:
            raise ValueError(
                f'{name} expert count must be positive and N/K must be positive multiples of 128')
        expected_sf_shape = (num_experts, n // 128, k // 128)
        if tuple(sf.shape) != expected_sf_shape:
            raise ValueError(
                f'{name} scale-factor shape must be {expected_sf_shape}, got {tuple(sf.shape)}')
        return weight, sf

    l1_fp8, l1_sf = validate_pair('l1_weights', l1_weights)
    l2_fp8, l2_sf = validate_pair('l2_weights', l2_weights)
    if (l1_fp8.shape[0] != l2_fp8.shape[0] or
        l1_fp8.shape[1] != 2 * l2_fp8.shape[2] or
        l1_fp8.shape[2] != l2_fp8.shape[1]):
        raise ValueError(
            'SM90 MegaMoE weights must have shapes (E, 2*IH, H) and (E, H, IH)')
    if l2_fp8.shape[1] % 256 != 0:
        raise ValueError('SM90 MegaMoE hidden must be a multiple of 256 for combine vectorization')
    if l1_fp8.device != l2_fp8.device:
        raise ValueError('L1 and L2 SM90 MegaMoE weights must be on the same device')

    l1_transformed = (_interleave_weights(l1_fp8), l1_sf)
    l2_transformed = (l2_fp8, l2_sf)
    return l1_transformed, l2_transformed


def fp8_mega_moe(y: torch.Tensor,
                 l1_weights: Tuple[torch.Tensor, torch.Tensor],
                 l2_weights: Tuple[torch.Tensor, torch.Tensor],
                 sym_buffer: SM90SymmBuffer,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 recipe: Tuple[int, int, int] = (128, 128, 128),
                 activation: str = 'swiglu',
                 activation_clamp: Optional[float] = None,
                 fast_math: bool = True):
    """SM90 (Hopper) MegaMoE entry point.

    Expects FP8 e4m3 weights and block-(128, 128) float scale factors. The
    weight SF layout matches the convention used by ``DeepSeekV4FlashFp8`` /
    DeepEP, so the same SF tensors can be physically shared between the
    DeepEP path and this kernel.
    """
    if not isinstance(sym_buffer, SM90SymmBuffer):
        raise ValueError('SM90 FP8 MegaMoE requires an SM90SymmBuffer')
    _C.fp8_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )


# Every accepted WEAVE_PRECISION_MODE / mma_type spelling (superset of the historical tuple).
WEAVE_PRECISION_MODES = _precision.ACCEPTED_SPELLINGS


def _precision_mode_from_env() -> Optional[str]:
    """Canonical precision from WEAVE_PRECISION_MODE (then WEAVE_KNOB_PRECISION_MODE), or None."""
    found = _precision.precision_from_env()
    return None if found is None else found[0]


def _normalize_mma_type(mma_type: str) -> str:
    """Canonical precision name: 'fp8xfp4', 'fp4xfp4', 'fp8xfp8' or 'bf16'."""
    return _precision.canonical_precision(mma_type)


def _buffer_precision(sym_buffer) -> Optional[str]:
    """Backend precision a symmetric buffer was built for (None if unknown)."""
    if isinstance(sym_buffer, SM90SymmBuffer):
        return "fp8xfp8"
    if isinstance(sym_buffer, SymmBuffer):
        return _precision.symm_buffer_precision(sym_buffer.mma_type)
    return None


def weave_mega_moe(y: torch.Tensor,
                   l1_weights,
                   l2_weights,
                   sym_buffer,
                   mma_type: Optional[str] = None,
                   *,
                   precision_mode: Optional[str] = None,
                   routing_stats=None,
                   num_global_tokens: Optional[int] = None,
                   **kwargs):
    """Unified MegaMoE entry point for B300 / H20 / MI308X.

    ``mma_type`` is the single first-class precision switch (``WEAVE_PRECISION_MODE``
    codes in brackets; spellings are case-insensitive):

    =====================  =====================================================
    mma_type               dispatched kernel
    =====================  =====================================================
    ``fp8xfp4`` [0]        SM10x (SM100 / SM103) ``sm100_fp8_fp4_mega_moe.cuh``
    ``fp4xfp4`` [1]        SM10x NVFP4 ``sm100_fp4_fp4_mega_moe.cuh`` (alias ``nvfp4``)
    ``fp8xfp8`` [2]        H20 SM90 ``sm90_fp8_mega_moe.cuh`` (alias ``fp8``)
    ``bf16`` [3]           SM10x BF16 ``sm100_bf16_mega_moe.cuh`` (alias ``bf16xbf16``)
    =====================  =====================================================

    Precedence: ``mma_type`` / ``precision_mode`` (must agree when both are given)
    > ``WEAVE_PRECISION_MODE`` > the precision the symmetric buffer was built for.
    ``None`` and ``'auto'`` mean "not specified".  The resolved precision is
    checked against the GPU (SM10x: 0/1/3, SM90: 2) and against the buffer: a
    buffer built for another precision has a different layout, so a mismatch
    raises ``ValueError`` instead of launching a kernel on the wrong layout.
    All keyword arguments are forwarded to the selected backend unchanged, so
    the existing DeepGEMM functions remain 100% intact.

    Zero-config tuning (``deep_gemm/mega/autotune.py``): when no WEAVE_* tuning
    variable is set, the launch uses the calibrated profile written by
    ``python -m deep_gemm.mega.autotune`` for this GPU / EP size / MoE shape
    (bucket nearest to ``S_global = tokens * EP``, adapted by the four calibrated
    routing rules on eager calls), or architecture-family cold-start defaults
    when no profile exists.  Explicit WEAVE_* variables always win (manual
    mode); ``WEAVE_AUTOTUNE=0`` keeps the compiled defaults.  ``routing_stats``
    (an ``autotune.RoutingStats`` or its dict) skips the on-device statistics
    pass; ``num_global_tokens`` overrides ``tokens * EP`` when ranks hold
    different token counts (all ranks must pass the same value).
    """
    device = y.device if isinstance(y, torch.Tensor) else None
    mode = _precision.resolve_weave_precision(
        mma_type, precision_mode, buffer_precision=_buffer_precision(sym_buffer),
        arch=_precision.current_gpu_arch(device))
    if mode in ("fp8xfp4", "fp8xfp8") and isinstance(sym_buffer, (SymmBuffer, SM90SymmBuffer)):
        selection = _autotune.select_launch_env(sym_buffer, y, mode, routing_stats, num_global_tokens)
        with _autotune.scoped_env(selection.env):
            return _weave_dispatch(mode, y, l1_weights, l2_weights, sym_buffer, kwargs)
    return _weave_dispatch(mode, y, l1_weights, l2_weights, sym_buffer, kwargs)


def _weave_dispatch(mode, y, l1_weights, l2_weights, sym_buffer, kwargs):
    if mode == "fp8xfp4":
        if not isinstance(sym_buffer, SymmBuffer):
            raise ValueError("fp8xfp4 requires a SymmBuffer")
        return fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, **kwargs)
    if mode == "fp4xfp4":
        if not isinstance(sym_buffer, SymmBuffer) or sym_buffer.mma_type != "fp4xfp4":
            raise ValueError("fp4xfp4/NVFP4 requires an fp4xfp4 SymmBuffer")
        return fp4_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, **kwargs)
    if mode == "fp8xfp8":
        if not isinstance(sym_buffer, SM90SymmBuffer):
            raise ValueError("fp8xfp8 (SM90) requires an SM90SymmBuffer")
        return fp8_mega_moe(y, l1_weights, l2_weights, sym_buffer, **kwargs)
    if not isinstance(sym_buffer, SymmBuffer) or sym_buffer.mma_type != "bf16xbf16":
        raise ValueError("bf16 requires a bf16xbf16 SymmBuffer")
    return bf16_mega_moe(y, l1_weights, l2_weights, sym_buffer, **kwargs)


# ---------------------------------------------------------------------------
# Weave unified three-hardware policy API.
#
# These names are pure Python and do not require a compiled extension to be
# imported directly from deep_gemm.mega.cost_model.  When the package's native
# _C extension is available, the C++ kernels use the identical equations from
# deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh.
# ---------------------------------------------------------------------------
from .cost_model import (  # noqa: E402
    HARDWARE as WEAVE_HARDWARE,
    CANDIDATES as WEAVE_CANDIDATES,
    HardwareSpec as WeaveHardwareSpec,
    WorkloadSpec as WeaveWorkloadSpec,
    SpatialCandidate as WeaveSpatialCandidate,
    Breakdown as WeaveBreakdown,
    encode_r_mask as weave_encode_r_mask,
    evaluate as weave_evaluate_policy,
    select_policy as weave_select_policy,
)


# ---------------------------------------------------------------------------
# Unified Spatial / Temporal / Implementation knob space: 14 public knobs
# (Spatial 5, Temporal 3, Implementation 6).  The six retired knobs stay internal
# macros / diagnostics, WEAVE_ADAPT_GEMM_CONFIG / WEAVE_ADAPT_BM stay accepted but
# are not public (host shape-adaptive GEMM tuple selection is always on), and the
# two renamed GEMM1/GEMM2 knobs keep their legacy environment
# names as transparent aliases (see deep_gemm/mega/knobs.py).
# ---------------------------------------------------------------------------
from .knobs import (  # noqa: E402
    KNOBS as WEAVE_KNOBS,
    REQUIRED as WEAVE_KNOB_REQUIRED,
    CATEGORIES as WEAVE_KNOB_CATEGORIES,
    RETIRED_INTERNAL as WEAVE_KNOB_RETIRED_INTERNAL,
    LEGACY_ALIASES as WEAVE_KNOB_LEGACY_ALIASES,
    by_name as weave_knob_by_name,
    as_dicts as weave_knob_dicts,
)


def get_weave_knob_schema():
    """Return the public knob schema (name, category, values, mechanism) of the 14 public knobs."""
    try:
        return _C.weave_knobs.schema()
    except AttributeError:
        return weave_knob_dicts()


def get_weave_knob_names():
    """Return the public knob names in schema order."""
    return [entry["name"] for entry in get_weave_knob_schema()]


def get_weave_knob_defaults(mma_type: Optional[str] = "fp8xfp4"):
    """Return the compiled defaults for one backend/precision selection.

    Any WEAVE_PRECISION_MODE spelling is accepted; ``None`` / ``'auto'`` follow the
    environment, and ``'mi308x'`` selects the MI308X defaults.
    """
    try:
        return dict(_C.weave_knobs.defaults(_precision.knob_defaults_key(mma_type)))
    except AttributeError:
        return {k.name: k.default for k in WEAVE_KNOBS}
