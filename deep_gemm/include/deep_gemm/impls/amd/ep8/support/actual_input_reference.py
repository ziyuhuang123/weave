"""Independent natural-weight reference consuming the exact gathered inputs."""
from __future__ import annotations
import hashlib


def tensor_sha(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def quant_dequant(value):
    import torch
    maximum = float(torch.finfo(torch.float8_e4m3fnuz).max)
    rows,width = value.shape
    blocks = value.float().reshape(rows,width//128,128)
    amax = blocks.abs().amax(-1)
    scale = amax * (1.0/maximum)
    safe_scale = torch.where(scale == 0,torch.ones_like(scale),scale)
    quant = (blocks/safe_scale[...,None]).clamp(-maximum,maximum).to(torch.float8_e4m3fnuz)
    return (quant.float()*safe_scale[...,None]).reshape(rows,width),quant.reshape(rows,width),safe_scale


def dequant_weight(raw,scale,out_rows,in_cols):
    import torch
    return (raw.view(torch.float8_e4m3fnuz).float().reshape(
        out_rows//128,128,in_cols//128,128)*scale[:,None,:,None]).reshape(out_rows,in_cols)


def compute(*,sources,natural,rank,world,spec,device,rccl_group,save_route_stages=True):
    import torch
    import torch.distributed as dist
    assert len(sources) == world == 8
    hidden,inter = spec.hidden,spec.intermediate
    local_tokens = sources[0]['x_bf16'].shape[0]
    local_experts = spec.experts//world
    # FP32 matmul is the independent numerical reference, with explicit graph
    # BF16/FP8 boundaries. No CK-shuffled tensor is passed into this function.
    torch.backends.cuda.matmul.allow_tf32 = False
    x_rows,ids_rows,weight_rows,checks = [],[],[],[]
    for source,row in enumerate(sources):
        x = row['x_bf16']
        ids = row['topk_ids']
        weights = row['topk_weights'].to(torch.float16).to(torch.float32).contiguous()
        assert tuple(x.shape) == (local_tokens,hidden)
        assert tuple(ids.shape) == tuple(weights.shape) == (local_tokens,spec.topk)
        xq,_,_ = quant_dequant(x.to(device))
        x_rows.append(xq)
        ids_rows.append(ids.cpu().numpy())
        weight_rows.append(weights.to(device))
        checks.append({'source_rank':source,'activation_sha256':tensor_sha(x),
                       'topk_ids_sha256':tensor_sha(ids),'route_weights_fp16_rounded_sha256':tensor_sha(weights)})
    x_all = torch.stack(x_rows)
    partial = torch.zeros((world,local_tokens,hidden),dtype=torch.float32,device=device)
    route_stages = {}
    expert_checks = []
    for lid in range(local_experts):
        expert = rank*local_experts+lid
        locations = [(src,tok,slot) for src,ids in enumerate(ids_rows)
                     for tok in range(local_tokens) for slot in range(spec.topk)
                     if int(ids[tok,slot]) == expert]
        if not locations:
            continue
        src = torch.tensor([p[0] for p in locations],dtype=torch.long,device=device)
        tok = torch.tensor([p[1] for p in locations],dtype=torch.long,device=device)
        route_w = torch.stack([weight_rows[s][t,k] for s,t,k in locations])
        w13 = dequant_weight(natural['w13_u8'][lid],natural['sf13'][lid],2*inter,hidden)
        w2 = dequant_weight(natural['w2_u8'][lid],natural['sf2'][lid],hidden,inter)
        gate_up = x_all[src,tok] @ w13.T
        before_round = torch.nn.functional.silu(gate_up[:,:inter])*gate_up[:,inter:]
        # Raw GEMM1 writes BF16 before its existing per128 FP8 quantizer.
        g1_bf16 = before_round.to(torch.bfloat16)
        iq,fp8,scales = quant_dequant(g1_bf16)
        route_out = ((iq @ w2.T)*route_w[:,None]).to(torch.bfloat16)
        partial.reshape(-1,hidden).index_add_(0,src*local_tokens+tok,route_out.float())
        if save_route_stages:
            for i,(s,t,k) in enumerate(locations):
                route_stages[(s,t,k)] = {'global_expert':expert,
                    'g1_bf16':g1_bf16[i].cpu(),'quant_fp8_u8':fp8[i].view(torch.uint8).cpu(),
                    'quant_scale':scales[i].cpu(),'g2_bf16':route_out[i].cpu()}
        expert_checks.append({'local_expert':lid,'global_expert':expert,'routes':len(locations)})
        del w13,w2,gate_up,before_round,g1_bf16,iq,fp8,scales,route_out
    # The candidate emits a BF16 owner partial, then a BF16 home result.
    send = partial.to(torch.bfloat16).contiguous().reshape(-1)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv,send,group=rccl_group)
    expected = recv.reshape(world,local_tokens,hidden).float().sum(0).to(torch.bfloat16)
    meta = {'schema':'weave-m16-actual-input-independent-reference/1',
            'rank':rank,'world':world,'topk':spec.topk,'source_checks':checks,
            'route_source':'exact gathered prepared x_bf16/topk_ids/topk_weights; no histogram reconstruction',
            'weights':'independent natural-layout FP8 bytes and128x128 scales',
            'g1_boundary':'SiLU(gate)*up -> BF16 RNE -> per128 FNUZ FP8 quant/dequant',
            'route_weight_boundary':'FP16-rounded metadata -> FP32 multiply -> BF16 route output',
            'owner_home_boundary':'FP32 owner sum -> BF16 transport -> FP32 cross-owner sum -> BF16 final',
            'expert_checks':expert_checks,'expected_sha256_bf16':tensor_sha(expected)}
    del x_rows,x_all,partial,send,recv
    torch.cuda.synchronize(device)
    return expected,meta,route_stages
