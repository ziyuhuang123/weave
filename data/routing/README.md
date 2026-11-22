# MoE Routing Statistics

Aggregated Mixture-of-Experts (MoE) routing statistics used by the Weave evaluation
and cost-model experiments. Every file here is a small, pure statistics artifact
(JSON or CSV) computed from captured router outputs. The route-capture tensors
themselves (`.pt`/`.bin`) are large binary files and are **not** included in this
repository; only the derived counts, histograms, placements and checksums are kept.

## Layout

```
data/routing/
  paper/                 paper-level expert-imbalance statistics
  ep8_eplb/              EP8 routing loads and expert-parallel load balancing
    ep8/summary.json
    eplb/summary.json
    eplb/eplb_vs_baseline_loads.csv
    eplb/redundancy_sweep.json
    eplb/placements/*.json
    datasets/{coig,dolly,metamath,sharegpt}/*_summary.json
    datasets/sharegpt/routing_per_expert_{summary.json,counts.csv}
  amd_ep16/              per-model EP16 routing statistics (AMD MI308X)
  amd_ep8_hist/          per-model EP8 routing histograms for the cost model
```

## Models and datasets

Model keys are `DeepSeek-V4-Flash`, `Qwen3.5-122B-A10B`, `Qwen3.5-35B-A3B`,
`Qwen3.5-397B-A17B`, `minimax_m3` and `qwen38max`. Dataset-conditioned captures use
the public instruction corpora `sharegpt`, `coig`, `dolly` and `metamath`.

## Files and fields

### `paper/`

- `me_hist.json` - object keyed by `<model>_S<tokens>`. Each value holds `bins`
  (16 log-scale buckets of routes per expert, from `0` to `>16384`) and `counts`
  (number of experts falling in each bucket).
- `me_stats.csv` - one row per (model, sequence length): mean/median/max routes per
  expert, coefficient of variation, zero-route fraction, expert-count and
  token-mass fractions over route-count bands, and per-rank load imbalance ratios
  under 8-way expert parallelism.
- `per_rank_stats.json` - list of 24 records, one per (model, `S`) with
  `S` in {128, 1024, 4096, 20000}. Fields: `stem` (model), `S`, `E` (experts),
  `K` (top-k), `H` (hidden size), `I` (intermediate size), `recv_dedup` and
  `recv_pairs` (per-rank received token volumes before/after deduplication),
  `touched` (distinct peers per rank), `dedup_factor`, and max-over-mean ratios.

### `ep8_eplb/`

- `ep8/summary.json` - baseline EP8 routing: 7 captures keyed
  `<model>_S<tokens>`, each with `E`, `K`, `EP`, `tokens`, `tokens_per_rank`,
  `experts_per_rank`, `routes_per_rank` (8 values) and `rank_balance_ratio`
  (max over mean rank load).
- `eplb/summary.json` - 22 captures with redundant-expert replication. In addition
  to the baseline fields it records `physical_experts`, `redundant_experts`,
  `physical_experts_per_rank`, `baseline_routes_per_rank`,
  `baseline_rank_balance_ratio`, `max_replicas_per_expert` and
  `experts_with_replicas`.
- `eplb/eplb_vs_baseline_loads.csv` - per (capture, rank) columns `baseline_routes`
  and `eplb_routes`.
- `eplb/redundancy_sweep.json` - 22 captures, each mapping the number of added
  redundant experts (`+0`, `+8`, `+16`, `+32`, `+64`) to the resulting rank-balance
  ratio.
- `eplb/placements/*.json` - one placement per capture: `model`, `E`, `EP`,
  `num_physical`, `num_redundant`, `physical_per_rank`, `phy2log` (logical expert
  served by each physical slot), `logcnt` (replica count per logical expert) and
  `rank_slots` (`EP` x `physical_per_rank` physical ids per rank).
- `datasets/<corpus>/<model>_summary.json` - per-dataset capture summary: `model`,
  `arch`, `E`, `K`, `EP`, `tokens_global_S`, `tokens_per_rank`, `routes_total`,
  `routes_per_dst_rank` (8 values), `rank_balance_ratio`, `expert_route_min`,
  `expert_route_max`, `expert_route_mean`, `nonzero_experts`, plus free-text
  `source` and `route_note` provenance notes.
- `datasets/sharegpt/routing_per_expert_summary.json` - per-sequence-length,
  per-expert route-count summary for the ShareGPT captures.
- `datasets/sharegpt/routing_per_expert_counts.csv` - long-format counts with
  columns `model`, `seqlen`, `expert_id`, `ep8_rank`, `tokens_routed`; covers 20
  model/dataset variants over sequence lengths from 16 to 262144.

### `amd_ep16/`

One `routing_ep16_<model>.stats.json` per model for the 16-rank EP16 AMD MI308X
MegaMoE experiments. Fields: `model`, `E`, `TOPK`, `el` (local experts per rank),
`R` (ranks), `T_TOK`, `bin` (capture file name), `record_bytes`, `records`,
`requires_harness_TOPK`, `routes_per_rank` (16 values), `routes_per_rank_min`,
`routes_per_rank_max`, `routes_per_rank_mean`, `routes_per_rank_max_min_ratio`,
`rows_conserved`, `dup_topk`, `bad_items` and `check`.

### `amd_ep8_hist/`

One `routing_ep8_<model>_S<tokens>.json` per model and token count for the EP8 AMD
MI308X cost model. Fields: `schema`, `model`, `S`, `R`, `T_per_rank`, `E`, `K`,
`el_ep8`, `src_lineage`, `selfcheck` (`bad_id`, `dup_topk`, `bad_w_full`,
`rows_total`), `rows_per_dst` (`min`, `mean`, `max`, `max/min`), `tiles_per_dst_g0`
(`s1_min`, `s1_max`, `s2_min`, `s2_max`, `gemm_workers`),
`max_src_share_per_dst` (8 values) and `worst_max_src_share`.

## Provenance and sanitization

- All values were computed from router outputs of the listed models on the listed
  public instruction corpora. Dataset summaries record the sampling recipe and
  routing function in their `source` and `route_note` fields.
- Two metadata fields were rewritten for this public release. No numeric value was
  altered: for every touched file, all integer and floating-point leaves were
  compared before and after the edit (identical Python `repr`, identical numeric
  literal count) and the only structural differences are the two fields below.
  - `amd_ep8_hist/*.json`, field `src_lineage`: an internal free-text lineage note
    was replaced with
    `EP8 prefix (ranks 0-7) derived from real ShareGPT token routing traces`.
  - `amd_ep16/*.json`, field `bin`: an absolute capture path was reduced to the
    capture file name.
- All files are ASCII-only UTF-8, contain no internal host paths, and no
  credentials. Binary route captures (`.pt`/`.bin`), raw dataset JSONL files and
  tokenizer/weight shards are intentionally excluded.