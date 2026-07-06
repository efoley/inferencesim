# Speculative low-cost architectures: how far can you get without HBM?

This is a design study, not a product catalogue. The machines in
`presets_spec.py` (`lpddr-swarm-*`, `cxl-*`) exist to push one question through
the simulator: **how close to HBM-class serving can you get on commodity parts
-- LPDDR instead of HBM, a low-latency on-board fabric, commodity Ethernet
between boxes, and disaggregated CXL memory?**

All numbers below are `--efficiency auto` (vendor-appropriate derating), fp8
weights + fp8 KV. Prices and power splits are the softest inputs; every figure
is a best-effort approximation grounded in the JEDEC/CXL spec points cited in
`presets_spec.py`. Reproduce any row with the command under its table.

## The thesis: decode is a bandwidth-cost arbitrage

A decode step streams the weights (and the KV cache) from DRAM once per token,
so tokens/s tracks **aggregate DRAM bandwidth**, not FLOP/s. HBM buys that
bandwidth at a punishing $/(GB/s) and pJ/bit. LPDDR5X delivers ~273 GB/s per
package (the proven GB10/DGX-Spark figure over 256-bit LPDDR5X-8533) at perhaps
~1/10 the $/GB and a fraction of the energy. The bet: if a **low-latency**
fabric lets you aggregate many small LPDDR packages' bandwidth cheaply, you
reach HBM-class *aggregate* bandwidth on commodity silicon -- the interconnect
is the part you have to earn.

## Building blocks (no HBM, by design)

| tile | compute (fp8) | DRAM | bandwidth | FLOP:byte | ~power |
|---|---|---|---|---|---|
| `LPDDR5X tile` | 128 TF | 32 GB LPDDR5X | 273 GB/s | 469 | ~85 W |
| `LPDDR6 tile` | 192 TF | 48 GB LPDDR6 | 460 GB/s | 417 | ~108 W |
| `CXL compute tile` | 256 TF | 256 GB pooled DDR5 | 512 GB/s (8× CXL 3.0 x16) | 500 | ~160 W |

Compute is sized at ~450 FLOP/byte -- memory-leaning, like a serving GPU
(H100 ~590, GB10 ~915), not a compute-heavy Tenstorrent part -- so the tiles
are balanced for batched decode rather than wasted on FLOP they can't feed.
On-chip NoC/SRAM are sized above DRAM so the memory stack stays the min-cut
(asserted in `test_presets_spec.py`).

## LPDDR swarms vs the HBM incumbents

llama-3.1-70b, batch 64, prompt 2048, output 256:

| machine | chips | TPOT | output tok/s | decode ceiling | J/tok | $/M out |
|---|---|---|---|---|---|---|
| `gb300-nvl72` (HBM) | 72 | 5.9 ms | 43.5 k | 96.9 k | 1.43 | $0.70 |
| `dgx-h100` (HBM) | 8 | 9.9 ms | 2.58 k | 6.45 k | 2.04 | $1.01 |
| `tt-quietbox` (GDDR6) | 4 | 118 ms | 361 | 540 | 2.77 | $0.38 |
| **`lpddr-swarm-64`** | 64 | 38.6 ms | 1.43 k | 3.32 k | 3.25 | $0.94 |
| **`lpddr6-swarm-64`** | 64 | 24.4 ms | 2.14 k | 5.25 k | 2.62 | $0.76 |

```bash
inferencesim run --hardware lpddr6-swarm-64 --model llama-3.1-70b \
    --tp 32 --batch 64 --prompt 2048 --output 256 \
    --weight-dtype fp8 --kv-dtype fp8 --efficiency auto
```

**Reading it.** A single 64-tile LPDDR6 box (~$175k, ~5.6 kW) lands between a
DGX H100 and a full NVL72 rack on per-token latency and throughput, at cost/token
competitive with a DGX H100 -- on memory that never touches an HBM fab. The
LPDDR5X box is a generation behind (273 vs 460 GB/s/tile) and shows it. Neither
touches the NVL72's absolute throughput -- 72 HBM chips on NVLink is a different
weight class -- but that is $3.5M and 62 kW. The swarm's pitch is *tokens per
dollar of memory*, and there it is live.

The load-bearing assumption is the fabric: TP=32 keeps `comm` at 11-15% of TPOT
only because the on-board fabric is fat (200-256 GB/s/dir) and low-latency
(0.2 µs). On a slow fabric a swarm of wimpy chips drowns in allreduce -- which
is exactly why the box-to-box story below is deliberately different.

## Commodity Ethernet between boxes: scale out with DP, never TP/PP

The pod (`lpddr-swarm-pod`, 4 boxes × 64 tiles) joins boxes with **commodity
400 GbE RoCE** (50 GB/s/dir, ~5 µs) -- an order of magnitude slower and
higher-latency than the on-board fabric. The simulator makes the discipline
concrete (same llama-70b workload):

| pod config | replicas | output tok/s | $/M out | note |
|---|---|---|---|---|
| PP=4 across Ethernet | 1 | 1.73 k | $3.28 | pipeline stages on 50 GB/s -- **anti-pattern** |
| TP=32 in-box, DP=8 | 8 | 5.71 k | $1.00 | linear 4× over one box, same $/tok |
| TP=16 in-box, DP=16 | 16 | 7.95 k | $0.72 | smaller TP groups → less collective tax |

```bash
inferencesim run --hardware lpddr-swarm-pod --model llama-3.1-70b \
    --tp 16 --batch 64 --prompt 2048 --output 256 \
    --weight-dtype fp8 --kv-dtype fp8 --efficiency auto
```

**Finding.** Commodity Ethernet is entirely adequate box-to-box *if you only
ask it to carry independent data-parallel replicas* -- DP replicas don't
communicate during steady-state decode, so the slow link is idle where it
matters. Push a bandwidth-heavy tensor- or pipeline-parallel group across it and
throughput collapses (the PP=4 row). The right knob is TP inside the box, DP
across boxes; smaller in-box TP groups even improve $/tok by cutting the
collective tax. The Ethernet is for scale, the on-board fabric is for sharding.

## Disaggregated CXL memory: a capacity tier, not a bandwidth tier

The `cxl-*` machines serve compute from a shared pool of cheap CXL-attached
DDR5 (256 GB/tile, 512 GB/s over 8× CXL 3.0 x16). Capacity and bandwidth scale
independently; the pool is enormous and cheap.

| machine | model | fits? | mem/chip | output tok/s | $/M out |
|---|---|---|---|---|---|
| `cxl-moe-pod` (32 tiles, 8 TB pool) | deepseek-v3 671B | ✅ 28/256 GB | plenty | 595 | $4.58 |
| `lpddr-swarm-64` (64 tiles, 2 TB) | deepseek-v3 671B | ✅ 13/32 GB | tight | 513 | $2.61 |

```bash
inferencesim run --hardware cxl-moe-pod --model deepseek-v3 \
    --tp 8 --ep 4 --batch 128 --prompt 4096 --output 512 \
    --weight-dtype fp8 --kv-dtype fp8 --efficiency auto
```

**The honest result.** For *serving* -- which is bandwidth-bound -- CXL
disaggregation trades away exactly what decode needs. A 671B MoE already fits
across 64 distributed LPDDR tiles (5 GB of weights each), and those 64 tiles
carry **17.5 TB/s** of aggregate bandwidth versus the CXL pod's ~16 TB/s at
higher latency and higher cost/token. Distributing memory across many compute
tiles gives you capacity *and* bandwidth that scale together; pooling it behind
CXL links gives you capacity while the links cap the bandwidth.

Where CXL genuinely wins is the regime this study does *not* reward on
throughput: capacity you cannot buy by adding compute -- a very large KV/context
tier, or many models held resident behind few compute tiles, with memory and
compute provisioned on independent budgets.

**Modelling caveat (important).** This simulator streams weights from a single
`dram`, so the CXL tiles model the pool *as* that DRAM -- the CXL links are the
bandwidth min-cut by construction. A genuine two-tier "small fast local DRAM +
big slow pool" hierarchy (hot weights local, cold experts/KV pooled) is not
expressible in the single-DRAM path model, and it is precisely the design that
would let CXL keep its capacity edge without paying full CXL bandwidth on every
byte. Treat the `cxl-*` rows as the conservative, everything-from-the-pool
bound, not the ceiling of what disaggregation can do.

## Takeaways

1. **LPDDR swarms are real.** A no-HBM 64-tile LPDDR6 box serves 70B between a
   DGX H100 and an NVL72 on latency, at DGX-competitive $/tok -- the whole win
   is aggregate bandwidth bought cheaply, *contingent on a low-latency on-board
   fabric* to keep TP collectives from eating it.
2. **Commodity Ethernet scales boxes fine -- for DP only.** Keep sharding
   (TP/PP) on the fat in-box fabric; let the slow Ethernet carry independent
   replicas, where it is idle during decode.
3. **CXL is a capacity play, not a throughput play.** Under a single-tier
   memory model it loses to distributed LPDDR on serving throughput and cost;
   its edge is independent capacity scaling, which a two-tier model (future
   work) would be needed to reward.
