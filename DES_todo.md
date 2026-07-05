# Discrete-event engine — remaining plan

Status of the DES as of this PR, and the roadmap to take it from
pipeline-stage granularity down to the expanded chip graph.

## What exists today

`inferencesim/des.py` — `DESEngine` simulates a real task graph instead of
summing op times analytically:

- **Tasks**: one `_Task` per (round, microbatch, pipeline stage, layer),
  plus collective and p2p-hop tasks, each with explicit `deps`.
- **Resources** (FIFO, one server each): `u{s}` stage execution unit,
  `c{s}` stage collective fabric, `h{s}` stage outbound p2p link.
- **Scheduler**: `schedule()` — deterministic list scheduling; a task starts
  when its deps are done and its resource is free. Returns finish times.
- **Measurement**: `_decode_wall` runs `decode_rounds` pipeline rounds and
  reports the mean steady-state round period after `warmup`; `_prefill_wall`
  walks the stages once (single request).
- **Service times**: reuse the roofline `time_op` math, so any divergence
  between engines is pure scheduling/contention, never unit costs.
- **Output**: `Phase.wall_time` carries the measured duration through the
  existing power/cost pipeline unchanged.

What's emergent (not assumed): pipeline microbatch overlap, the real cost of
unbalanced stages (`n_layers % pp != 0`), LM-head/hop overlap, serial
prefill fill/drain.

Also already built and waiting to be consumed:

- `Graph.expand()` — materialises counted groups (`sram` ×35 →
  `sram[0..34]`) and grouped edges into concrete per-instance links.
- `Graph.max_flow()` — aggregate bandwidth crediting parallel routes,
  invariant under `expand()`.
- Edge `pattern`s (`interleave`/`all`) and selectors (`sram[0:8]`).

## The gap

The DES resources are still *lumped abstractions* (`u{s}` = "a stage's tp
chips' compute+DRAM as one server"). Service times come from the roofline
`max(flops/peak, bytes/bw)` per op. So within a stage there is no
contention: DRAM-bank conflicts, NoC-hop sharing, SRAM-bank pressure, and
compute/DRAM overlap are all still analytic. The point of the hierarchical
graph + `expand()` was to let the DES walk the *real* nodes and edges — that
connection is not wired yet.

## Roadmap

### 1. Walk the expanded chip graph (the headline item)

Drive per-chip work against `chip_graph.expand()` resources instead of one
`u{s}` server.

- [x] **Resource model from graph**: `graphdes.ChipModel` expands the chip
      graph and turns MEMORY/COMPUTE instances into FIFO resources, SWITCH
      nodes into shared (processor-sharing) resources, and concrete edges
      into FIFO link resources, driven by the same `sched.schedule()` core.
- [x] **Lower an `Op` to a sub-task chain over the path**: `op_wall` lowers a
      COMPUTE op to per-tile read → NoC hop(s) → SRAM → compute → write-back
      task chains (store-and-forward, one task per constrained element). Tile
      size is the `--tile-fill` knob against per-core SRAM capacity (now
      enforced: it sets the tile count). Byte count sizes *memory* tiles only;
      a compute-bearing op makes at least one tile per core so FLOPs spread
      over the whole pool (roofline-consistent). Modelling an op that is
      genuinely too serial to fill the chip (e.g. decode attention with few
      heads) is future work: it needs op-structure metadata in `ops.py`
      (heads, query blocks), not derivable from byte counts.
- [x] **Bank/port assignment**: tiles round-robin over banks and cores using
      the interleave convention `expand()` already wires (tile `i` → bank
      `i % n_banks`, core `i % n_cores`), so bank conflicts are modelled, not
      averaged. (Explicit address→bank *hashing* knob still open — only
      round-robin is implemented.)
- [x] **Compute/DRAM overlap becomes emergent**: reads and math sit on
      separate resources with double buffering (`1/tile_fill` buffers), so
      overlap falls out of the schedule; tested `wall < mem_t + compute_t`.
- [x] **Validation**: degenerate tests in `tests/test_graphdes.py` — single
      bank → `bytes/bw`, single core → `flops/rate`, two banks halve the
      wall, shared NoC caps throughput, mixed ops overlap — reproduce the
      closed forms exactly, and the engine-level graph-DES is a never-optimistic
      refinement of the lumped DES (`tests/test_des.py`).

### 2. Contention & queueing fidelity

- [x] **k-server resources**: `Resource(servers=k)` in `sched.py` serves k
      tasks concurrently from a k-slot pool (k=1 reproduces the old single
      `free[resource]` behaviour exactly). The stage-level engine still
      declares only 1-server FIFO resources until the expanded-graph PR
      consumes the new mode.
- [x] **Link duplexing & sharing**: `Resource(shared=True)` in `sched.py`
      models processor sharing (N concurrent flows each get bw/N; a single
      flow is identical to FIFO). Not yet wired into stage links — the
      expanded-graph PR will map bandwidth-shared edges onto it.
- [x] **Collective internals**: `collectives.py` expands each collective into
      its per-step link transfers on per-member outbound-link resources, named
      for the fabric they egress onto (`s{s}.l{i}.out` egress port on a
      switched fabric, `.cw`/`.ccw` cables on a ring), which also carry the
      pipeline hops (member 0) -- so collective/hop and collective/collective
      contention emerges. Link resources carry only *bandwidth occupancy*
      (`bytes/bw`); propagation *latency* rides the dependency chain (barrier /
      propagation tasks, unique per collective instance / message so concurrent
      instances never falsely serialise their flight times), never a link. So
      each expansion reproduces its closed form exactly in isolation and
      diverges only under genuine bandwidth contention. Ring allreduce = 2(g-1)
      barrier-separated steps (bandwidth-optimal on RING and ALL_TO_ALL alike);
      MoE all-to-all is g-1 per-member messages on a switched fabric
      (occupancies serialise on the egress port, one flight time on the exit
      barrier), or shortest-way store-and-forward routing on a RING; MESH_2D
      falls back to the closed form (no preset uses it). `engine.ring_allreduce_time`
      and the switched all-to-all closed form are the oracles (exact, rel=1e-9);
      a hand-computed g=4 case oracles the routed ring all-to-all. Also fixed a
      bridge bug where interconnect topology (now load-bearing) was written to
      the fabric switch's meta but read from the chip composite's, so a RING
      system silently round-tripped to ALL_TO_ALL; the switch node's meta is now
      its single home.
    - Ingress/incast is unmodeled: each member has an *egress* resource, but
      simultaneous *arrivals* at a member are unbounded -- harmless for uniform
      collectives (per-member ingress load equals egress load), but a hot-expert
      MoE all-to-all would incast onto the busy owners and needs a per-member
      ingress resource before that imbalance is faithful.

### 3. Heterogeneity (lift the homogeneous-chip restriction)

- [ ] `bridge.system_from_graph` currently rejects >1 distinct chip
      composite. The DES has no such need — let it consume a graph with
      mixed chips / cards directly (e.g. a Grace CPU node + GPU nodes, or
      hot vs harvested dies).
- [ ] **Per-instance heterogeneity**: MoE hot-expert imbalance, one
      throttled chip, a harvested 132-of-140-core die — expressible via
      selectors + `disabled`/derated instance attributes (needs a small
      node-attribute addition).

### 4. Whole-system dynamics (beyond one phase in isolation)

- [ ] **Continuous batching**: simulate prefill and decode *interleaved* on
      the same resources (today they're timed as separate phases and
      combined analytically in `simulate.py`). Captures prefill/decode
      interference, chunked prefill, and admission effects.
- [ ] **Request-level arrival process**: Poisson/trace-driven arrivals →
      real queueing latency (p50/p99 TTFT & TPOT), not just steady-state
      averages. The scheduler already produces per-task timelines to
      histogram.
- [ ] **KV-cache growth over a request**: attention cost currently uses
      mean context; step it per token so long-sequence tails show up.

### 5. Engine plumbing & ergonomics

- [x] **Convergence control**: `DESEngine(decode_rounds=None)` (the new
      default) auto-grows the round count -- starting at `max(8, 2*pp)`,
      doubling and rebuilding the graph -- until two successive round-period
      estimates agree within `rtol` (default 1e-3) or `max_rounds` (default
      256) is hit; the outcome (rounds, converged, rel delta) lands on
      `engine.last_convergence`.  Passing explicit `decode_rounds`/`warmup`
      still pins the old fixed run byte-for-byte (`warmup` defaults to
      `decode_rounds // 2`).
- [x] **Event-driven core**: moved to `sched.py`, a proper event loop with
      a ready heap plus lazy-invalidated (epoch-tagged) departure events for
      shared resources. All-FIFO graphs still schedule identically to the
      old list scheduler (the interface is unchanged for the stage engine).
- [x] **Timeline export**: `chrome_trace()` emits per-resource task
      timelines (one process per resource, tasks packed into non-overlapping
      tid lanes); `inferencesim run --engine des --trace out.json` writes
      both phases to a Perfetto-loadable file.
- [x] **`Report` surface**: `Phase.resource_busy`/`resource_span` and
      `Report.resource_util` carry per-resource utilisation, rendered as a
      per-phase line in the report (roofline output is unchanged).

## Validation philosophy

Every fidelity step must keep a **degenerate case that reproduces the
simpler model**, tested:

- pp=1 → DES == roofline (already tested).
- single bank / single port / infinite SRAM → expanded DES == lumped DES.
- closed-form ring == expanded-ring collective under no contention.

This keeps the engines a strict refinement hierarchy (roofline ⊂ stage-DES
⊂ graph-DES) rather than three models that happen to disagree.
