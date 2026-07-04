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

- [ ] **Resource model from graph**: turn each expanded node (DRAM bank,
      NoC switch/port, per-core SRAM, matrix engine) into a FIFO (or
      k-server) resource, and each edge into a link resource with its
      bandwidth/latency. Reuse the `_Task`/`schedule()` core.
- [ ] **Lower an `Op` to a sub-task chain over the path**: a GEMM tile
      becomes read-from-DRAM-bank(s) → traverse NoC port(s) → land in
      SRAM → matrix-engine compute → write-back, each hop a task on the
      corresponding resource. Tile size becomes a knob (SRAM capacity is
      the real constraint — currently stored but never enforced).
- [ ] **Bank/port assignment**: use the edge `pattern` (interleave/all)
      and selectors to decide which bank a tile's address maps to, so bank
      conflicts are modelled, not averaged. Address→bank hashing knob.
- [ ] **Compute/DRAM overlap becomes emergent**: with reads and math on
      separate resources, double-buffering overlap falls out of the
      schedule instead of `max(compute_t, mem_t)`.
- [ ] **Validation**: with a single bank, single port and infinite SRAM,
      the expanded DES must collapse to today's `max(flops/peak, bytes/bw)`
      per op (regression test, analogous to the pp=1 == roofline test).

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
- [ ] **Collective internals**: expand ring/all-to-all into their actual
      per-step link transfers over the topology (`Topology.RING` /
      `MESH_2D` / `ALL_TO_ALL`) instead of the closed-form
      `ring_allreduce_time`; contention with concurrent hop traffic then
      emerges. Keep the closed form as the validation oracle.

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

- [ ] **Convergence control**: auto-grow `decode_rounds` until the measured
      round period stabilises within a tolerance, instead of fixed
      `decode_rounds/warmup`.
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
