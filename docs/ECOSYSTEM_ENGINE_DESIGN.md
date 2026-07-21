# VoxelEco — Real-Scale Ecosystem Engine: Design Notes

North star: **a real-world ecosystem simulator that looks good** — real physics, real
biology, at real distances. This doc scopes the two goals that outgrow the current Godot
build: (A) much-further *simulated* distance, and (B) ecology/biology that is 1:1 with
real life. Both are "design first" tasks; a custom engine is on the table.

Status quo (shipped): GPU falling-sand hydrology + erosion, vegetation→herbivore→predator
ecology rendered as grass/critters over a world mesh, and a `bio_period` time-dilation
that already puts metabolism/breeding/growth on a real-life clock (movement stays
real-time; fast-forward to 3600× to watch days pass).

---

## A. Much-further simulated distance

### The wall
The active sim is one camera-following window. It is **already at the engine's ceiling**:
~3.17 B cells = **102 m × 102 m** footprint (× 37.8 m band) at 5 cm voxels. The cap is not
VRAM (a 4090 holds it in ~12.7 GB) — it is **32-bit cell addressing across three ≤4 GB
Godot storage buffers**. uint32 tops out at 4.29 B cells, so a 4th buffer isn't fully
addressable. You cannot get "much further" by turning a knob; you trade resolution/vertical
range, or you change the architecture.

### Option 1 — Multi-resolution nested sim (within Godot, medium effort)
Simulate at several resolutions like a clipmap, coarsening with distance:
- **L0 (near, 5 cm)**: the current fine voxel window — full falling-sand water, erosion,
  individual voxel agents. ~50 m radius.
- **L1…Ln (10 cm → 40 cm → …)**: coarser voxel windows sharing the camera center, each the
  same cell budget but covering 4×/16×/… the area. Run cheaper physics out here.
- **Coupling**: inner window's edge state feeds the next ring; conserve water mass across
  the boundary (flux in = flux out). This is the hard part — falling-sand is
  resolution-dependent, so the coarse rings need a *different* fluid model that agrees at
  the seam (see below), not the same kernel at bigger voxels.
- **Background (heightfield)**: beyond the coarsest ring, keep today's render-only far
  terrain, but give it a **shallow-water / pipe-model flow** heightfield sim (cheap, 2.5D)
  so rivers/lakes exist km-scale without volumetric voxels.

Payoff: km-scale *active* water/terrain, detail only where you look. Cost: real work on
cross-resolution coupling + a second (heightfield) fluid solver. Still bounded by uint32
per level, but each level is its own budget so total reach multiplies.

### Option 2 — Custom compute engine (large effort, removes the wall)
Godot's RenderingDevice is the constraint (4 GB/buffer, uint32, no out-of-core). A
purpose-built Vulkan/CUDA compute core removes it:
- **64-bit / paged indexing** and **out-of-core streaming** (GigaVoxels / sparse voxel
  DAG — the user has study material on this): only resident bricks live in VRAM; the rest
  page from disk/host as the camera moves. Active *simulated* set can be far larger than
  VRAM.
- **Sparse bricking**: air/solid-rock regions cost nothing; only active (water, surface,
  agents) bricks are stored and stepped. Most of a world is inert → huge effective reach.
- **Same multi-res idea as Option 1**, but without the 4.29 B ceiling and with proper
  streaming instead of a single teleporting window.
- Render stays feasible: reuse the meshing/agent-emit ideas; the engine hands Godot (or a
  custom renderer) instance/heightfield buffers, or we go fully custom.

Recommendation: **prototype Option 1's multi-res coupling first** (it de-risks the coupling
math and the coarse fluid model, which Option 2 also needs), and treat Option 2 as the
destination once the multi-res model is proven. The coupling + shallow-water solver is the
reusable core either way.

### Open technical questions
- Coarse fluid model that agrees with fine falling-sand at a seam (shallow-water pipe model
  vs. cellular). Mass conservation across resolution jumps is the acceptance test.
- Agent handoff across levels (individual voxel agents ↔ coarse/statistical agents).
- Streaming state preservation (today a window recenter re-generates; real streaming must
  carry live water/erosion/agents across the move).

---

## B. Ecology/biology 1:1 with real life

`bio_period` set the real-life *timescale*. Making it *identical to real life* is a
data-and-behaviour program, phased:

### B1 — Per-organism calibration (data-driven)
Replace the hand-tuned energy constants with a **species table** of real biological
parameters, each mapped to sim units via the existing calibration (5 cm voxel, 30 tick/s):
- basal metabolic rate → energy drain / real-day; starvation survival window
- forage intake rate, diet (which PLANT/prey), bite size
- gestation length, litter size, sexual-maturity age, inter-birth interval
- lifespan + senescence (age-driven mortality — not modelled yet)
- body size (drives the render scale + step length + speed)
Start with 1–2 real species end-to-end (e.g. rabbit + fox, or sheep + wolf) sourced from
real numbers, verify each rate in isolation, then expand the table. Per-process factors
replace the single global `bio_period`.

### B2 — Richer real behaviours
Today: grazers seek foliage; predators pursue nearest prey. Real behaviour adds:
- **prey**: flee/vigilance (sense predators, run — deliberately deferred so far), herding,
  refuge-seeking
- **predators**: stalking, energy-aware hunting (don't chase hopeless prey), territory
- **foraging**: energy-optimal movement (patch choice), not just nearest-food gradient
- **life stages**: juveniles, aging, disease/parasite load
Keep the GPU-agent, zero-cross-buffer-coupling pattern that has worked so far.

### B3 — Environmental coupling
Real ecology is driven by the environment the sim already has (water) plus:
- **seasons + day/night + temperature** → growth rate, breeding windows, behaviour
- **weather** (rain already exists) → soil moisture → growth (already partially coupled via
  saturation-gated vegetation)
- **spatial heterogeneity**: soil quality, sunlight/slope aspect → carrying capacity
This is where "looks good" and "is real" converge — seasonal color, dawn light, drought.

### Testing under real-life rates
Headless demographic tests can't run real months of ticks. Keep the `bio_period = 1`
(fast) regime as the **mechanism test** (dynamics are scale-invariant), and add
rate-calibration checks that assert a single process matches its real target (e.g. "a fed
grazer's energy integrates to X over a sim-day"), rather than running whole life cycles.

---

## Suggested next increments (small → large)
1. Prey fleeing (B2) — highest-impact behaviour, fits the current agent model.
2. Age/lifespan + a first real species pair with real numbers (B1).
3. Seasons/day-night + growth coupling (B3) — also a big "looks good" win.
4. Multi-res coupling + coarse shallow-water solver prototype (A, Option 1).
5. Custom compute engine with sparse/streaming bricks (A, Option 2) — the endgame.
