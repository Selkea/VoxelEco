# VoxelEco — 3D voxel falling-sand ecosystem simulator (v0.1)

A physically-grounded world built in **Godot 4.7**, aiming for real-life
behaviour over game abstraction. Everything is literal **material voxels** —
rock, soil, sand, water, air — obeying local falling-sand physics on a 3D
grid. v0.1 is **hydrology first**: rain falls as water voxels that flow
downhill, pool into ponds, evaporate, and erode soil into sand as they run.
Vegetation and animals layer on top of this abiotic substrate later.

**Play:** double-click `play.bat`.  **Edit in Godot:** `edit.bat`.

## Controls

Creative-mode free flight (no gravity / collision):

| Input | Action |
|---|---|
| mouse | Look around |
| W A S D | Fly forward / left / back / right (in the look direction) |
| Space / Ctrl | Fly up / down |
| Shift (hold) | Sprint (4x) |
| mouse wheel | Fly speed down / up |
| Esc | Release / re-grab the mouse cursor |
| 1 – 7 | Speed: 1x / 2x / 4x / 8x / 16x / 32x / 64x (1x = real time) |
| Q / E | Less / more rain (mm/hour) — the world starts **calm**; press E to make it rain |
| C | Toggle a cross-section (see subsurface moisture / the water table) |
| T | Toggle worldgen **blended** ↔ **terraced** (regenerates in place) |
| P | Pause / resume |
| R | Generate a fresh world |

## How it works

- **The world** is a 3D grid of material cells (`bedrock, stone, soil, sand,
  water, air`) stored as one flat byte array.
- **Falling-sand physics** runs per cell: water falls, runs down slopes via
  down-diagonals, spreads to find its level, and evaporates when exposed to
  air; heavier materials sink through lighter ones. Materials differ in
  **cohesion**: loose **sand** slumps to a shallow angle of repose, saturated
  **mud** flows (mudslides), intact **soil** is cohesive and holds hillsides
  (it only fails if undercut or eroded loose), and **grass** is rooted and
  never slides. So freshly generated terrain sits at rest — slides happen for
  real reasons (undercutting, erosion loosening soil to sand, waterlogging to
  mud), not just because the land was stamped from noise.
- **Materials erode at different rates**: sand is already the mobile product,
  soil loosens to sand at the base rate, stone weathers ~20x slower (to soil),
  and bedrock never erodes. Only **flowing** water erodes — water that is
  actually falling or running downhill — so rivers, rain runoff and waterfalls
  carve the terrain while a **calm lake leaves its banks intact** instead of
  slowly fraying its own shoreline at storm-river rates.
- **Materials transform under conditions**: soil that saturates past a
  threshold turns to **mud** (soft, slumps on slopes, washes away fast) and
  dries back to soil when it drains. Stone weathers to soil, soil erodes to
  sand.
- **Vegetation**: **grass** colonizes moist, lit surface soil; it binds the
  ground (eroding ~10x slower than bare soil), and its roots draw down the
  saturation of nearby voxels — keeping the soil it holds from waterlogging
  into mud. Drown it or bury it and it dies back to soil. Soil holds a base
  **field-capacity** moisture against gravity, so vegetation stays supplied.
- **Soil moisture & infiltration**: each permeable cell (soil/sand/mud/grass)
  carries a water-saturation level (packed into the voxel, so wetness moves with
  the soil). Surface water soaks into unsaturated ground and **percolates downward**
  until it hits impermeable rock or already-saturated soil, forming a water
  table; stone and bedrock are impermeable so water pools/flows over them.
  Saturated ground renders darker (wet earth), and the whole subsurface is
  visible with the cross-section key.
- **GPU compute physics**: the per-cell step runs as a Vulkan compute shader
  using **Margolus partitioning** — disjoint 2x2x2 blocks per thread, offset
  alternating each tick (Noita's conflict-free checkerboard idea at per-thread
  granularity, single buffer, in-place). Deterministic per seed via hashed
  RNG. Measured ~**940x faster** than the optimized CPU step (250 heavy rain
  ticks: 5 ms vs 4.7 s). Falls back to the CPU step transparently when no
  RenderingDevice exists (plain --headless).
- **Noita-style dirty chunks**: the shader flags 16x16-column regions whose
  cells changed; only those chunks get remeshed. A settled landscape costs
  nothing to keep on screen.
- **Rendering**: each chunk builds two face-culled meshes — opaque
  vertex-coloured terrain and a translucent water surface — so interior
  voxels cost nothing.
- **Hierarchical chunk-based worldgen** produces a finished landscape *at
  rest*. Voxels group 20x20 into a **block** (1 m) and 20x20 blocks into a
  **chunk** (20 m). The base heightfield is generated **at block (1 m)
  resolution** — one height per block, "as if 1 block = 1 voxel" — from a
  chunk-scale landform band plus per-block relief. The 5 cm voxels then fill
  between those block heights one of two ways (**T** toggles, title bar shows
  which), from the **same** heightfield: **blended** smooth-interpolates the
  surrounding block heights (smooth natural terrain), or **terraced** snaps each
  block to a flat 1 m plateau at a 1 m-quantised height so the world reads as
  clean **1 m voxel-cubes** (a block resembles a voxel) — while the
  physics/water/erosion keep simulating on the 5 cm grid underneath. Both are
  pure functions of world (x,z) + seed sampling only continuous world-space
  noise (never the world size or a fixed centre), so the terrain is **seamless
  across any chunk boundary** and comes out identical however the world is
  windowed — the groundwork for streaming. Land above the water line
  is **vegetated (grassy)**; basins fill with **lakes** on pre-saturated beds
  (so lakes sit on firm wet ground and don't seep or slump); the exposed
  shoreline stays firm soil and no loose sand is pre-placed, so nothing
  avalanches on the first tick. Rain, flow, and erosion take it from there.
- **Fully GPU pipeline**: terrain generation (fbm noise), physics, and
  surface extraction all run in compute. The renderer is a MultiMesh of
  surface voxels whose instance buffer is written by a compute pass — the
  CPU never loops over cells. A fresh 6.3M-cell world generates, storms,
  and renders in under 2 seconds.
- **Zero-readback rendering**: compute runs on the engine's main
  RenderingDevice and the emit pass writes the MultiMesh instance buffers
  directly in VRAM (`multimesh_get_buffer_rd_rid`); the only per-frame
  readback is a 16-byte instance counter.
- **1 m blocks with voxel-tinted tops**: the fine 5 cm sim is drawn as chunky
  1 m blocks, but each block's **top is skinned with its real 5 cm surface
  voxels**, so the blocky terrain still shows per-voxel colour detail. One
  thread per 5 cm column lays a 5 cm tile at the block-rounded top height,
  tinted by its own surface voxel (material + wet/dry saturation + per-voxel
  jitter); the block bodies/sides stay solid 1 m cubes (one per 20x20x20 block,
  coloured by their centre material — grass over soil over stone). Draw cost is
  the surface area (a few million tiles), not the sim volume (billions of
  voxels), so the map can be large while the detailed hydrology runs underneath.
  `VOX_RENDER=block` draws the coarser 1 m-block-with-voxel-tinted-tops path.
- **Streaming (endless world)**: the sim buffer is a **window that follows the
  camera**. As you fly near an edge, the window recenters on you and regenerates
  at the new world origin; because worldgen is deterministic per world-coordinate
  (verified bit-exact seamless), the overlapping terrain is identical, so the
  world scrolls seamlessly and extends forever. The emit writes voxels at their
  **world** positions (buffer + window origin) so the render lands where the
  camera is. Distance fog fades the window's far edge into sky. *(The window's
  live sim state resets on recenter — a full-window regen; state-preserving
  toroidal streaming is a future refinement.)*
- **Scale**: the default fly window is **1280x1280x128 voxels at ~5 cm** = a
  64 m x 64 m x 6.4 m plot (~210M sim cells), sized so the full per-voxel
  renderer flies smoothly. A single fixed (non-streamed) map is capped by a hard
  ceiling: **a single GPU storage buffer is 32-bit-sized in Godot**, so the
  cells buffer (4 bytes/cell) tops out at ~4 GB ≈ **1.05B cells** regardless of
  VRAM — past that it silently truncates, so oversized `VOX_SIZE` requests are
  clamped with a warning. Streaming sidesteps this by keeping the resident
  window small while the world itself is unbounded.

## Physical scale & calibration

The world is calibrated to real units, not tuned by feel:

- **Space**: 1 voxel = **5 cm**, so the default 512x512x192 world is a
  25.6 m x 25.6 m plot, 9.6 m tall. One water voxel dropped in a column adds
  **50 mm** of depth.
- **Time**: **30 ticks = 1 simulated second**, so one simulated hour is
  108,000 ticks. At 1x the sim runs in real time; keys 1-7 fast-forward.
- **Rain** is set in **mm/hour** (WMO scale: <2.5 light, 2.5-7.5 moderate,
  >7.5 heavy, >50 violent). The per-tick spawn probability is derived so the
  deposited depth matches the rate — verified by `--sim` with `VOX_CALIB=1`
  (40 mm/h for 1 h deposits 40.0 mm, 0.1% error).
- **Evaporation** is set in **mm/day** (open water is ~1-10; default 5). At
  this scale that is deliberately slow: a puddle takes ~10 days to lose one
  5 cm voxel layer, so under real rates water pools and *persists* rather than
  vanishing between frames — as it does in reality.
- **Erosion** is the one genuinely scale-free process (mm/year geologically,
  cm/hour in a storm rill); it is pinned to the fast fluvial regime
  (~5 cm of bank retreat per hour of sustained flow) so landscape change is
  visible over sim-hours.

Because a 5 cm voxel is a coarse unit of water (50 mm), realistic rain
accumulates gradually — a heavy storm needs sim-minutes to puddle and
sim-hours to flood. Fast-forward (keys 1-7) is the intended way to watch
weather play out.

The whole simulation is a pure, node-free `VoxWorld` object, so it runs and
self-tests with no rendering.

## Self-testing

```
<godot> --headless --path . -- --sim    # ticks the sim, asserts water flows,
                                         # pools on low ground, and mass is conserved
<godot> --path . -- --shot               # runs a rain-then-drain cycle, saves a screenshot
```

## Source map

| File | Responsibility |
|---|---|
| `scripts/vox_world.gd` | The whole simulation: terrain gen, falling-sand physics, rain, erosion. No nodes. |
| `scripts/vox_view.gd` | Turns world state into face-culled terrain + water meshes |
| `scripts/game.gd` | Owns world + view, orbit camera, lighting, controls, headless test/screenshot |

Art: none — voxels are vertex-coloured by material.

## Natural next steps

Proper pressure-based water (true levelling / hydrostatic flow), sediment the
water carries and deposits (deltas, meanders), a closed water cycle
(evaporation → clouds → rain), Noita-style velocity particles for splashes
and waterfalls, then the ecology: vegetation colonizing moist soil,
herbivores, predators. Next perf frontier: GPU meshing (the sim no longer
bottlenecks; GDScript mesh building does).
