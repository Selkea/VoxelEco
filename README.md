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
| B | Toggle render: full **5cm voxels** ↔ **1m blocks** (voxel-tinted tops) |
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
- **Floating-origin rendering**: the sim runs at world coordinates ~1e5 (the
  streaming origin sits far out in positive space for the toroidal wrap math), but
  drawing 5 cm cubes that far from the origin cracks thin seams between voxels as
  float32 loses precision in the view transform. So the emit writes instances in a
  **local frame** (position − window origin, ~0..W) and the camera is offset by the
  same origin — the scene is drawn near zero however far you stream. A lifted near
  plane (far:near ~7000:1, not 40000:1), 4x MSAA, and a ~2% cube inflation (so
  neighbouring voxels overlap and seal the hairline junctions between them and
  between Y levels — same-material overlaps are invisible) finish the job.
- **1 m blocks with voxel-skinned faces** (default): the fine 5 cm sim is drawn
  as chunky 1 m blocks (flat 1 m tops, 1 m-aligned steps), but **every visible
  face is skinned with its real 5 cm voxels** — so cliff sides show the strata
  (grass on top, soil, then stone) at full per-voxel detail, not flat colour.
  One thread per 5 cm column emits a shell of 5 cm voxel-cubes for the block-
  snapped terrain: the top voxel plus, where a neighbouring block is shorter,
  the side voxels down to it, each tinted by its own voxel (material + wet/dry
  saturation + jitter). Interior voxels are occluded so none are emitted; draw
  cost is the (block-snapped) surface area, not the sim volume, so the map can
  be large while the detailed hydrology runs underneath. Press **B** (or
  `VOX_RENDER=voxel`) for the full per-5 cm-voxel renderer — the true 5 cm shape
  (fine steps) rather than chunky blocks. Instance buffers are sized for both,
  so **B** flips modes live with no reallocation.
- **Toroidal streaming (endless, state-preserving world)**: the sim buffer is a
  **torus that follows the camera**. Buffer slot = world column mod W, so as you
  fly, only the freshly-entered edge strip regenerates (one thin `regen_strip`)
  while the rest of the window keeps its **live water and erosion** — verified:
  after advancing a chunk, 0 kept-region cells change. Physics wraps around the
  torus but skips the moving "seam" (the join at the window's far edge, where
  buffer-adjacent columns are world-far), so nothing flows across it. Worldgen
  is deterministic and bit-exact seamless per world-coordinate, so the terrain
  scrolls without a ripple and extends forever. Gen + emit map buffer slots to
  world coords with an all-unsigned wrap (this GPU miscomputes signed subtraction
  that goes negative), and the window rides in positive world space. Distance
  fog fades the far edge into sky.
- **Vertical-tracking band (tall worlds)**: the world has **256 m of vertical
  relief** (5120 voxels at 5 cm) but the resident sim buffer is only a thin
  **height band** (default ~45 m) that **rides up and down with the terrain
  surface** under the camera. Worldgen returns a true world-Y surface spanning the
  full relief over **wide (hundreds-of-metres) features**, so 256 m of relief reads
  as gentle mountains inside any one view rather than spikes — the peaks and
  valleys reveal themselves as you fly. Only the near-surface slice is stored
  (`gen_oy` = the band's world-Y floor); everything below the band is implicit deep
  rock, everything above is implicit air, so a tall world costs the same cells as a
  shallow one. As the surface drifts past a deadband the band re-centres and
  regenerates; horizontal motion stays toroidal/state-preserving as long as you fly
  at roughly constant elevation. This is what lets a 256 m-tall world keep a wide
  (~50 m) footprint under the 1-billion-cell single-buffer ceiling.
- **Scale**: the default fly window is a **1024x1024 footprint (51 m) with a
  ~45 m resident band at 5 cm** (~805M sim cells), sized so the full per-voxel
  renderer flies smoothly. A single fixed (non-streamed) map is capped by a hard
  ceiling: **a single GPU storage buffer is 32-bit-sized in Godot**, so the
  cells buffer (4 bytes/cell) tops out at ~4 GB ≈ **1.05B cells** regardless of
  VRAM — past that it silently truncates, so oversized `VOX_SIZE` requests are
  clamped with a warning. Toroidal streaming (horizontal) and the vertical-tracking
  band (up/down) together keep the resident window small while the world itself is
  unbounded sideways and 256 m tall.

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
