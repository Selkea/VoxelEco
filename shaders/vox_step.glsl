#version 450
// VoxelEco falling-sand physics on the GPU.
//
// Margolus partitioning: the grid is split into disjoint 2x2x2 blocks and one
// thread rearranges each block, so there are no write races by construction.
// The partition offset alternates 0/1 each tick so material crosses block
// boundaries over successive ticks. Randomness is a PCG hash of
// (block, tick, seed) — fully deterministic for a given seed.
//
// mode 0 = physics step   mode 1 = rain   mode 2 = pack bytes for readback
// mode 3 = emit surface-voxel instances for MultiMesh   mode 4 = worldgen

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(set = 0, binding = 0, std430) restrict buffer CellsBuf { uint cells[]; };
// MULTI-BUFFER CELLS: Godot caps a single storage buffer at 4 GB (32-bit byte
// size), so bigger worlds split across up to THREE buffers at linear index
// boundaries (p.cells_split / p.cells_split2, whole y-slabs). cget/cset route
// every access; smaller worlds set the unused splits to the cell count so the
// spare buffers (64-byte stubs) are never touched. uint32 indexing caps the
// total at ~4.29B cells.
layout(set = 0, binding = 8, std430) restrict buffer CellsBuf2 { uint cells2[]; };
layout(set = 0, binding = 9, std430) restrict buffer CellsBuf3 { uint cells3[]; };
// RAY-CAST renderer (GigaVoxels-inspired): per-column ground heights + a coarse
// per-16x16-tile max grid accelerate a heightfield ray march — draw cost scales
// with SCREEN RESOLUTION, not instance count. heights = first-air-above-ground
// per column (contiguous ground; airborne falling grains are ignored).
layout(set = 0, binding = 10, std430) restrict buffer HeightsBuf { uint heights[]; };
layout(set = 0, binding = 11, std430) restrict buffer HMaxBuf { uint hmax[]; };
// camera for the ray pass: eye/forward/right/up in the LOCAL render frame, plus
// sun direction and tan(fov)*aspect factors — updated per frame, tiny buffer
layout(set = 0, binding = 12, std430) restrict buffer CamBuf {
	vec4 cam_eye; vec4 cam_fwd; vec4 cam_right; vec4 cam_up; vec4 cam_sun;
};
layout(set = 0, binding = 13, rgba8) uniform restrict writeonly image2D out_img;
layout(set = 0, binding = 1, std430) restrict buffer PackBuf { uint packed_out[]; };
layout(set = 0, binding = 2, std430) restrict buffer StatsBuf { uint rained; uint evaporated; uint absorbed; };
// Noita-style dirty tracking, one flag per 16x16-column mesh chunk: settled
// regions are never remeshed. Flags are set here, read+cleared by the CPU.
layout(set = 0, binding = 3, std430) restrict buffer DirtyBuf { uint dirty[]; };
// MultiMesh instance streams (16 floats each: 3x4 transform rows + color),
// filled by the emit pass so the CPU never loops over cells to mesh
layout(set = 0, binding = 4, std430) restrict buffer SolidInst { float solid_inst[]; };
layout(set = 0, binding = 5, std430) restrict buffer WaterInst { float water_inst[]; };
layout(set = 0, binding = 6, std430) restrict buffer InstCount { uint n_solid; uint n_water; uint cap_solid; uint cap_water; };
// active-block gating (per 16x16-column chunk, same grid as dirty): the physics
// only steps chunks with recent activity, and settled regions sleep. A change
// wakes a chunk (+1-chunk margin) to KEEPALIVE; do_decay counts it back down, so a
// chunk stays awake for KEEPALIVE ticks after its last change (covers both Margolus
// offsets). This skips the cell READS for inert regions, which is the whole cost.
layout(set = 0, binding = 7, std430) restrict buffer ActiveBuf { uint awake[]; };
const uint KEEPALIVE = 4u;

layout(push_constant) uniform Params {
	uint W; uint H; uint D; uint tick;
	uint seedv; uint rain_thr; uint mode; uint offset;
	// calibrated per-tick probabilities (rain is an integer threshold out of
	// 2^24 for finer resolution than a float32 near zero)
	float evap_prob; float erode_prob; uint cut_z;
	// world-space origin of this buffer, in voxels (chunk streaming): worldgen
	// samples noise at (local + origin) so a chunk generated at any world
	// position lines up seamlessly with its neighbours. int via bit-reinterpret.
	uint gen_ox; uint gen_oz;
	uint gen_flags;   // bit0: 1 = terraced (flat 1m plateaus), 0 = blended (smooth)
	// toroidal streaming: worldgen only fills the buffer-slot strip
	// [x0, x0+width) x [z0, z0+depth) (mod W/D) — the freshly-entered edge —
	// packed lo|hi<<16. Full regen = width/depth == W/D.
	uint strip_x; uint strip_z;
	// vertical-tracking band: world-Y voxel of the buffer floor (buffer y 0). The
	// buffer stores world-Y [gen_oy, gen_oy+H); gen/emit map buffer y -> world y by
	// adding it. Below the band is implicit bedrock, above it is implicit air.
	uint gen_oy;
	// first linear cell index stored in cells2 / cells3 (multi-buffer splits;
	// unused splits = total cell count, so the spare buffers are never touched)
	uint cells_split; uint cells_split2;
	// distance LOD: camera position in the LOCAL render frame (voxels) and the
	// near radius. Columns within lod_r of the camera render as fine 5 cm faces
	// (do_face_emit); farther terrain as coarse 1 m block quads (do_lod_emit).
	// lod_r == 0 disables LOD (everything fine) — tests and small worlds.
	uint lod_cx; uint lod_cz; uint lod_r;
	// ray-cast output image size (mode 16)
	uint img_w; uint img_h;
} p;

// route a linear cell index to its buffer (see CellsBuf2/3). Spatially adjacent
// cells share a buffer (splits are whole y-slabs), so the branch is coherent.
uint cget(uint i) {
	if (i < p.cells_split) { return cells[i]; }
	if (i < p.cells_split2) { return cells2[i - p.cells_split]; }
	return cells3[i - p.cells_split2];
}
void cset(uint i, uint v) {
	if (i < p.cells_split) { cells[i] = v; }
	else if (i < p.cells_split2) { cells2[i - p.cells_split] = v; }
	else { cells3[i - p.cells_split2] = v; }
}

const uint CHUNK = 16u;

// flag the chunks around (x, z) — a border cell change alters the visible
// faces of cells in the adjacent chunk, so flag with a 1-cell margin
void mark_dirty(uint x, uint z) {
	uint cw = (p.W + CHUNK - 1u) / CHUNK;
	uint x0 = (x > 0u ? x - 1u : 0u) / CHUNK;
	uint x1 = min(x + 1u, p.W - 1u) / CHUNK;
	uint z0 = (z > 0u ? z - 1u : 0u) / CHUNK;
	uint z1 = min(z + 1u, p.D - 1u) / CHUNK;
	// mark for remeshing (dirty) AND wake the chunk for physics (active). The
	// 1-chunk margin wakes neighbours so material can flow across a sleep boundary.
	uint i0 = z0 * cw + x0;              dirty[i0] = 1u; awake[i0] = KEEPALIVE;
	if (x1 != x0) { uint i = z0 * cw + x1; dirty[i] = 1u; awake[i] = KEEPALIVE; }
	if (z1 != z0) { uint i = z1 * cw + x0; dirty[i] = 1u; awake[i] = KEEPALIVE; }
	if (x1 != x0 && z1 != z0) { uint i = z1 * cw + x1; dirty[i] = 1u; awake[i] = KEEPALIVE; }
}

const uint AIR = 0u;
const uint BEDROCK = 1u;
const uint STONE = 2u;
const uint SOIL = 3u;
const uint SAND = 4u;
const uint WATER = 5u;
const uint MUD = 6u;      // waterlogged soil: soft, flows, washes away fast
const uint GRASS = 7u;    // vegetated soil: roots bind it, drink the ground dry

// A cell is one uint: material in byte 0, ground-water SATURATION
// (0..sat_cap, phenomenological % of pore space) in byte 1. Wetness therefore
// travels with the material when it moves, and needs no extra buffer.
// Bytes 2-3 are free — reserved for a planned per-cell TEMPERATURE field
// (drives freezing/thawing, evaporation rate, snow) with the same
// travels-with-the-cell property.
uint MAT(uint c) { return c & 0xFFu; }
uint SAT(uint c) { return (c >> 8u) & 0xFFu; }
uint PACK(uint m, uint s) { return (m & 0xFFu) | (min(s, 255u) << 8u); }

bool is_soil(uint m) { return m == SOIL || m == SAND || m == MUD || m == GRASS; }

// per-material pore capacity (0 = impermeable rock; sand drains so holds less
// against gravity, soil holds more). Reaching cap = fully saturated -> pools.
uint sat_cap(uint m) {
	if (m == SAND) return 70u;
	if (m == SOIL || m == MUD || m == GRASS) return 120u;
	return 0u;                       // STONE / BEDROCK / AIR / WATER: impermeable
}
bool permeable(uint m) { return is_soil(m); }

// permeability = how fast water seeps INTO and THROUGH a material (hydraulic
// conductivity), as a per-tick chance. Coarse sand drains fast; waterlogged
// mud is nearly sealed. This rate-limits infiltration so surface water levels
// out first and only then soaks in, instead of a whole voxel vanishing per
// tick (which made lake edges wick into the shore before they could settle).
float permeability(uint m) {
	if (m == SAND) return 0.060;
	if (m == SOIL) return 0.022;
	if (m == GRASS) return 0.018;
	if (m == MUD) return 0.008;
	return 0.0;
}

// erosion resistance: detachment rate multiplier vs the base erode_prob.
// mud washes away fastest; soil is the reference; grass roots bind it 10x
// tighter; stone weathers 20x slower; sand is already the mobile product.
float erosion_mult(uint m) {
	if (m == MUD) return 3.0;
	if (m == SOIL) return 1.0;
	if (m == GRASS) return 0.1;
	if (m == STONE) return 0.05;
	return 0.0;
}

// angle of repose: how readily a grain slumps sideways down a slope. Intact
// SOIL is COHESIVE (held by structure/roots) so it holds hillsides and doesn't
// avalanche — only LOOSE sand piles to a shallow angle, saturated MUD flows
// (mudslides), and grass is rooted. This is what keeps freshly generated
// terrain at rest instead of landsliding; erosion loosens soil into sand,
// and THAT sand then slumps (a localized slide) as it should.
float repose(uint m) {
	if (m == SAND) return 0.55;
	if (m == MUD) return 0.5;
	// SOIL and GRASS are fully cohesive: they hold any slope (even the sharp 1 m
	// cliffs of terraced worldgen) and never slump grain-by-grain. They still
	// fall straight down when undercut, and erosion loosens soil into SAND —
	// which is the material that actually slumps. Loose/wet material flows;
	// intact ground holds.
	return 0.0;                       // SOIL + GRASS
}

const uint ABSORB = 34u;   // pore-fill added when a water voxel soaks in (~3 to saturate)
const uint PERC = 9u;      // saturation draining one cell downward per contact
const uint FIELD_CAP = 40u;// moisture soil holds against gravity (capillary); only
                           // the excess percolates down, so ground stays moist
const uint MUD_WET = 112u; // soil at/above this saturation turns to mud...
const uint MUD_DRY = 55u;  // ...and mud below this dries back to soil
const uint UPTAKE = 3u;    // saturation a grass cell draws from the ground per tick
const uint GRASS_FLOOR = 24u; // grass draws surrounding moisture down only to here
const float GROW = 0.004;  // per-tick chance moist surface soil sprouts grass

uint dens(uint m) {
	if (m == AIR) return 0u;
	if (m == WATER) return 1u;
	if (m == STONE) return 9u;
	if (m == BEDROCK) return 99u;
	return 3u;                        // SOIL / SAND / MUD / GRASS
}

bool movable(uint m) { return m == WATER || is_soil(m); }

uint cidx(uint x, uint y, uint z) { return x + z * p.W + y * p.W * p.D; }

// world coordinate of buffer slot s on the toroidal window (which covers world
// [origin, origin+len); buffer slot = world mod len). Used by gen + render so a
// streamed window generates/draws at its true world position. ALL-UNSIGNED with
// no negative intermediate: this driver miscomputes `int - int` that goes
// negative (wraps as uint, shifting the modulo by 2^32 mod len). The streaming
// camera is kept in positive world space so origin >= 0.
uint world_coord(uint s, uint origin, uint len) {
	uint seam = origin % len;            // origin mod len, buffer column of the join
	return (origin - seam) + s + (s < seam ? len : 0u);
}

uint pcg(uint v) {
	v = v * 747796405u + 2891336453u;
	v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
	return (v >> 22u) ^ v;
}

uint g_state;
float rnd() { g_state = pcg(g_state); return float(g_state) * (1.0 / 4294967296.0); }

// flatten a 4x4x4-local dispatch into a linear id (for rain / pack modes)
uint flat_id() {
	return (gl_GlobalInvocationID.x * 4u + gl_GlobalInvocationID.y) * 4u
			+ gl_GlobalInvocationID.z;
}

void do_rain() {
	uint id = flat_id();
	if (id >= p.W * p.D) { return; }
	g_state = pcg(id ^ pcg(p.tick ^ p.seedv) ^ 0x9e3779b9u);
	if ((pcg(g_state) & 0xFFFFFFu) < p.rain_thr) {
		uint x = id % p.W;
		uint z = id / p.W;
		uint i = cidx(x, p.H - 1u, z);
		if (MAT(cget(i)) == AIR) {
			cset(i, WATER);
			atomicAdd(rained, 1u);
			mark_dirty(x, z);
		}
	}
}

void do_pack() {
	uint n = p.W * p.H * p.D;
	uint id = flat_id();
	if (id * 4u >= n) { return; }
	uint b = id * 4u;
	// pack only the MATERIAL byte of each cell for CPU-side meshing/analysis
	uint v = MAT(cget(b));
	if (b + 1u < n) { v |= MAT(cget(b + 1u)) << 8u; }
	if (b + 2u < n) { v |= MAT(cget(b + 2u)) << 16u; }
	if (b + 3u < n) { v |= MAT(cget(b + 3u)) << 24u; }
	packed_out[id] = v;
}

// Is the water in local cell j actively flowing — able to fall or run downhill?
// Pooled, level water rests on full support and must NOT erode: a calm lake does
// not cut its banks at storm-river rates, which was fraying every shoreline over
// time. Only descending water detaches material. Checks the four cells directly
// and diagonally below j, all inside this 2x2x2 block, so it's race-free; j must
// be an upper cell for 'below' to be in-block, and the Margolus offset flip
// evaluates the lower half of each water-soil contact on the next tick.
bool water_flowing(uint cc[8], uint j) {
	if (MAT(cc[j]) != WATER) { return false; }
	if ((j & 2u) == 0u) { return false; }       // lower cell: below is out-of-block
	uint lo = j & ~2u;                            // cell directly below (in block)
	return MAT(cc[lo]) == AIR || MAT(cc[lo ^ 1u]) == AIR
		|| MAT(cc[lo ^ 4u]) == AIR || MAT(cc[lo ^ 5u]) == AIR;
}

// local cell index bits: bit0 = x, bit1 = y, bit2 = z
void do_step() {
	uvec3 base = gl_GlobalInvocationID * 2u + uvec3(p.offset);
	// Toroidal window: x/z wrap so the buffer is a torus, EXCEPT at the seam
	// (buffer column origin mod W/D) where buffer-adjacent cells are world-far —
	// skip Margolus blocks straddling it so nothing flows across the join. The
	// seam sits at the window's far edge, away from the camera. y never wraps.
	uint sx = p.gen_ox % p.W;   // seam column (origin >= 0, so plain uint mod)
	uint sz = p.gen_oz % p.D;
	if ((base.x + 1u) % p.W == sx || (base.z + 1u) % p.D == sz) { return; }
	// ACTIVE-BLOCK GATE: skip inert chunks BEFORE touching any cells — the cell
	// reads are the cost, so gating here (one tiny flag read, cached per workgroup)
	// is what actually saves time. Chunk index matches mark_dirty / dirty.
	uint cw = (p.W + CHUNK - 1u) / CHUNK;
	if (awake[(min(base.z, p.D - 1u) / CHUNK) * cw + (min(base.x, p.W - 1u) / CHUNK)] == 0u) { return; }
	uint c[8];
	bool ib[8];
	uint ids[8];
	bool any_active = false;
	for (uint l = 0u; l < 8u; l++) {
		uvec3 pos = base + uvec3(l & 1u, (l >> 1u) & 1u, (l >> 2u) & 1u);
		pos.x %= p.W;   // toroidal wrap in x/z
		pos.z %= p.D;
		bool ok = pos.y < p.H;
		ib[l] = ok;
		ids[l] = ok ? cidx(pos.x, pos.y, pos.z) : 0u;
		c[l] = ok ? cget(ids[l]) : BEDROCK;
		any_active = any_active || movable(MAT(c[l]));
	}
	if ((p.mode >> 8u) == 32u) {
		// partition self-test: count how many threads touch each cell.
		for (uint l = 0u; l < 8u; l++) {
			if (ib[l]) {
				uint ai = ids[l];   // routed atomic (partition self-test only)
				if (ai < p.cells_split) { atomicAdd(cells[ai], 1u); }
				else if (ai < p.cells_split2) { atomicAdd(cells2[ai - p.cells_split], 1u); }
				else { atomicAdd(cells3[ai - p.cells_split2], 1u); }
			}
		}
		return;
	}
	if (!any_active) { return; }
	g_state = pcg((gl_GlobalInvocationID.x * 73856093u)
			^ (gl_GlobalInvocationID.y * 19349663u)
			^ (gl_GlobalInvocationID.z * 83492791u)
			^ pcg(p.tick * 2654435761u + p.seedv));

	const uint bottoms[4] = uint[4](0u, 1u, 4u, 5u);

	uint rules = p.mode >> 8u;
	if (rules == 0u) { rules = 0xFFu; }
	// 1. gravity: heavier sinks into lighter directly below (in-block pairs)
	if ((rules & 1u) != 0u)
	for (uint k = 0u; k < 4u; k++) {
		uint b = bottoms[k];
		uint t = b + 2u;
		if (movable(MAT(c[t])) && dens(MAT(c[t])) > dens(MAT(c[b]))) {
			uint tmp = c[t]; c[t] = c[b]; c[b] = tmp;   // swap carries wetness
		}
	}

	// 2. down-diagonals: water runs downhill, grains topple at their repose
	if ((rules & 2u) != 0u)
	for (uint k = 0u; k < 4u; k++) {
		uint t = bottoms[k] + 2u;
		uint m = MAT(c[t]);
		if (!movable(m)) { continue; }
		bool is_water = m == WATER;
		if (!is_water && rnd() >= repose(m)) { continue; }
		// bottom-layer cells that are diagonal (differ in x and/or z)
		uint r = uint(rnd() * 3.0);
		for (uint a = 0u; a < 3u; a++) {
			uint pick = (r + a) % 3u;
			// enumerate the 3 bottom cells other than the one directly below
			uint below = t - 2u;
			uint cand = below ^ 1u;                       // differs in x
			if (pick == 1u) { cand = below ^ 4u; }        // differs in z
			if (pick == 2u) { cand = (below ^ 1u) ^ 4u; } // differs in both
			if (is_water ? (MAT(c[cand]) == AIR) : (dens(MAT(c[cand])) < dens(m))) {
				uint tmp = c[t]; c[t] = c[cand]; c[cand] = tmp;
				break;
			}
		}
	}

	// 3. water finds its level and then STOPS. Water that can't fall (something
	// directly below it) flows sideways ONLY toward a cell it can then drain
	// out of — i.e. the target has AIR below it. So mounds flatten and water
	// runs downhill to fill low ground, but a level, contained body has nowhere
	// lower to go and comes fully to rest (no perpetual edge-shuffling). This
	// also leaves lone airborne drops untouched (their below is air -> they
	// fall via gravity, not this rule), so rain stays vertical.
	// Top-layer pairs only, so both 'below' cells (a^2, b^2) are in this block.
	//   FUTURE: WIND adds a deliberate directional bias here.
	const uint lat_a[4] = uint[4](2u, 6u, 2u, 3u);
	const uint lat_b[4] = uint[4](3u, 7u, 6u, 7u);
	if ((rules & 4u) != 0u)
	for (uint k = 0u; k < 4u; k++) {
		uint a = lat_a[k];
		uint b = lat_b[k];
		// a -> b: water at a can't fall (below-a solid/water), target b is air
		// with air below it (drainable). And the mirror, b -> a.
		bool ab = MAT(c[a]) == WATER && MAT(c[b]) == AIR
				&& MAT(c[a ^ 2u]) != AIR && MAT(c[b ^ 2u]) == AIR;
		bool ba = MAT(c[b]) == WATER && MAT(c[a]) == AIR
				&& MAT(c[b ^ 2u]) != AIR && MAT(c[a ^ 2u]) == AIR;
		if (ab || ba) {
			uint tmp = c[a]; c[a] = c[b]; c[b] = tmp;
		}
	}

	// 3b. infiltration: surface water soaks into unsaturated ground, and
	// ground water percolates downward until it hits an impermeable layer
	// or already-saturated soil (forming a water table).
	if ((rules & 64u) != 0u)
	for (uint k = 0u; k < 4u; k++) {
		uint b = bottoms[k];
		uint t = b + 2u;
		uint mb = MAT(c[b]);
		uint mt = MAT(c[t]);
		if (mt == WATER && permeable(mb) && SAT(c[b]) < sat_cap(mb)
				&& rnd() < permeability(mb)) {
			// water resting on thirsty ground soaks in at the ground's seepage
			// rate (not instantly), so it can level out first
			c[b] = PACK(mb, min(SAT(c[b]) + ABSORB, sat_cap(mb)));
			c[t] = AIR;
			atomicAdd(absorbed, 1u);
		} else if (permeable(mt) && permeable(mb)
				&& SAT(c[t]) > FIELD_CAP && SAT(c[b]) < sat_cap(mb)
				&& rnd() < permeability(mt)) {
			// only water above field capacity drains, at the material's seepage
			// rate; the rest is held by capillary action (keeps ground moist)
			uint mv = min(min(SAT(c[t]) - FIELD_CAP, PERC), sat_cap(mb) - SAT(c[b]));
			c[t] = PACK(mt, SAT(c[t]) - mv);
			c[b] = PACK(mb, SAT(c[b]) + mv);
		}
	}

	// 4. evaporation: exposed bottom-layer water under in-block air
	if ((rules & 8u) != 0u)
	for (uint k = 0u; k < 4u; k++) {
		uint b = bottoms[k];
		if (MAT(c[b]) == WATER && MAT(c[b + 2u]) == AIR && rnd() < p.evap_prob) {
			c[b] = AIR;
			atomicAdd(evaporated, 1u);
		}
	}

	// 5. erosion: material in FLOWING water detaches, at a per-material rate.
	// stone weathers slowly to soil; soil loosens to (mobile) sand; sand is
	// already the mobile product. Any absorbed water is carried along. Only
	// water that is actually moving (falling / running downhill) erodes — a
	// still lake resting against its bank leaves it intact, so shorelines stop
	// fraying and the water they hold stops shifting.
	if ((rules & 16u) != 0u)
	for (uint l = 0u; l < 8u; l++) {
		uint ml = MAT(c[l]);
		float em = erosion_mult(ml);
		if (em <= 0.0) { continue; }
		if ((water_flowing(c, l ^ 1u) || water_flowing(c, l ^ 2u) || water_flowing(c, l ^ 4u))
				&& rnd() < p.erode_prob * em) {
			uint into = ml == STONE ? SOIL : SAND;
			c[l] = PACK(into, min(SAT(c[l]), sat_cap(into)));
		}
	}

	// 6. material transitions + vegetation, per vertical pair (in-block only)
	if ((rules & 128u) != 0u)
	for (uint k = 0u; k < 4u; k++) {
		uint b = bottoms[k];
		uint t = b + 2u;
		// waterlogging: only SURFACE soil (air directly above) that saturates
		// turns to soft mud — submerged/buried soil stays firm, so lake beds
		// don't become sliding mud. Mud dries back to soil anywhere. Formation
		// is tested on the bottom cell, whose 'above' is the in-block top cell.
		if (MAT(c[b]) == SOIL && SAT(c[b]) >= MUD_WET && MAT(c[t]) == AIR)
			c[b] = PACK(MUD, SAT(c[b]));
		else if (MAT(c[b]) == MUD && SAT(c[b]) <= MUD_DRY) c[b] = PACK(SOIL, SAT(c[b]));
		if (MAT(c[t]) == MUD && SAT(c[t]) <= MUD_DRY) c[t] = PACK(SOIL, SAT(c[t]));

		// grass on the bottom cell (its 'above' is the in-block top cell):
		// grows on moist, lit surface soil; reverts if drowned/buried/parched
		uint mb = MAT(c[b]);
		uint mt = MAT(c[t]);
		if (mb == SOIL && mt == AIR && SAT(c[b]) >= 8u && SAT(c[b]) <= 108u
				&& rnd() < GROW) {
			c[b] = PACK(GRASS, SAT(c[b]));
		} else if (mb == GRASS && mt != AIR) {
			// drowned or buried (no light) -> vegetation dies back to soil
			c[b] = PACK(SOIL, SAT(c[b]));
		}

		// grass roots draw surrounding ground moisture down toward GRASS_FLOOR
		// (absorbing water from nearby voxels) — it never drinks itself dry,
		// and by lowering saturation it holds waterlogging/mud at bay
		if (MAT(c[t]) == GRASS && SAT(c[b]) > GRASS_FLOOR) {
			c[b] = PACK(MAT(c[b]), max(SAT(c[b]) - UPTAKE, GRASS_FLOOR));
		}
	}

	bool changed = false;
	for (uint l = 0u; l < 8u; l++) {
		if (ib[l] && cget(ids[l]) != c[l]) {
			cset(ids[l], c[l]);
			changed = true;
		}
	}
	if (changed) { mark_dirty(base.x, base.z); mark_dirty(base.x + 1u, base.z + 1u); }
}

// material albedo in LINEAR space (vertex/instance colors skip sRGB conversion)
vec3 mat_color(uint m) {
	if (m == BEDROCK) return vec3(0.042, 0.051, 0.065);
	if (m == STONE) return vec3(0.257, 0.266, 0.295);
	if (m == SOIL) return vec3(0.282, 0.145, 0.048);
	if (m == SAND) return vec3(0.552, 0.423, 0.174);
	if (m == MUD) return vec3(0.115, 0.072, 0.040);
	if (m == GRASS) return vec3(0.086, 0.210, 0.052);
	return vec3(1.0);
}

void write_inst(bool water, uint slot, vec3 origin, vec4 col, float scale) {
	uint b = slot * 16u;
	// 3x4 transform rows (scaled diagonal basis), then color
	float t[12] = float[12](scale, 0.0, 0.0, origin.x,
			0.0, scale, 0.0, origin.y,
			0.0, 0.0, scale, origin.z);
	for (uint k = 0u; k < 12u; k++) {
		if (water) { water_inst[b + k] = t[k]; } else { solid_inst[b + k] = t[k]; }
	}
	if (water) {
		water_inst[b+12u] = col.r; water_inst[b+13u] = col.g;
		water_inst[b+14u] = col.b; water_inst[b+15u] = col.a;
	} else {
		solid_inst[b+12u] = col.r; solid_inst[b+13u] = col.g;
		solid_inst[b+14u] = col.b; solid_inst[b+15u] = col.a;
	}
}

// GREEDY-MESH RENDER: write an oriented, stretched QUAD instance. The base mesh is
// a unit quad in local XZ [0,1]^2 (normal +Y), so a local vertex (lx,0,lz) maps to
// org + lx*ax + lz*az. ax/az are the two in-plane world edge vectors already scaled
// to the quad's size; ay is the outward normal (for lighting). Material is double-
// sided so winding doesn't matter. One quad can cover a whole merged face.
void write_quad(bool water, uint slot, vec3 ax, vec3 ay, vec3 az, vec3 org, vec4 col) {
	uint b = slot * 16u;
	float t[12] = float[12](ax.x, ay.x, az.x, org.x,
			ax.y, ay.y, az.y, org.y,
			ax.z, ay.z, az.z, org.z);
	for (uint k = 0u; k < 12u; k++) {
		if (water) { water_inst[b + k] = t[k]; } else { solid_inst[b + k] = t[k]; }
	}
	if (water) {
		water_inst[b+12u] = col.r; water_inst[b+13u] = col.g;
		water_inst[b+14u] = col.b; water_inst[b+15u] = col.a;
	} else {
		solid_inst[b+12u] = col.r; solid_inst[b+13u] = col.g;
		solid_inst[b+14u] = col.b; solid_inst[b+15u] = col.a;
	}
}

// emit a 1x1 quad on face k of a voxel at local (wx, Y, wz). k: 0 +X 1 -X 2 +Y
// 3 -Y 4 +Z 5 -Z. Stage 1a: one quad per exposed face (per-voxel, not yet merged),
// which should render identically to the cube emit but as flat faces.
void emit_face(bool water, uint k, float wx, float Y, float wz, vec4 col) {
	vec3 ax, ay, az, org;
	if (k == 0u)      { ax = vec3(0,1,0); az = vec3(0,0,1); ay = vec3( 1,0,0); org = vec3(wx+1.0, Y, wz); }
	else if (k == 1u) { ax = vec3(0,1,0); az = vec3(0,0,1); ay = vec3(-1,0,0); org = vec3(wx, Y, wz); }
	else if (k == 2u) { ax = vec3(1,0,0); az = vec3(0,0,1); ay = vec3(0, 1,0); org = vec3(wx, Y+1.0, wz); }
	else if (k == 3u) { ax = vec3(1,0,0); az = vec3(0,0,1); ay = vec3(0,-1,0); org = vec3(wx, Y, wz); }
	else if (k == 4u) { ax = vec3(1,0,0); az = vec3(0,1,0); ay = vec3(0,0, 1); org = vec3(wx, Y, wz+1.0); }
	else              { ax = vec3(1,0,0); az = vec3(0,1,0); ay = vec3(0,0,-1); org = vec3(wx, Y, wz); }
	if (water) { uint s = atomicAdd(n_water, 1u); if (s < cap_water) { write_quad(true, s, ax, ay, az, org, col); } }
	else       { uint s = atomicAdd(n_solid, 1u); if (s < cap_solid) { write_quad(false, s, ax, ay, az, org, col); } }
}

// one thread per COLUMN: emit exposed voxel FACES as quads (stage 1a of greedy
// meshing — machinery test, not yet merged). Same per-column walk + early-out as
// do_emit; same colour, so the result should match the cube render face-for-face.
void do_face_emit() {
	uint id = flat_id();
	if (id >= p.W * p.D) { return; }
	uint x = id % p.W;
	uint z = id / p.W;
	if (z < p.cut_z) { return; }
	float wx = float(world_coord(x, p.gen_ox, p.W) - p.gen_ox);
	float wz = float(world_coord(z, p.gen_oz, p.D) - p.gen_oz);
	// distance LOD: fine 5 cm faces only within lod_r of the camera (local frame);
	// the far terrain comes from do_lod_emit's coarse block quads. lod_r 0 = all fine.
	if (p.lod_r > 0u) {
		float ddx = wx - float(p.lod_cx);
		float ddz = wz - float(p.lod_cz);
		if (ddx * ddx + ddz * ddz > float(p.lod_r) * float(p.lod_r)) { return; }
	}
	uint buried = 0u;
	uint yy = p.H;
	while (yy > 0u) {
		yy--;
		uint cid = cidx(x, yy, z);
		uint raw = cget(cid);
		uint m = MAT(raw);
		if (m == AIR) { buried = 0u; continue; }
		bool is_water = m == WATER;
		float Y = float(int(yy) + int(p.gen_oy));
		uint nbr[6];
		int solid_n = 0;
		for (uint k = 0u; k < 6u; k++) {
			int dx = k==0u?1:(k==1u?-1:0);
			int dy = k==2u?1:(k==3u?-1:0);
			int dz = k==4u?1:(k==5u?-1:0);
			int nx = int(x)+dx, ny = int(yy)+dy, nz = int(z)+dz;
			uint nm = AIR;
			bool oob = nx<0 || nx>=int(p.W) || ny>=int(p.H) || nz<0 || nz>=int(p.D);
			if (ny < 0) { nm = BEDROCK; }
			else if (nz < int(p.cut_z)) { nm = AIR; }   // cross-section reveal stays open
			// sideways past the window edge counts SOLID: the camera lives inside a
			// streamed window, so the perimeter "diorama" walls are never visible —
			// at a 1260-voxel band they were ~2.7M instances (half the draw) of wall.
			else if (oob && dy == 0) { nm = BEDROCK; }
			else if (!oob) { nm = MAT(cget(cidx(uint(nx), uint(ny), uint(nz)))); }
			nbr[k] = nm;
			if (nm != AIR && nm != WATER) { solid_n += 1; }
		}
		vec4 col;
		if (is_water) { col = vec4(1.0); }
		else {
			float jit = 1.0 + (float(pcg(cid * 2654435761u) & 255u) / 255.0 * 0.24 - 0.12);
			float ao = 1.0 - float(solid_n) * 0.05;
			vec3 base = mat_color(m);
			if (m == SOIL || m == SAND) {
				float wet = clamp(float(SAT(raw)) / float(sat_cap(m)), 0.0, 1.0);
				base = mix(base, vec3(0.09, 0.05, 0.03), wet * 0.8);
			}
			col = vec4(base * jit * ao, 1.0);
		}
		for (uint k = 0u; k < 6u; k++) {
			uint nm = nbr[k];
			bool open = is_water ? (nm == AIR) : (nm == AIR || nm == WATER);
			if (open) { emit_face(is_water, k, wx, Y, wz, col); }
		}
		if (solid_n == 6) { buried += 1u; if (buried >= 2u) { break; } }
		else { buried = 0u; }
	}
}

// one thread per COLUMN: walk it top-down and emit every exposed voxel, then stop
// once a cell is buried in solid on all six sides — in this sim nothing below such
// a cell is ever visible (no caves; water sits on top, the subsurface is solid
// soil/stone). This skips the deep buried interior the old per-cell dispatch
// scanned (940M threads -> W*D threads), the same trick the block shell uses, at
// full 5 cm detail. Exposure/colour logic is unchanged.
void do_emit() {
	uint id = flat_id();
	if (id >= p.W * p.D) { return; }
	uint x = id % p.W;
	uint z = id / p.W;
	if (z < p.cut_z) { return; }   // whole column hidden behind the cross-section
	uint buried = 0u;
	uint yy = p.H;
	while (yy > 0u) {
		yy--;
		uint cid = cidx(x, yy, z);
		uint raw = cget(cid);
		uint m = MAT(raw);
		if (m == AIR) { buried = 0u; continue; }
		bool is_water = m == WATER;
		// exposure test mirrors the face-cull rules: solids show against
		// AIR/WATER, water only against AIR; OOB sides count as exposed
		// (diorama cut faces), the underside does not
		bool exposed = false;
		int solid_n = 0;
		for (uint k = 0u; k < 6u; k++) {
			int dx = k == 0u ? 1 : (k == 1u ? -1 : 0);
			int dy = k == 2u ? 1 : (k == 3u ? -1 : 0);
			int dz = k == 4u ? 1 : (k == 5u ? -1 : 0);
			int nx = int(x) + dx;
			int ny = int(yy) + dy;
			int nz = int(z) + dz;
			uint nm = AIR;
			bool oob = nx < 0 || nx >= int(p.W) || ny >= int(p.H) || nz < 0 || nz >= int(p.D);
			if (ny < 0) { nm = BEDROCK; }
			else if (nz < int(p.cut_z)) { nm = AIR; }   // across the cut = open (reveal section)
			else if (!oob) { nm = MAT(cget(cidx(uint(nx), uint(ny), uint(nz)))); }
			if (nm != AIR && nm != WATER) { solid_n += 1; }
			if (is_water ? (nm == AIR) : (nm == AIR || nm == WATER)) { exposed = true; }
		}
		if (exposed) {
			// FLOATING ORIGIN: emit relative to the window origin (world_coord -
			// origin, in [0, W/D)) so 5 cm cubes are drawn near zero, not at world
			// coords ~1e5 where float32 cracks seams; the camera is offset to match.
			vec3 origin = vec3(float(world_coord(x, p.gen_ox, p.W) - p.gen_ox) + 0.5,
					float(int(yy) + int(p.gen_oy)) + 0.5,
					float(world_coord(z, p.gen_oz, p.D) - p.gen_oz) + 0.5);
			if (is_water) {
				uint slot = atomicAdd(n_water, 1u);
				if (slot < cap_water) { write_inst(true, slot, origin, vec4(1.0), 1.0); }
			} else {
				// per-voxel tint jitter + crude AO from buried-ness
				float jit = 1.0 + (float(pcg(cid * 2654435761u) & 255u) / 255.0 * 0.24 - 0.12);
				float ao = 1.0 - float(solid_n) * 0.05;
				vec3 base = mat_color(m);
				// darken dry-able ground (soil/sand) toward wet earth as it saturates
				if (m == SOIL || m == SAND) {
					float wet = clamp(float(SAT(raw)) / float(sat_cap(m)), 0.0, 1.0);
					base = mix(base, vec3(0.09, 0.05, 0.03), wet * 0.8);
				}
				vec4 col = vec4(base * jit * ao, 1.0);
				uint slot = atomicAdd(n_solid, 1u);
				if (slot < cap_solid) { write_inst(false, slot, origin, col, 1.0); }
			}
		}
		// buried in solid on all six sides -> everything below in this column is
		// hidden too (heightfield terrain, no caves); stop after a couple to be safe
		if (solid_n == 6) { buried += 1u; if (buried >= 2u) { break; } }
		else { buried = 0u; }
	}
}

// one thread per 1 m block COLUMN (20x20 voxels): render the fine 5 cm sim as a
// heightmap of solid 1 m cubes. Scan the column's centre for the surface, round
// its height to whole blocks, then stack cubes bottom-to-surface: the TOP cube
// is always coloured by the true surface material (grass cap / lake water),
// body cubes by the material at their own centre (soil / stone / water). This
// decouples render cost from sim resolution, so the 5 cm hydrology can cover a
// much larger map, and the surface never drops out or reads as bare stone.
void do_block_emit() {
	uint nbx = (p.W + 19u) / 20u;
	uint nbz = (p.D + 19u) / 20u;
	uint id = flat_id();
	if (id >= nbx * nbz) { return; }
	uint bz = id / nbx;
	uint bx = id % nbx;
	uint x0 = bx * 20u, z0 = bz * 20u;
	if (z0 + 19u < p.cut_z) { return; }   // block-aligned cross-section
	uint cx = min(x0 + 10u, p.W - 1u);
	uint cz = min(z0 + 10u, p.D - 1u);
	// surface = top-most non-air voxel in the centre column (full scan)
	int surfy = -1;
	uint surfmat = AIR;
	for (uint yy = p.H; yy > 0u; ) {
		yy--;
		uint m = MAT(cget(cidx(cx, yy, cz)));
		if (m != AIR) { surfy = int(yy); surfmat = m; break; }
	}
	if (surfy < 0) { return; }                       // empty column
	uint ntop = max((uint(surfy) + 10u) / 20u, 1u);  // height rounded to whole blocks
	for (uint by = 0u; by < ntop; by++) {
		uint y0 = by * 20u;
		uint m = surfmat;                            // top cube = surface material
		if (by + 1u < ntop) {                        // body cube = its own centre
			m = MAT(cget(cidx(cx, min(y0 + 10u, p.H - 1u), cz)));
			if (m == AIR) { m = SOIL; }
		}
		vec3 center = vec3(float(int(x0)) + 10.0,        // local frame (floating origin)
				float(int(y0) + int(p.gen_oy)) + 10.0,
				float(int(z0)) + 10.0);
		if (m == WATER) {
			uint slot = atomicAdd(n_water, 1u);
			if (slot < cap_water) { write_inst(true, slot, center, vec4(1.0), 20.0); }
		} else {
			float shade = 0.74 + 0.26 * clamp(float(y0) / float(p.H), 0.0, 1.0);
			// per-cube tint jitter (hash of block position) so each 1m block
			// varies individually, like the per-voxel tint of the fine renderer
			uint h = pcg((x0 * 73856093u) ^ (y0 * 19349663u) ^ (z0 * 83492791u));
			float jit = 1.0 + (float(h & 255u) / 255.0 * 0.24 - 0.12);
			uint slot = atomicAdd(n_solid, 1u);
			if (slot < cap_solid) { write_inst(false, slot, center, vec4(mat_color(m) * shade * jit, 1.0), 20.0); }
		}
	}
}

// full surface colour of one voxel, including per-voxel tint jitter and the
// wet-darkening of saturated soil/sand (mirrors the fine per-voxel renderer)
vec3 surf_color(uint raw, uint m, uint id) {
	float jit = 1.0 + (float(pcg(id * 2654435761u) & 255u) / 255.0 * 0.24 - 0.12);
	vec3 base = mat_color(m);
	if (m == SOIL || m == SAND) {
		float wet = clamp(float(SAT(raw)) / float(sat_cap(m)), 0.0, 1.0);
		base = mix(base, vec3(0.09, 0.05, 0.03), wet * 0.8);
	}
	return base * jit;
}

// block-snapped solid top (in voxels) of the block at (bx, bz): its centre
// column's surface, rounded to whole 1 m blocks, so a whole 20x20 block shares
// one flat top. 0 if the column is empty.
uint block_top_h(uint bx, uint bz) {
	uint cx = min(bx * 20u + 10u, p.W - 1u);
	uint cz = min(bz * 20u + 10u, p.D - 1u);
	for (uint yy = p.H; yy > 0u; ) {
		yy--;
		if (MAT(cget(cidx(cx, yy, cz))) != AIR) { return max((yy + 10u) / 20u, 1u) * 20u; }
	}
	return 0u;
}

// 1 m blocks skinned with 5 cm voxels on EVERY visible face. One thread per
// 5 cm column emits a shell of 5 cm voxel-cubes for its column of the block-
// snapped terrain: the top voxel (always exposed) plus, where an adjacent block
// is shorter, the side voxels down to that neighbour's height — each tinted by
// its own real voxel (grass on top, soil/stone strata down the cliffs). The
// shape stays chunky (flat 1 m tops, 1 m-aligned steps) while every face shows
// the fine per-voxel detail; interior voxels are occluded, so none are emitted.
void do_skin_emit() {
	uint id = flat_id();
	if (id >= p.W * p.D) { return; }
	uint x = id % p.W;
	uint z = id / p.W;
	if (z < p.cut_z) { return; }
	uint bx = x / 20u, bz = z / 20u;
	uint topf = block_top_h(bx, bz);
	if (topf == 0u) { return; }
	// this column's own surface voxel (fallback tint where the snap rounds the
	// block top above the real 5 cm terrain)
	uint sm = SOIL; uint sraw = 0u;
	for (uint yy = topf; yy > 0u; ) {
		yy--; uint raw = cget(cidx(x, yy, z));
		if (MAT(raw) != AIR) { sm = MAT(raw); sraw = raw; break; }
	}
	// lowest exposed voxel: the top is always exposed; each block edge whose
	// neighbouring block is shorter exposes this column's side down to it
	uint low = topf - 1u;
	bool xhi = x % 20u == 19u && x + 1u < p.W;
	bool xlo = x % 20u == 0u  && x > 0u;
	bool zhi = z % 20u == 19u && z + 1u < p.D;
	bool zlo = z % 20u == 0u  && z > 0u;
	if (xhi) { uint hn = block_top_h(bx + 1u, bz); if (hn < topf) { low = min(low, hn); } }
	if (xlo) { uint hn = block_top_h(bx - 1u, bz); if (hn < topf) { low = min(low, hn); } }
	if (zhi) { uint hn = block_top_h(bx, bz + 1u); if (hn < topf) { low = min(low, hn); } }
	if (zlo) { uint hn = block_top_h(bx, bz - 1u); if (hn < topf) { low = min(low, hn); } }
	// diagonal corner columns: if only the DIAGONAL block is shorter, the 4 checks
	// above miss it and a pinhole opens at the block corner. Seal it by dropping to
	// the diagonal neighbour's height too.
	if (xhi && zhi) { uint hn = block_top_h(bx + 1u, bz + 1u); if (hn < topf) { low = min(low, hn); } }
	if (xhi && zlo) { uint hn = block_top_h(bx + 1u, bz - 1u); if (hn < topf) { low = min(low, hn); } }
	if (xlo && zhi) { uint hn = block_top_h(bx - 1u, bz + 1u); if (hn < topf) { low = min(low, hn); } }
	if (xlo && zlo) { uint hn = block_top_h(bx - 1u, bz - 1u); if (hn < topf) { low = min(low, hn); } }
	float wx = float(world_coord(x, p.gen_ox, p.W) - p.gen_ox);   // local frame (floating origin)
	float wz = float(world_coord(z, p.gen_oz, p.D) - p.gen_oz);
	for (uint y = low; y < topf; y++) {
		uint raw = cget(cidx(x, y, z));
		uint m = MAT(raw);
		if (m == AIR) { m = sm; raw = sraw; }
		vec3 c = vec3(wx + 0.5, float(int(y) + int(p.gen_oy)) + 0.5, wz + 0.5);
		uint vid = x + z * p.W + y * p.W * p.D;   // stable per-voxel id for tint jitter
		if (m == WATER) {
			uint s = atomicAdd(n_water, 1u);
			if (s < cap_water) { write_inst(true, s, c, vec4(1.0), 1.0); }
		} else {
			uint s = atomicAdd(n_solid, 1u);
			if (s < cap_solid) { write_inst(false, s, c, vec4(surf_color(raw, m, vid), 1.0), 1.0); }
		}
	}
}

// ---- distance LOD: coarse far terrain (mode 11) ----

// exact surface top (buffer y of the top solid/water cell + 1) of one column
uint col_top(uint x, uint z) {
	uint yy = p.H;
	while (yy > 0u) {
		yy--;
		if (MAT(cget(cidx(x, yy, z))) != AIR) { return yy + 1u; }
	}
	return 0u;
}

// representative top of a 1 m block: the MIN of its centre + 4 corner columns,
// returned as (top, column x, column z) — the COLUMN MATTERS: the surface
// material must be sampled from the column that set the min (sampling another
// column at that height reads its subsoil and paints far terrain brown/grey).
// Using the min means the far quad never pokes above the fine near geometry in
// the LOD-boundary overlap ring — worst case it sits slightly recessed, hidden
// behind the skirts, which reads fine at distance.
uvec3 lod_top(uint bx, uint bz) {
	uint x0 = bx * 20u, z0 = bz * 20u;
	uint x1 = min(x0 + 19u, p.W - 1u), z1 = min(z0 + 19u, p.D - 1u);
	uint cx = min(x0 + 10u, p.W - 1u), cz = min(z0 + 10u, p.D - 1u);
	uvec3 best = uvec3(col_top(cx, cz), cx, cz);
	uint t = col_top(x0, z0); if (t < best.x) { best = uvec3(t, x0, z0); }
	t = col_top(x1, z0); if (t < best.x) { best = uvec3(t, x1, z0); }
	t = col_top(x0, z1); if (t < best.x) { best = uvec3(t, x0, z1); }
	t = col_top(x1, z1); if (t < best.x) { best = uvec3(t, x1, z1); }
	return best;
}

// FAR LOD: one thread per 1 m block (20x20 columns). Beyond the near radius,
// emit ONE top quad per block plus side skirts down to each lower neighbour —
// the correct silhouette at distance for ~1/400th the instances of fine columns.
// Overlaps the fine region by one block so the boundary ring can't gap open.
void do_lod_emit() {
	if (p.lod_r == 0u) { return; }
	uint nbxl = (p.W + 19u) / 20u;
	uint nbzl = (p.D + 19u) / 20u;
	uint id = flat_id();
	if (id >= nbxl * nbzl) { return; }
	uint bx = id % nbxl;
	uint bz = id / nbxl;
	uint x0 = bx * 20u, z0 = bz * 20u;
	if (z0 + 19u < p.cut_z) { return; }
	// skip blocks straddling the toroidal seam (their columns are world-far apart)
	uint xe = min(x0 + 19u, p.W - 1u), ze = min(z0 + 19u, p.D - 1u);
	float wx0 = float(world_coord(x0, p.gen_ox, p.W) - p.gen_ox);
	float wz0 = float(world_coord(z0, p.gen_oz, p.D) - p.gen_oz);
	if (float(world_coord(xe, p.gen_ox, p.W) - p.gen_ox) - wx0 != float(xe - x0)) { return; }
	if (float(world_coord(ze, p.gen_oz, p.D) - p.gen_oz) - wz0 != float(ze - z0)) { return; }
	// the fine pass owns the near disc; keep blocks whose centre is beyond
	// lod_r - 28 (one block + margin of overlap into the fine region)
	float ddx = (wx0 + 10.0) - float(p.lod_cx);
	float ddz = (wz0 + 10.0) - float(p.lod_cz);
	float rr = max(float(p.lod_r) - 28.0, 0.0);
	if (ddx * ddx + ddz * ddz < rr * rr) { return; }
	uvec3 bt = lod_top(bx, bz);
	uint top = bt.x;
	if (top == 0u) { return; }
	uint ccx = bt.y, ccz = bt.z;   // sample material at the min column's own surface
	uint traw = cget(cidx(ccx, top - 1u, ccz));
	uint tm = MAT(traw);
	bool iw = tm == WATER;
	float Y = float(int(top) + int(p.gen_oy));
	// per-block tint id keeps the coarse quads from reading as one flat colour
	uint bid = x0 + z0 * p.W;
	vec4 tcol = iw ? vec4(1.0) : vec4(surf_color(traw, tm, bid), 1.0);
	if (iw) {
		uint sl = atomicAdd(n_water, 1u);
		if (sl < cap_water) { write_quad(true, sl, vec3(20,0,0), vec3(0,1,0), vec3(0,0,20), vec3(wx0, Y, wz0), tcol); }
	} else {
		uint sl = atomicAdd(n_solid, 1u);
		if (sl < cap_solid) { write_quad(false, sl, vec3(20,0,0), vec3(0,1,0), vec3(0,0,20), vec3(wx0, Y, wz0), tcol); }
	}
	// skirt colour: the SURFACE material. A skirt mostly stands in for a run of
	// gentle steps whose visible faces are grass tops and 1-voxel grass risers —
	// colouring it subsoil brown painted every distant ridge brown. The side
	// normal's lighting darkens it naturally, like the fine render's risers.
	vec4 scol = tcol;
	for (uint k = 0u; k < 4u; k++) {
		int nx = int(bx) + (k == 0u ? 1 : (k == 1u ? -1 : 0));
		int nz = int(bz) + (k == 2u ? 1 : (k == 3u ? -1 : 0));
		if (nx < 0 || nx >= int(nbxl) || nz < 0 || nz >= int(nbzl)) { continue; }
		uint nt = lod_top(uint(nx), uint(nz)).x;
		if (nt >= top || nt == 0u) { continue; }
		float h0 = float(int(nt) + int(p.gen_oy));
		float hh = float(top - nt);
		vec3 ax, ay, az, org;
		if (k == 0u)      { ax = vec3(0,hh,0); az = vec3(0,0,20); ay = vec3( 1,0,0); org = vec3(wx0+20.0, h0, wz0); }
		else if (k == 1u) { ax = vec3(0,hh,0); az = vec3(0,0,20); ay = vec3(-1,0,0); org = vec3(wx0, h0, wz0); }
		else if (k == 2u) { ax = vec3(20,0,0); az = vec3(0,hh,0); ay = vec3(0,0, 1); org = vec3(wx0, h0, wz0+20.0); }
		else              { ax = vec3(20,0,0); az = vec3(0,hh,0); ay = vec3(0,0,-1); org = vec3(wx0, h0, wz0); }
		if (iw) {
			uint sl = atomicAdd(n_water, 1u);
			if (sl < cap_water) { write_quad(true, sl, ax, ay, az, org, vec4(1.0)); }
		} else {
			uint sl = atomicAdd(n_solid, 1u);
			if (sl < cap_solid) { write_quad(false, sl, ax, ay, az, org, scol); }
		}
	}
}

// ---- GPU worldgen: value-noise fbm heightfield, bowl basin, ridge ----

float vhash(vec2 q) {
	uint h = pcg(uint(int(q.x)) * 374761393u ^ uint(int(q.y)) * 668265263u ^ p.seedv);
	return float(h) * (1.0 / 4294967296.0);
}

float vnoise(vec2 q) {
	vec2 i = floor(q);
	vec2 f = q - i;
	vec2 u = f * f * (3.0 - 2.0 * f);
	float a = vhash(i);
	float b = vhash(i + vec2(1.0, 0.0));
	float c = vhash(i + vec2(0.0, 1.0));
	float d = vhash(i + vec2(1.0, 1.0));
	return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 q) {
	float v = 0.0;
	float amp = 0.5;
	for (int o = 0; o < 4; o++) {
		v += vnoise(q) * amp;
		q *= 2.03;
		amp *= 0.5;
	}
	return v;   // ~0..1
}

// ---------- hierarchical terrain: chunk -> block -> voxel ----------
// Voxels are grouped 20x20 into a BLOCK (1 m) and 20x20 blocks into a CHUNK
// (20 m). Surface height is a pure function of world (x,z) + seed that sums
// bands of continuous value-noise per level of the hierarchy. Because it only
// samples world-space noise — never the world size or a fixed centre — the
// topography is seamless across every chunk boundary and comes out identical
// however the world is windowed or streamed (streaming-ready).
const float BLOCK_VOX = 20.0;       // voxels per block edge (1 m)
const float CHUNK_VOX = 400.0;      // voxels per chunk edge (20 blocks, 20 m)

// VERTICAL-TRACKING WORLD: the terrain surface is a world-Y value spanning the
// full RELIEF (256 m), but the sim buffer is only a thin height BAND (p.H) that
// rides up/down with the surface (gen_oy = band's world-Y floor). So the surface
// amplitude is decoupled from the band height — it is this fixed RELIEF, not p.H.
// Features are DELIBERATELY WIDE (hundreds of metres) so 256 m of relief reads as
// gentle mountains within any small window, not spikes; only the near-surface is
// stored, deep rock below the band is implicit bedrock, high air above is implicit.
const float RELIEF = 5120.0;        // total vertical relief in voxels (256 m at 5 cm)
const float SEA_Y  = RELIEF * 0.30; // fixed sea level, world-Y voxels

// Terrain height sampled at a BLOCK grid point (bc in block units, i.e. world
// voxels / 20), returned as a WORLD-Y voxel value in ~[0, RELIEF]. Wide bands:
// a continental landform (~600 m, squared toward its tails so terrain spends time
// as plains and peaks), hills (~120 m), and per-block detail (~25 m). All widths
// are absolute world scales so relief is gentle regardless of the window size.
float block_height(vec2 bc, float H, vec2 s) {
	vec2 w = bc * BLOCK_VOX;   // block lattice back to world voxels for wide sampling
	// Features are WIDE so 256 m of relief is a gentle slope inside a ~50 m window
	// (fits the resident band) yet valleys-to-peaks span the full RELIEF as you fly.
	// Kept within ~[0, RELIEF] so the trackable band can always reach the surface.
	// "gentle-plus" steepness (~1.5x the widest setting): a few hills & valleys per
	// view, still well inside the resident band. Widths must match the gpu_world.gd
	// CPU mirror (_block_height) exactly or the band mis-places.
	float cn = fbm(w / 21000.0 + s) - 0.5;                // ~1.05 km continental
	float chunk = cn * (1.0 + 2.0 * abs(cn));             // plains & peaks
	float hill = fbm(w / 2700.0 + s * 2.0 + 31.7) - 0.5;  // ~135 m hills
	float det = fbm(w / 600.0 + s * 4.0 + 91.3) - 0.5;    // ~30 m detail
	return RELIEF * (0.5 + chunk * 0.44 + hill * 0.06 + det * 0.02);
}

// Surface height at a voxel column. BOTH modes read the SAME block (1 m)
// heightfield — they differ only in how the 5 cm voxels fill between block
// samples (gen_flags bit0): TERRACED gives each block a flat plateau snapped to
// 1 m steps (clean 1 m voxel-cubes), BLENDED smooth-interpolates the 4
// surrounding block heights (the same terrain, just smoothed — no extra bands).
// Both are pure functions of world (x,z)+seed and the block grid aligns to
// chunk edges (20 per chunk), so either tiles seamlessly.
float world_height(vec2 w) {
	float H = float(p.H);
	vec2 s = vec2(float(p.seedv & 0xFFFFu), float((p.seedv >> 16) & 0xFFFFu)) * 0.618;
	if ((p.gen_flags & 1u) != 0u) {
		// TERRACED: flat per-block footprint, height snapped to 1 m steps
		vec2 bc = floor(w / BLOCK_VOX);
		return floor(block_height(bc, H, s) / BLOCK_VOX) * BLOCK_VOX;
	}
	// BLENDED: bilinear (smoothstep) interpolation of the same 4 block heights
	vec2 bcf = w / BLOCK_VOX;
	vec2 i = floor(bcf);
	vec2 f = bcf - i;
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(
		mix(block_height(i, H, s),                block_height(i + vec2(1.0, 0.0), H, s), u.x),
		mix(block_height(i + vec2(0.0, 1.0), H, s), block_height(i + vec2(1.0, 1.0), H, s), u.x),
		u.y);
}

// one thread per column: carve the whole column
void do_gen() {
	uint id = flat_id();
	if (id >= p.W * p.D) { return; }
	uint x = id % p.W;
	uint z = id / p.W;
	// toroidal edge regen: only fill the targeted buffer-slot strip (modular so
	// it may wrap around the buffer); full regen has width == W / depth == D.
	uint sx0 = p.strip_x & 0xFFFFu, sw = p.strip_x >> 16u;
	uint sz0 = p.strip_z & 0xFFFFu, sd = p.strip_z >> 16u;
	if (((x + p.W - sx0) % p.W) >= sw) { return; }
	if (((z + p.D - sz0) % p.D) >= sd) { return; }
	// world column of this buffer slot on the torus; the seamless world-space
	// noise makes the joins invisible.
	vec2 w = vec2(float(world_coord(x, p.gen_ox, p.W)), float(world_coord(z, p.gen_oz, p.D)));
	// world-Y of the surface (0..RELIEF) and the fixed sea level. The buffer only
	// covers the band [gen_oy, gen_oy+H); each cell's world-Y = gen_oy + y. Below
	// the band is implicit deep rock (the buffer floor is a solid boundary), above
	// is implicit air, so the sim rides a thin near-surface slice of a tall world.
	int top = int(world_height(w));
	int sea = int(SEA_Y);                        // basins below this start as lakes
	int oy = int(p.gen_oy);
	// NEVER generate an empty column: if the surface falls below the band floor
	// (stale band during streaming, or a narrow dip the tracker's sampling
	// missed), clamp it to a thin floor — a sunken pit renders and simulates
	// fine, where an empty column was a hole clean through the world.
	if (top < oy + 2) { top = oy + 2; }
	for (uint y = 0u; y < p.H; y++) {
		int wy = oy + int(y);                    // this cell's world-Y
		uint m = AIR;
		uint s0 = 0u;
		if (wy == 0) { m = BEDROCK; }
		else if (wy < top - 4) { m = STONE; }
		else if (wy < top) { m = SOIL; s0 = 40u; }   // field-capacity moisture
		// the land surface that stands above the water line starts vegetated;
		// low ground stays firm soil (loose sand isn't pre-placed — it forms
		// where water erodes, so freshly generated slopes don't avalanche)
		if (m == SOIL && wy == top - 1 && top > sea) { m = GRASS; s0 = 40u; }
		// submerged ground generates already SATURATED so the lakes sit on wet
		// beds and don't seep away. The one exception is the ground's exposed
		// SURFACE cell at the water's EDGE (a waterline column, top==sea, so its
		// top-1 surface has AIR above): saturated to capacity it would be over the
		// mud threshold and instantly waterlog into sliding mud that avalanches
		// into the lake. Under deep water (top<sea) the surface keeps its cap.
		if ((m == SOIL || m == SAND) && wy < sea
				&& (top < sea || wy < top - 1)) { s0 = sat_cap(m); }
		// fill low basins and valleys with standing water up to the water line
		if (m == AIR && wy < sea) { m = WATER; s0 = 0u; }
		cset(cidx(x, y, z), PACK(m, s0));
	}
	// wake the freshly generated chunk so the physics settles it (then it sleeps)
	uint cw = (p.W + CHUNK - 1u) / CHUNK;
	awake[(z / CHUNK) * cw + (x / CHUNK)] = KEEPALIVE;
}

// one thread per chunk: count the active flag down. A chunk woken by mark_dirty is
// set to KEEPALIVE and decays 1/tick, so it stays awake KEEPALIVE ticks after its
// last change, then sleeps and the physics stops reading its cells.
void do_decay() {
	uint cw = (p.W + CHUNK - 1u) / CHUNK;
	uint cd = (p.D + CHUNK - 1u) / CHUNK;
	uint id = flat_id();
	if (id >= cw * cd) { return; }
	if (awake[id] > 0u) { awake[id] -= 1u; }
}

// ---- far field (mode 12): render-only terrain to the horizon ----
// Beyond the resident sim window, terrain is drawn straight from the worldgen
// heightfield — a pure function of world (x,z) + seed — with no cells and no sim.
// Three square clip-rings of growing tile size follow the camera out to FAR_END
// (8 km at 5 cm). Tiles snap to their own absolute grid so they don't swim as the
// camera moves; each emits a flat top quad, plus skirts down to its lower east /
// south neighbours (west/north edges are covered by THOSE tiles' skirts).
// Basins below the sea line render as a flat water plane at SEA_Y, like the gen.
const float FAR_END = 160000.0;                              // 8 km
const float RING_TILE[3]  = float[3](40.0, 160.0, 640.0);    // 2 m / 8 m / 32 m
const float RING_OUTER[3] = float[3](8000.0, 32000.0, FAR_END);

void do_far_emit() {
	uint ring = p.offset;               // reused push field: ring index 0..2
	float tile = RING_TILE[ring];
	float outer = RING_OUTER[ring];
	float inner = ring == 0u ? 0.0 : RING_OUTER[ring - 1u];
	uint side = uint(outer * 2.0 / tile);
	uint id = flat_id();
	if (id >= side * side) { return; }
	float camx = float(p.gen_ox + p.lod_cx);   // camera in world voxels
	float camz = float(p.gen_oz + p.lod_cz);
	float gx = floor(camx / tile) * tile + (float(id % side) - float(side / 2u)) * tile;
	float gz = floor(camz / tile) * tile + (float(id / side) - float(side / 2u)) * tile;
	float dx = gx + tile * 0.5 - camx;
	float dz = gz + tile * 0.5 - camz;
	float d = max(abs(dx), abs(dz));           // square-ring metric
	// rings OVERLAP inward by one tile: at a ring boundary the coarser ring's
	// heights differ slightly (coarser sampling), and skirts don't bridge across
	// ring sizes — the overlap row tucks under the finer ring and seals the seam
	if (d >= outer || d < inner - tile) { return; }
	// the resident window renders its own footprint — skip tiles inside it, but
	// keep one tile of overlap tucked UNDER its edge (the window's perimeter
	// walls are culled, so a lower far tile would otherwise show a gap)
	float lx = gx + tile * 0.5 - float(p.gen_ox);
	float lz = gz + tile * 0.5 - float(p.gen_oz);
	if (lx >= tile && lx <= float(p.W) - tile && lz >= tile && lz <= float(p.D) - tile) { return; }
	float h = world_height(vec2(gx + tile * 0.5, gz + tile * 0.5));
	float sea = SEA_Y;
	bool iw = h < sea;
	float Y = iw ? sea : h;
	// per-tile tint jitter (absolute tile coords, so it's stable as rings move)
	uint th = pcg(uint(int(gx / tile)) * 73856093u ^ uint(int(gz / tile)) * 83492791u ^ p.seedv);
	float jit = 1.0 + (float(th & 255u) / 255.0 * 0.20 - 0.10);
	vec4 col = iw ? vec4(1.0) : vec4(mat_color(GRASS) * jit, 1.0);
	vec3 org = vec3(gx - float(p.gen_ox), Y, gz - float(p.gen_oz));   // local frame
	if (iw) {
		uint sl = atomicAdd(n_water, 1u);
		if (sl < cap_water) { write_quad(true, sl, vec3(tile,0,0), vec3(0,1,0), vec3(0,0,tile), org, col); }
		return;                                   // the sea is flat: no skirts
	}
	uint sl = atomicAdd(n_solid, 1u);
	if (sl < cap_solid) { write_quad(false, sl, vec3(tile,0,0), vec3(0,1,0), vec3(0,0,tile), org, col); }
	// skirts on EVERY edge where this tile is the higher side (the lower
	// neighbour never emits that edge, so covering only E/S left cracks wherever
	// the ground rose to the east or south). Clamped up to the sea plane.
	for (uint k = 0u; k < 4u; k++) {
		vec2 off = k == 0u ? vec2(tile, 0.0) : (k == 1u ? vec2(-tile, 0.0)
				: (k == 2u ? vec2(0.0, tile) : vec2(0.0, -tile)));
		float hn = max(world_height(vec2(gx + tile * 0.5, gz + tile * 0.5) + off), sea);
		if (h - hn < 0.5) { continue; }
		hn -= 20.0;            // apron: overshoot 1 m down so sampling mismatches
		float hh = h - hn;     // (ring boundaries, window edge) can't open cracks
		float lx0 = gx - float(p.gen_ox);
		float lz0 = gz - float(p.gen_oz);
		vec3 ax, ay, az, sorg;
		if (k == 0u)      { ax = vec3(0,hh,0); az = vec3(0,0,tile); ay = vec3( 1,0,0); sorg = vec3(lx0 + tile, hn, lz0); }
		else if (k == 1u) { ax = vec3(0,hh,0); az = vec3(0,0,tile); ay = vec3(-1,0,0); sorg = vec3(lx0, hn, lz0); }
		else if (k == 2u) { ax = vec3(tile,0,0); az = vec3(0,hh,0); ay = vec3(0,0, 1); sorg = vec3(lx0, hn, lz0 + tile); }
		else              { ax = vec3(tile,0,0); az = vec3(0,hh,0); ay = vec3(0,0,-1); sorg = vec3(lx0, hn, lz0); }
		uint s2 = atomicAdd(n_solid, 1u);
		if (s2 < cap_solid) { write_quad(false, s2, ax, ay, az, sorg, col); }
	}
}

// diagnostic (mode 13): count EMPTY columns — a column with no cells at all is a
// hole in the world (its surface fell outside the resident band), which renders
// as a missing chunk. Counts into the stats 'rained' slot.
void do_holes() {
	uint id = flat_id();
	if (id >= p.W * p.D) { return; }
	if (col_top(id % p.W, id / p.W) == 0u) { atomicAdd(rained, 1u); }
}

// ---- ray-cast renderer (modes 14-16), GigaVoxels-inspired ----
// Instead of emitting instances, march eye rays through the ground heightfield:
// per-column heights + a per-16x16-tile max grid let rays skip empty space, so
// draw cost scales with SCREEN RESOLUTION, not window size or instance count.

// mode 14: per-column CONTIGUOUS ground height (scan up from the floor to the
// first air). Airborne grains (falling rain/sand) are ignored by this renderer.
// Cells are indexed by BUFFER SLOT (torus), but rays march in the LOCAL frame —
// so heights are written at the column's LOCAL index (world_coord - origin),
// unshuffling the toroidal seam once here instead of per ray step.
void do_heights() {
	uint id = flat_id();
	if (id >= p.W * p.D) { return; }
	uint x = id % p.W;
	uint z = id / p.W;
	uint y = 0u;
	while (y < p.H && MAT(cget(cidx(x, y, z))) != AIR) { y++; }
	uint lx = world_coord(x, p.gen_ox, p.W) - p.gen_ox;
	uint lz = world_coord(z, p.gen_oz, p.D) - p.gen_oz;
	heights[lz * p.W + lx] = y;
}

// mode 15: per-tile max of the column heights (16x16 columns, the chunk grid)
void do_hmax() {
	uint cw = (p.W + CHUNK - 1u) / CHUNK;
	uint cd = (p.D + CHUNK - 1u) / CHUNK;
	uint id = flat_id();
	if (id >= cw * cd) { return; }
	uint x0 = (id % cw) * CHUNK, z0 = (id / cw) * CHUNK;
	uint m = 0u;
	for (uint j = 0u; j < CHUNK; j++) {
		for (uint i = 0u; i < CHUNK; i++) {
			uint x = min(x0 + i, p.W - 1u), z = min(z0 + j, p.D - 1u);
			m = max(m, heights[z * p.W + x]);
		}
	}
	hmax[id] = m;
}

// mode 16: one thread per pixel — march the ray through the heightfield.
// Two-level DDA: step 5 cm columns, but on entering a 16-column tile whose max
// height stays below the ray, jump straight to the tile's exit. On a hit, shade
// the actual CELL (same per-voxel colour/jitter/wetness as the raster emit).
void do_raycast() {
	uint id = flat_id();
	if (id >= p.img_w * p.img_h) { return; }
	int px = int(id % p.img_w);
	int py = int(id / p.img_w);
	float u = (float(px) + 0.5) / float(p.img_w) * 2.0 - 1.0;
	float v = 1.0 - (float(py) + 0.5) / float(p.img_h) * 2.0;
	vec3 o = cam_eye.xyz;
	o.y -= float(p.gen_oy);   // camera rides in world-Y; heights are band-local
	vec3 d = normalize(cam_fwd.xyz + cam_right.xyz * (u * cam_right.w) + cam_up.xyz * (v * cam_up.w));
	float W = float(p.W), D = float(p.D);
	// clip to the window's footprint
	float t0 = 0.0, t1 = 1.0e9;
	if (abs(d.x) < 1e-6) {
		if (o.x < 0.0 || o.x > W) { imageStore(out_img, ivec2(px, py), vec4(0.0)); return; }
	} else {
		float ta = -o.x / d.x, tb = (W - o.x) / d.x;
		t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb));
	}
	if (abs(d.z) < 1e-6) {
		if (o.z < 0.0 || o.z > D) { imageStore(out_img, ivec2(px, py), vec4(0.0)); return; }
	} else {
		float ta = -o.z / d.z, tb = (D - o.z) / d.z;
		t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb));
	}
	if (t1 <= t0) { imageStore(out_img, ivec2(px, py), vec4(0.0)); return; }
	float t = t0 + 1e-4;
	// column DDA state
	int cx = clamp(int(floor(o.x + d.x * t)), 0, int(p.W) - 1);
	int cz = clamp(int(floor(o.z + d.z * t)), 0, int(p.D) - 1);
	int sx = d.x > 0.0 ? 1 : -1;
	int sz = d.z > 0.0 ? 1 : -1;
	float dtx = abs(d.x) > 1e-6 ? abs(1.0 / d.x) : 1.0e9;
	float dtz = abs(d.z) > 1e-6 ? abs(1.0 / d.z) : 1.0e9;
	float tmx = abs(d.x) > 1e-6 ? (float(cx + (sx > 0 ? 1 : 0)) - o.x) / d.x : 1.0e9;
	float tmz = abs(d.z) > 1e-6 ? (float(cz + (sz > 0 ? 1 : 0)) - o.z) / d.z : 1.0e9;
	int axis = d.y < 0.0 ? 1 : 0;   // last stepped axis: 0=x, 2=z (1 = top face)
	uint cw = (p.W + CHUNK - 1u) / CHUNK;
	int guard = 0;
	while (t < t1 && guard < 8192) {
		guard++;
		// tile fast path: on tile entry, if the ray stays above the tile's max
		// height for the whole crossing, jump to the tile's exit in one step
		int tx = cx >> 4, tz = cz >> 4;
		float tileExit = min(t1,
				min(abs(d.x) > 1e-6 ? (float((tx + (sx > 0 ? 1 : 0)) << 4) - o.x) / d.x : 1.0e9,
				    abs(d.z) > 1e-6 ? (float((tz + (sz > 0 ? 1 : 0)) << 4) - o.z) / d.z : 1.0e9));
		float hmx = float(hmax[uint(tz) * cw + uint(tx)]);
		float yA = o.y + d.y * t;
		float yE = o.y + d.y * tileExit;
		if (min(yA, yE) > hmx + 0.001) {
			if (d.y >= 0.0) { break; }        // rising above everything: sky
			// STRICTLY forward: at |pos|~1000 a 1e-4 step is ~1 float ulp, so the
			// jump could round back into the same tile and loop until the guard
			t = max(t + 0.01, tileExit + 0.01);
			cx = int(floor(o.x + d.x * t)); cz = int(floor(o.z + d.z * t));
			if (cx < 0 || cx >= int(p.W) || cz < 0 || cz >= int(p.D)) { break; }
			tmx = abs(d.x) > 1e-6 ? (float(cx + (sx > 0 ? 1 : 0)) - o.x) / d.x : 1.0e9;
			tmz = abs(d.z) > 1e-6 ? (float(cz + (sz > 0 ? 1 : 0)) - o.z) / d.z : 1.0e9;
			continue;
		}
		// per-column test over this column's t-span [t, tNext]
		float tNext = min(min(tmx, tmz), t1);
		float h = float(heights[uint(cz) * p.W + uint(cx)]);
		float y0 = o.y + d.y * t;
		if (y0 < h) {
			// entered the column below its top: SIDE face hit at t (camera-in-
			// ground start shades as a top so the screen never goes garbage)
			int cy = clamp(int(floor(y0)), 0, int(p.H) - 1);
			vec3 n = axis == 0 ? vec3(float(-sx), 0.0, 0.0)
					: (axis == 2 ? vec3(0.0, 0.0, float(-sz)) : vec3(0.0, 1.0, 0.0));
			uint bx = (uint(cx) + p.gen_ox) % p.W;   // local -> buffer slot (torus)
			uint bz = (uint(cz) + p.gen_oz) % p.D;
			uint raw = cget(cidx(bx, uint(cy), bz));
			uint m = MAT(raw);
			if (m == AIR) { m = SOIL; }
			uint vid = bx + bz * p.W + uint(cy) * p.W * p.D;
			vec3 base = m == WATER ? vec3(0.024, 0.13, 0.34) : surf_color(raw, m, vid);
			float ndl = max(dot(n, -normalize(cam_sun.xyz)), 0.0);
			vec3 c = base * (0.42 + 0.72 * ndl);
			imageStore(out_img, ivec2(px, py), vec4(pow(c, vec3(1.0 / 2.2)), 1.0));
			return;
		}
		if (d.y < 0.0) {
			float yN = o.y + d.y * tNext;
			if (yN < h) {
				// crosses the top plane inside this column: TOP face hit
				float tt = (h - o.y) / d.y;
				int cy = clamp(int(h) - 1, 0, int(p.H) - 1);
				uint bx = (uint(cx) + p.gen_ox) % p.W;   // local -> buffer slot (torus)
				uint bz = (uint(cz) + p.gen_oz) % p.D;
				uint raw = cget(cidx(bx, uint(cy), bz));
				uint m = MAT(raw);
				if (m == AIR) { m = SOIL; }
				uint vid = bx + bz * p.W + uint(cy) * p.W * p.D;
				vec3 base = m == WATER ? vec3(0.024, 0.13, 0.34) : surf_color(raw, m, vid);
				float ndl = max(dot(vec3(0.0, 1.0, 0.0), -normalize(cam_sun.xyz)), 0.0);
				vec3 c = base * (0.42 + 0.72 * ndl);
				imageStore(out_img, ivec2(px, py), vec4(pow(c, vec3(1.0 / 2.2)), 1.0));
				return;
			}
		}
		// advance to the next column
		if (tmx < tmz) { t = tmx; cx += sx; tmx += dtx; axis = 0; }
		else           { t = tmz; cz += sz; tmz += dtz; axis = 2; }
		if (cx < 0 || cx >= int(p.W) || cz < 0 || cz >= int(p.D)) { break; }
	}
	imageStore(out_img, ivec2(px, py), vec4(0.0));   // window exit: sky / far field shows
}

void main() {
	uint mode = p.mode & 0xFFu;      // high bits carry the debug rules mask
	if (mode == 1u) { do_rain(); }
	else if (mode == 2u) { do_pack(); }
	else if (mode == 3u) { do_emit(); }
	else if (mode == 4u) { do_gen(); }
	else if (mode == 5u) { do_block_emit(); }
	else if (mode == 6u) { do_skin_emit(); }
	else if (mode == 7u) { do_decay(); }
	else if (mode == 8u) { do_face_emit(); }
	else if (mode == 11u) { do_lod_emit(); }
	else if (mode == 12u) { do_far_emit(); }
	else if (mode == 13u) { do_holes(); }
	else if (mode == 14u) { do_heights(); }
	else if (mode == 15u) { do_hmax(); }
	else if (mode == 16u) { do_raycast(); }
	else { do_step(); }
}
