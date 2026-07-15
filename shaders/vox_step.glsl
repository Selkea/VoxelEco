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
} p;

const uint CHUNK = 16u;

// flag the chunks around (x, z) — a border cell change alters the visible
// faces of cells in the adjacent chunk, so flag with a 1-cell margin
void mark_dirty(uint x, uint z) {
	uint cw = (p.W + CHUNK - 1u) / CHUNK;
	uint x0 = (x > 0u ? x - 1u : 0u) / CHUNK;
	uint x1 = min(x + 1u, p.W - 1u) / CHUNK;
	uint z0 = (z > 0u ? z - 1u : 0u) / CHUNK;
	uint z1 = min(z + 1u, p.D - 1u) / CHUNK;
	dirty[z0 * cw + x0] = 1u;
	if (x1 != x0) { dirty[z0 * cw + x1] = 1u; }
	if (z1 != z0) { dirty[z1 * cw + x0]= 1u; }
	if (x1 != x0 && z1 != z0) { dirty[z1 * cw + x1] = 1u; }
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
		if (MAT(cells[i]) == AIR) {
			cells[i] = WATER;
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
	uint v = MAT(cells[b]);
	if (b + 1u < n) { v |= MAT(cells[b + 1u]) << 8u; }
	if (b + 2u < n) { v |= MAT(cells[b + 2u]) << 16u; }
	if (b + 3u < n) { v |= MAT(cells[b + 3u]) << 24u; }
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
		c[l] = ok ? cells[ids[l]] : BEDROCK;
		any_active = any_active || movable(MAT(c[l]));
	}
	if ((p.mode >> 8u) == 32u) {
		// partition self-test: count how many threads touch each cell.
		for (uint l = 0u; l < 8u; l++) {
			if (ib[l]) { atomicAdd(cells[ids[l]], 1u); }
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
		if (ib[l] && cells[ids[l]] != c[l]) {
			cells[ids[l]] = c[l];
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

// one thread per cell: emit an instance for every exposed voxel
void do_emit() {
	uint n = p.W * p.H * p.D;
	uint id = flat_id();
	if (id >= n) { return; }
	uint raw = cells[id];
	uint m = MAT(raw);
	if (m == AIR) { return; }
	uint y = id / (p.W * p.D);
	uint rem = id % (p.W * p.D);
	uint z = rem / p.W;
	uint x = rem % p.W;
	if (z < p.cut_z) { return; }   // hidden behind the cross-section plane
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
		int ny = int(y) + dy;
		int nz = int(z) + dz;
		uint nm = AIR;
		bool oob = nx < 0 || nx >= int(p.W) || ny >= int(p.H) || nz < 0 || nz >= int(p.D);
		if (ny < 0) { nm = BEDROCK; }
		else if (nz < int(p.cut_z)) { nm = AIR; }   // across the cut = open (reveal section)
		else if (!oob) { nm = MAT(cells[cidx(uint(nx), uint(ny), uint(nz))]); }
		if (nm != AIR && nm != WATER) { solid_n += 1; }
		if (is_water ? (nm == AIR) : (nm == AIR || nm == WATER)) { exposed = true; }
	}
	if (!exposed) { return; }
	// place the voxel at its WORLD position (toroidal buffer slot -> world) so a
	// streamed window renders where the camera actually is
	vec3 origin = vec3(float(world_coord(x, p.gen_ox, p.W)) + 0.5,
			float(y) + 0.5, float(world_coord(z, p.gen_oz, p.D)) + 0.5);
	if (is_water) {
		uint slot = atomicAdd(n_water, 1u);
		if (slot < cap_water) { write_inst(true, slot, origin, vec4(1.0), 1.0); }
		return;
	}
	// per-voxel tint jitter + crude AO from buried-ness
	float jit = 1.0 + (float(pcg(id * 2654435761u) & 255u) / 255.0 * 0.24 - 0.12);
	float ao = 1.0 - float(solid_n) * 0.05;
	vec3 base = mat_color(m);
	// darken the dry-able ground (soil/sand) toward wet earth as it saturates;
	// mud/grass already carry their own wet/vegetated tone
	if (m == SOIL || m == SAND) {
		float wet = clamp(float(SAT(raw)) / float(sat_cap(m)), 0.0, 1.0);
		base = mix(base, vec3(0.09, 0.05, 0.03), wet * 0.8);
	}
	vec4 col = vec4(base * jit * ao, 1.0);
	uint slot = atomicAdd(n_solid, 1u);
	if (slot < cap_solid) { write_inst(false, slot, origin, col, 1.0); }
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
		uint m = MAT(cells[cidx(cx, yy, cz)]);
		if (m != AIR) { surfy = int(yy); surfmat = m; break; }
	}
	if (surfy < 0) { return; }                       // empty column
	uint ntop = max((uint(surfy) + 10u) / 20u, 1u);  // height rounded to whole blocks
	for (uint by = 0u; by < ntop; by++) {
		uint y0 = by * 20u;
		uint m = surfmat;                            // top cube = surface material
		if (by + 1u < ntop) {                        // body cube = its own centre
			m = MAT(cells[cidx(cx, min(y0 + 10u, p.H - 1u), cz)]);
			if (m == AIR) { m = SOIL; }
		}
		vec3 center = vec3(float(int(x0) + int(p.gen_ox)) + 10.0, float(y0) + 10.0,
				float(int(z0) + int(p.gen_oz)) + 10.0);
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
		if (MAT(cells[cidx(cx, yy, cz)]) != AIR) { return max((yy + 10u) / 20u, 1u) * 20u; }
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
		yy--; uint raw = cells[cidx(x, yy, z)];
		if (MAT(raw) != AIR) { sm = MAT(raw); sraw = raw; break; }
	}
	// lowest exposed voxel: the top is always exposed; each block edge whose
	// neighbouring block is shorter exposes this column's side down to it
	uint low = topf - 1u;
	if (x % 20u == 19u && x + 1u < p.W) { uint hn = block_top_h(bx + 1u, bz); if (hn < topf) { low = min(low, hn); } }
	if (x % 20u == 0u  && x > 0u)       { uint hn = block_top_h(bx - 1u, bz); if (hn < topf) { low = min(low, hn); } }
	if (z % 20u == 19u && z + 1u < p.D) { uint hn = block_top_h(bx, bz + 1u); if (hn < topf) { low = min(low, hn); } }
	if (z % 20u == 0u  && z > 0u)       { uint hn = block_top_h(bx, bz - 1u); if (hn < topf) { low = min(low, hn); } }
	float wx = float(world_coord(x, p.gen_ox, p.W));   // world pos (torus)
	float wz = float(world_coord(z, p.gen_oz, p.D));
	for (uint y = low; y < topf; y++) {
		uint raw = cells[cidx(x, y, z)];
		uint m = MAT(raw);
		if (m == AIR) { m = sm; raw = sraw; }
		vec3 c = vec3(wx + 0.5, float(y) + 0.5, wz + 0.5);
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

// Terrain height sampled at a BLOCK grid point (bc in block units, i.e. world
// voxels / 20). This is the base heightfield resolution — one value per 1 m
// block — so the world generates "as if 1 block = 1 voxel". It sums the CHUNK
// band (broad landforms, squared toward its tails so terrain spends time as
// plains and peaks) and a per-block variation band.
float block_height(vec2 bc, float H, vec2 s) {
	// CHUNK band: 20 blocks per chunk, so bc/20 == world/CHUNK_VOX
	float cn = fbm(bc / 20.0 + s) - 0.5;
	float chunk = cn * (1.0 + 2.4 * abs(cn));
	// per-block relief on the block lattice
	float blk = fbm(bc * 0.5 + s * 2.0 + 31.7) - 0.5;
	return H * 0.42 + chunk * H * 1.35 + blk * H * 0.09;
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
	float hgt = world_height(w);
	// signed math throughout: with uint, top-4 underflows for shallow
	// columns and fills them with stone to the sky
	int top = clamp(int(hgt), 2, int(p.H) - 4);
	int water_level = int(float(p.H) * 0.26);   // basins below this start as lakes
	for (uint y = 0u; y < p.H; y++) {
		int yi = int(y);
		uint m = AIR;
		uint s0 = 0u;
		if (yi == 0) { m = BEDROCK; }
		else if (yi < top - 4) { m = STONE; }
		else if (yi < top) { m = SOIL; s0 = 40u; }   // field-capacity moisture
		// the land surface that stands above the water line starts vegetated;
		// low ground stays firm soil (loose sand isn't pre-placed — it forms
		// where water erodes, so freshly generated slopes don't avalanche)
		if (m == SOIL && yi == top - 1 && top > water_level) { m = GRASS; s0 = 40u; }
		// submerged ground generates already SATURATED so the lakes sit on wet
		// beds and don't seep away. The one exception is the ground's exposed
		// SURFACE cell at the water's EDGE (a waterline column, top==water_level,
		// so its top-1 surface has AIR above): saturated to capacity it would be
		// over the mud threshold and instantly waterlog into sliding mud that
		// avalanches into the lake. Under deep water (top<water_level) the surface
		// keeps its cap — water above exempts it from mud, and a firm bed can't
		// seep. So: cap everything submerged except that exposed shoreline layer.
		if ((m == SOIL || m == SAND) && yi < water_level
				&& (top < water_level || yi < top - 1)) { s0 = sat_cap(m); }
		// fill low basins and valleys with standing water up to the water line
		if (m == AIR && yi < water_level) { m = WATER; s0 = 0u; }
		cells[cidx(x, y, z)] = PACK(m, s0);
	}
}

void main() {
	uint mode = p.mode & 0xFFu;      // high bits carry the debug rules mask
	if (mode == 1u) { do_rain(); }
	else if (mode == 2u) { do_pack(); }
	else if (mode == 3u) { do_emit(); }
	else if (mode == 4u) { do_gen(); }
	else if (mode == 5u) { do_block_emit(); }
	else if (mode == 6u) { do_skin_emit(); }
	else { do_step(); }
}
