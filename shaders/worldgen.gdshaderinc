// SHARED WORLDGEN NOISE for the world-mesh shaders (terrain + water). This
// is the render-side port of the sim's generator — constants and formulas
// MUST stay in sync with vox_step.glsl (and gpu_world.gd's CPU mirror):
// wavelengths 21000/2700/600, RELIEF, SEA_Y = RELIEF*0.30, BLOCK_VOX.
uniform int seed_v = 0;                  // world seed (vox_step p.seedv)
uniform vec2 origin = vec2(0.0);         // local->world offset (gen_origin), voxels
uniform bool terraced = false;           // gen_flags bit0 (T toggle)

const float BLOCK_VOX = 20.0;
const float RELIEF = 5120.0;
const float SEA_Y = RELIEF * 0.30;

uint pcg(uint v) {
	v = v * 747796405u + 2891336453u;
	v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
	return (v >> 22u) ^ v;
}

float vhash(vec2 q) {
	uint h = pcg(uint(int(q.x)) * 374761393u ^ uint(int(q.y)) * 668265263u ^ uint(seed_v));
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
	return v;
}

// analytic-erosion strength — MUST match gpu_world.gd ERODE_K and vox_step.glsl
// or the sim band mis-places against this render surface.
const float ERODE_K = 18.0;
// continental domain-warp amplitude (in continental-noise cells) — MUST match the
// other two mirrors. Bends the value-noise lattice the erosion damping rides so it
// reads as organic landforms, not an axis-aligned grid (visible as squares from
// altitude). 0 = no warp.
const float WARP_A = 1.6;

// value noise WITH gradient: (value, d/dx, d/dy) in q-space. Smoothstep weight
// u = f*f*(3-2f) has derivative du = 6*f*(1-f); the bilinear mix expands to
// a + k1*ux + k2*uy + k3*ux*uy, whose partials are closed-form. Value channel
// equals vnoise. Keep bit-identical to gpu_world.gd _vnoise_d.
vec3 vnoise_d(vec2 q) {
	vec2 i = floor(q);
	vec2 f = q - i;
	vec2 u = f * f * (3.0 - 2.0 * f);
	vec2 du = 6.0 * f * (1.0 - f);
	float a = vhash(i);
	float b = vhash(i + vec2(1.0, 0.0));
	float c = vhash(i + vec2(0.0, 1.0));
	float d = vhash(i + vec2(1.0, 1.0));
	float k1 = b - a;
	float k2 = c - a;
	float k3 = a - b - c + d;
	float val = a + k1 * u.x + k2 * u.y + k3 * u.x * u.y;
	float gx = du.x * (k1 + k3 * u.y);
	float gy = du.y * (k2 + k3 * u.x);
	return vec3(val, gx, gy);
}

// derivative-damped fbm = ANALYTIC hydraulic erosion (iq / de Carpentier): each
// octave is divided by (1 + k*|accumulated slope|^2), so already-steep flanks
// suppress the finer octaves — valleys flatten/widen and ridges stay sharp,
// breaking plain fbm's fake up/down symmetry. Stays a pure function of world
// coords (no stateful droplet pass), so it streams seamlessly and evaluates the
// same here, in the sim, and in the CPU mirror. k=0 == plain fbm.
float fbm_erode(vec2 q, float k) {
	float v = 0.0;
	float amp = 0.5;
	vec2 d = vec2(0.0);
	for (int o = 0; o < 4; o++) {
		vec3 n = vnoise_d(q);
		// damp by the slope from COARSER octaves ONLY (accumulate AFTER using it):
		// value noise has axis-aligned separable gradients, so damping an octave by
		// its OWN gradient prints a lattice grid at that octave's wavelength (visible
		// as axis-aligned "brickwork" relief from altitude). Damping by only the
		// coarser slope keeps the damping field smoother than the octave it shapes —
		// fine detail still fades on steep terrain (erosion), no same-frequency grid.
		v += amp * n.x / (1.0 + k * dot(d, d));
		d += n.yz;
		q *= 2.03;
		amp *= 0.5;
	}
	return v;
}

float block_height(vec2 bc, vec2 s) {
	vec2 w = bc * BLOCK_VOX;
	// Erode ONLY the continental term: it drives the landmass shape (coastline,
	// km-scale valleys and ridgelines) and its erosion "grid" is at the ~21 km
	// continental wavelength — far too coarse to perceive. The hill/detail terms
	// stay plain fbm: their erosion grids fall at 2700/600-vox wavelengths, which
	// read as visible axis-aligned brickwork relief from altitude.
	vec2 wc = w / 21000.0 + s;
	wc += (vec2(fbm(wc * 1.9 + 5.2), fbm(wc * 1.9 + 13.9)) - 0.5) * WARP_A;   // domain warp
	float cn = fbm_erode(wc, ERODE_K) - 0.5;
	float chunk = cn * (1.0 + 2.0 * abs(cn));
	float hill = fbm(w / 2700.0 + s * 2.0 + 31.7) - 0.5;
	float det = fbm(w / 600.0 + s * 4.0 + 91.3) - 0.5;
	return RELIEF * (0.5 + chunk * 0.44 + hill * 0.06 + det * 0.02);
}

float world_height(vec2 w) {
	vec2 s = vec2(float(uint(seed_v) & 0xFFFFu), float((uint(seed_v) >> 16u) & 0xFFFFu)) * 0.618;
	if (terraced) {
		vec2 bc = floor(w / BLOCK_VOX);
		return floor(block_height(bc, s) / BLOCK_VOX) * BLOCK_VOX;
	}
	vec2 bcf = w / BLOCK_VOX;
	vec2 i = floor(bcf);
	vec2 f = bcf - i;
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(
		mix(block_height(i, s), block_height(i + vec2(1.0, 0.0), s), u.x),
		mix(block_height(i + vec2(0.0, 1.0), s), block_height(i + vec2(1.0, 1.0), s), u.x),
		u.y);
}
