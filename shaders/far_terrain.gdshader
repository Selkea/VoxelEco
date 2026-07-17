// UNIFIED WORLD-MESH TERRAIN: one displaced clipmap surface from the camera to
// the 8 km horizon. Inside the resident sim window, vertices sample the SIM's
// per-column surface textures (terra/tcol, written by do_heights every refresh
// — erosion, settling and water levels all show up); outside, the same
// worldgen noise the sim generates from. The two crossfade over BLEND_M just
// inside the window edge, so there is no renderer boundary anywhere.
// Shaded PBR: the engine's sun/sky/ACES/shadows light everything uniformly.
// Units are VOXELS in the local render frame; Y is world-Y voxels.
shader_type spatial;
render_mode cull_disabled;

uniform int seed_v = 0;                  // world seed (vox_step p.seedv)
uniform vec2 origin = vec2(0.0);         // local->world offset (gen_origin), voxels
uniform bool terraced = false;           // gen_flags bit0 (T toggle)
uniform float px_angle = 0.0015;         // radians per screen pixel (speckle fade)
// sim window surface data (Texture2DRD): r = solid ground top, g = fluid top,
// both in LOCAL band Y (add gen_oy for world-Y). win_size = (W, D); x <= 0
// disables sim sampling (pure procedural rings).
uniform sampler2D sim_terra : filter_linear, repeat_disable;
uniform sampler2D sim_col : filter_linear, repeat_disable;
uniform vec2 win_size = vec2(0.0);
uniform float gen_oy = 0.0;
const float BLEND_M = 200.0;             // sim->proc crossfade band (10 m)

const float BLOCK_VOX = 20.0;
const float RELIEF = 5120.0;
const float SEA_Y = RELIEF * 0.30;

#include "res://shaders/sim_surface.gdshaderinc"

varying vec3 v_local;                    // local-frame position (displaced)
varying float v_water;                   // 1 = procedural sea surface
varying vec3 v_nrm;                      // world-axis surface normal
varying float v_simf;                    // sim-data weight at this vertex

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

float block_height(vec2 bc, vec2 s) {
	vec2 w = bc * BLOCK_VOX;
	float cn = fbm(w / 21000.0 + s) - 0.5;
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

// how strongly this local position uses SIM data (0 outside the window,
// ramping to 1 over BLEND_M inside its edge)
float sim_w(vec2 lp) {
	if (win_size.x <= 0.0) { return 0.0; }
	vec2 e = min(lp, win_size - lp);
	return clamp(min(e.x, e.y) / BLEND_M, 0.0, 1.0);
}

// terrain height at a local position: sim ground inside the window,
// procedural outside, crossfaded — plus the sea clamp on the pure-proc side
// (window water is a separate translucent mesh over the real lakebed)
float terrain_h(vec2 lp, float f, out float wtr) {
	float hs = 0.0;
	wtr = 0.0;
	if (f > 0.0) {
		hs = sim_tex_smooth(sim_terra, lp, win_size).r + gen_oy;
		if (f >= 1.0) { return hs; }
	}
	float hp = world_height(lp + origin);
	if (hp < SEA_Y) { wtr = 1.0 - f; hp = SEA_Y; }
	return mix(hp, hs, f);
}

void vertex() {
	vec3 lp3 = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec2 lp = lp3.xz;
	float f = sim_w(lp);
	v_simf = f;
	float wtr;
	float h = terrain_h(lp, f, wtr);
	v_water = wtr;
	// finite-difference normals: a fine stride near the camera keeps shading
	// faithful to small sim relief, but at grazing distance a fine stride
	// tracks every residual 5 cm quantization step as contour banding — so
	// widen it with view distance; the procedural far field stays at 20
	float dist = length(lp3 - INV_VIEW_MATRIX[3].xyz);
	float ns = mix(20.0, clamp(dist * 0.02, 4.0, 20.0), f);
	float hx = terrain_h(lp + vec2(ns, 0.0), sim_w(lp + vec2(ns, 0.0)), wtr);
	float hz = terrain_h(lp + vec2(0.0, ns), sim_w(lp + vec2(0.0, ns)), wtr);
	v_nrm = normalize(vec3(h - hx, ns, h - hz));
	v_local = vec3(lp.x, h, lp.y);
	NORMAL = v_nrm;
	VERTEX = vec3(VERTEX.x, h - MODEL_MATRIX[3].y, VERTEX.z);
}

// one speckle level: SMOOTH value noise at a cell scale, fading out once the
// cells shrink toward a pixel. Constant-per-cell hash lattices were tried
// first and their axis-aligned squares read as a checkerboard from altitude —
// interpolated noise gives organic mottling at every viewing distance.
float spek(vec2 w, float scale, float amp, float pxw, uint salt) {
	float fade = clamp((scale / max(pxw, 1e-4) - 1.0) * 0.5, 0.0, 1.0);
	if (fade <= 0.0) { return 0.0; }
	vec2 q = w / scale + float(salt & 1023u) * 17.31;
	return (vnoise(q) - 0.5) * 2.0 * amp * fade;
}

void fragment() {
	NORMAL = normalize((VIEW_MATRIX * vec4(v_nrm, 0.0)).xyz) * (FRONT_FACING ? 1.0 : -1.0);
	vec2 w = v_local.xz + origin;
	vec3 alb;
	if (v_water > 0.5) {
		alb = vec3(0.024, 0.13, 0.34);        // procedural far sea
		ROUGHNESS = 0.15;
		SPECULAR = 0.4;
	} else {
		vec3 proc = vec3(0.086, 0.210, 0.052);   // mat_color(GRASS)
		// smooth reconstruction of the baked column colors too: per-voxel hash
		// jitter reads as surface roughness at travel height, so average it
		// over the same ~2-voxel kernel (material boundaries only soften 10 cm)
		alb = v_simf > 0.0
				? mix(proc, sim_tex_smooth(sim_col, v_local.xz, win_size).rgb, v_simf)
				: proc;
		float pxw = length(VERTEX) * px_angle;
		float tint = 1.0
				+ spek(w, 1.0, 0.03, pxw, 0u)
				+ spek(w, 5.0, 0.02, pxw, 668265263u)
				+ spek(w, 20.0, 0.03, pxw, 1336530526u)
				+ spek(w, 100.0, 0.03, pxw, 2004795789u);
		alb *= max(tint, 0.2);
		ROUGHNESS = 1.0;
		SPECULAR = 0.05;
	}
	ALBEDO = alb;
}
