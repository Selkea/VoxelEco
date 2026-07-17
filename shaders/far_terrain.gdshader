// UNIFIED WORLD-MESH TERRAIN: one displaced clipmap surface from the camera to
// the 8 km horizon. Inside the resident sim window, vertices sample the SIM's
// per-column surface textures (terra/tcol, written by do_heights every refresh
// — erosion, settling and water levels all show up); outside, the same
// worldgen noise the sim generates from. The two crossfade over BLEND_M just
// inside the window edge, so there is no renderer boundary anywhere.
// This shader draws GROUND ONLY — every water surface (sim lakes AND the
// procedural sea) is the single translucent water mesh (water_mesh.gdshader),
// so water compositing is identical everywhere and no handoff can show.
// Shaded PBR: the engine's sun/sky/ACES/shadows light everything uniformly.
// Units are VOXELS in the local render frame; Y is world-Y voxels.
shader_type spatial;
render_mode cull_disabled;

uniform float px_angle = 0.0015;         // radians per screen pixel (speckle fade)
// sim window surface data (Texture2DRD): r = solid ground top, g = fluid top,
// both in LOCAL band Y (add gen_oy for world-Y). win_size = (W, D); x <= 0
// disables sim sampling (pure procedural rings).
uniform sampler2D sim_terra : filter_linear, repeat_disable;
uniform sampler2D sim_col : filter_linear, repeat_disable;
uniform vec2 win_size = vec2(0.0);
uniform float gen_oy = 0.0;

#include "res://shaders/worldgen.gdshaderinc"
#include "res://shaders/sim_surface.gdshaderinc"

varying vec3 v_local;                    // local-frame position (displaced)
varying vec3 v_nrm;                      // world-axis surface normal
varying float v_simf;                    // sim-data weight at this vertex

// how strongly this local position uses SIM data (0 outside the window,
// ramping to 1 over BLEND_M inside its edge)
float sim_w(vec2 lp) {
	if (win_size.x <= 0.0) { return 0.0; }
	vec2 e = min(lp, win_size - lp);
	return clamp(min(e.x, e.y) / BLEND_M, 0.0, 1.0);
}

// GROUND height at a local position: sim ground inside the window,
// procedural outside, crossfaded bed-to-bed. Water never displaces this
// surface — the translucent water mesh above carries every water surface.
float terrain_h(vec2 lp, float f) {
	if (f > 0.0) {
		float hs = sim_tex_smooth(sim_terra, lp, win_size).r + gen_oy;
		if (f >= 1.0) { return hs; }
		return mix(world_height(lp + origin), hs, f);
	}
	return world_height(lp + origin);
}

void vertex() {
	vec3 lp3 = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec2 lp = lp3.xz;
	float f = sim_w(lp);
	v_simf = f;
	float h = terrain_h(lp, f);
	// finite-difference normals: a fine stride near the camera keeps shading
	// faithful to small sim relief, but at grazing distance a fine stride
	// tracks every residual 5 cm quantization step as contour banding — so
	// widen it with view distance; the procedural far field stays at 20
	float dist = length(lp3 - INV_VIEW_MATRIX[3].xyz);
	float ns = mix(20.0, clamp(dist * 0.02, 4.0, 20.0), f);
	float hx = terrain_h(lp + vec2(ns, 0.0), sim_w(lp + vec2(ns, 0.0)));
	float hz = terrain_h(lp + vec2(0.0, ns), sim_w(lp + vec2(0.0, ns)));
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
	vec3 proc = vec3(0.086, 0.210, 0.052);   // mat_color(GRASS)
	// wet the procedural grass toward mud BELOW the sea line, BEFORE blending
	// with the sim's baked bed colour. Applied after the sim/proc blend (and
	// scaled by 1-f) it left the crossfade band showing dry green grass over
	// submerged ground — a green square along the whole window rim, obvious
	// underwater. Pre-wetting the proc endpoint makes a submerged blend band
	// go mud->mud; at f=1 the sim colour wins so its own baked wet tone is
	// untouched. Per-fragment (from v_local.y) also kills the old LoD facets.
	float sub = clamp((SEA_Y - v_local.y) * 0.25, 0.0, 1.0);
	proc = mix(proc, vec3(0.13, 0.085, 0.05), sub);
	// smooth reconstruction of the baked column colors too: per-voxel hash
	// jitter reads as surface roughness at travel height, so average it
	// over the same ~2-voxel kernel (material boundaries only soften 10 cm)
	vec3 alb = v_simf > 0.0
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
	ALBEDO = alb;
}
