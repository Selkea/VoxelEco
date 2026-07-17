// SMOOTH SIM-SURFACE RECONSTRUCTION, shared by the terrain and water shaders.
// do_heights writes whole-voxel column heights, so plain bilinear sampling
// shows every 5 cm quantization step as a terrace/stair. Reconstructing with
// a cubic B-spline instead (4 hardware bilinear taps) rolls those steps into
// a C1-continuous surface: single-voxel noise is averaged away while any
// feature wider than ~2 voxels (channels, banks, dunes) keeps its shape.
// lp is the local column position in voxels (texel k is centred at lp = k),
// wsz the sim window size (W, D).

// sim->proc crossfade band just inside the window edge (10 m). Terrain and
// water must agree on it: the water shader reconstructs the exact ground the
// terrain draws to decide where the sheet is submerged.
const float BLEND_M = 200.0;

vec4 sim_tex_smooth(sampler2D tex, vec2 lp, vec2 wsz) {
	vec2 b = floor(lp);
	vec2 f = lp - b;
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;
	vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) * (1.0 / 6.0);
	vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) * (1.0 / 6.0);
	vec2 w2 = (1.0 + 3.0 * (f + f2 - f3)) * (1.0 / 6.0);
	vec2 w3 = f3 * (1.0 / 6.0);
	// fold the 4x4 kernel into 2x2 bilinear taps (g0/g1 never reach zero)
	vec2 g0 = w0 + w1;
	vec2 g1 = w2 + w3;
	vec2 t0 = b - 1.0 + w1 / g0;
	vec2 t1 = b + 1.0 + w3 / g1;
	vec2 s = 1.0 / wsz;
	return (texture(tex, (vec2(t0.x, t0.y) + 0.5) * s) * g0.x
		  + texture(tex, (vec2(t1.x, t0.y) + 0.5) * s) * g1.x) * g0.y
		 + (texture(tex, (vec2(t0.x, t1.y) + 0.5) * s) * g0.x
		  + texture(tex, (vec2(t1.x, t1.y) + 0.5) * s) * g1.x) * g1.y;
}
