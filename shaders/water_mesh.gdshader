// THE water surface — one translucent mesh for ALL water, sim and
// procedural: a fine grid over the resident window (+apron), then a flat
// skirt at SEA_Y out to the horizon. Inside the window, vertices sample the
// fluid-top channel of the sim surface texture, so lakes, pools and rising
// floods are the actual simulated water; across the apron the level relaxes
// to the global sea line; beyond it the surface is exactly flat. Depth
// (which drives colour/alpha) comes from the sim texture inside and is
// re-derived per-FRAGMENT from the worldgen noise outside, crossfaded over
// the apron — one shader, one compositing model, no boundary anywhere.
// Fragments with no meaningful depth discard, leaving dry ground untouched.
shader_type spatial;
render_mode cull_disabled;

uniform sampler2D sim_terra : filter_linear, repeat_disable;
uniform vec2 win_size = vec2(0.0);
uniform float gen_oy = 0.0;
// reach (voxels) of the fine grid beyond the window rect: the zone where
// the sim water level relaxes to the sea line and sim depth crossfades to
// procedural depth
uniform float apron = 0.0;

#include "res://shaders/worldgen.gdshaderinc"
#include "res://shaders/sim_surface.gdshaderinc"

varying float v_depth;    // sim fluid top minus ground top (voxels)
varying float v_pw;       // procedural weight: 0 in-window .. 1 past apron
varying vec2 v_lp;        // local-frame xz (for per-fragment proc depth)

void vertex() {
	vec3 lp3 = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	v_lp = lp3.xz;
	vec2 lpc = clamp(lp3.xz, vec2(0.5), win_size - 0.5);
	vec2 hw = sim_tex_smooth(sim_terra, lpc, win_size).rg;
	v_depth = hw.g - hw.r;
	vec2 e = min(lp3.xz, win_size - lp3.xz);
	float dout = max(-min(e.x, e.y), 0.0);
	float t = clamp(dout / max(apron, 1.0), 0.0, 1.0);
	v_pw = t;
	float y;
	if (t >= 1.0) {
		// past the apron the surface is EXACTLY flat: the horizon skirt is
		// single quads whose T-junctions along the fine grid's edge stay
		// crack-free only because both sides are bit-identical SEA_Y
		y = SEA_Y;
	} else {
		// dry columns keep the surface hugging the ground slightly below the
		// terrain (fragments discard anyway) so shoreline tris don't stretch
		y = hw.g + gen_oy - (v_depth < 0.5 ? 0.35 : 0.0);
		y = mix(y, SEA_Y, t);
	}
	VERTEX = vec3(VERTEX.x, y - MODEL_MATRIX[3].y, VERTEX.z);
}

void fragment() {
	float d = v_depth;
	if (v_pw > 0.0) {
		// outside the window the bed is the worldgen noise, evaluated per
		// fragment: per-vertex depth inherited mesh density and printed
		// LoD artefacts on the mirror-flat surface
		d = mix(v_depth, SEA_Y - world_height(v_lp + origin), v_pw);
	}
	if (d < 0.6) { discard; }
	// deepen the blue with depth; shallows stay translucent over the bed
	float dp = 1.0 - exp(-d * 0.10);
	ALBEDO = mix(vec3(0.10, 0.30, 0.45), vec3(0.024, 0.13, 0.34), dp);
	ALPHA = clamp(0.55 + dp * 0.45, 0.0, 0.97);
	ROUGHNESS = mix(0.08, 0.15, dp);
	SPECULAR = mix(0.55, 0.4, dp);
}
