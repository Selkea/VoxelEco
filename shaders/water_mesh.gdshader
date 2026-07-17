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
// RAW (unsmoothed) surface data: the wet/dry decision reads this per-pixel.
// The bilateral prefilter blends ground and fluid heights across gentle
// shorelines, so smoothed depth underestimates near the waterline — the
// discard retreated into the lake, exposing a strip of bed under a
// hovering water rim.
uniform sampler2D sim_raw : filter_linear, repeat_disable;
uniform vec2 win_size = vec2(0.0);
uniform float gen_oy = 0.0;
// reach (voxels) of the fine grid beyond the window rect: the zone where
// the sim water level relaxes to the sea line and sim depth crossfades to
// procedural depth
uniform float apron = 0.0;

#include "res://shaders/worldgen.gdshaderinc"
#include "res://shaders/sim_surface.gdshaderinc"

varying float v_wy;       // water surface world-y (voxels)
varying float v_pw;       // procedural weight: 0 in-window .. 1 past apron
varying vec2 v_lp;        // local-frame xz (for per-fragment proc depth)

void vertex() {
	vec3 lp3 = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	v_lp = lp3.xz;
	vec2 lpc = clamp(lp3.xz, vec2(0.5), win_size - 0.5);
	vec2 hw = sim_tex_smooth(sim_terra, lpc, win_size).rg;
	float sdep = hw.g - hw.r;
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
		y = hw.g + gen_oy - (sdep < 0.5 ? 0.35 : 0.0);
		y = mix(y, SEA_Y, t);
	}
	v_wy = y;
	VERTEX = vec3(VERTEX.x, y - MODEL_MATRIX[3].y, VERTEX.z);
}

void fragment() {
	vec2 lpc = clamp(v_lp, vec2(0.5), win_size - 0.5);
	vec2 hs = sim_tex_smooth(sim_terra, lpc, win_size).rg;
	vec2 raw = texture(sim_raw, (lpc + 0.5) / win_size).rg;
	// tint depth, PER-PIXEL: inside the window the RAW sim texture gives the
	// true per-column depth; outside, the blended surface over the worldgen
	// bed (per-vertex values inherit mesh density and print LoD artefacts)
	float draw = raw.g - raw.r;
	float d = draw;
	// the GROUND the terrain mesh draws at this pixel — smoothed sim inside
	// the window, worldgen outside, crossfaded over BLEND_M inside the edge
	// (mirrors far_terrain's terrain_h exactly)
	vec2 e = min(v_lp, win_size - v_lp);
	float f = clamp(min(e.x, e.y) / BLEND_M, 0.0, 1.0);
	float grd = hs.r + gen_oy;
	if (f < 1.0) {
		float wh = world_height(v_lp + origin);
		grd = mix(wh, grd, f);
		d = mix(d, v_wy - wh, v_pw);
	}
	// WET/DRY is ONE smooth test: keep the sheet only where its surface sits
	// above the DRAWN ground. Both sides are B-spline-smoothed (v_wy from the
	// smoothed fluid channel, grd the exact ground far_terrain rasterizes), so
	// the waterline is a continuous curve — testing raw/quantized depth here
	// instead snapped the shoreline to voxel columns and shed blocky slivers
	// across shallow shelves. In dry columns the fluid channel equals the
	// ground channel, so v_wy hugs (just under) the terrain and discards
	// cleanly — no chording across dry hollows, no film over land. Raw depth
	// stays for TINT only, where quantization is invisible.
	if (v_wy - grd < 0.05) { discard; }
	// deepen the blue with depth; shallows stay translucent over the bed
	float dp = 1.0 - exp(-d * 0.10);
	// ALPHA ramps from 0 at the waterline to near-opaque within ~1.5 voxels
	// of depth. This does double duty: it feathers the very edge (no hard
	// voxel-snapped shoreline, no slivers, and the faint films over dry
	// sub-sea-level hollows vanish since their RAW depth is exactly 0), AND
	// it keeps the lake reading as SOLID water. The old fixed 0.55 alpha
	// floor left the whole shallow shelf half-transparent, so over the brown
	// mud bed — meters wide on a gentle shelf — it smeared into a milky,
	// sky-reflecting band up close. A quick ramp leaves only a thin
	// translucent rim at the shore.
	ALBEDO = mix(vec3(0.10, 0.30, 0.45), vec3(0.024, 0.13, 0.34), dp);
	ALPHA = clamp(d * (1.0 / 1.5), 0.0, 0.95);
	ROUGHNESS = mix(0.08, 0.15, dp);
	SPECULAR = mix(0.55, 0.4, dp);
}
