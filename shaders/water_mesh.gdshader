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
		// water LEVEL to relax toward the sea line across the apron. A WET sim
		// column carries its own surface (a perched lake keeps a level above the
		// sea); a DRY column has no water of its own, so the covering water is the
		// SEA at SEA_Y — NOT the column's ground. Relaxing the dry GROUND toward
		// SEA_Y instead lifted the sheet into a tall ramp wherever the window edge
		// sat on land above the sea: the apron clamps to that high edge column and
		// blends it down, standing the sea surface visibly above the flat lake and
		// over the bed (the "lifting edge"). At a real shore the ground already
		// sits at ~SEA_Y, so this matches the old hug there — it only changes the
		// high-dry-land case. The tiny bias keeps dry columns just under the sea so
		// they discard cleanly against ground at exactly the sea line.
		float wlev = (sdep < 0.5) ? (SEA_Y - 0.35) : (hw.g + gen_oy);
		y = mix(wlev, SEA_Y, t);
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
	// CRISP WATER EXTENT, agreeing with the TERRAIN's own waterline so the two
	// meshes never disagree (world-mesh / "ray off" must look like "ray on"):
	// water exists exactly where the DRAWN ground (grd — the same smoothed-inside,
	// worldgen-outside height far_terrain rasterizes) sits below the sea line, or
	// the sim actually holds water here. far_terrain paints submerged bed (mud)
	// wherever grd < SEA_Y; gating the water on the SAME grd guarantees every
	// browned strip is covered by water and no dry grass is filmed over. Testing
	// worldgen height here instead left a band near shore where the smoothed bed
	// had already dipped below the sea (terrain browned it) but worldgen had not
	// (water discarded) — an exposed mud rim with the water visibly lifting above
	// it. draw is the RAW per-column sim depth (sim lakes perched above the sea
	// line stay through the draw>=0.5 escape).
	if (grd >= SEA_Y && draw < 0.5) { discard; }
	if (v_wy - grd < 0.05) { discard; }
	// deepen the blue with depth; shallows stay translucent over the bed
	float dp = 1.0 - exp(-d * 0.10);
	// ALPHA rises from 0 at the waterline toward near-opaque as depth grows.
	// It feathers the very edge (no hard voxel-snapped shoreline, no slivers,
	// and the faint films over dry sub-sea-level hollows vanish since their RAW
	// depth is exactly 0) AND keeps the lake reading as SOLID water. Use a
	// SMOOTH exponential approach, not a linear ramp into a clamp: the old
	// clamp(d/1.5, 0, 0.95) saturated with a hard CORNER at d=1.5 voxels, and on
	// a gently sloped shelf — where that corner spreads across many pixels up
	// close — the slope break read as a distinct line ringing the shore (the bed
	// abruptly stopped showing through there). The exponential has no knee
	// anywhere, so the shelf fades into open water as one continuous gradient at
	// every viewing distance; the rate keeps the translucent rim about as thin
	// as the old ramp, so the milky half-transparent band the earlier 0.55 floor
	// caused stays gone.
	ALBEDO = mix(vec3(0.10, 0.30, 0.45), vec3(0.024, 0.13, 0.34), dp);
	// OPAQUE water with only a thin shoreline feather — matching the ray path,
	// which writes one solid colour so shallows read as clean light blue instead
	// of the bed showing through. The old translucent ramp (0.95*(1-exp(-d*1.5)))
	// let the green bed bleed through the near-shore shallows, washing them into a
	// broad pale band that foreshortened into a big smear at grazing angles — the
	// lake "looked different when approaching". Depth still reads through COLOUR
	// (light blue -> navy over dp, the SAME ramp the ray uses), so the lake keeps
	// its shallow/deep gradient; only the bed dilution is gone. The exp feather
	// still has no knee, so the waterline stays a soft continuous edge.
	ALPHA = 1.0 - exp(-d * 5.0);
	ROUGHNESS = mix(0.08, 0.15, dp);
	SPECULAR = mix(0.55, 0.4, dp);
}
