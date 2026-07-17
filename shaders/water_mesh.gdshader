// SIM WATER SURFACE: a translucent displaced plane over the resident window.
// Vertices sample the fluid-top channel of the sim surface texture, so lakes,
// pools and rising floods are the actual simulated water; fragments with no
// meaningful depth discard, leaving dry ground untouched. The far sea beyond
// the window is drawn (opaque) by the terrain shader's procedural branch.
shader_type spatial;
render_mode cull_disabled;

uniform sampler2D sim_terra : filter_linear, repeat_disable;
uniform vec2 win_size = vec2(0.0);
uniform float gen_oy = 0.0;
// reach (voxels) beyond the window rect: wet edges carry their water level
// outward, relaxing to the global sea line so the handoff to the opaque
// procedural sea happens at matched heights
uniform float apron = 0.0;
const float SEA_Y = 5120.0 * 0.30;

#include "res://shaders/sim_surface.gdshaderinc"

varying float v_depth;    // fluid top minus ground top (voxels)

void vertex() {
	vec3 lp3 = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec2 lpc = clamp(lp3.xz, vec2(0.5), win_size - 0.5);
	vec2 hw = sim_tex_smooth(sim_terra, lpc, win_size).rg;
	v_depth = hw.g - hw.r;
	// dry columns keep the surface hugging the ground slightly below the
	// terrain (fragments discard anyway) so shoreline triangles don't stretch
	float y = hw.g + gen_oy - (v_depth < 0.5 ? 0.35 : 0.0);
	// apron profile: converge to a hair ABOVE the sea line and stay there
	// through the zone where the terrain's bed->sea wall can land (sea starts
	// at 464; its first vertex row lands up to a ring cell later) — the
	// wall's top edge sits exactly at the surface and pokes through anything
	// lower as a dark grazing line. Only the final 64 vox duck 1.25 under so
	// the rim itself hides beneath the opaque sea.
	vec2 e = min(lp3.xz, win_size - lp3.xz);
	float dout = max(-min(e.x, e.y), 0.0);
	float inner = max(apron - 64.0, 1.0);
	y = mix(y, SEA_Y + 0.25, clamp(dout / 464.0, 0.0, 1.0))
			- 1.25 * clamp((dout - inner) / 64.0, 0.0, 1.0);
	VERTEX = vec3(VERTEX.x, y - MODEL_MATRIX[3].y, VERTEX.z);
}

void fragment() {
	if (v_depth < 0.6) { discard; }
	// deepen the blue with depth; shallows stay translucent over the bed.
	// Deep water converges to the EXACT material of the terrain shader's
	// opaque procedural sea (albedo/roughness/specular, near-opaque), so the
	// window/apron handoff to the far sea carries no tone seam and the
	// submerged bed cliff at the apron's end can't show through.
	float dp = 1.0 - exp(-v_depth * 0.10);
	ALBEDO = mix(vec3(0.10, 0.30, 0.45), vec3(0.024, 0.13, 0.34), dp);
	ALPHA = clamp(0.55 + dp * 0.45, 0.0, 0.97);
	ROUGHNESS = mix(0.08, 0.15, dp);
	SPECULAR = mix(0.55, 0.4, dp);
}
