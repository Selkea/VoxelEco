// PROCEDURAL GRASS TUFT — the living meadow's foliage layer, drawn as instanced
// blade clusters over the smooth world-mesh terrain instead of voxel cubes. One
// MultiMesh instance per plant-bearing column (placed/scaled/tinted by the compute
// emit, see do_grass_emit); the base mesh is a fan of tapered blades (real opaque
// geometry — no alpha, so nothing to sort and no shimmer). This shader gives the
// blades a base->tip colour gradient, lawn-like lighting, and a gentle wind sway.
shader_type spatial;
render_mode cull_disabled;   // blades are single-sided quads; show both faces

// height fraction (0 base .. 1 tip) rides UV.y so it survives per-instance scaling
varying float v_h;

uniform float wind_amp = 0.18;    // sway amplitude in voxels at the blade tip
uniform float wind_speed = 1.6;

void vertex() {
	v_h = UV.y;
	// world XZ of this instance (floating-origin local frame is fine — it only
	// sets the sway phase, so tufts don't all lean in lockstep)
	vec2 wpos = MODEL_MATRIX[3].xz;
	float phase = wpos.x * 0.15 + wpos.y * 0.17;
	// two summed waves = a less periodic gust; weighted by height^1.5 so the base
	// stays planted and only the tip bends
	float sway = (sin(TIME * wind_speed + phase) + 0.4 * sin(TIME * wind_speed * 2.3 + phase * 1.7));
	sway *= wind_amp * pow(v_h, 1.5);
	VERTEX.x += sway;
	VERTEX.z += sway * 0.6;
	// light the blades like the ground (world-up normal) rather than per-blade
	// facets — grass reads as a soft lawn, not a cluster of dark shards
	NORMAL = (VIEW_MATRIX * vec4(0.0, 1.0, 0.0, 0.0)).xyz;
}

void fragment() {
	// base (shaded, cool green) -> tip (brighter, yellow-green); per-instance COLOR
	// carries the emit's meadow tint variation so a field isn't one flat green
	vec3 base = vec3(0.11, 0.24, 0.07);
	vec3 tip  = vec3(0.44, 0.60, 0.20);
	vec3 g = mix(base, tip, v_h) * COLOR.rgb;
	// a touch of ambient occlusion crushed into the very base of the tuft
	g *= mix(0.7, 1.0, smoothstep(0.0, 0.35, v_h));
	ALBEDO = g;
	ROUGHNESS = 0.9;
	SPECULAR = 0.15;
}
