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

varying float v_depth;    // fluid top minus ground top (voxels)

void vertex() {
	vec3 lp3 = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec2 uv = (lp3.xz + 0.5) / win_size;
	vec2 hw = texture(sim_terra, uv).rg;
	v_depth = hw.g - hw.r;
	// dry columns keep the surface hugging the ground slightly below the
	// terrain (fragments discard anyway) so shoreline triangles don't stretch
	float y = hw.g + gen_oy - (v_depth < 0.5 ? 0.35 : 0.0);
	VERTEX = vec3(VERTEX.x, y - MODEL_MATRIX[3].y, VERTEX.z);
}

void fragment() {
	if (v_depth < 0.6) { discard; }
	// deepen the blue with depth; shallows stay translucent over the bed
	float dp = 1.0 - exp(-v_depth * 0.10);
	ALBEDO = mix(vec3(0.10, 0.30, 0.45), vec3(0.016, 0.09, 0.24), dp);
	ALPHA = clamp(0.55 + dp * 0.4, 0.0, 0.92);
	ROUGHNESS = 0.08;
	SPECULAR = 0.55;
}
