shader_type spatial;
render_mode cull_back, unshaded;
// Toon / cel terrain: the per-instance voxel colour lit by a hard-banded sun —
// three flat lighting steps and a touch of sky ambient instead of the smooth
// PBR response. Normals come from the quad instances (world space via MODEL).

uniform vec3 sun_dir = vec3(-0.5, -0.7, -0.5);   // set from the scene's sun

varying vec3 vcol;
varying vec3 vnorm;

void vertex() {
	vcol = COLOR.rgb;                              // instance colour (linear)
	vnorm = (MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz;
}

void fragment() {
	float nd = clamp(dot(normalize(vnorm), normalize(-sun_dir)), 0.0, 1.0);
	float band = floor(nd * 3.0 + 0.5) / 3.0;      // 4 hard light bands
	float lum = 0.35 + 0.75 * band;
	ALBEDO = vcol * lum;
}
