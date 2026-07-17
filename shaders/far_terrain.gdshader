// FAR-FIELD CLIPMAP MESH: a continuous heightfield mesh replacing the old
// instanced far tiles. The terrain surface is a pure function of world (x,z)
// + seed, so static grid rings that ride the camera displace their vertices
// right here in the vertex shader by evaluating the SAME worldgen noise the
// sim uses (ported from vox_step.glsl — the constants must match it and the
// gpu_world.gd CPU mirror exactly or the far field won't line up with the
// streamed window). Units are VOXELS in the local render frame; Y is world-Y
// voxels, exactly like the instanced emit.
shader_type spatial;
render_mode cull_disabled;   // heightfield seen from above; winding-proof (repo rule)

uniform int seed_v = 0;                  // world seed (vox_step p.seedv)
uniform vec2 origin = vec2(0.0);         // local->world offset (gen_origin), voxels
uniform bool terraced = false;           // gen_flags bit0 (T toggle)
// resident-window rect in LOCAL coords: fragments inside are discarded — the
// window draws itself (rays or fine/LOD instances). Shrunk ~one block by the
// caller so the mesh tucks under the window edge (no crack at the boundary).
uniform vec2 win_min = vec2(1.0);
uniform vec2 win_max = vec2(-1.0);       // min > max = no exclusion
uniform float normal_step = 20.0;        // finite-difference step (voxels)
// radians per screen pixel (2*tan(fov/2)/viewport_h), set by the game — used
// to fade each speckle lattice as its cells go subpixel without derivatives
uniform float px_angle = 0.0015;

const float BLOCK_VOX = 20.0;
const float RELIEF = 5120.0;
const float SEA_Y = RELIEF * 0.30;

varying vec3 v_local;                    // local-frame position (for discard/colour)
varying float v_water;                   // 1 = sea surface vertex
varying float v_hraw;                    // raw ground height (voxels)

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

void vertex() {
	vec3 lp = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;   // local render frame
	vec2 w = lp.xz + origin;                            // world voxels
	float h = world_height(w);
	v_hraw = h;
	v_water = h < SEA_Y ? 1.0 : 0.0;
	float y = max(h, SEA_Y);                            // basins fill to the sea line
	// finite-difference normal from the same height function (sea stays flat)
	float hx = max(world_height(w + vec2(normal_step, 0.0)), SEA_Y);
	float hz = max(world_height(w + vec2(0.0, normal_step)), SEA_Y);
	NORMAL = normalize(vec3(max(h, SEA_Y) - hx, normal_step, max(h, SEA_Y) - hz));
	v_local = vec3(lp.x, y, lp.z);
	// write displaced position in the node's LOCAL space (undo the node offset)
	VERTEX = vec3(VERTEX.x, y - MODEL_MATRIX[3].y, VERTEX.z);
}

// one speckle lattice level: hashed tint at a given cell scale, faded out as
// the cells go subpixel (pxw = world voxels per pixel)
float spek(vec2 w, float scale, float amp, float pxw, uint salt) {
	float fade = clamp(scale / max(pxw, 1e-4) - 0.5, 0.0, 1.0);
	uint h = pcg(uint(int(floor(w.x / scale))) * 2654435761u
			^ uint(int(floor(w.y / scale))) * 40503u ^ salt);
	return (float(h & 255u) / 255.0 - 0.5) * 2.0 * amp * fade;
}

void fragment() {
	// the resident window draws itself — open a hole for it
	if (v_local.x > win_min.x && v_local.x < win_max.x
			&& v_local.z > win_min.y && v_local.z < win_max.y) {
		discard;
	}
	// the underside shows where the (eroded, deposited) sim terrain stands a
	// little above this pure-gen surface at the window edge — light it like
	// the topside instead of a black backface
	NORMAL = FRONT_FACING ? NORMAL : -NORMAL;
	vec2 w = v_local.xz + origin;
	if (v_water > 0.5) {
		ALBEDO = vec3(0.05, 0.18, 0.35);
		ROUGHNESS = 0.15;
		SPECULAR = 0.6;
	} else {
		// multi-scale tint speckle matching the close-up per-voxel jitter: a
		// hash lattice per scale (1 voxel, 25 cm, 1 m, 5 m), each fading out
		// as its cells go subpixel. Near the window edge the mesh speckles
		// per-voxel exactly like the real terrain; far hills keep a coarser
		// grain instead of going flat.
		// world voxels per screen pixel from view distance (no derivatives)
		float pxw = length(VERTEX) * px_angle;
		float tint = 1.0
				+ spek(w, 1.0, 0.12, pxw, 0u)             // per-voxel (5 cm)
				+ spek(w, 5.0, 0.10, pxw, 668265263u)     // 25 cm
				+ spek(w, 20.0, 0.12, pxw, 1336530526u)   // per-block (1 m)
				+ spek(w, 100.0, 0.07, pxw, 2004795789u); // 5 m (far hills)
		// mat_color(GRASS) dimmed ~18%: the window's 5 cm voxels average darker
		// (per-voxel jitter/AO/wetness), so straight grass albedo reads brighter
		// than the terrain it continues — this meets the window in the middle
		ALBEDO = vec3(0.086, 0.210, 0.052) * 0.84 * max(tint, 0.2);
		ROUGHNESS = 1.0;
		SPECULAR = 0.0;
	}
}
