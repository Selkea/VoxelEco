// FAR-FIELD CLIPMAP MESH: a continuous heightfield mesh replacing the old
// instanced far tiles. The terrain surface is a pure function of world (x,z)
// + seed, so static grid rings that ride the camera displace their vertices
// right here in the vertex shader by evaluating the SAME worldgen noise the
// sim uses (ported from vox_step.glsl — the constants must match it and the
// gpu_world.gd CPU mirror exactly or the far field won't line up with the
// streamed window). Units are VOXELS in the local render frame; Y is world-Y
// voxels, exactly like the instanced emit.
shader_type spatial;
// UNSHADED: lit manually with the EXACT formula the ray renderer uses
// (ray_shade in vox_step.glsl), so the window and the far field are the same
// math at every view angle — no SSAO/Fresnel/PBR divergence to calibrate away.
// The engine still tonemaps (ACES) + sRGB-encodes this output, mirroring the
// ray path's own ACES fit + gamma. tone_scale absorbs the tiny residual
// between the two pipelines (calibrated by pixel measurement).
render_mode unshaded, cull_disabled;

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
uniform vec3 sun_dir = vec3(0.0, -1.0, 0.0);   // light travel direction (cam_sun)
uniform vec3 tone_scale = vec3(0.948, 0.940, 0.883);   // residual pipeline calibration

const float BLOCK_VOX = 20.0;
const float RELIEF = 5120.0;
const float SEA_Y = RELIEF * 0.30;

varying vec3 v_local;                    // local-frame position (for discard/colour)
varying float v_water;                   // 1 = sea surface vertex
varying float v_hraw;                    // raw ground height (voxels)
varying vec3 v_nrm;                      // world-axis surface normal

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
	v_nrm = normalize(vec3(max(h, SEA_Y) - hx, normal_step, max(h, SEA_Y) - hz));
	v_local = vec3(lp.x, y, lp.z);
	// write displaced position in the node's LOCAL space (undo the node offset)
	VERTEX = vec3(VERTEX.x, y - MODEL_MATRIX[3].y, VERTEX.z);
}

// one speckle lattice level: hashed tint at a given cell scale, faded out as
// the cells go subpixel (pxw = world voxels per pixel)
float spek(vec2 w, float scale, float amp, float pxw, uint salt) {
	// full only once a cell covers ~3 pixels — mirrors how real voxel jitter
	// averages away with distance (the old earlier fade-in over-grained the
	// mesh right at the window seam)
	float fade = clamp((scale / max(pxw, 1e-4) - 1.0) * 0.5, 0.0, 1.0);
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
	vec3 n = normalize(v_nrm) * (FRONT_FACING ? 1.0 : -1.0);
	vec2 w = v_local.xz + origin;
	vec3 alb;
	if (v_water > 0.5) {
		alb = vec3(0.024, 0.13, 0.34);   // same water base the rays shade with
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
				+ spek(w, 5.0, 0.05, pxw, 668265263u)     // 25 cm
				+ spek(w, 20.0, 0.05, pxw, 1336530526u)   // per-block (1 m)
				+ spek(w, 100.0, 0.03, pxw, 2004795789u); // 5 m (far hills)
		alb = vec3(0.086, 0.210, 0.052) * max(tint, 0.2);   // mat_color(GRASS)
		// 1 m contour lines like the window's terrace-step shadows: darken a
		// ~1-voxel band at each whole-metre height (density tracks slope for
		// free — flat ground crosses metre levels rarely), fading as the band
		// goes subpixel. This carries the window's signature texture across
		// the renderer seam.
		float cd = abs(v_hraw / 20.0 - round(v_hraw / 20.0)) * 20.0;
		float cfade = clamp(2.0 / max(pxw, 1e-4) - 0.2, 0.0, 1.0);
		alb *= 1.0 - (1.0 - smoothstep(0.5, 1.2, cd)) * cfade * 0.35;
	}
	// STAIR-MODEL lighting, matching how the rays actually light voxel terrain:
	// blended slopes render as stair-stepped 5 cm TOP faces (normal straight
	// up, one constant sun term everywhere) with occasional dark RISER faces
	// whose share of the surface grows with slope. Smooth Lambert normals here
	// made every away-facing slope darker than the window — the visible LOD
	// line. Same light colours as ray_shade; engine ACES+sRGB mirrors its
	// ACES fit + gamma; tone_scale is the pixel-measured residual.
	vec3 sdir = -normalize(sun_dir);
	vec3 amb = vec3(0.28, 0.34, 0.50);
	vec3 sun = vec3(0.81, 0.79, 0.82);
	vec3 lit_top = alb * (amb + sun * max(sdir.y, 0.0));
	vec2 hn = vec2(n.x, n.z);
	float hl = length(hn);
	float ndl_side = hl > 1e-4 ? max(dot(hn / hl, sdir.xz), 0.0) : 0.0;
	vec3 lit_side = alb * (amb + sun * ndl_side);
	float riser = clamp((1.0 - n.y) * 3.0, 0.0, 0.5);   // riser share by slope
	vec3 lit = mix(lit_top, lit_side, riser);
	ALBEDO = lit * 0.98 * tone_scale;
}
