#version 450

// COARSE SHALLOW-WATER SOLVER (virtual-pipe model, Mei et al. 2007 — "Fast
// Hydraulic Erosion Simulation and Visualization on GPU"). This is the 2.5D
// background fluid the multi-res design calls for: one water COLUMN per coarse
// cell over a terrain heightfield, connected to its 4 neighbours by "virtual
// pipes" whose flow is driven by the water-surface height difference. It covers
// a huge footprint cheaply (a heightfield, not volumetric voxels) so rivers and
// lakes can exist at km scale beyond the fine falling-sand window — see
// docs/ECOSYSTEM_ENGINE_DESIGN.md section A, Option 1.
//
// Why the pipe model: it is unconditionally MASS-CONSERVING. Each cell's total
// outflow over a tick is clamped to the water it actually holds (the K factor
// below), so depth can never go negative and, with reflective boundaries (pipes
// off the grid edge carry zero), the summed water column is invariant to float
// precision. Mass conservation is the acceptance test for the coarse solver.
//
// Grid layout: N x N cells, row-major (idx = x + z*N). Three storage buffers:
//   terr  [N*N]      terrain floor height b   (world voxels)
//   water [N*N]      water column height d     (voxels; surface = b + d)
//   flux  [N*N*4]    outflow to L,R,T,B pipes  (volume rate, voxel^3 / tick)
// The solver is single-buffered: the flux pass reads heights and writes flux;
// the depth pass reads flux and writes depth; a barrier between them (issued by
// the host) keeps the two half-steps ordered. Every cell writes only its OWN
// flux / depth, so there is no read-after-write hazard within a pass.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict buffer TerrBuf  { float terr[]; };
layout(set = 0, binding = 1, std430) restrict buffer WaterBuf { float water[]; };
layout(set = 0, binding = 2, std430) restrict buffer FluxBuf  { float flux[]; };

layout(push_constant, std430) uniform Params {
	uint  N;      // grid side (cells)
	uint  mode;   // 0 init, 1 flux, 2 depth, 3 add-water
	float L;      // cell size = pipe length = pipe width (world voxels)
	float dt;     // timestep
	float g;      // gravity
	float A;      // virtual-pipe cross-section
	float p0;     // mode param: init base height / add-water amount
	float p1;     // mode param: init slope-x  / add-water region reuse
	float p2;     // mode param: init slope-z
	float p3;     // mode param: init kind (0 plane, 1 bowl) / bowl curvature
	uint  u0;     // add-water region x0
	uint  u1;     // add-water region z0
	uint  u2;     // add-water region x1
	uint  u3;     // add-water region z1
	uint  pad0;
	uint  pad1;
} p;

const uint L_ = 0u;   // pipe indices within a cell's flux[base + dir]
const uint R_ = 1u;
const uint T_ = 2u;   // -z ("top")
const uint B_ = 3u;   // +z ("bottom")

uint cidx(uint x, uint z) { return x + z * p.N; }

// ---- mode 0: initialise the grid --------------------------------------------
// Synthetic terrain for the solver unit tests: a tilted plane (kind 0) or a
// radial bowl (kind 1). Water and flux start empty. The worldgen-heightfield
// terrain used by the real far field is filled by the host on the CPU / a
// future mode, not here, so this file stays independent of the worldgen noise.
void do_init(uint x, uint z) {
	uint i = cidx(x, z);
	float b;
	if (p.p3 <= 0.0) {
		// plane (no curvature): base + slope per cell in x and z
		b = p.p0 + p.p1 * float(x) + p.p2 * float(z);
	} else {
		// bowl: paraboloid rising from the grid centre outward
		float cx = float(x) - float(p.N) * 0.5;
		float cz = float(z) - float(p.N) * 0.5;
		b = p.p0 + p.p3 * (cx * cx + cz * cz);
	}
	terr[i] = b;
	water[i] = 0.0;
	uint fb = i * 4u;
	flux[fb + L_] = 0.0;
	flux[fb + R_] = 0.0;
	flux[fb + T_] = 0.0;
	flux[fb + B_] = 0.0;
}

// ---- mode 1: flux update ----------------------------------------------------
// For each of the 4 pipes, accelerate the outflow by the water-surface height
// drop to that neighbour: f += dt * A * g * dh / L, clamped at >= 0 (pipes only
// push out; the neighbour's own pipe handles the reverse). Off-grid pipes stay
// zero -> reflective, mass-conserving boundary. Finally scale all 4 outflows by
// K so the tick's total outflow can't exceed the water actually present; this
// is what makes depth stay non-negative and mass exactly conserved.
void do_flux(uint x, uint z) {
	uint i = cidx(x, z);
	float d = water[i];
	float hs = terr[i] + d;                 // this cell's water-surface height
	uint fb = i * 4u;
	float coeff = p.dt * p.A * p.g / p.L;

	// neighbour surface heights; a wall (off grid) reads as +inf drop = no outflow
	float fL = flux[fb + L_];
	float fR = flux[fb + R_];
	float fT = flux[fb + T_];
	float fB = flux[fb + B_];
	if (x > 0u)        { fL = max(0.0, fL + coeff * (hs - (terr[cidx(x-1u, z)] + water[cidx(x-1u, z)]))); } else { fL = 0.0; }
	if (x + 1u < p.N)  { fR = max(0.0, fR + coeff * (hs - (terr[cidx(x+1u, z)] + water[cidx(x+1u, z)]))); } else { fR = 0.0; }
	if (z > 0u)        { fT = max(0.0, fT + coeff * (hs - (terr[cidx(x, z-1u)] + water[cidx(x, z-1u)]))); } else { fT = 0.0; }
	if (z + 1u < p.N)  { fB = max(0.0, fB + coeff * (hs - (terr[cidx(x, z+1u)] + water[cidx(x, z+1u)]))); } else { fB = 0.0; }

	// positivity clamp: outflow volume this tick <= water volume in the column
	float out_sum = (fL + fR + fT + fB) * p.dt;
	float avail = d * p.L * p.L;
	float K = 1.0;
	if (out_sum > 1e-12 && out_sum > avail) { K = avail / out_sum; }
	flux[fb + L_] = fL * K;
	flux[fb + R_] = fR * K;
	flux[fb + T_] = fT * K;
	flux[fb + B_] = fB * K;
}

// ---- mode 2: depth update ---------------------------------------------------
// Net volume change = inflow (each neighbour's pipe pointing back at us) minus
// this cell's total outflow, integrated over dt, spread over the cell area.
// Reads only flux (written last pass) + own depth; writes own depth.
void do_depth(uint x, uint z) {
	uint i = cidx(x, z);
	uint fb = i * 4u;
	float outflow = flux[fb + L_] + flux[fb + R_] + flux[fb + T_] + flux[fb + B_];
	float inflow = 0.0;
	if (x > 0u)       { inflow += flux[cidx(x-1u, z) * 4u + R_]; }   // left  neighbour flowing right into us
	if (x + 1u < p.N) { inflow += flux[cidx(x+1u, z) * 4u + L_]; }   // right neighbour flowing left  into us
	if (z > 0u)       { inflow += flux[cidx(x, z-1u) * 4u + B_]; }   // top   neighbour flowing down  into us
	if (z + 1u < p.N) { inflow += flux[cidx(x, z+1u) * 4u + T_]; }   // bottom neighbour flowing up   into us
	float dV = p.dt * (inflow - outflow);
	float d = water[i] + dV / (p.L * p.L);
	water[i] = max(d, 0.0);   // guard the last ulp; the K clamp already prevents real drainage past empty
}

// ---- mode 3: add water ------------------------------------------------------
// Raise the water column by p0 across the rectangle [u0,u2) x [u1,u3). Used by
// the host to inject a source / a rain blob without a CPU upload.
void do_add(uint x, uint z) {
	if (x < p.u0 || x >= p.u2 || z < p.u1 || z >= p.u3) { return; }
	water[cidx(x, z)] += p.p0;
}

void main() {
	uint x = gl_GlobalInvocationID.x;
	uint z = gl_GlobalInvocationID.y;
	if (x >= p.N || z >= p.N) { return; }
	if      (p.mode == 0u) { do_init(x, z); }
	else if (p.mode == 1u) { do_flux(x, z); }
	else if (p.mode == 2u) { do_depth(x, z); }
	else if (p.mode == 3u) { do_add(x, z); }
}
