class_name ShallowWater
extends RefCounted
## Coarse 2.5D shallow-water solver (virtual-pipe model) — the km-scale
## background fluid from docs/ECOSYSTEM_ENGINE_DESIGN.md section A. One water
## column per coarse cell over a terrain heightfield; neighbours exchange water
## through pressure-driven "virtual pipes" (see shaders/shallow_water.glsl). It
## is deliberately INDEPENDENT of the fine falling-sand sim: its own compute
## pipeline and buffers on its own RenderingDevice, so it can be built, tested
## and reasoned about in isolation before it is coupled to the fine window.
##
## By default it spins up a LOCAL RenderingDevice (create_local_rendering_device),
## so the solver and its mass-conservation test need no GpuWorld and no shared
## VRAM. Pass an existing device to `new()` to run it on the main device later
## (when the coarse water is rendered / coupled to the fine window).

var rd: RenderingDevice
var _own_rd := false          # true if we created the device and must free it
var shader: RID
var pipeline: RID
var terr_buf: RID
var water_buf: RID
var flux_buf: RID
var uniform_set: RID

var N := 0                    # grid side (cells)
var L := 4.0                  # cell size = pipe length/width (world voxels)
var dt := 0.08                # timestep
var g := 9.81                 # gravity
var A := 1.0                  # virtual-pipe cross-section
var _groups := 0             # dispatch groups per axis (N / 8, rounded up)

const WG := 8                 # local_size_x/y in the shader
const MODE_INIT := 0
const MODE_FLUX := 1
const MODE_DEPTH := 2
const MODE_ADD := 3
const PC_SIZE := 64           # push constant bytes (16 * 4; multiple of 16)

## n = grid side (cells). device: reuse an existing RenderingDevice (main
## device, for rendering/coupling) or leave null to create an isolated local one.
func _init(n: int, cell_size: float = 4.0, device: RenderingDevice = null) -> void:
	N = n
	L = cell_size
	if device != null:
		rd = device
	else:
		rd = RenderingServer.create_local_rendering_device()
		_own_rd = true
	if rd == null:
		push_error("ShallowWater: no RenderingDevice (headless dummy driver?) — needs a windowed/GPU context")
		return
	var src := RDShaderSource.new()
	src.source_compute = FileAccess.get_file_as_string("res://shaders/shallow_water.glsl")
	var spirv := rd.shader_compile_spirv_from_source(src)
	if spirv.compile_error_compute != "":
		push_error("ShallowWater shader compile failed:\n" + spirv.compile_error_compute)
		return
	shader = rd.shader_create_from_spirv(spirv)
	pipeline = rd.compute_pipeline_create(shader)
	var cells := N * N
	terr_buf = rd.storage_buffer_create(cells * 4)          # float per cell
	water_buf = rd.storage_buffer_create(cells * 4)
	flux_buf = rd.storage_buffer_create(cells * 4 * 4)      # 4 pipes per cell
	var us: Array[RDUniform] = []
	for pair: Array in [[0, terr_buf], [1, water_buf], [2, flux_buf]]:
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = pair[0]
		u.add_id(pair[1])
		us.append(u)
	uniform_set = rd.uniform_set_create(us, shader, 0)
	_groups = ceili(N / float(WG))

func ok() -> bool:
	return rd != null and pipeline.is_valid()

# push constant: [N, mode, L, dt, g, A, p0..p3, u0..u3, pad, pad]
func _pc(mode: int, p0 := 0.0, p1 := 0.0, p2 := 0.0, p3 := 0.0,
		u0 := 0, u1 := 0, u2 := 0, u3 := 0) -> PackedByteArray:
	var pc := PackedByteArray()
	pc.resize(PC_SIZE)
	pc.encode_u32(0, N)
	pc.encode_u32(4, mode)
	pc.encode_float(8, L)
	pc.encode_float(12, dt)
	pc.encode_float(16, g)
	pc.encode_float(20, A)
	pc.encode_float(24, p0)
	pc.encode_float(28, p1)
	pc.encode_float(32, p2)
	pc.encode_float(36, p3)
	pc.encode_u32(40, u0)
	pc.encode_u32(44, u1)
	pc.encode_u32(48, u2)
	pc.encode_u32(52, u3)
	pc.encode_u32(56, 0)
	pc.encode_u32(60, 0)
	return pc

func _dispatch(pc: PackedByteArray) -> void:
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	rd.compute_list_set_push_constant(cl, pc, PC_SIZE)
	rd.compute_list_dispatch(cl, _groups, _groups, 1)
	rd.compute_list_end()
	if _own_rd:
		rd.submit()
		rd.sync()

## fill terrain with a tilted plane: b = base + slope_x*x + slope_z*z (per cell)
func init_plane(base: float, slope_x: float, slope_z: float) -> void:
	_dispatch(_pc(MODE_INIT, base, slope_x, slope_z, 0.0))

## fill terrain with a radial bowl: b = base + curv*((x-cx)^2 + (z-cz)^2)
func init_bowl(base: float, curv: float) -> void:
	_dispatch(_pc(MODE_INIT, base, 0.0, 0.0, curv))

## raise the water column by `amount` across cells [x0,x1) x [z0,z1)
func add_water(x0: int, z0: int, x1: int, z1: int, amount: float) -> void:
	_dispatch(_pc(MODE_ADD, amount, 0.0, 0.0, 0.0, x0, z0, x1, z1))

## advance n solver ticks (flux half-step, barrier, depth half-step, barrier)
func step(n: int) -> void:
	if not ok():
		return
	var fpc := _pc(MODE_FLUX)
	var dpc := _pc(MODE_DEPTH)
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	for k in range(n):
		rd.compute_list_set_push_constant(cl, fpc, PC_SIZE)
		rd.compute_list_dispatch(cl, _groups, _groups, 1)
		rd.compute_list_add_barrier(cl)
		rd.compute_list_set_push_constant(cl, dpc, PC_SIZE)
		rd.compute_list_dispatch(cl, _groups, _groups, 1)
		rd.compute_list_add_barrier(cl)
	rd.compute_list_end()
	if _own_rd:
		rd.submit()
		rd.sync()

## pull the water-depth grid to the CPU (row-major, x + z*N)
func read_water() -> PackedFloat32Array:
	if not ok():
		return PackedFloat32Array()
	return rd.buffer_get_data(water_buf).to_float32_array()

## pull the terrain-height grid to the CPU
func read_terr() -> PackedFloat32Array:
	if not ok():
		return PackedFloat32Array()
	return rd.buffer_get_data(terr_buf).to_float32_array()

## total water VOLUME (sum of depth * cell area) — the mass-conservation metric
func total_water() -> float:
	var w := read_water()
	var s := 0.0
	for d in w:
		s += d
	return s * L * L

func free_gpu() -> void:
	if rd == null:
		return
	for r in [uniform_set, pipeline, shader, terr_buf, water_buf, flux_buf]:
		if r.is_valid():
			rd.free_rid(r)
	if _own_rd:
		rd.free()   # local device: we own it
	rd = null
