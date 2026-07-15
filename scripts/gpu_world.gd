class_name GpuWorld
extends VoxWorld
## GPU-accelerated VoxWorld running on Godot's MAIN RenderingDevice, so the
## emit pass writes MultiMesh instance buffers directly in VRAM — zero
## readback, no CPU loops anywhere in worldgen, physics, or rendering.
## Physics is a Margolus 2x2x2 compute kernel (Noita's conflict-free
## checkerboard at per-thread granularity). Falls back to CPU stepping
## transparently when no RenderingDevice exists (plain --headless).

var rd: RenderingDevice
var gpu_ok := false
var shader: RID
var pipeline: RID
var cells_buf: RID
var pack_buf: RID
var stats_buf: RID
var dirty_buf: RID
var inst_count_buf: RID
var uniform_set: RID
var _solid_target: RID    # multimesh buffers (bound by the view) or placeholders
var _water_target: RID
var _placeholder_a: RID
var _placeholder_b: RID

var chunk_w := 0
var chunk_d := 0
var solid_cap := 0
var water_cap := 0
var _need_gpu_gen := false

var _step_groups := Vector3i()
var _col_groups := 0      # one thread per column (rain / worldgen)
var _cell_groups := 0     # one thread per cell (emit)
var _pack_groups := 0

var rules_mask := 0   # debug: bit0 gravity, 1 diagonal, 2 lateral, 3 evap, 4 erosion; 0 = all
# world-space origin of this buffer in voxels — worldgen samples noise at
# (local + origin) so a chunk generates seamlessly at any world position.
var gen_origin_x := 0
var gen_origin_z := 0
var gen_flags := 0        # bit0: 1 = terraced worldgen, 0 = blended

## grids past this size generate on the GPU instead of a GDScript loop
const CPU_GEN_LIMIT := 400_000

func _generate() -> void:
	if W * D * H <= CPU_GEN_LIMIT:
		super._generate()
	else:
		_need_gpu_gen = true   # leave cells AIR; the gen kernel fills them

func _init(seed_v: int = 0, w: int = 64, d: int = 64, h: int = 40) -> void:
	super(seed_v, w, d, h)
	chunk_w = (W + 15) / 16
	chunk_d = (D + 15) / 16
	rd = RenderingServer.get_rendering_device()
	if rd == null:
		push_warning("GpuWorld: no RenderingDevice — falling back to CPU stepping")
		if _need_gpu_gen:
			super._generate()   # slow but functional CPU fallback
		return
	var src := RDShaderSource.new()
	src.source_compute = FileAccess.get_file_as_string("res://shaders/vox_step.glsl")
	var spirv := rd.shader_compile_spirv_from_source(src)
	if spirv.compile_error_compute != "":
		push_error("GpuWorld shader compile failed:\n" + spirv.compile_error_compute)
		return
	shader = rd.shader_create_from_spirv(spirv)
	pipeline = rd.compute_pipeline_create(shader)

	var n := cell.size()
	if _need_gpu_gen:
		var blank := PackedByteArray()
		blank.resize(n * 4)          # all AIR; the gen kernel fills it
		cells_buf = rd.storage_buffer_create(n * 4, blank)
	else:
		var ints := PackedInt32Array()
		ints.resize(n)
		for i in range(n):
			ints[i] = cell[i]
		cells_buf = rd.storage_buffer_create(n * 4, ints.to_byte_array())
	var words := (n + 3) / 4
	pack_buf = rd.storage_buffer_create(words * 4)
	stats_buf = rd.storage_buffer_create(12, PackedByteArray([0,0,0,0,0,0,0,0,0,0,0,0]))
	var zeros := PackedByteArray()
	zeros.resize(chunk_w * chunk_d * 4)
	dirty_buf = rd.storage_buffer_create(zeros.size(), zeros)
	solid_cap = W * D * 4
	water_cap = W * D * 2
	var caps := PackedInt32Array([0, 0, solid_cap, water_cap]).to_byte_array()
	inst_count_buf = rd.storage_buffer_create(16, caps)
	# tiny placeholders until the view binds real multimesh buffers
	_placeholder_a = rd.storage_buffer_create(64)
	_placeholder_b = rd.storage_buffer_create(64)
	_solid_target = _placeholder_a
	_water_target = _placeholder_b
	_rebuild_uniform_set()

	_step_groups = Vector3i(
		ceili(W / 2.0 / 4.0), ceili(H / 2.0 / 4.0), ceili(D / 2.0 / 4.0))
	_col_groups = ceili(W * D / 64.0)
	_cell_groups = ceili(W * D * H / 64.0)
	_pack_groups = ceili(words / 64.0)
	gpu_ok = true
	if OS.get_environment("VOX_GENMODE") == "terraced":
		gen_flags = 1
	if _need_gpu_gen:
		var cl := rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(cl, pipeline)
		rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
		rd.compute_list_set_push_constant(cl, _pc(4, 0), PC_SIZE)
		rd.compute_list_dispatch(cl, _col_groups, 1, 1)
		rd.compute_list_end()

## regenerate the whole buffer as if its lower corner sits at world (ox, oz).
## This is the chunk-streaming primitive: the same kernel fills any chunk at any
## world position, and the seamless world-space noise makes the joins invisible.
func regen(ox: int, oz: int) -> void:
	gen_origin_x = ox
	gen_origin_z = oz
	if not gpu_ok:
		return
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	rd.compute_list_set_push_constant(cl, _pc(4, 0), PC_SIZE)
	rd.compute_list_dispatch(cl, _col_groups, 1, 1)
	rd.compute_list_end()

func _rebuild_uniform_set() -> void:
	var bufs := [cells_buf, pack_buf, stats_buf, dirty_buf,
			_solid_target, _water_target, inst_count_buf]
	var us: Array[RDUniform] = []
	for b in range(bufs.size()):
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = b
		u.add_id(bufs[b])
		us.append(u)
	uniform_set = rd.uniform_set_create(us, shader, 0)

## the view hands us its MultiMesh storage buffers: the emit pass writes
## instance transforms/colors straight into them on the GPU
func bind_instance_buffers(solid_rid: RID, water_rid: RID) -> void:
	if not gpu_ok:
		return
	_solid_target = solid_rid
	_water_target = water_rid
	_rebuild_uniform_set()

const PC_SIZE := 56   # push constant byte size (must match the shader struct)

func _pc(mode: int, offset: int) -> PackedByteArray:
	# 56-byte push constant: 11 uints + 2 floats + gen origin (2) + 1 pad. rain
	# is an integer threshold out of 2^24; evap/erode are float probabilities.
	var pc := PackedByteArray()
	pc.resize(PC_SIZE)
	pc.encode_u32(0, W)
	pc.encode_u32(4, H)
	pc.encode_u32(8, D)
	pc.encode_u32(12, tick_count)
	pc.encode_u32(16, seed_value)
	pc.encode_u32(20, int(clampf(rain_prob, 0.0, 1.0) * 16777216.0))
	pc.encode_u32(24, mode | (rules_mask << 8))
	pc.encode_u32(28, offset)
	pc.encode_float(32, evap_prob)
	pc.encode_float(36, erode_prob)
	pc.encode_u32(40, cut_z)
	pc.encode_s32(44, gen_origin_x)   # signed origin (two's complement); int(uint) in-shader
	pc.encode_s32(48, gen_origin_z)
	pc.encode_u32(52, gen_flags)
	return pc

func step() -> void:
	if not gpu_ok:
		super.step()
		return
	run(1)

## queue n physics ticks; on the main device they execute with the frame
## (any readback below flushes them)
func run(n: int) -> void:
	if not gpu_ok:
		for i in range(n):
			super.step()
		return
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	for k in range(n):
		tick_count += 1
		rd.compute_list_set_push_constant(cl, _pc(1, 0), PC_SIZE)      # rain
		rd.compute_list_dispatch(cl, _col_groups, 1, 1)
		rd.compute_list_add_barrier(cl)
		rd.compute_list_set_push_constant(cl, _pc(0, tick_count & 1), PC_SIZE)  # physics
		rd.compute_list_dispatch(cl, _step_groups.x, _step_groups.y, _step_groups.z)
		rd.compute_list_add_barrier(cl)
	rd.compute_list_end()

## did the physics change any cell since the last check? (reads + clears the
## dirty-chunk flags). Lets the renderer skip re-emitting a static world — no
## re-emit means the instance buffer is untouched, so a still surface can't
## shimmer from the emit's non-deterministic instance order.
func any_dirty_and_clear() -> bool:
	if not gpu_ok:
		return true
	var d := rd.buffer_get_data(dirty_buf).to_int32_array()
	var any := false
	for v in d:
		if v != 0:
			any = true
			break
	if any:
		var zeros := PackedByteArray()
		zeros.resize(chunk_w * chunk_d * 4)
		rd.buffer_update(dirty_buf, 0, zeros.size(), zeros)
	return any

## dispatch the emit pass (writes multimesh buffers in VRAM) and read back
## only the two instance counters — 16 bytes, the sole per-frame readback
func dispatch_emit() -> PackedInt32Array:
	if not gpu_ok:
		return PackedInt32Array()
	rd.buffer_update(inst_count_buf, 0, 8, PackedByteArray([0,0,0,0,0,0,0,0]))
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	rd.compute_list_set_push_constant(cl, _pc(3, 0), PC_SIZE)
	rd.compute_list_dispatch(cl, _cell_groups, 1, 1)
	rd.compute_list_end()
	var counts := rd.buffer_get_data(inst_count_buf, 0, 8).to_int32_array()
	return PackedInt32Array([mini(counts[0], solid_cap), mini(counts[1], water_cap)])

## Pull GPU state back to the CPU: cell bytes (tests/analysis), water stats,
## dirty-chunk flags. Stalls the pipe — fine for tests, not per-frame.
func sync_cells() -> void:
	if not gpu_ok:
		return
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	rd.compute_list_set_push_constant(cl, _pc(2, 0), PC_SIZE)          # pack
	rd.compute_list_dispatch(cl, _pack_groups, 1, 1)
	rd.compute_list_end()
	cell = rd.buffer_get_data(pack_buf).slice(0, cell.size())
	var st := rd.buffer_get_data(stats_buf).to_int32_array()
	water_added = st[0]
	water_evaporated = st[1]
	water_absorbed = st[2]
	dirty_chunks = rd.buffer_get_data(dirty_buf).to_int32_array()
	var zeros := PackedByteArray()
	zeros.resize(chunk_w * chunk_d * 4)
	rd.buffer_update(dirty_buf, 0, zeros.size(), zeros)

## push the CPU-side `cell` bytes back up to the GPU buffer (small worlds /
## test scaffolding only — this loops over cells in GDScript)
func upload_cells() -> void:
	if not gpu_ok:
		return
	var n := cell.size()
	var ints := PackedInt32Array()
	ints.resize(n)
	for i in range(n):
		ints[i] = cell[i]
	rd.buffer_update(cells_buf, 0, n * 4, ints.to_byte_array())

func reset_water_stats() -> void:
	super.reset_water_stats()
	if gpu_ok:
		rd.buffer_update(stats_buf, 0, 12, PackedByteArray([0,0,0,0,0,0,0,0,0,0,0,0]))

func free_gpu() -> void:
	if rd == null:
		return
	# free our resources, never the shared main device
	for r in [uniform_set, pipeline, shader, cells_buf, pack_buf, stats_buf,
			dirty_buf, inst_count_buf, _placeholder_a, _placeholder_b]:
		if r.is_valid():
			rd.free_rid(r)
	rd = null
	gpu_ok = false
