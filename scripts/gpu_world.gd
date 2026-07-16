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
var cells_buf2: RID    # second half of a >4GB world (see cells_split)
# first linear cell index living in cells_buf2. Godot caps one storage buffer at
# 4 GB (32-bit byte size), so bigger worlds split at a whole y-slab boundary; the
# shader's cget/cset route each access. Fits-in-one worlds set split = cell count.
var cells_split := 0
var pack_buf: RID
var stats_buf: RID
var dirty_buf: RID
var active_buf: RID    # per-chunk physics keepalive (active-block skipping)
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
var _cell_groups := 0     # one thread per cell (voxel emit)
var _pack_groups := 0
var _block_groups := 0    # one thread per 1m block (block emit)
var _decay_groups := 0    # one thread per chunk (active-flag decay)
var nbx := 0
var nby := 0
var nbz := 0
# render coarsened 1m blocks (one cube per 20x20x20 voxels) rather than 5cm
# voxels — decouples render cost from the fine sim so the map can be large.
# VOX_RENDER=voxel forces the old per-voxel path (small worlds only).
var block_render := true
var mesh_render := true    # greedy-mesh quads (the fast default); overrides block/voxel
var can_toggle_render := false   # buffers sized for both render modes (live toggle)
var _pack_full := false   # pack buffer grown to full size on first sync_cells

var rules_mask := 0   # debug: bit0 gravity, 1 diagonal, 2 lateral, 3 evap, 4 erosion; 0 = all
# world-space origin of this buffer in voxels — worldgen samples noise at
# (local + origin) so a chunk generates seamlessly at any world position.
var gen_origin_x := 0
var gen_origin_z := 0
var gen_flags := 0        # bit0: 1 = terraced worldgen, 0 = blended
# toroidal edge regen: packed x0|width<<16 buffer-slot strip (0 = full width)
var gen_strip_x := 0
var gen_strip_z := 0
# vertical-tracking band: world-Y voxel of the buffer floor. The buffer stores
# world-Y [gen_oy, gen_oy+H); gen/emit add it to map buffer y -> world y. Tracked
# so the thin band rides the terrain surface through a 256 m-relief world.
var gen_oy := 0

# total vertical relief and sea level in voxels — MUST match the shader's RELIEF
# and SEA_Y. The surface spans [0, RELIEF]; only a p.H-tall band is ever resident.
const RELIEF := 5120        # 256 m at 5 cm/voxel
const SEA_Y := 1536         # RELIEF * 0.30

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
	# split point: worlds over ~4 GB of cells split across two buffers at a whole
	# y-slab (VOX_FORCESPLIT=1 forces a split on small worlds for the equivalence
	# test); otherwise everything lives in buffer A and B is a stub.
	if n * 4 > 4_000_000_000 or OS.get_environment("VOX_FORCESPLIT") != "":
		cells_split = (H / 2) * W * D
	else:
		cells_split = n
	if _need_gpu_gen:
		# no CPU upload — the gen kernel writes every cell, so allocating a
		# multi-GB zeroed CPU array just to upload it would blow out RAM on big
		# worlds. Create the VRAM buffers uninitialised; do_gen fills them in _init.
		cells_buf = rd.storage_buffer_create(cells_split * 4)
		cells_buf2 = rd.storage_buffer_create(maxi((n - cells_split) * 4, 64))
	else:
		var ints := PackedInt32Array()
		ints.resize(n)
		for i in range(n):
			ints[i] = cell[i]
		var bytes := ints.to_byte_array()
		cells_buf = rd.storage_buffer_create(cells_split * 4, bytes.slice(0, cells_split * 4))
		cells_buf2 = rd.storage_buffer_create(maxi((n - cells_split) * 4, 64),
				bytes.slice(cells_split * 4) if n > cells_split else PackedByteArray())
	var words := (n + 3) / 4          # pack-dispatch thread count (see _pack_groups)
	# pack buffer (~1 byte/cell) is only read by sync_cells (tests/analysis),
	# never in the interactive/render path — allocate a stub and grow it on the
	# first sync_cells so huge worlds don't pay for it.
	pack_buf = rd.storage_buffer_create(64)
	stats_buf = rd.storage_buffer_create(12, PackedByteArray([0,0,0,0,0,0,0,0,0,0,0,0]))
	var zeros := PackedByteArray()
	zeros.resize(chunk_w * chunk_d * 4)
	dirty_buf = rd.storage_buffer_create(zeros.size(), zeros)
	active_buf = rd.storage_buffer_create(zeros.size(), zeros)   # all asleep until gen wakes
	nbx = (W + 19) / 20
	nby = (H + 19) / 20
	nbz = (D + 19) / 20
	# default: draw all 5cm voxels (the true fine surface); VOX_RENDER=block draws
	# the coarser 1m block shell instead (fewer instances, chunky look). Allocate
	# for BOTH modes (per-voxel is the larger) so the render can be toggled live
	# (B key) with no reallocation — unless the per-voxel count is too many for one
	# buffer, then lock to the block shell.
	block_render = OS.get_environment("VOX_RENDER") == "block"
	var skin_cap := W * D + nbx * nby * nbz
	can_toggle_render = W * D * 4 <= 60_000_000
	if can_toggle_render:
		solid_cap = W * D * 4
		water_cap = W * D * 2
	else:
		block_render = true
		solid_cap = skin_cap
		water_cap = skin_cap
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
	_block_groups = ceili(nbx * nbz / 64.0)   # one thread per 1m block column
	_decay_groups = ceili(chunk_w * chunk_d / 64.0)   # one thread per 16x16 chunk
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
		# free the CPU-side cell mirror (VoxWorld allocated W*D*H bytes) — the sim
		# lives entirely in VRAM now. sync_cells restores it on demand for tests.
		cell = PackedByteArray()

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

## regenerate with the vertical band auto-placed under the terrain surface at the
## window centre (band_oy_for). Convenience for spinning up a world whose surface
## sits in a tall-relief world — the tests and headless spin-ups use it.
func regen_tracked(ox: int, oz: int) -> void:
	gen_oy = band_oy_for(ox + W * 0.5, oz + D * 0.5)
	regen(ox, oz)

## toroidal streaming: shift the window's world origin without regenerating —
## the buffer keeps its live sim state; only regen_strip refills entered edges.
func set_origin(ox: int, oz: int) -> void:
	gen_origin_x = ox
	gen_origin_z = oz

## regenerate ONLY the buffer-slot strip [x0, x0+width) x [z0, z0+depth) (mod
## W/D) at the current origin — the freshly-entered edge — leaving the rest of
## the window (and its water/erosion) untouched. This is what makes streaming
## state-preserving instead of resetting the whole window each recenter.
func regen_strip(x0: int, width: int, z0: int, depth: int) -> void:
	if not gpu_ok:
		return
	gen_strip_x = (x0 & 0xFFFF) | (width << 16)
	gen_strip_z = (z0 & 0xFFFF) | (depth << 16)
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	rd.compute_list_set_push_constant(cl, _pc(4, 0), PC_SIZE)
	rd.compute_list_dispatch(cl, _col_groups, 1, 1)
	rd.compute_list_end()
	gen_strip_x = 0   # restore full width for any later whole-buffer regen
	gen_strip_z = 0

# ---- CPU mirror of the shader's blended world_height, so the band can be placed
# under the camera without a GPU readback. Must stay bit-compatible with the GLSL
# (same pcg hash, same fbm, same bands) — see block_height/world_height in the
# shader. Only the blended surface is mirrored; the band's margins absorb the tiny
# terraced-vs-blended difference. All hashing emulates 32-bit unsigned wraparound.
func _pcg(v: int) -> int:
	v = (v * 747796405 + 2891336453) & 0xFFFFFFFF
	v = (((v >> ((v >> 28) + 4)) ^ v) * 277803737) & 0xFFFFFFFF
	return ((v >> 22) ^ v) & 0xFFFFFFFF

func _vhash(qx: float, qy: float) -> float:
	var a := (int(qx) * 374761393) & 0xFFFFFFFF
	var b := (int(qy) * 668265263) & 0xFFFFFFFF
	return float(_pcg(a ^ b ^ (seed_value & 0xFFFFFFFF))) * (1.0 / 4294967296.0)

func _vnoise(qx: float, qy: float) -> float:
	var ix := floorf(qx)
	var iy := floorf(qy)
	var fx := qx - ix
	var fy := qy - iy
	var ux := fx * fx * (3.0 - 2.0 * fx)
	var uy := fy * fy * (3.0 - 2.0 * fy)
	return lerpf(lerpf(_vhash(ix, iy), _vhash(ix + 1.0, iy), ux),
			lerpf(_vhash(ix, iy + 1.0), _vhash(ix + 1.0, iy + 1.0), ux), uy)

func _fbm(qx: float, qy: float) -> float:
	var v := 0.0
	var amp := 0.5
	for o in range(4):
		v += _vnoise(qx, qy) * amp
		qx *= 2.03
		qy *= 2.03
		amp *= 0.5
	return v

# terrain_steep multiplies steepness on top of the shipped baseline (1.0 = shipped
# "gentle-plus"): higher packs ridges & valleys closer together = steeper per view.
# At 1.0 the wavelengths below MUST match the shader's block_height exactly, or the
# band mis-places; other values are PREVIEW ONLY (the CPU mirror draws the relief
# overview, VOX_RELIEFSHOT — it does not change the shader gen).
var terrain_steep := 1.0

func _block_height(bcx: float, bcz: float) -> float:
	var wx := bcx * 20.0
	var wz := bcz * 20.0
	var sx := float(seed_value & 0xFFFF) * 0.618
	var sz := float((seed_value >> 16) & 0xFFFF) * 0.618
	var k := terrain_steep
	var cn := _fbm(wx / (21000.0 / k) + sx, wz / (21000.0 / k) + sz) - 0.5
	var chunk := cn * (1.0 + 2.0 * absf(cn))
	var hill := _fbm(wx / (2700.0 / k) + sx * 2.0 + 31.7, wz / (2700.0 / k) + sz * 2.0 + 31.7) - 0.5
	var det := _fbm(wx / (600.0 / k) + sx * 4.0 + 91.3, wz / (600.0 / k) + sz * 4.0 + 91.3) - 0.5
	return float(RELIEF) * (0.5 + chunk * 0.44 + hill * 0.06 + det * 0.02)

## terrain surface world-Y (voxels) at world column (wx, wz) — blended mode
func surface_world_y(wx: float, wz: float) -> float:
	var bcfx := wx / 20.0
	var bcfz := wz / 20.0
	var ix := floorf(bcfx)
	var iz := floorf(bcfz)
	var fx := bcfx - ix
	var fz := bcfz - iz
	var ux := fx * fx * (3.0 - 2.0 * fx)
	var uz := fz * fz * (3.0 - 2.0 * fz)
	return lerpf(lerpf(_block_height(ix, iz), _block_height(ix + 1.0, iz), ux),
			lerpf(_block_height(ix, iz + 1.0), _block_height(ix + 1.0, iz + 1.0), ux), uz)

## world-Y the band floor should sit at so the whole resident footprint's surface
## fits inside the band, with a little subsurface below and air above. Samples the
## surface across the window (centred on wx,wz) so the band covers the local relief
## — not just the centre column — then centres the covered range in the band. This
## is the tracker's target: it keeps this near-surface slice resident as the world
## scrolls through the 256 m-relief world. Clamped so the band stays in [0, RELIEF].
func band_oy_for(wx: float, wz: float) -> int:
	var lo := 1e9
	var hi := -1e9
	var stepx := float(W) / 8.0
	var stepz := float(D) / 8.0
	for iz in range(-4, 5):
		for ix in range(-4, 5):
			var s := surface_world_y(wx + ix * stepx, wz + iz * stepz)
			lo = minf(lo, s)
			hi = maxf(hi, s)
	# centre the [lo, hi] surface band in the buffer, biased slightly downward so a
	# few metres of subsurface (soil / water table) are always resident below it
	var mid := (lo + hi) * 0.5
	var oy := int(round(mid - float(H) * 0.5 + float(H) * 0.08))
	return clampi(oy, 0, maxi(RELIEF - H, 0))

func _rebuild_uniform_set() -> void:
	var bufs := [cells_buf, pack_buf, stats_buf, dirty_buf,
			_solid_target, _water_target, inst_count_buf, active_buf, cells_buf2]
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

const PC_SIZE := 72   # push constant byte size (must match the shader struct)

func _pc(mode: int, offset: int) -> PackedByteArray:
	# 64-byte push constant. rain is an integer threshold out of 2^24; evap/erode
	# are float probabilities; strip_x/z pack (x0 | width<<16) for toroidal edge
	# regen (0 -> full width, i.e. regenerate the whole buffer).
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
	pc.encode_u32(56, gen_strip_x if gen_strip_x != 0 else (W << 16))
	pc.encode_u32(60, gen_strip_z if gen_strip_z != 0 else (D << 16))
	pc.encode_u32(64, gen_oy)          # world-Y voxel of the buffer floor (band)
	pc.encode_u32(68, cells_split)     # first cell index in cells_buf2 (multi-buffer)
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
		rd.compute_list_set_push_constant(cl, _pc(7, 0), PC_SIZE)               # sleep/wake decay
		rd.compute_list_dispatch(cl, _decay_groups, 1, 1)
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
	if mesh_render:
		rd.compute_list_set_push_constant(cl, _pc(8, 0), PC_SIZE)   # greedy-mesh faces (per-column)
		rd.compute_list_dispatch(cl, _col_groups, 1, 1)
	elif block_render:
		rd.compute_list_set_push_constant(cl, _pc(6, 0), PC_SIZE)   # 1m blocks + voxel-tinted tops
		rd.compute_list_dispatch(cl, _col_groups, 1, 1)
	else:
		rd.compute_list_set_push_constant(cl, _pc(3, 0), PC_SIZE)   # per-voxel (per-column walk)
		rd.compute_list_dispatch(cl, _col_groups, 1, 1)
	rd.compute_list_end()
	var counts := rd.buffer_get_data(inst_count_buf, 0, 8).to_int32_array()
	return PackedInt32Array([mini(counts[0], solid_cap), mini(counts[1], water_cap)])

## Pull GPU state back to the CPU: cell bytes (tests/analysis), water stats,
## dirty-chunk flags. Stalls the pipe — fine for tests, not per-frame.
func sync_cells() -> void:
	if not gpu_ok:
		return
	if not _pack_full:
		# grow the stub pack buffer to full size on first use, rebind it
		rd.free_rid(pack_buf)
		var words := (W * D * H + 3) / 4
		pack_buf = rd.storage_buffer_create(words * 4)
		_pack_full = true
		_rebuild_uniform_set()
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline)
	rd.compute_list_bind_uniform_set(cl, uniform_set, 0)
	rd.compute_list_set_push_constant(cl, _pc(2, 0), PC_SIZE)          # pack
	rd.compute_list_dispatch(cl, _pack_groups, 1, 1)
	rd.compute_list_end()
	cell = rd.buffer_get_data(pack_buf).slice(0, W * D * H)
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
	var bytes := ints.to_byte_array()
	rd.buffer_update(cells_buf, 0, mini(n, cells_split) * 4, bytes.slice(0, mini(n, cells_split) * 4))
	if n > cells_split:
		rd.buffer_update(cells_buf2, 0, (n - cells_split) * 4, bytes.slice(cells_split * 4))

func reset_water_stats() -> void:
	super.reset_water_stats()
	if gpu_ok:
		rd.buffer_update(stats_buf, 0, 12, PackedByteArray([0,0,0,0,0,0,0,0,0,0,0,0]))

func free_gpu() -> void:
	if rd == null:
		return
	# free our resources, never the shared main device
	for r in [uniform_set, pipeline, shader, cells_buf, cells_buf2, pack_buf, stats_buf,
			dirty_buf, active_buf, inst_count_buf, _placeholder_a, _placeholder_b]:
		if r.is_valid():
			rd.free_rid(r)
	rd = null
	gpu_ok = false
