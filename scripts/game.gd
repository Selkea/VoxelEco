extends Node3D
## VoxelEco root: owns the VoxWorld sim + VoxView renderer, an orbit camera,
## lighting, time/rain controls, and the headless --sim / --shot modes.

var world: VoxWorld
var view: VoxView
var cam: Camera3D
# real-time clock: at 1x, one simulated second passes per wall second,
# independent of frame rate. A tick is 1/TICK_RATE simulated seconds.
const TICK_RATE := 30.0
const SPEEDS := [1, 2, 4, 8, 16, 32, 64]
const MAX_TICKS_PER_FRAME := 96   # keeps extreme speeds from death-spiraling
var speed_mult := 1               # 0 = paused
var time_acc := 0.0
var mesh_acc := 0.0
var orbit := 0.7            # camera yaw
var pitch := 0.42          # lower angle reads the terrain relief
var dist := 92.0   # scaled to world size in _ready
var dragging := false
var _title_acc := 0.0

func _init() -> void:
	_add_action("pause", [KEY_SPACE])
	for i in range(SPEEDS.size()):
		_add_action("speed_%d" % (i + 1), [KEY_1 + i])
	_add_action("rain_up", [KEY_E])
	_add_action("rain_down", [KEY_Q])
	_add_action("restart", [KEY_R])
	_add_action("cut", [KEY_C])
	_add_action("genmode", [KEY_T])   # toggle blended / terraced worldgen

const CHUNK_VOX := 400   # 20 blocks x 20 voxels = 20 m chunk edge

func _world_size() -> Vector3i:
	# height is a FIXED vertical extent (voxels), decoupled from horizontal span:
	# a streamed world grows sideways in chunks, not upward. VOX_H overrides it.
	var hh := OS.get_environment("VOX_H").to_int()
	# VOX_CHUNKS=n makes an n x n grid of whole 20 m chunks (a fixed multi-chunk
	# world); VOX_SIZE=n makes an arbitrary n-voxel-wide world.
	var nc := OS.get_environment("VOX_CHUNKS").to_int()
	if nc > 0:
		var wv := nc * CHUNK_VOX
		return Vector3i(wv, wv, hh if hh > 0 else 128)
	var sz := OS.get_environment("VOX_SIZE").to_int()
	if sz > 0:
		return Vector3i(sz, sz, hh if hh > 0 else maxi(24, sz * 3 / 8))
	return Vector3i(512, 512, hh if hh > 0 else 192)

func _ready() -> void:
	var ws := _world_size()
	world = GpuWorld.new(12345, ws.x, ws.y, ws.z)
	if not world.gpu_ok:
		world.prime()      # the CPU fallback path needs its active set
	view = VoxView.new()
	view.world = world
	view.use_instances = world.gpu_ok
	add_child(view)
	if view.use_instances:
		world.bind_instance_buffers(view.solid_buffer_rid(), view.water_buffer_rid())
	_refresh_view(true)

	var env := WorldEnvironment.new()
	var e := Environment.new()
	var sky_mat := ProceduralSkyMaterial.new()
	sky_mat.sky_top_color = Color("#5083c2")
	sky_mat.sky_horizon_color = Color("#c3d9ea")
	sky_mat.ground_bottom_color = Color("#41454e")
	sky_mat.ground_horizon_color = Color("#c3d9ea")
	var sky := Sky.new()
	sky.sky_material = sky_mat
	e.background_mode = Environment.BG_SKY
	e.sky = sky
	e.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	e.ambient_light_energy = 0.55
	e.tonemap_mode = Environment.TONE_MAPPER_ACES
	e.tonemap_exposure = 0.92
	e.ssao_enabled = true
	e.ssao_intensity = 2.2
	e.fog_enabled = true
	e.fog_light_color = Color("#c3d9ea")
	e.fog_density = 0.055 / (ws.x * 1.12)   # constant haze regardless of world size
	env.environment = e
	add_child(env)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-48, -38, 0)
	sun.light_energy = 1.15
	sun.light_color = Color("#fff2dd")
	sun.shadow_enabled = true
	sun.shadow_blur = 1.4
	sun.directional_shadow_max_distance = 340.0
	add_child(sun)

	cam = Camera3D.new()
	add_child(cam)
	dist = world.W * 1.12
	_place_cam()

	if OS.get_environment("VOX_GENPROBE") != "":
		world.sync_cells()
		var mid := world.D / 2
		var line := ""
		for x in range(0, world.W, 8):
			var top := 0
			for y in range(world.H):
				var mm: int = world.cell[world.idx(x, y, mid)]
				if mm != VoxWorld.AIR and mm != VoxWorld.WATER:
					top = y
			line += "%3d" % top
		print("terrain tops along z=%d: %s" % [mid, line])
		var cx := world.W / 2
		var mats := {}
		for y in range(world.H):
			var mm2: int = world.cell[world.idx(cx, y, mid)]
			mats[y] = mm2
		var col := ""
		for y in range(world.H):
			col += str(mats[y])
		print("center column materials (y 0->%d): %s" % [world.H - 1, col])
		get_tree().quit()
		return
	if "--sim" in OS.get_cmdline_user_args():
		_run_sim_test()
	if "--shot" in OS.get_cmdline_user_args():
		_take_screenshot()

func _place_cam() -> void:
	var c := Vector3(world.W * 0.5, world.H * 0.22, world.D * 0.5)
	if world.cut_z > 0:
		c = Vector3(world.W * 0.5, 28.0, world.cut_z)   # frame the cross-section wall
	if OS.get_environment("VOX_CUBETEST") != "":
		c = Vector3(32.5, 9.5, 32.5)   # aim dead at the probe cube
	var off := Vector3(cos(orbit) * cos(pitch), sin(pitch), sin(orbit) * cos(pitch)) * dist
	cam.position = c + off
	cam.look_at(c, Vector3.UP)

func _process(dt: float) -> void:
	if Input.is_action_just_pressed("pause"):
		speed_mult = 0 if speed_mult > 0 else 1
	for i in range(SPEEDS.size()):
		if Input.is_action_just_pressed("speed_%d" % (i + 1)):
			speed_mult = SPEEDS[i]
	if Input.is_action_just_pressed("rain_up"):
		world.set_rain_mm_hr(minf(world.rain_mm_hr + 10.0, 300.0))
	if Input.is_action_just_pressed("rain_down"):
		world.set_rain_mm_hr(maxf(world.rain_mm_hr - 10.0, 0.0))
	if Input.is_action_just_pressed("cut"):
		# toggle a cross-section to see subsurface moisture / the water table
		world.cut_z = 0 if world.cut_z > 0 else world.D / 2
		_refresh_view(true)
	if Input.is_action_just_pressed("genmode") and world is GpuWorld:
		# flip blended <-> terraced worldgen and regenerate this world in place
		var gw := world as GpuWorld
		gw.gen_flags ^= 1
		gw.regen(gw.gen_origin_x, gw.gen_origin_z)
		gw.reset_water_stats()
		_refresh_view(true)
	if Input.is_action_just_pressed("restart"):
		world.free_gpu()
		var wsz := _world_size()
		world = GpuWorld.new(Time.get_ticks_msec(), wsz.x, wsz.y, wsz.z)
		if not world.gpu_ok:
			world.prime()
		view.world = world
		view.use_instances = world.gpu_ok
		if view.use_instances:
			world.bind_instance_buffers(view.solid_buffer_rid(), view.water_buffer_rid())
		_refresh_view(true)

	if speed_mult > 0:
		time_acc += dt * speed_mult
		var ticks := int(time_acc * TICK_RATE)
		if ticks > 0:
			time_acc -= ticks / TICK_RATE
			world.run(mini(ticks, MAX_TICKS_PER_FRAME))

	mesh_acc += dt
	if mesh_acc >= 0.07:       # refresh the render ~14 Hz
		mesh_acc = 0.0
		_refresh_view()
	_place_cam()
	_title_acc += dt
	if _title_acc > 0.5:
		_title_acc = 0.0
		var sim_s := int(world.tick_count / TICK_RATE)
		var genmode := "terraced" if (world is GpuWorld and (world as GpuWorld).gen_flags & 1) else "blended"
		DisplayServer.window_set_title("VoxelEco — %s | %s | sim %02d:%02d:%02d | %s | rain %d mm/h | %d fps" % [
			"GPU" if world.gpu_ok else "CPU",
			genmode,
			sim_s / 3600, (sim_s / 60) % 60, sim_s % 60,
			"paused" if speed_mult == 0 else str(speed_mult) + "x",
			int(world.rain_mm_hr),
			int(Engine.get_frames_per_second())])

func _refresh_view(force := false) -> void:
	if view.use_instances:
		# only re-emit when the sim actually changed (or forced on first show /
		# restart). A static world keeps its instance buffer, so it can't flicker.
		if force or world.any_dirty_and_clear():
			var counts: PackedInt32Array = world.dispatch_emit()
			if counts.size() == 2:
				view.set_visible_counts(counts[0], counts[1])
	else:
		world.sync_cells()
		view.rebuild(world.dirty_chunks)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			dragging = event.pressed
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
			dist = maxf(30.0, dist - 6.0)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			dist = minf(220.0, dist + 6.0)
	elif event is InputEventMouseMotion and dragging:
		orbit -= event.relative.x * 0.007
		pitch = clampf(pitch - event.relative.y * 0.007, 0.12, 1.45)

func _add_action(action: String, keys: Array) -> void:
	if InputMap.has_action(action):
		return
	InputMap.add_action(action)
	for k in keys:
		var ev := InputEventKey.new()
		ev.physical_keycode = k
		InputMap.action_add_event(action, ev)

## Debug: run a full rain-then-drain cycle so water sheds off the hills and
## collects as a clean lake, then save a frame to review the 3D scene.
func _take_screenshot() -> void:
	if OS.get_environment("VOX_CUBETEST") != "":
		# winding probe: a single floating cube, no sim
		for i in range(world.cell.size()):
			world.cell[i] = VoxWorld.AIR
		world.cell[world.idx(32, 9, 32)] = VoxWorld.STONE
		world.upload_cells()
		speed_mult = 0
		if OS.get_environment("VOX_ORBIT") != "":
			orbit = float(OS.get_environment("VOX_ORBIT"))
		if OS.get_environment("VOX_PITCH") != "":
			pitch = float(OS.get_environment("VOX_PITCH"))
		dist = 9.0
		_refresh_view(true)
		_place_cam()
		await get_tree().create_timer(0.3).timeout
		get_viewport().get_texture().get_image().save_png("user://shot.png")
		print("SHOT saved: ", ProjectSettings.globalize_path("user://shot.png"))
		get_tree().quit()
		return
	if OS.get_environment("VOX_SHORESHOT") != "":
		# long run (rain off), low camera near the shore — reproduces the
		# "stranded water / eroded bank" the user sees minutes into a session
		world.set_rain_mm_hr(0.0)
		var total := OS.get_environment("VOX_SHORETICKS").to_int()
		if total <= 0: total = 20000
		var stepn := 0
		while stepn < total:
			world.run(mini(500, total - stepn))
			stepn += 500
		pitch = float(OS.get_environment("VOX_PITCH")) if OS.get_environment("VOX_PITCH") != "" else 0.45
		orbit = float(OS.get_environment("VOX_ORBIT")) if OS.get_environment("VOX_ORBIT") != "" else 0.7
		dist = world.W * (float(OS.get_environment("VOX_DIST")) if OS.get_environment("VOX_DIST") != "" else 1.05)
		_refresh_view(true)
		_place_cam()
		await get_tree().create_timer(0.4).timeout
		get_viewport().get_texture().get_image().save_png("user://shot.png")
		print("SHORESHOT saved (%d ticks): " % total, ProjectSettings.globalize_path("user://shot.png"))
		get_tree().quit()
		return
	# the world generates already grassy with lakes in the basins; just let the
	# lakes settle/soak into their beds for a moment, then capture
	world.set_rain_mm_hr(0.0)
	world.run(150)
	_refresh_view(true)
	await get_tree().create_timer(0.4).timeout
	if OS.get_environment("VOX_FLICKERTEST") != "":
		# freeze the sim (still lake) and run the NORMAL refresh path several
		# times, then diff two renders. With the dirty-gate, a still world isn't
		# re-emitted, so the frames should be identical (no shimmer).
		speed_mult = 0
		var a := get_viewport().get_texture().get_image()
		for reps in range(4):
			# FORCE a re-emit (reshuffles instance order) to test the worst case:
			# does the render change just because the buffer order changed?
			var counts: PackedInt32Array = world.dispatch_emit()
			view.set_visible_counts(counts[0], counts[1])
			await get_tree().create_timer(0.15).timeout
		var b := get_viewport().get_texture().get_image()
		var diff := 0
		for y in range(0, a.get_height(), 2):
			for x in range(0, a.get_width(), 2):
				var ca := a.get_pixel(x, y)
				var cb := b.get_pixel(x, y)
				if absf(ca.r - cb.r) + absf(ca.g - cb.g) + absf(ca.b - cb.b) > 0.03:
					diff += 1
		var total := (a.get_width() / 2) * (a.get_height() / 2)
		print("FLICKERTEST: %d of %d sampled pixels changed between re-emits (%.2f%%)" % [
			diff, total, 100.0 * diff / total])
		print("FLICKER FREE" if diff < total / 500 else "FLICKERING (%d px)" % diff)
		get_tree().quit()
		return
	var img := get_viewport().get_texture().get_image()
	img.save_png("user://shot.png")
	print("SHOT saved: ", ProjectSettings.globalize_path("user://shot.png"))
	get_tree().quit()

## Headless self-test: tick the pure sim and assert the water cycle works —
## rain accumulates, water sinks to low ground and pools, mass is conserved
## (added = standing + evaporated), nothing crashes.
func _run_sim_test() -> void:
	var w := GpuWorld.new(12345)
	if not w.gpu_ok:
		w.prime()
	w.rules_mask = OS.get_environment("VOX_RULES").to_int() if OS.get_environment("VOX_RULES") != "" else 0
	print("backend: %s rules_mask=%d" % ["GPU compute" if w.gpu_ok else "CPU fallback", w.rules_mask])
	if OS.get_environment("VOX_GENSTABLE") != "":
		# is freshly generated terrain at rest, or does it landslide? count solid
		# (non-water) cells whose material changes per tick — movement = slumping
		var gw := GpuWorld.new(12345, 160, 160, 60)
		gw.set_rain_mm_hr(0.0)
		if not gw.gpu_ok:
			print("GENSTABLE: no GPU"); get_tree().quit(); return
		var solid := func(m: int) -> bool:
			return m == VoxWorld.SOIL or m == VoxWorld.SAND or m == VoxWorld.STONE \
				or m == VoxWorld.MUD or m == VoxWorld.GRASS
		var names := {VoxWorld.AIR: "air", VoxWorld.SOIL: "soil", VoxWorld.SAND: "sand",
			VoxWorld.STONE: "stone", VoxWorld.MUD: "mud", VoxWorld.GRASS: "grass",
			VoxWorld.WATER: "water", VoxWorld.BEDROCK: "rock"}
		print("GENSTABLE: material transitions per tick (src->dst: count):")
		gw.sync_cells()
		var prev := gw.cell.duplicate()
		for t in range(5):
			gw.run(1)
			gw.sync_cells()
			var by := {}
			for i in range(gw.cell.size()):
				if prev[i] != gw.cell[i]:
					var k := "%s->%s" % [names.get(prev[i], str(prev[i])), names.get(gw.cell[i], str(gw.cell[i]))]
					by[k] = by.get(k, 0) + 1
			var keys := by.keys()
			keys.sort_custom(func(a, b): return by[a] > by[b])
			var parts := PackedStringArray()
			for k in keys:
				parts.append("%s:%d" % [k, by[k]])
			print("  tick %d: %s" % [t + 1, ", ".join(parts)])
			prev = gw.cell.duplicate()
		get_tree().quit()
		return
	if OS.get_environment("VOX_SEAMTEST") != "":
		# streaming seam check: a chunk generated STANDALONE at its world origin
		# must be bit-identical to that same region generated inside a larger
		# world. If so, chunks can be generated independently and still tile
		# seamlessly — the core guarantee the streaming runtime relies on.
		var Hh := 96
		var big := GpuWorld.new(777, 800, 800, Hh)   # 2x2 chunks, generated whole
		var one := GpuWorld.new(777, 400, 400, Hh)    # one chunk, standalone
		one.regen(400, 0)                             # as world-chunk (1,0)
		if not big.gpu_ok or not one.gpu_ok:
			print("SEAMTEST: no GPU"); get_tree().quit(); return
		big.sync_cells(); one.sync_cells()
		var mismatch := 0
		var solid := 0
		for z in range(400):
			for y in range(Hh):
				var brow := (400) + z * 800 + y * 800 * 800
				var orow := z * 400 + y * 400 * 400
				for x in range(400):
					var a: int = one.cell[orow + x]
					var b: int = big.cell[brow + x]
					if a != VoxWorld.AIR: solid += 1
					if a != b: mismatch += 1
		print("SEAMTEST: chunk(1,0) standalone vs in-context: %d of %d cells differ (%d non-air)" \
			% [mismatch, 400 * 400 * Hh, solid])
		print("SEAMLESS" if mismatch == 0 else "SEAM MISMATCH")
		big.free_gpu(); one.free_gpu()
		get_tree().quit()
		return
	if OS.get_environment("VOX_SHORETEST") != "":
		# long run, rain off: does the shoreline stay clean, or does water get
		# stranded above the lake as the bank erodes/shifts? reports the water
		# surface-height spread + perched water + erosion product (sand).
		var gw := GpuWorld.new(12345, 160, 160, 60)
		gw.set_rain_mm_hr(0.0)
		if OS.get_environment("VOX_NOEROSION") != "":
			gw.rules_mask = 0xEF   # all rules except erosion (bit 16)
		if not gw.gpu_ok:
			print("SHORETEST: no GPU"); get_tree().quit(); return
		print("SHORETEST: rain=0 erosion=%s" % ["off" if gw.rules_mask == 0xEF else "on"])
		var checkpoints := [0, 2000, 8000, 20000]
		var done := 0
		for cp in checkpoints:
			gw.run(cp - done)
			done = cp
			gw.sync_cells()
			# per-column highest water cell; histogram those heights
			var tops := {}
			var sand := 0
			var mud := 0
			for z in range(gw.D):
				for x in range(gw.W):
					var hi := -1
					for y in range(gw.H):
						var m: int = gw.cell[x + z * gw.W + y * gw.W * gw.D]
						if m == VoxWorld.WATER: hi = y
						elif m == VoxWorld.SAND: sand += 1
						elif m == VoxWorld.MUD: mud += 1
					if hi >= 0: tops[hi] = tops.get(hi, 0) + 1
			# lake level = most common water-surface height
			var lvl := 0
			var best := 0
			for h in tops:
				if tops[h] > best: best = tops[h]; lvl = h
			var perched := 0    # water-surface columns sitting above the lake line
			for h in tops:
				if h > lvl + 1: perched += tops[h]
			print("  t=%d: lake level y=%d (%d cols), perched cols>lvl+1=%d, sand=%d mud=%d" \
				% [cp, lvl, best, perched, sand, mud])
		get_tree().quit()
		return
	if OS.get_environment("VOX_LAKETEST") != "":
		# generate a real world WITH lakes (GPU-gen needs >400k cells) and check
		# the lakes come to rest instead of forever seeping into their beds
		var lw := GpuWorld.new(12345, 160, 160, 60)
		lw.set_rain_mm_hr(OS.get_environment("VOX_RAIN").to_float())   # default 0 -> calm
		if not lw.gpu_ok:
			print("LAKETEST: no GPU"); get_tree().quit(); return
		lw.run(400)                        # settle
		# measure the VISIBLE surface: topmost water y per column, and how many
		# columns' surface height changes each tick over several consecutive
		# ticks (catches 2-tick offset oscillation that single samples miss)
		var surf := func() -> PackedInt32Array:
			lw.sync_cells()
			var h := PackedInt32Array()
			h.resize(lw.W * lw.D)
			for z in range(lw.D):
				for x in range(lw.W):
					var top := -1
					for y in range(lw.H - 1, -1, -1):
						if lw.cell[lw.idx(x, y, z)] == VoxWorld.WATER:
							top = y
							break
					h[z * lw.W + x] = top
			return h
		print("LAKETEST: columns whose water surface moved, per tick (settled):")
		var prev_h: PackedInt32Array = surf.call()
		for t in range(8):
			lw.run(1)
			var h: PackedInt32Array = surf.call()
			var moved := 0
			var wet := 0
			for i in range(h.size()):
				if h[i] >= 0 or prev_h[i] >= 0:
					wet += 1
				if h[i] != prev_h[i]:
					moved += 1
			print("  tick +%d: %d of %d wet columns changed surface height" % [t + 1, moved, wet])
			prev_h = h
		get_tree().quit()
		return
	if OS.get_environment("VOX_SETTLETEST") != "":
		# drop a tall narrow water column onto a flat floor and watch how fast
		# its surface levels out (max column height should collapse quickly)
		for i in range(w.cell.size()):
			w.cell[i] = VoxWorld.AIR
		for z in range(w.D):
			for x in range(w.W):
				w.cell[w.idx(x, 0, z)] = VoxWorld.STONE   # flat impermeable floor
		var cx := w.W / 2
		var cz := w.D / 2
		for dz in range(-3, 4):
			for dx in range(-3, 4):
				for y in range(1, 33):
					w.cell[w.idx(cx + dx, y, cz + dz)] = VoxWorld.WATER
		w.upload_cells()
		w.set_rain_mm_hr(0.0)
		print("SETTLE: 7x7x32 water column (top at y=32), tallest column vs ticks:")
		var prev := 0
		for cp in [20, 40, 80, 160, 320]:
			w.run(cp - prev)
			prev = cp
			w.sync_cells()
			var tallest := 0
			for z in range(w.D):
				for x in range(w.W):
					for y in range(w.H - 1, 0, -1):
						if w.cell[w.idx(x, y, z)] == VoxWorld.WATER:
							tallest = maxi(tallest, y)
							break
			print("  tick %4d: tallest water column y=%d" % [cp, tallest])
		# is it actually STILL now? count water voxels that move over one tick
		var before := w.cell.duplicate()
		w.run(1)
		w.sync_cells()
		var moved := 0
		for i in range(w.cell.size()):
			if (before[i] == VoxWorld.WATER) != (w.cell[i] == VoxWorld.WATER):
				moved += 1
		print("  residual churn after settling: %d water cells changed in 1 tick" % moved)
		print("SETTLE STILL" if moved < 20 else "SETTLE RESTLESS (%d)" % moved)
		get_tree().quit()
		return
	if OS.get_environment("VOX_DROPTEST") != "":
		# rain must fall STRAIGHT: drop water voxels in one column of open air
		# and confirm they don't drift sideways over several ticks
		for i in range(w.cell.size()):
			w.cell[i] = VoxWorld.AIR
		w.upload_cells()
		w.set_rain_mm_hr(0.0)
		var cx := w.W / 2
		var cz := w.D / 2
		for dy in range(2, 14, 2):
			w.cell[w.idx(cx, w.H - dy, cz)] = VoxWorld.WATER
		w.upload_cells()
		w.run(5)             # a few ticks of falling
		w.sync_cells()
		var max_dev := 0
		var n := 0
		for z in range(w.D):
			for x in range(w.W):
				for y in range(w.H):
					if w.cell[w.idx(x, y, z)] == VoxWorld.WATER:
						n += 1
						max_dev = maxi(max_dev, maxi(absi(x - cx), absi(z - cz)))
		print("DROPTEST: %d water voxels, max horizontal drift from column = %d" % [n, max_dev])
		print("DROP PASS" if max_dev == 0 else "DROP FAIL (rain drifting sideways)")
		get_tree().quit()
		return
	if OS.get_environment("VOX_CALIB") != "":
		# prove the calibration: rain at a known mm/h for a known sim-time must
		# deposit the matching water depth (each voxel = VOXEL_M*1000 mm).
		var R := 40.0                    # mm/hour (heavy rain)
		var hours := 1.0
		w.set_rain_mm_hr(R)
		w.reset_water_stats()
		var ticks := int(hours * 3600.0 * VoxWorld.TICK_RATE)
		w.run(ticks)
		w.sync_cells()
		var voxel_mm := VoxWorld.VOXEL_M * 1000.0
		var implied_mm := float(w.water_added) / float(w.W * w.D) * voxel_mm
		var target_mm := R * hours
		var err := absf(implied_mm - target_mm) / target_mm
		print("CALIB: %.0f mm/h for %.0f h over %dx%d -> %d voxels added" % [
			R, hours, w.W, w.D, w.water_added])
		print("  implied depth %.2f mm vs target %.2f mm (err %.2f%%)" % [
			implied_mm, target_mm, err * 100.0])
		print("  derived rates @ %d ticks/s, %.0f cm voxels:" % [
			int(VoxWorld.TICK_RATE), VoxWorld.VOXEL_M * 100.0])
		print("    rain  %.0f mm/h -> p=%s /column/tick" % [w.rain_mm_hr, str(w.rain_prob)])
		print("    evap  %.0f mm/day -> p=%s /voxel/tick" % [w.evap_mm_day, str(w.evap_prob)])
		print("    erode %.0f cm/hr flow -> p=%s /contact/tick" % [
			VoxWorld.EROSION_CM_PER_HR, str(w.erode_prob)])
		print("CALIB PASS" if err < 0.05 else "CALIB FAIL")
		get_tree().quit()
		return
	if OS.get_environment("VOX_PARTTEST") != "":
		# zero all cells, run one step per offset with the counting write path,
		# then verify every cell was written exactly once per step
		for i in range(w.cell.size()):
			w.cell[i] = 0
		w.upload_cells()
		w.set_rain_mm_hr(0.0)
		w.rules_mask = 32
		w.run(2)               # offsets 1 then 0
		w.sync_cells()
		var census := {}
		for i in range(w.cell.size()):
			census[w.cell[i]] = census.get(w.cell[i], 0) + 1
		print("partition census (want all cells == 2, edges may be 1): %s" % [census])
		get_tree().quit()
		return
	if OS.get_environment("VOX_PREFILL") != "":
		# control experiment: flat water slab, no rain, tick-by-tick counts
		w.set_rain_mm_hr(0.0)
		for z in range(w.D):
			for x in range(w.W):
				if (x + z) % 2 == 0:
					w.cell[w.idx(x, int(OS.get_environment("VOX_PREFILL")), z)] = VoxWorld.WATER
		w.upload_cells()
		w.sync_cells()
		print("prefill: water=%d" % w.water_count())
		var row := int(OS.get_environment("VOX_PREFILL"))
		for t in range(4):
			var before := w.cell.duplicate()
			w.run(1)
			w.sync_cells()
			var census := {}
			for i in range(w.cell.size()):
				census[w.cell[i]] = census.get(w.cell[i], 0) + 1
			print("  after tick %d (offset=%d): water=%d census=%s" % [
				t + 1, w.tick_count & 1, w.water_count(), census])
			# find a block that LOST water and dump its full 2x2x2 state
			var off: int = w.tick_count & 1
			var dumped := 0
			for bz in range(0, w.D / 2 - 1):
				for bx in range(0, w.W / 2 - 1):
					if dumped >= 3:
						break
					var x0: int = bx * 2 + off
					var z0: int = bz * 2 + off
					var wb := 0
					var wa := 0
					var pre := []
					var post := []
					for dy in range(2):
						for dz in range(2):
							for dx in range(2):
								var i := w.idx(x0 + dx, row + dy, z0 + dz)
								pre.append(before[i])
								post.append(w.cell[i])
								if before[i] == VoxWorld.WATER: wb += 1
								if w.cell[i] == VoxWorld.WATER: wa += 1
					if wa < wb:
						dumped += 1
						print("    block(%d,%d,%d) lost water %d->%d  pre=%s post=%s" % [
							x0, row, z0, wb, wa, pre, post])
		get_tree().quit()
		return
	var t0 := Time.get_ticks_msec()
	w.run(60)                  # settle terrain
	var settle_ms := Time.get_ticks_msec() - t0
	w.sync_cells()
	print("settle 60 steps: %d ms" % settle_ms)
	var initial := w.water_count()
	w.reset_water_stats()
	w.rain_prob = 0.04   # stress rain for the pooling test (not a real rate)
	t0 = Time.get_ticks_msec()
	w.run(500)           # long enough to saturate the ground, then pool on top
	var rain_ms := Time.get_ticks_msec() - t0
	w.sync_cells()
	print("rain 500 steps: %d ms (water=%d)" % [rain_ms, w.water_count()])
	if w.gpu_ok:
		# CPU reference timing for the same rain workload
		var c := VoxWorld.new(12345)
		c.prime()
		c.run(60)
		c.rain_prob = 0.04
		t0 = Time.get_ticks_msec()
		c.run(500)
		print("CPU reference, rain 500 steps: %d ms  (GPU speedup: %.1fx)" % [
			Time.get_ticks_msec() - t0,
			float(Time.get_ticks_msec() - t0) / maxf(1.0, rain_ms)])

	# mean height of standing water vs mean height of the terrain surface:
	# if water pools in low ground, its mean height is below the land's.
	var standing := 0
	var min_water_y := w.H
	var pooled := 0        # water resting on solid/water (a real pool, not a droplet mid-fall)
	var mud := 0
	var grass := 0
	var sand := 0
	for z in range(w.D):
		for x in range(w.W):
			for y in range(w.H):
				var mm: int = w.cell[w.idx(x, y, z)]
				if mm == VoxWorld.WATER:
					standing += 1
					min_water_y = mini(min_water_y, y)
					if y == 0 or w.cell[w.idx(x, y - 1, z)] != VoxWorld.AIR:
						pooled += 1
				elif mm == VoxWorld.MUD:
					mud += 1
				elif mm == VoxWorld.GRASS:
					grass += 1
				elif mm == VoxWorld.SAND:
					sand += 1
	print("materials: mud=%d grass=%d sand=%d (sand=erosion product under flow)" % [mud, grass, sand])
	# conservation now has three sinks: standing surface water, water absorbed
	# into the ground, and evaporation. added = standing + absorbed + evaporated.
	var expected: int = initial + w.water_added - w.water_evaporated - w.water_absorbed
	var leak: int = standing - expected
	print("water: standing=%d  pooled=%d  min_y=%d  added=%d absorbed=%d evap=%d leak=%d" % [
		standing, pooled, min_water_y, w.water_added, w.water_absorbed, w.water_evaporated, leak])
	var ok := true
	if standing < 500:
		ok = false; print("FAIL: no surface water at all")
	if w.water_absorbed < 10000:
		ok = false; print("FAIL: infiltration not soaking water into the ground")
	if min_water_y > w.H * 0.4:
		ok = false; print("FAIL: no water reached low ground")
	if pooled < 500:
		ok = false; print("FAIL: water not forming real pools")
	if mud < 100:
		ok = false; print("FAIL: saturated soil not turning to mud")
	if absi(leak) > maxi(50, standing / 50):
		ok = false; print("FAIL: water mass not conserved (leak=%d)" % leak)
	print("SIM PASS" if ok else "SIM FAIL")
	get_tree().quit()
