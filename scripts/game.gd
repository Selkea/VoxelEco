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
var orbit := 0.7            # camera yaw (screenshot/diorama camera only)
var pitch := 0.42          # lower angle reads the terrain relief
var dist := 92.0   # scaled to world size in _ready
var dragging := false
var _title_acc := 0.0
# creative-mode fly camera (interactive play): free-fly, no gravity/collision
var interactive := false
var _freeze_cam := false   # non-interactive shots that set their own camera
var _band_acc := 0.0       # throttle for the vertical-tracking band check
var _lod_cam_last := Vector2(1e12, 1e12)   # camera pos at the last LOD emit
var _emit_dir := Vector2.ZERO              # view direction at the last emit (cone)
var _env: Environment
# shader looks, cycled with V: 0 standard, 1 pixel-art, 2 toon, 3 toon+pixel
const LOOK_NAMES := ["standard", "pixel", "toon", "retro"]
var _look := 0
var _toon_mat: ShaderMaterial
var _pixel_rect: ColorRect
var _sun: DirectionalLight3D
var _ray_size := Vector2i.ZERO   # viewport size at the last ray setup (resize check)
# ray supersampling: rays per axis vs viewport pixels (SSAA — the ray path has
# no MSAA). DEFAULT 1: at 2 the displayed image comes out vertically shifted /
# banded (display-path bug, under investigation) — VOX_RAYSS=2 to experiment.
var _ray_ss := maxf(OS.get_environment("VOX_RAYSS").to_float(), 1.0) \
		if OS.get_environment("VOX_RAYSS") != "" else 1.0
var fly_pos := Vector3()
var fly_yaw := 0.0         # radians, mouse-look
var fly_pitch := -0.5
var fly_speed := 30.0      # metres... voxels / second (scroll to change)
var mouse_captured := false

func _init() -> void:
	_add_action("pause", [KEY_P])
	for i in range(SPEEDS.size()):
		_add_action("speed_%d" % (i + 1), [KEY_1 + i])
	_add_action("rain_up", [KEY_E])
	_add_action("rain_down", [KEY_Q])
	_add_action("restart", [KEY_R])
	_add_action("cut", [KEY_C])
	_add_action("genmode", [KEY_T])   # toggle blended / terraced worldgen
	_add_action("render_toggle", [KEY_B])   # per-voxel <-> 1m blocks (voxel tops)
	_add_action("look_cycle", [KEY_V])      # cycle shader looks: standard/pixel/toon/retro
	_add_action("ray_toggle", [KEY_G])      # raster instances <-> ray-cast renderer
	# creative fly controls (Minecraft-style): WASD move, Space/Shift up/down
	_add_action("fly_fwd", [KEY_W])
	_add_action("fly_back", [KEY_S])
	_add_action("fly_left", [KEY_A])
	_add_action("fly_right", [KEY_D])
	_add_action("fly_up", [KEY_SPACE])
	_add_action("fly_down", [KEY_CTRL])

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
		return _clamp_cells(Vector3i(sz, sz, hh if hh > 0 else maxi(24, sz * 3 / 8)))
	# default map: a 2048x2048 footprint (102 m) with a 756-voxel (37.8 m) resident
	# BAND that vertically tracks the terrain surface (gen_oy). The world's true
	# vertical relief is 256 m (GpuWorld.RELIEF) but only this near-surface band is
	# ever stored — ~3.17B sim cells, SPLIT ACROSS THREE GPU BUFFERS (a single
	# Godot storage buffer caps at 4 GB; cget/cset in the shader route each
	# access). The draw stays 60fps-sized at this width because of DISTANCE LOD:
	# fine 5 cm faces near the camera, coarse 1 m block quads beyond (VOX_LODR).
	# VOX_H overrides the band height, not the relief.
	return Vector3i(2048, 2048, hh if hh > 0 else 756)

# The cells now span up to THREE storage buffers (each 32-bit-size-capped at
# ~4 GB in Godot), so the ceiling is ~3.17B cells (~12.7 GB VRAM). Past that a
# buffer would silently truncate and the world read back as garbage, so clamp
# the height and warn rather than generate a broken map. (uint32 cell indexing
# tops out at 4.29B cells, so a fourth buffer wouldn't be fully addressable.)
func _clamp_cells(ws: Vector3i) -> Vector3i:
	const MAX_CELLS := 3_170_000_000
	var cells := ws.x * ws.y * ws.z
	if cells <= MAX_CELLS:
		return ws
	var h := maxi(24, MAX_CELLS / (ws.x * ws.y))
	push_warning("World %dx%dx%d = %.1fB cells exceeds the ~3.17B three-buffer limit; clamping height to %d." % [ws.x, ws.y, ws.z, cells / 1e9, h])
	return Vector3i(ws.x, ws.y, h)

func _ready() -> void:
	var ws := _world_size()
	world = GpuWorld.new(12345, ws.x, ws.y, ws.z)
	if world is GpuWorld:
		# distance LOD: fine faces within this radius (voxels) of the camera,
		# coarse 1m block quads beyond. Default on for wide windows; VOX_LODR
		# overrides (0 = off). Small/test worlds render all-fine as before.
		var lr := OS.get_environment("VOX_LODR")
		(world as GpuWorld).lod_r = lr.to_int() if lr != "" else (800 if ws.x >= 1536 else 0)
		# far field: heightfield-only vista out to 8 km beyond the sim window
		# (VOX_FAR=0 disables). Needs the LOD camera, so it rides lod_r.
		(world as GpuWorld).far_field = OS.get_environment("VOX_FAR") != "0" \
				and (world as GpuWorld).lod_r > 0
	if not world.gpu_ok:
		world.prime()      # the CPU fallback path needs its active set
	view = VoxView.new()
	view.world = world
	view.use_instances = world.gpu_ok
	add_child(view)
	if view.use_instances:
		world.bind_instance_buffers(view.solid_buffer_rid(), view.water_buffer_rid())
	# far field as a clipmap MESH (default): continuous heightfield rings
	# displaced in the vertex shader by the worldgen noise — smooth connected
	# slopes instead of instanced tiles, and zero emit instances out there.
	# VOX_FARMESH=0 falls back to the old instanced far tiles.
	if view.use_instances and world is GpuWorld and (world as GpuWorld).far_field \
			and OS.get_environment("VOX_FARMESH") != "0":
		(world as GpuWorld).far_tiles = false
		view.build_far_mesh(world as GpuWorld, OS.get_environment("VOX_RENDER") == "")
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
	e.ssao_enabled = OS.get_environment("VOX_SSAO") != "0"
	e.ssao_intensity = 2.2
	e.fog_enabled = false   # no distance fog (the streaming window's far edge is a hard cut)
	env.environment = e
	_env = e
	add_child(env)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-48, -38, 0)
	sun.light_energy = 1.15
	sun.light_color = Color("#fff2dd")
	sun.shadow_enabled = true
	sun.shadow_blur = 1.4
	sun.directional_shadow_max_distance = 340.0
	add_child(sun)
	_sun = sun

	# --- shader looks (V cycles; env vars set the starting look/params) ---
	# Both are built once here and toggled live: VOX_PIXEL=n / VOX_LEVELS=k /
	# VOX_DITHER=0 configure the pixel-art post; VOX_TOON=1 starts cel-shaded;
	# VOX_LOOK=0..3 picks the starting look directly.
	if view.use_instances:
		_toon_mat = ShaderMaterial.new()
		_toon_mat.shader = load("res://shaders/toon_terrain.gdshader")
		_toon_mat.set_shader_parameter("sun_dir", -sun.global_transform.basis.z)
	var pm := ShaderMaterial.new()
	pm.shader = load("res://shaders/pixel_post.gdshader")
	if OS.get_environment("VOX_PIXEL") != "":
		pm.set_shader_parameter("pixel_size", maxi(1, OS.get_environment("VOX_PIXEL").to_int()))
	if OS.get_environment("VOX_LEVELS") != "":
		pm.set_shader_parameter("levels", maxi(2, OS.get_environment("VOX_LEVELS").to_int()))
	pm.set_shader_parameter("dither", OS.get_environment("VOX_DITHER") != "0")
	if OS.get_environment("VOX_DITHERSTR") != "":
		pm.set_shader_parameter("dither_strength", clampf(OS.get_environment("VOX_DITHERSTR").to_float(), 0.0, 1.0))
	var lay := CanvasLayer.new()
	lay.layer = 10   # above the ray-cast overlay so the pixel post covers it too
	_pixel_rect = ColorRect.new()
	_pixel_rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	_pixel_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_pixel_rect.material = pm
	_pixel_rect.visible = false
	lay.add_child(_pixel_rect)
	add_child(lay)
	if OS.get_environment("VOX_LOOK") != "":
		_look = clampi(OS.get_environment("VOX_LOOK").to_int(), 0, 3)
	else:
		var px := OS.get_environment("VOX_PIXEL") != ""
		var tn := OS.get_environment("VOX_TOON") != ""
		_look = 3 if px and tn else (1 if px else (2 if tn else 0))
	_apply_look()

	cam = Camera3D.new()
	# far plane covers the whole far-field vista (8 km rings + square corners)
	var far_on := world is GpuWorld and (world as GpuWorld).far_field
	cam.far = 400_000.0 if far_on else maxf(2000.0, world.W * 1.6)
	# a 0.05 near plane against a 2000 far plane wastes almost all depth precision
	# up close, so coincident voxel-cube faces z-fight into thin seams. 1 unit = 1
	# voxel (5 cm); a 0.3-voxel near plane (1.5 cm) is closer than you fly yet lifts
	# the far:near ratio ~7x, and 4x MSAA smooths the high-contrast cube edges.
	cam.near = float(OS.get_environment("VOX_NEAR")) if OS.get_environment("VOX_NEAR") != "" else 0.3
	add_child(cam)
	var msaa := OS.get_environment("VOX_MSAA")   # 0/2/4/8; default 4x
	get_viewport().msaa_3d = ([Viewport.MSAA_DISABLED, Viewport.MSAA_2X, Viewport.MSAA_4X, Viewport.MSAA_8X][{"0":0,"2":1,"4":2,"8":3}.get(msaa, 2)] as Viewport.MSAA)
	dist = world.W * 1.12
	interactive = not ("--sim" in OS.get_cmdline_user_args() or "--shot" in OS.get_cmdline_user_args())
	if interactive:
		# Start the streaming window at a large positive world origin (chunk-
		# aligned) so the toroidal coordinate math — which is unsigned — stays
		# valid whichever way you fly, with kilometres of room in every direction.
		var base := 100000
		var surf := float(world.H) * 0.45   # fallback (CPU world): mid-band
		if world is GpuWorld:
			var gw := world as GpuWorld
			# place the resident band under the terrain surface at the window centre,
			# then generate: the band rides the 256 m-relief world as a thin slice
			var cx := base + world.W * 0.5
			var cz := base + world.D * 0.5
			surf = gw.surface_world_y(cx, cz)
			gw.gen_oy = gw.band_oy_for(cx, cz)
			gw.regen(base, base)
			view.set_stream_origin(base, base)
		# camera hovers above the surface (world-Y), looking down over the vista
		fly_pos = Vector3(base + world.W * 0.5, surf + world.H * 0.4, base + world.D * 0.5)
		fly_yaw = 0.0
		fly_pitch = -0.6
		_capture_mouse(true)
		_place_fly()
	else:
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
	# the ray-cast renderer is the interactive default (G toggles back to the
	# instanced raster; VOX_RENDER=mesh/block/voxel starts on the raster path).
	# Shots/tests keep the raster unless they opt in with VOX_RENDER=ray.
	var rmode_env := OS.get_environment("VOX_RENDER")
	if rmode_env == "ray":
		_set_ray(true)
	elif rmode_env == "" and world is GpuWorld and view.use_instances:
		# DEFAULT: unified world mesh. The sim stays voxels; the render is one
		# displaced surface sampling the sim surface textures (see vox_view).
		(world as GpuWorld).world_mesh = true
		(world as GpuWorld).update_heights()
		_refresh_view(true)
	if "--sim" in OS.get_cmdline_user_args():
		_run_sim_test()
	if "--shot" in OS.get_cmdline_user_args():
		_take_screenshot()

func _capture_mouse(on: bool) -> void:
	mouse_captured = on
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if on else Input.MOUSE_MODE_VISIBLE

func _chunk_snap(v: float) -> int:
	return int(floor(v / CHUNK_VOX)) * CHUNK_VOX

## chunk streaming: the sim buffer is a window that follows the camera. When the
## camera drifts near an edge, recenter the window on it and regenerate at the
## new world origin. Because worldgen is deterministic per world-coordinate, the
## overlapping terrain is identical, so the world scrolls seamlessly and extends
## forever. (The window's live sim state resets on recenter — a full-window
## regen; state-preserving toroidal streaming is a later refinement.)
func _stream() -> void:
	if not (world is GpuWorld) or not view.use_instances:
		return
	var gw := world as GpuWorld
	var margin := world.W * 0.16
	var ox := gw.gen_origin_x
	var oz := gw.gen_origin_z
	var nox := ox
	var noz := oz
	if fly_pos.x - ox < margin or fly_pos.x - ox > world.W - margin:
		nox = _chunk_snap(fly_pos.x - world.W * 0.5)
	if fly_pos.z - oz < margin or fly_pos.z - oz > world.D - margin:
		noz = _chunk_snap(fly_pos.z - world.D * 0.5)
	if nox == ox and noz == oz:
		return
	var dx := absi(nox - ox)
	var dz := absi(noz - oz)
	# check the vertical band BEFORE cutting strips: strips generate at the
	# CURRENT gen_oy, and if the entered terrain has drifted past the band's
	# deadband they'd come out clipped or empty (the "missing chunk" bug). A
	# band move needs a full-window regen anyway, so do that instead.
	var noy := gw.band_oy_for(nox + world.W * 0.5, noz + world.D * 0.5)
	if absi(noy - gw.gen_oy) >= int(world.H * 0.12):
		gw.gen_oy = noy
		gw.regen(nox, noz)
		gw.reset_water_stats()
	elif dx >= world.W or dz >= world.D:
		gw.regen(nox, noz)          # teleport: whole window is new
		gw.reset_water_stats()
	else:
		# TOROIDAL: shift the origin and regenerate ONLY the freshly-entered edge
		# strips, keeping the rest of the window's water/erosion state intact.
		gw.set_origin(nox, noz)
		if nox != ox:
			gw.regen_strip(posmod(mini(ox, nox), world.W), dx, 0, world.D)
		if noz != oz:
			gw.regen_strip(0, world.W, posmod(mini(oz, noz), world.D), dz)
	view.set_stream_origin(nox, noz)
	_refresh_view(true)

## vertical-tracking band: keep the thin resident band centred on the terrain
## surface under the camera as it flies through the 256 m-relief world. When the
## surface drifts past a deadband, shift the band to re-centre and regenerate. The
## band is a vertical slice (not a torus), so a shift is a full-window regen — but
## it only fires when you change ELEVATION; flying at a roughly constant height
## keeps the toroidal horizontal streaming's live water/erosion intact. Throttled
## (band_oy_for samples the whole footprint, which is a bit of CPU noise work).
func _track_band() -> void:
	if not (world is GpuWorld) or not view.use_instances:
		return
	var gw := world as GpuWorld
	var target := gw.band_oy_for(fly_pos.x, fly_pos.z)
	if absi(target - gw.gen_oy) < int(world.H * 0.12):
		return
	gw.gen_oy = target
	gw.regen(gw.gen_origin_x, gw.gen_origin_z)   # re-fill the window at the new band Y
	gw.reset_water_stats()
	_refresh_view(true)

## FLOATING ORIGIN: the emit draws the world in a local frame relative to the
## window origin (world_coord - gen_origin), so the camera is offset by the same
## origin. This keeps rendered coordinates small (~0..W) even when the sim runs at
## world coords ~1e5, where float32 would otherwise crack seams between the voxels.
func _render_off() -> Vector3:
	if world is GpuWorld:
		return Vector3(world.gen_origin_x, 0.0, world.gen_origin_z)
	return Vector3.ZERO

func _place_fly() -> void:
	cam.position = fly_pos - _render_off()
	cam.basis = Basis(Vector3.UP, fly_yaw) * Basis(Vector3.RIGHT, fly_pitch)

## creative-mode free flight: WASD in the look direction, Space/Shift straight
## up/down, Ctrl to sprint, scroll to change speed. No gravity, no collision.
func _fly(dt: float) -> void:
	if mouse_captured:
		var basis := Basis(Vector3.UP, fly_yaw) * Basis(Vector3.RIGHT, fly_pitch)
		var move := Vector3.ZERO
		if Input.is_action_pressed("fly_fwd"): move -= basis.z
		if Input.is_action_pressed("fly_back"): move += basis.z
		if Input.is_action_pressed("fly_right"): move += basis.x
		if Input.is_action_pressed("fly_left"): move -= basis.x
		if Input.is_action_pressed("fly_up"): move += Vector3.UP
		if Input.is_action_pressed("fly_down"): move -= Vector3.UP
		if move.length() > 0.001:
			var speed := fly_speed * (4.0 if Input.is_key_pressed(KEY_SHIFT) else 1.0)
			fly_pos += move.normalized() * speed * dt
	_place_fly()

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
	if Input.is_action_just_pressed("render_toggle") and world is GpuWorld \
			and (world as GpuWorld).can_toggle_render:
		# flip between full per-voxel and 1m-block (voxel-tinted tops) rendering
		var gw := world as GpuWorld
		gw.block_render = not gw.block_render
		_refresh_view(true)
	if Input.is_action_just_pressed("look_cycle"):
		_look = (_look + 1) % 4
		_apply_look()
	if Input.is_action_just_pressed("ray_toggle") and world is GpuWorld:
		_set_ray(not (world as GpuWorld).ray_render)
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
	if interactive:
		_fly(dt)
		_stream()
		_band_acc += dt
		if _band_acc > 0.2:
			_band_acc = 0.0
			_track_band()
		_place_fly()   # re-sync camera to the (possibly shifted) render origin
	elif not _freeze_cam:
		_place_cam()
	# far clipmap mesh rides the camera (per-ring grid snap + world offset sync)
	if cam != null and world is GpuWorld:
		var gwf := world as GpuWorld
		var vph := get_viewport().get_visible_rect().size.y
		view.update_far_mesh(Vector2(cam.position.x, cam.position.z),
				Vector2(gwf.gen_origin_x, gwf.gen_origin_z), (gwf.gen_flags & 1) != 0,
				2.0 * tan(deg_to_rad(cam.fov * 0.5)) / maxf(vph, 1.0),
				float(gwf.gen_oy))
	# ray-cast renderer: march the frame's rays from the final camera pose
	if world is GpuWorld and (world as GpuWorld).ray_render and cam != null:
		var gwr := world as GpuWorld
		var vps := get_viewport().get_visible_rect().size
		if Vector2i(int(vps.x), int(vps.y)) != _ray_size:
			gwr.setup_raycast(int(vps.x * _ray_ss), int(vps.y * _ray_ss))
			_ray_size = Vector2i(int(vps.x), int(vps.y))
			view.set_ray_mode(true, gwr.ray_tex)
		var b := cam.global_transform.basis
		gwr.raycast(cam.position, -b.z, b.x, b.y,
				-_sun.global_transform.basis.z,
				tan(deg_to_rad(cam.fov * 0.5)), vps.x / vps.y)
	_title_acc += dt
	if _title_acc > 0.5:
		_title_acc = 0.0
		var sim_s := int(world.tick_count / TICK_RATE)
		var genmode := "terraced" if (world is GpuWorld and (world as GpuWorld).gen_flags & 1) else "blended"
		var rmode := "blocks" if (world is GpuWorld and (world as GpuWorld).block_render) else "voxels"
		DisplayServer.window_set_title("VoxelEco — %s | %s | %s | %s | sim %02d:%02d:%02d | %s | rain %d mm/h | %d fps" % [
			"GPU" if world.gpu_ok else "CPU",
			rmode, genmode, LOOK_NAMES[_look],
			sim_s / 3600, (sim_s / 60) % 60, sim_s % 60,
			"paused" if speed_mult == 0 else str(speed_mult) + "x",
			int(world.rain_mm_hr),
			int(Engine.get_frames_per_second())])

## toggle the ray-cast renderer (G): rays draw the window terrain into an image
## overlay; the instanced emit switches to far-field-only behind it
func _set_ray(on: bool) -> void:
	if not (world is GpuWorld) or not view.use_instances:
		return
	var gw := world as GpuWorld
	gw.ray_render = on
	if on:
		var vps := get_viewport().get_visible_rect().size
		gw.setup_raycast(int(vps.x * _ray_ss), int(vps.y * _ray_ss))
		_ray_size = Vector2i(int(vps.x), int(vps.y))
		gw.update_heights()
	view.set_ray_mode(on, gw.ray_tex)
	_refresh_view(true)   # re-emit: far-only in ray mode, full otherwise

## apply the current shader look (V cycles): pixel-art post overlay on/off and
## the terrain material swapped between standard PBR and the cel/toon shader
func _apply_look() -> void:
	if _pixel_rect != null:
		_pixel_rect.visible = _look == 1 or _look == 3
	if _toon_mat != null and view.use_instances:
		view.solid_mm.material_override = _toon_mat if (_look == 2 or _look == 3) else view.solid_mat

func _refresh_view(force := false) -> void:
	if view.use_instances:
		# distance LOD follows the camera: aim the near disc at the camera's local-
		# frame position, and re-emit when the camera has moved far enough that the
		# LOD boundary would visibly lag (even if the sim itself is static).
		if world is GpuWorld and (world as GpuWorld).lod_r > 0:
			var gw := world as GpuWorld
			var lc: Vector3
			if interactive:
				lc = fly_pos
			elif cam != null:
				lc = cam.position + _render_off()
			else:   # first refresh in _ready, before the camera exists
				lc = Vector3(gw.gen_origin_x + world.W * 0.5, 0.0, gw.gen_origin_z + world.D * 0.5)
			gw.lod_cx = maxi(0, int(lc.x) - gw.gen_origin_x)
			gw.lod_cz = maxi(0, int(lc.z) - gw.gen_origin_z)
			if Vector2(lc.x, lc.z).distance_to(_lod_cam_last) > 40.0:   # 2 m
				force = true
			# camera-cone emit culling: only geometry near the view cone is
			# emitted; turning past the margin forces a re-emit. Steep pitches
			# widen the projected cone, so the margin grows and near-vertical
			# views disable it entirely.
			# (non-interactive owners — tests, shots — manage gw.cone_* themselves)
			if interactive and cam != null:
				var fwd := -cam.global_transform.basis.z
				var fxz := Vector2(fwd.x, fwd.z)
				var flen := fxz.length()
				if flen < 0.35 or OS.get_environment("VOX_CONE") == "0":
					gw.cone_cos = -2.0                    # near-vertical: emit all
				else:
					fxz /= flen
					gw.cone_dir = fxz
					var aspect := get_viewport().get_visible_rect().size.aspect()
					var hfov := atan(tan(deg_to_rad(cam.fov * 0.5)) * aspect)
					gw.cone_cos = cos(minf(hfov + 0.7 + (1.0 - flen), PI))
					if fxz.dot(_emit_dir) < cos(0.1):     # turned ~6 deg since last emit
						force = true
			if force or world.any_dirty_and_clear():
				_lod_cam_last = Vector2(lc.x, lc.z)
				_emit_dir = gw.cone_dir
				if gw.ray_render or gw.world_mesh:
					gw.update_heights()   # rays read the heightfield; refresh with the sim
				var counts: PackedInt32Array = world.dispatch_emit()
				if counts.size() == 2:
					view.set_visible_counts(counts[0], counts[1])
			return
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
	if not interactive:
		return
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT and not mouse_captured:
			_capture_mouse(true)                                  # click to re-grab
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
			fly_speed = minf(fly_speed * 1.25, 4000.0)            # faster (up to 200 m/s)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			fly_speed = maxf(fly_speed / 1.25, 2.0)               # slower
	elif event is InputEventMouseMotion and mouse_captured:
		fly_yaw -= event.relative.x * 0.005
		fly_pitch = clampf(fly_pitch - event.relative.y * 0.005, -1.55, 1.55)
	elif event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		_capture_mouse(not mouse_captured)                        # Esc frees/grabs cursor

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
	if OS.get_environment("VOX_FPSTEST") != "" and world is GpuWorld:
		# measure the DRAW (rasterization) side: real render of the band from a
		# representative fly view, reporting GPU render time/frame, primitives and
		# instances. Complements VOX_PERF (which measures the emit/physics compute).
		var gw := world as GpuWorld
		var base := 100000
		var cx := base + world.W * 0.5
		var cz := base + world.D * 0.5
		var surf := gw.surface_world_y(cx, cz)
		gw.gen_oy = gw.band_oy_for(cx, cz)
		gw.regen(base, base)
		view.set_stream_origin(base, base)
		_freeze_cam = true
		speed_mult = 0
		world.set_rain_mm_hr(0.0)
		world.run(40)
		var ro := _render_off()
		var vy := OS.get_environment("VOX_CAMY").to_float()
		if vy > 0.0:
			# vista check: rise high and look level at the horizon (far field)
			cam.position = Vector3(cx, surf + vy, cz - 20.0) - ro
			cam.look_at(Vector3(cx, surf + vy * 0.6, cz + 8000.0) - ro, Vector3.UP)
		else:
			cam.position = Vector3(cx, surf + 30.0, cz - 20.0) - ro   # representative fly view
			cam.look_at(Vector3(cx, surf - 139.0, cz + 227.0) - ro, Vector3.UP)   # -34 deg (default pitch)
		var vprid := get_viewport().get_viewport_rid()
		RenderingServer.viewport_set_measure_render_time(vprid, true)
		gw.lod_cx = maxi(0, int(cx) - gw.gen_origin_x)     # LOD near disc at the camera
		gw.lod_cz = maxi(0, int(cz - 20.0) - gw.gen_origin_z)
		if OS.get_environment("VOX_CONE") != "0":
			# camera-cone emit culling, same margin the interactive path uses
			var fw := -cam.global_transform.basis.z
			var fx := Vector2(fw.x, fw.z)
			if fx.length() >= 0.35:
				gw.cone_dir = fx.normalized()
				var asp := get_viewport().get_visible_rect().size.aspect()
				gw.cone_cos = cos(minf(atan(tan(deg_to_rad(cam.fov * 0.5)) * asp) \
						+ 0.7 + (1.0 - fx.length()), PI))
		if gw.ray_render:
			gw.update_heights()                            # heights for the new origin
		var cnt: PackedInt32Array = gw.dispatch_emit()
		view.set_visible_counts(cnt[0], cnt[1])
		if gw.ray_render:
			# time the ray pass itself (the per-frame cost of the ray renderer)
			var bb := cam.global_transform.basis
			var vps2 := get_viewport().get_visible_rect().size
			var rsync := func() -> void: gw.rd.buffer_get_data(gw.stats_buf)
			gw.raycast(cam.position, -bb.z, bb.x, bb.y, -_sun.global_transform.basis.z,
					tan(deg_to_rad(cam.fov * 0.5)), vps2.x / vps2.y)
			rsync.call()
			var rt0 := Time.get_ticks_usec()
			for i in range(30):
				gw.raycast(cam.position, -bb.z, bb.x, bb.y, -_sun.global_transform.basis.z,
						tan(deg_to_rad(cam.fov * 0.5)), vps2.x / vps2.y)
			rsync.call()
			print("FPSTEST ray pass: %.2f ms/frame at %dx%d" % [
				(Time.get_ticks_usec() - rt0) / 30.0 / 1000.0, int(vps2.x), int(vps2.y)])
		for i in range(15):
			await get_tree().process_frame                       # warm up
		var gpu := 0.0
		var cpu := 0.0
		var N := 60
		for i in range(N):
			await get_tree().process_frame
			gpu += RenderingServer.viewport_get_measured_render_time_gpu(vprid)
			cpu += RenderingServer.viewport_get_measured_render_time_cpu(vprid)
		var prims := RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME)
		print("FPSTEST: draw GPU %.2f ms/frame | CPU %.2f ms | %d instances | %.1fM primitives" \
			% [gpu / N, cpu / N, cnt[0] + cnt[1], prims / 1e6])
		get_viewport().get_texture().get_image().save_png("user://shot.png")
		get_tree().quit()
		return
	if OS.get_environment("VOX_RELIEFSHOT") != "" and world is GpuWorld:
		# terrain-shape PREVIEW: a single band view only shows a ~51 m slice, so to
		# judge how mountainous the world is, draw a wide oblique overview of the true
		# surface (surface_world_y) over ~2.5 km at true 1:1 scale. VOX_STEEP scales
		# how tightly ridges/valleys are packed (1 = shipped/gentle).
		var gw := world as GpuWorld
		gw.terrain_steep = float(OS.get_environment("VOX_STEEP")) if OS.get_environment("VOX_STEEP") != "" else 1.0
		_freeze_cam = true
		speed_mult = 0
		view.visible = false                               # hide the band's multimesh
		_env.fog_enabled = false                           # fog washes out km-scale distance
		var exag := 2.0                                     # vertical exaggeration (shape legibility)
		var N := 180
		var span := 50000.0                                # 2.5 km at 5 cm
		var base := 100000.0
		var cellw := span / N
		var sea := float(gw.SEA_Y)
		var relief := float(gw.RELIEF)
		var col := func(h: float) -> Color:
			if h <= sea + 1.0: return Color(0.16, 0.34, 0.58)          # water
			var t: float = clampf((h - sea) / (relief - sea), 0.0, 1.0)
			if t < 0.32: return Color(0.28, 0.5, 0.17).lerp(Color(0.46, 0.42, 0.24), t / 0.32)
			elif t < 0.66: return Color(0.46, 0.42, 0.24).lerp(Color(0.55, 0.55, 0.58), (t - 0.32) / 0.34)
			return Color(0.55, 0.55, 0.58).lerp(Color(0.96, 0.97, 1.0), (t - 0.66) / 0.34)
		var hs := PackedFloat32Array()
		hs.resize((N + 1) * (N + 1))
		for j in range(N + 1):
			for i in range(N + 1):
				hs[j * (N + 1) + i] = gw.surface_world_y(base + i * cellw, base + j * cellw)
		var st := SurfaceTool.new()
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		var addv := func(i: int, j: int) -> void:
			var h: float = hs[j * (N + 1) + i]
			st.set_color(col.call(h))
			st.add_vertex(Vector3(i * cellw, maxf(h, sea) * exag, j * cellw))
		for j in range(N):
			for i in range(N):
				addv.call(i, j); addv.call(i + 1, j); addv.call(i + 1, j + 1)
				addv.call(i, j); addv.call(i + 1, j + 1); addv.call(i, j + 1)
		st.generate_normals()
		var mi := MeshInstance3D.new()
		mi.mesh = st.commit()
		var mat := StandardMaterial3D.new()
		mat.vertex_color_use_as_albedo = true
		mat.roughness = 0.95
		mi.material_override = mat
		add_child(mi)
		var hmax := 0.0
		for h in hs: hmax = maxf(hmax, h)
		var ctr := Vector3(span * 0.5, (sea + (hmax - sea) * 0.35) * exag, span * 0.5)
		cam.position = Vector3(span * 0.5, hmax * exag + span * 0.16, -span * 0.42)
		cam.look_at(ctr, Vector3.UP)
		cam.far = span * 3.0
		await get_tree().create_timer(0.5).timeout
		get_viewport().get_texture().get_image().save_png("user://shot.png")
		print("RELIEFSHOT steep=%.1f: max height %.0f m over %.1f km (vertical exag %.1fx)" % [gw.terrain_steep, hmax * 0.05, span * 0.05 / 1000.0, exag])
		get_tree().quit()
		return
	if OS.get_environment("VOX_BANDSHOT") != "" and world is GpuWorld:
		# vertical-tracking band: place the thin resident band under the terrain
		# surface at the window centre (as the interactive view does), generate the
		# 256 m-relief world, and shoot an overview. Also sample the surface across a
		# wide area to confirm the full relief exists (a single 51 m window only sees
		# a gentle slice of it — the point of wide features).
		var gw := world as GpuWorld
		var base := OS.get_environment("VOX_BASE").to_int() if OS.get_environment("VOX_BASE") != "" else 100000
		var cx := base + world.W * 0.5
		var cz := base + world.D * 0.5
		var surf := gw.surface_world_y(cx, cz)
		gw.gen_oy = gw.band_oy_for(cx, cz)
		gw.regen(base, base)
		view.set_stream_origin(base, base)
		_freeze_cam = true
		world.set_rain_mm_hr(0.0)
		world.run(90)
		gw.lod_cx = maxi(0, int(cx) - gw.gen_origin_x)   # LOD near disc at window centre
		gw.lod_cz = maxi(0, int(cz) - gw.gen_origin_z)
		var cnt: PackedInt32Array = gw.dispatch_emit()
		view.set_visible_counts(cnt[0], cnt[1])
		var lo := 1e9
		var hi := -1e9
		for gz in range(0, 40000, 2000):
			for gx in range(0, 40000, 2000):
				var sy := gw.surface_world_y(base + gx, base + gz)
				lo = minf(lo, sy); hi = maxf(hi, sy)
		var wlo := 1e9
		var whi := -1e9
		for wz2 in range(0, world.D, 32):
			for wx2 in range(0, world.W, 32):
				var ws := gw.surface_world_y(base + wx2, base + wz2)
				wlo = minf(wlo, ws); whi = maxf(whi, ws)
		print("BANDSHOT: window surface relief = %.0f vox (%.1f m); band height = %d vox (%.1f m)" \
			% [whi - wlo, (whi - wlo) * 0.05, world.H, world.H * 0.05])
		print("BANDSHOT: centre surf=%.0f (%.1fm), band oy=%d..%d" % [surf, surf * 0.05, gw.gen_oy, gw.gen_oy + world.H])
		print("BANDSHOT: relief over 2km sample = %.0f..%.0f vox (%.1f..%.1f m) = %.1f m span" \
			% [lo, hi, lo * 0.05, hi * 0.05, (hi - lo) * 0.05])
		print("BANDSHOT: emit solid=%d water=%d instances" % [cnt[0], cnt[1]])
		var ro := _render_off()                            # floating origin: draw near 0
		if OS.get_environment("VOX_CLOSE") != "":
			# close low-angle view to inspect seams between individual voxels
			cam.position = Vector3(cx, surf + 10.0, cz - 70.0) - ro
			cam.look_at(Vector3(cx, surf - 6.0, cz) - ro, Vector3.UP)
		else:
			cam.position = Vector3(cx - world.W * 0.35, surf + world.H * 0.85, cz - world.D * 0.35) - ro
			cam.look_at(Vector3(cx, surf, cz) - ro, Vector3.UP)
		await get_tree().create_timer(0.4).timeout
		get_viewport().get_texture().get_image().save_png("user://shot.png")
		print("BANDSHOT saved: ", ProjectSettings.globalize_path("user://shot.png"))
		get_tree().quit()
		return
	if OS.get_environment("VOX_STREAMSHOT") != "" and world is GpuWorld:
		# streaming check: regenerate the window at a non-zero world origin and
		# aim the camera there. If the terrain renders at that world position (not
		# back at origin 0), the emit's world-offset — the core of streaming — works.
		var gw := world as GpuWorld
		var soff := 1200
		gw.regen(soff, soff)
		view.set_stream_origin(soff, soff)
		world.set_rain_mm_hr(0.0)
		world.run(60)
		var scnt: PackedInt32Array = gw.dispatch_emit()
		view.set_visible_counts(scnt[0], scnt[1])
		print("STREAM emit at origin %d: solid=%d water=%d instances" % [soff, scnt[0], scnt[1]])
		var wc := Vector3(soff + world.W * 0.5, world.H * 0.3, soff + world.D * 0.5)
		cam.position = Vector3(wc.x, world.W * 0.95, wc.z + 1.0)   # near top-down
		cam.look_at(wc, Vector3.FORWARD)
		await get_tree().create_timer(0.4).timeout
		get_viewport().get_texture().get_image().save_png("user://shot.png")
		print("STREAMSHOT saved (world origin %d,%d): " % [soff, soff], ProjectSettings.globalize_path("user://shot.png"))
		get_tree().quit()
		return
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
	if OS.get_environment("VOX_PERF") != "":
		# profile the default-size world: physics step (settled vs raining) and the
		# emit pass (per-voxel vs block shell). GPU compute on the main device is
		# async; a tiny buffer read flushes + syncs the queue, so timing run()/emit
		# followed by that read captures the GPU time.
		var ws := _world_size()
		# free the member world's VRAM first — at 3.17B cells (12.7 GB) two resident
		# worlds oversubscribe a 24 GB card and the probe measures paging, not compute.
		# Frees are deferred a few frames, so idle until the memory actually returns
		# (speed_mult 0 keeps _process from stepping the freed world meanwhile).
		speed_mult = 0
		(world as GpuWorld).free_gpu()
		for i in range(8):
			await get_tree().process_frame
		var gw := GpuWorld.new(999, ws.x, ws.y, ws.z)
		if not gw.gpu_ok:
			print("PERF: no GPU"); get_tree().quit(); return
		gw.regen_tracked(100000, 100000)
		# match the interactive LOD config so the emit numbers are the real ones
		var lre := OS.get_environment("VOX_LODR")
		gw.lod_r = lre.to_int() if lre != "" else (800 if ws.x >= 1536 else 0)
		gw.lod_cx = ws.x / 2
		gw.lod_cz = ws.y / 2
		var cells := ws.x * ws.y * ws.z
		# dummy instance buffers (64 B/instance: 12-float xform + 4-float colour) so
		# the emit can write; bind them as the emit targets
		var sbuf := gw.rd.storage_buffer_create(gw.solid_cap * 64)
		var wbuf := gw.rd.storage_buffer_create(gw.water_cap * 64)
		gw.bind_instance_buffers(sbuf, wbuf)
		var sync := func() -> void: gw.rd.buffer_get_data(gw.stats_buf)
		gw.run(80); sync.call()                                  # warm up: let it settle & sleep
		var t0 := Time.get_ticks_usec()
		gw.run(60); sync.call()
		var settled_us := (Time.get_ticks_usec() - t0) / 60.0
		gw.set_rain_mm_hr(60.0); gw.run(120); sync.call()        # build live water/erosion
		t0 = Time.get_ticks_usec()
		gw.run(60); sync.call()
		var active_us := (Time.get_ticks_usec() - t0) / 60.0
		gw.block_render = false
		t0 = Time.get_ticks_usec()
		for i in range(20): gw.dispatch_emit()
		var vox_us := (Time.get_ticks_usec() - t0) / 20.0
		var vcnt: PackedInt32Array = gw.dispatch_emit()
		gw.block_render = true
		t0 = Time.get_ticks_usec()
		for i in range(20): gw.dispatch_emit()
		var blk_us := (Time.get_ticks_usec() - t0) / 20.0
		var bcnt: PackedInt32Array = gw.dispatch_emit()
		print("PERF world %dx%dx%d = %.0fM cells" % [ws.x, ws.y, ws.z, cells / 1e6])
		print("PERF physics/tick: settled %.2f ms | raining %.2f ms  (30 ticks = 1 s sim)" % [settled_us / 1000.0, active_us / 1000.0])
		print("PERF emit: per-voxel %.2f ms (%d inst) | block shell %.2f ms (%d inst)" % [vox_us / 1000.0, vcnt[0] + vcnt[1], blk_us / 1000.0, bcnt[0] + bcnt[1]])
		print("PERF budget: at 60 fps a frame is 16.7 ms; at 1x sim that's ~2 physics ticks + 1 emit")
		gw.free_gpu()
		get_tree().quit()
		return
	if OS.get_environment("VOX_HOLETEST") != "":
		# holes = columns whose surface fell outside the resident band (they render
		# as missing chunks). Walk the world in window-sized steps, place the band
		# exactly as the game does, and count empty columns at each stop.
		speed_mult = 0
		(world as GpuWorld).free_gpu()
		for i in range(8):
			await get_tree().process_frame
		var ws := _world_size()
		var gw := GpuWorld.new(12345, ws.x, ws.y, ws.z)
		if not gw.gpu_ok:
			print("HOLETEST: no GPU"); get_tree().quit(); return
		var total := 0
		for k in range(10):
			var ox := 100000 + k * 4000          # 200 m steps across the world
			gw.regen_tracked(ox, 100000 + k * 2000)
			var holes := gw.count_holes()
			total += holes
			print("HOLETEST @(%d,%d): oy=%d holes=%d" % [ox, 100000 + k * 2000, gw.gen_oy, holes])
		# STREAMING pass: advance the window one chunk at a time exactly like
		# _stream() does (strips at the current band unless the band target has
		# drifted past the deadband). This is where the missing chunks came from:
		# strips generated at a stale band height as the terrain descends.
		var stream_mode := OS.get_environment("VOX_HOLESTREAM")
		for dirv: Vector2i in [Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1)]:
			gw.regen_tracked(100000, 100000)
			var o := Vector2i(100000, 100000)
			var worst := 0
			for k in range(100):
				var no := o + dirv * CHUNK_VOX
				var noy := gw.band_oy_for(no.x + world.W * 0.5, no.y + world.D * 0.5)
				if stream_mode != "old" and absi(noy - gw.gen_oy) >= int(gw.H * 0.12):
					gw.gen_oy = noy
					gw.regen(no.x, no.y)         # band moved: full regen (like _stream)
				else:
					gw.set_origin(no.x, no.y)
					if dirv.x != 0:
						gw.regen_strip(posmod(mini(o.x, no.x), gw.W), CHUNK_VOX, 0, gw.D)
					else:
						gw.regen_strip(0, gw.W, posmod(mini(o.y, no.y), gw.D), CHUNK_VOX)
				o = no
				var h2 := gw.count_holes()
				worst = maxi(worst, h2)
				total += h2
			print("HOLETEST stream dir %s: 100 chunks (2 km), worst step holes=%d" % [dirv, worst])
		print("HOLES TOTAL %d — %s" % [total, "CLEAN" if total == 0 else "WORLD HAS HOLES"])
		gw.free_gpu()
		get_tree().quit()
		return
	if OS.get_environment("VOX_SPLITTEST") != "":
		# multi-buffer equivalence: the same world (seed, band, origin) simulated
		# with its cells SPLIT across two buffers must be BIT-IDENTICAL to the
		# single-buffer run — gen, rain, physics and erosion are all deterministic
		# per seed, so any cget/cset routing bug shows up as a cell mismatch.
		var wa := GpuWorld.new(4242, 192, 192, 192)
		OS.set_environment("VOX_FORCESPLIT", "2")   # force THREE buffers on B
		var wb := GpuWorld.new(4242, 192, 192, 192)
		OS.set_environment("VOX_FORCESPLIT", "")
		if not wa.gpu_ok or not wb.gpu_ok:
			print("SPLITTEST: no GPU"); get_tree().quit(); return
		var nt := 192 * 192 * 192
		print("SPLITTEST: A splits=%d,%d/%d (single) vs B splits=%d,%d/%d (three buffers)" \
			% [wa.cells_split, wa.cells_split2, nt, wb.cells_split, wb.cells_split2, nt])
		wa.regen_tracked(100000, 100000)
		wb.gen_oy = wa.gen_oy               # identical band + origin
		wb.regen(100000, 100000)
		for w2: GpuWorld in [wa, wb]:
			w2.set_rain_mm_hr(80.0); w2.run(150)   # storm builds water + erosion state
			w2.set_rain_mm_hr(0.0); w2.run(60)     # then settle
		wa.sync_cells(); wb.sync_cells()
		var diff := 0
		for i in range(wa.cell.size()):
			if wa.cell[i] != wb.cell[i]: diff += 1
		print("SPLITTEST: %d of %d cells differ after storm+settle" % [diff, nt])
		print("SPLIT EQUIVALENT" if diff == 0 and wb.cells_split2 < nt else "SPLIT MISMATCH")
		wa.free_gpu(); wb.free_gpu()
		get_tree().quit()
		return
	if OS.get_environment("VOX_LISTTEST") != "":
		# indirect-dispatch equivalence: the same world stepped by the full-grid
		# gated dispatch vs the compacted awake-list indirect dispatch must be
		# BIT-IDENTICAL — the list path runs the same block set with the same
		# virtual-gid RNG seeds, so any divergence is a compaction/indexing bug.
		# VOX_LISTTEST=gg / ll / bb compare two same-mode worlds instead (a
		# determinism probe: nonzero gg/ll/bb diff = that mode races with itself)
		var lmode := OS.get_environment("VOX_LISTTEST")
		var modes := {"gg": ["grid", "grid"], "ll": ["", ""],
				"bb": ["bounded", "bounded"], "gb": ["grid", "bounded"]}.get(lmode, ["grid", ""]) as Array
		OS.set_environment("VOX_STEPMODE", modes[0])
		var wa := GpuWorld.new(4242, 192, 192, 192)
		OS.set_environment("VOX_STEPMODE", modes[1])
		var wb := GpuWorld.new(4242, 192, 192, 192)
		OS.set_environment("VOX_STEPMODE", "")
		if not wa.gpu_ok or not wb.gpu_ok:
			print("LISTTEST: no GPU"); get_tree().quit(); return
		var nt := 192 * 192 * 192
		print("LISTTEST: A %s vs B %s" % [wa.step_mode, wb.step_mode])
		wa.regen_tracked(100000, 100000)
		wb.gen_oy = wa.gen_oy               # identical band + origin
		wb.regen(100000, 100000)
		for w2: GpuWorld in [wa, wb]:
			w2.set_rain_mm_hr(80.0); w2.run(150)   # storm builds water + erosion state
			w2.set_rain_mm_hr(0.0); w2.run(60)     # then settle
		wa.sync_cells(); wb.sync_cells()
		var diff := 0
		for i in range(wa.cell.size()):
			if wa.cell[i] != wb.cell[i]: diff += 1
		print("LISTTEST: %d of %d cells differ after storm+settle" % [diff, nt])
		if lmode in ["gg", "ll", "bb"]:
			print("SELF DETERMINISTIC" if diff == 0 else "SELF RACE (%s)" % lmode)
		else:
			print("LIST EQUIVALENT" if diff == 0 and wb.step_mode == "indirect" \
					and wa.step_mode == "grid" else "LIST MISMATCH")
		wa.free_gpu(); wb.free_gpu()
		get_tree().quit()
		return
	if OS.get_environment("VOX_TRACKTEST") != "":
		# vertical-tracking band: fly a long horizontal path across the 256 m-relief
		# world, apply the same deadband band-shift the interactive tracker uses, and
		# assert the terrain surface stays resident in the band the whole way (never
		# clips out the top or bottom). gw is tiny — only its CPU noise (surface_world_y,
		# seeded) is used; the band footprint/height are the real defaults.
		var Wt := 2048.0
		var Ht := 756
		var gw := GpuWorld.new(777, 64, 64, 64)
		var base := 100000.0
		var band_target := func(cx: float, cz: float) -> int:
			var blo := 1e9
			var bhi := -1e9
			for iz in range(-4, 5):
				for ix in range(-4, 5):
					var s: float = gw.surface_world_y(cx + ix * Wt / 8.0, cz + iz * Wt / 8.0)
					blo = minf(blo, s); bhi = maxf(bhi, s)
			return clampi(int(round((blo + bhi) * 0.5 - Ht * 0.5 + Ht * 0.08)), 0, maxi(GpuWorld.RELIEF - Ht, 0))
		var oy: int = band_target.call(base + Wt * 0.5, base + Wt * 0.5)
		var shifts := 0
		var worst_bottom := 1e9   # min (surface - band floor): subsurface headroom
		var worst_top := 1e9      # min (band top - surface): air/peak headroom
		var lo := 1e9
		var hi := -1e9
		var steps := 70
		var stride := 400.0                               # 400 vox/step -> 1.4 km flight
		for i in range(steps):
			var camx := base + Wt * 0.5 + i * stride
			var camz := base + Wt * 0.5
			var target: int = band_target.call(camx, camz)
			if absi(target - oy) >= int(Ht * 0.12):
				oy = target
				shifts += 1
			for jz in range(-1, 2):
				for jx in range(-1, 2):
					var s: float = gw.surface_world_y(camx + jx * Wt * 0.4, camz + jz * Wt * 0.4)
					lo = minf(lo, s); hi = maxf(hi, s)
					worst_bottom = minf(worst_bottom, s - oy)
					worst_top = minf(worst_top, float(oy + Ht) - s)
		print("TRACKTEST: flew %.0f m; surface spanned %.1f..%.1f m (%.1f m relief traversed)" \
			% [steps * stride * 0.05, lo * 0.05, hi * 0.05, (hi - lo) * 0.05])
		print("TRACKTEST: band shifts=%d; min subsurface headroom=%.1f m, min air/peak headroom=%.1f m" \
			% [shifts, worst_bottom * 0.05, worst_top * 0.05])
		print("SURFACE STAYED RESIDENT" if worst_bottom > 0.0 and worst_top > 0.0 else "SURFACE CLIPPED")
		gw.free_gpu()
		get_tree().quit()
		return
	if OS.get_environment("VOX_GENSTABLE") != "":
		# is freshly generated terrain at rest, or does it landslide? count solid
		# (non-water) cells whose material changes per tick — movement = slumping
		var gw := GpuWorld.new(12345, 160, 160, 160)
		gw.set_rain_mm_hr(0.0)
		if not gw.gpu_ok:
			print("GENSTABLE: no GPU"); get_tree().quit(); return
		gw.regen_tracked(100000, 100000)   # place the band under the surface
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
		var Hh := 512
		var big := GpuWorld.new(777, 800, 800, Hh)   # 2x2 chunks, generated whole
		var one := GpuWorld.new(777, 400, 400, Hh)    # one chunk, standalone
		if not big.gpu_ok or not one.gpu_ok:
			print("SEAMTEST: no GPU"); get_tree().quit(); return
		# both worlds must use the SAME vertical band (gen_oy is a windowing choice,
		# not part of the world), or the vertical slice differs; horizontal gen is
		# seamless per world-coord. Share one oy computed for chunk (1,0)'s centre.
		var soy := big.band_oy_for(600, 200)
		big.gen_oy = soy; big.regen(0, 0)
		one.gen_oy = soy; one.regen(400, 0)           # as world-chunk (1,0)
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
	if OS.get_environment("VOX_TOROTEST") != "":
		# toroidal streaming: after advancing the window one chunk, the NEWLY-
		# ENTERED edge strip regenerates but the rest of the window keeps its live
		# sim state. Rain a while to build state, snapshot, advance, and confirm
		# only the strip changed.
		var Wt := 800
		var gw := GpuWorld.new(555, Wt, Wt, 512)
		if not gw.gpu_ok:
			print("TOROTEST: no GPU"); get_tree().quit(); return
		var base := 100400                     # chunk-aligned positive origin
		gw.regen_tracked(base, base)           # band under the surface (constant during the shift)
		gw.set_rain_mm_hr(80.0); gw.run(150)   # accumulate water + erosion state
		gw.set_rain_mm_hr(0.0); gw.run(60)
		gw.sync_cells()
		var before := gw.cell.duplicate()
		var stripx := posmod(base, Wt)          # buffer slots that will regenerate
		gw.set_origin(base + 400, base)          # fly east one chunk
		gw.regen_strip(stripx, 400, 0, Wt)
		gw.sync_cells()
		var kept_changed := 0
		var strip_changed := 0
		for i in range(before.size()):
			if before[i] == gw.cell[i]: continue
			if posmod((i % Wt) - stripx, Wt) < 400: strip_changed += 1
			else: kept_changed += 1
		print("TOROTEST: after +1 chunk: kept-region cells changed=%d (want 0), strip changed=%d" \
			% [kept_changed, strip_changed])
		print("STATE PRESERVED" if kept_changed == 0 and strip_changed > 0 else "STATE LOST")
		gw.free_gpu()
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
	w.regen_tracked(100000, 100000)   # place the band under the tall-world surface
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
