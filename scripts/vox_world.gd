class_name VoxWorld
extends RefCounted
## Pure 3D voxel falling-sand world — no nodes, no rendering, deterministic
## from its seed. Hydrology first: rain spawns water voxels that fall, flow
## downhill, pool into lakes, and evaporate; sand and soil topple to their
## angle of repose; flowing water erodes soil and carries sediment.
##
## Grid is a flat PackedByteArray, one material per cell. Only "active" cells
## (water + unsettled solids) are simulated each tick, so a settled landscape
## is cheap and the water is where the work goes.

enum { AIR, BEDROCK, STONE, SOIL, SAND, WATER, MUD, GRASS, PLANT, HERB, PRED }

# density indexed by material (heavier sinks through lighter; BEDROCK immovable).
# a plain Array, not a Dictionary — indexing is far faster in the hot loop.
# indices: AIR BEDROCK STONE SOIL SAND WATER MUD GRASS PLANT HERB PRED
# PLANT is immovable standing foliage; HERB is a mobile grazer's body voxel and PRED a
# predator's (all GPU-only, see vox_step.glsl do_vegetate / do_herbivore / do_predator)
# — dens 3 keeps them consistent with the shader's fall-through if referenced on the CPU.
const DENS := [0, 99, 9, 3, 3, 1, 3, 3, 3, 3, 3]
# the 4 horizontal directions, as parallel arrays (no per-cell allocation)
const DX := [1, -1, 0, 0]
const DZ := [0, 0, 1, -1]

var W := 64      # x
var D := 64      # z
var H := 40      # y (up)
var wd := 0      # W*D, stride between y layers

# ---------------- physical scale (real-world calibration) ----------------
# The sim is unit-less per tick; these map ticks/voxels onto real units so the
# weather rates below mean something. One voxel is a 5 cm cube, so a single
# water voxel dropped into a column adds 50 mm of DEPTH there, and one
# simulated hour is 108,000 ticks — every derivation follows from those two.
const VOXEL_M := 0.05             # metres per voxel edge
const TICK_RATE := 30.0           # simulation ticks per simulated second
const TICK_S := 1.0 / TICK_RATE
# erosion is wildly context-dependent in reality (mm/year geologically,
# cm/hour in a storm rill); we pick the fast fluvial regime so it's visible
# over sim-hours: a soil voxel in sustained flow detaches after this retreat.
const EROSION_CM_PER_HR := 5.0

var cell := PackedByteArray()
var active := {}          # index -> true : cells to simulate next tick
var rng := RandomNumberGenerator.new()
var seed_value := 0
var tick_count := 0
# weather in real units; the *_prob fields are the per-tick derivations the
# physics actually uses (set via the calibration helpers below)
var rain_mm_hr := 0.0             # calm by default; press E to make it rain
                                 # (mm/hour, WMO: >7.5 heavy, >50 violent)
var evap_mm_day := 5.0            # open-water evaporation (typical 1-10 mm/day)
var rain_prob := 0.0             # per top-column per tick
var evap_prob := 0.0             # per exposed water voxel per tick
var erode_prob := 0.0            # per soil-water contact per tick
var water_added := 0
var water_evaporated := 0
var water_absorbed := 0          # water voxels that soaked into the ground
var water_deposited := 0         # water voxels that settled out as a sediment (SAND) voxel

func _init(seed_v: int = 0, w: int = 64, d: int = 64, h: int = 40) -> void:
	W = w; D = d; H = h; wd = W * D
	seed_value = seed_v if seed_v != 0 else 12345
	rng.seed = seed_value
	cell.resize(W * D * H)
	set_rain_mm_hr(rain_mm_hr)
	set_evap_mm_day(evap_mm_day)
	# EROSION_CM_PER_HR of bank retreat per hour of contact -> one 5 cm voxel
	# every (5 / EROSION_CM_PER_HR) hours of continuous flow contact
	erode_prob = (EROSION_CM_PER_HR / (VOXEL_M * 100.0)) / (3600.0 * TICK_RATE)
	_generate()

## rain intensity in mm/hour -> probability a top column spawns a water voxel
## this tick. Each voxel = VOXEL_M*1000 mm of depth; 3600*TICK_RATE ticks/hour.
func set_rain_mm_hr(mm: float) -> void:
	rain_mm_hr = maxf(0.0, mm)
	rain_prob = (rain_mm_hr / (VOXEL_M * 1000.0)) / (3600.0 * TICK_RATE)

## evaporation in mm/day -> probability an exposed water voxel evaporates/tick.
func set_evap_mm_day(mm: float) -> void:
	evap_mm_day = maxf(0.0, mm)
	evap_prob = (evap_mm_day / (VOXEL_M * 1000.0)) / (86400.0 * TICK_RATE)

func idx(x: int, y: int, z: int) -> int:
	return x + z * W + y * wd

func in_b(x: int, y: int, z: int) -> bool:
	return x >= 0 and x < W and y >= 0 and y < H and z >= 0 and z < D

func mat(i: int) -> int:
	return cell[i]

## interface shared with GpuWorld (overridden there)
var dirty_chunks := PackedInt32Array()   # empty = "rebuild everything"
var cut_z := 0        # emit cross-section: hide voxels with z < cut_z (0 = whole world)

func run(n: int) -> void:
	for i in range(n):
		step()

func sync_cells() -> void:
	pass          # CPU cells are always current

func reset_water_stats() -> void:
	water_added = 0
	water_evaporated = 0
	water_absorbed = 0
	water_deposited = 0

func free_gpu() -> void:
	pass

func water_count() -> int:
	var n := 0
	for i in range(cell.size()):
		if cell[i] == WATER:
			n += 1
	return n

# ---------------- terrain generation ----------------

func _generate() -> void:
	var noise := FastNoiseLite.new()
	noise.seed = seed_value
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	noise.frequency = 0.028
	noise.fractal_octaves = 4
	var ridged := FastNoiseLite.new()
	ridged.seed = seed_value + 99
	ridged.noise_type = FastNoiseLite.TYPE_SIMPLEX
	ridged.frequency = 0.06
	var cx := W * 0.5
	var cz := D * 0.5
	for z in range(D):
		for x in range(W):
			# rolling hills
			var n := noise.get_noise_2d(x, z) * 0.5 + 0.5      # 0..1
			var hgt := 6.0 + n * (H * 0.55)
			# carve a broad central basin so water collects into a lake
			var r := Vector2(x - cx, z - cz).length() / (W * 0.5)
			var bowl := clampf(1.0 - r, 0.0, 1.0)
			hgt -= bowl * bowl * (H * 0.32)
			# a rocky ridge on one side for relief
			hgt += maxf(0.0, ridged.get_noise_2d(x * 1.0, z * 1.0)) * 6.0
			var top := clampi(int(hgt), 2, H - 4)
			for y in range(H):
				var m := AIR
				if y == 0:
					m = BEDROCK
				elif y < top - 4:
					m = STONE
				elif y < top - 1:
					m = SOIL
				elif y < top:
					m = SAND if top < 10 else SOIL   # low ground = sandy
				cell[idx(x, y, z)] = m

# ---------------- simulation ----------------

func _wake(i: int) -> void:
	active[i] = true

func _wake_around(x: int, y: int, z: int) -> void:
	var i := x + z * W + y * wd
	if y + 1 < H: active[i + wd] = true
	if y > 0: active[i - wd] = true
	if x + 1 < W: active[i + 1] = true
	if x > 0: active[i - 1] = true
	if z + 1 < D: active[i + W] = true
	if z > 0: active[i - W] = true

func _swap(a: int, b: int) -> void:
	var t := cell[a]
	cell[a] = cell[b]
	cell[b] = t

func step() -> void:
	tick_count += 1
	_rain()
	# process active cells bottom-up so a falling column resolves in one pass
	var order := active.keys()
	order.sort()               # ascending index ~ ascending y (y is the high bits)
	active = {}
	for key in order:
		var i: int = key
		var m: int = cell[i]
		if m == AIR or m == BEDROCK or m == STONE:
			continue
		var y: int = i / wd
		var rem: int = i % wd
		var z: int = rem / W
		var x: int = rem % W
		if m == WATER:
			_step_water(x, y, z, i)
		else:
			_step_grain(x, y, z, i, m)

func _rain() -> void:
	# expected spawns this tick = rain_prob per column x every column;
	# stochastic-round the fraction so low rates still spawn occasionally
	var expected := rain_prob * float(W * D)
	var count := int(expected)
	if rng.randf() < expected - float(count):
		count += 1
	for k in range(count):
		var x := rng.randi_range(0, W - 1)
		var z := rng.randi_range(0, D - 1)
		var i := idx(x, H - 1, z)
		if cell[i] == AIR:
			cell[i] = WATER
			water_added += 1
			_wake(i)

# water: fall, run downhill along diagonals, spread to level, evaporate, erode
func _step_water(x: int, y: int, z: int, i: int) -> void:
	# evaporate if exposed to open air above
	if y < H - 1 and cell[i + wd] == AIR and rng.randf() < evap_prob:
		cell[i] = AIR
		water_evaporated += 1
		_wake_around(x, y, z)
		return
	# straight down
	if y > 0 and DENS[cell[i - wd]] < 1:
		_swap(i, i - wd)
		_wake_around(x, y, z)
		_wake_around(x, y - 1, z)
		active[i] = true
		return
	var r := rng.randi() & 3          # random rotation over the 4 dirs, no alloc
	# down-diagonals: flow downhill over terrain
	if y > 0:
		for k in range(4):
			var kk: int = (r + k) & 3
			var nx: int = x + DX[kk]
			var nz: int = z + DZ[kk]
			if nx >= 0 and nx < W and nz >= 0 and nz < D:
				var di: int = nx + nz * W + (y - 1) * wd
				if cell[di] == AIR:
					_swap(i, di)
					_wake_around(x, y, z)
					_wake_around(nx, y - 1, nz)
					return
	# spread sideways to find its level
	for k in range(4):
		var kk: int = (r + k) & 3
		var nx: int = x + DX[kk]
		var nz: int = z + DZ[kk]
		if nx >= 0 and nx < W and nz >= 0 and nz < D:
			var si: int = nx + nz * W + y * wd
			if cell[si] == AIR:
				_swap(i, si)
				_wake_around(x, y, z)
				_wake_around(nx, y, nz)
				return
	# settled: only stay active if it could still evaporate
	if y < H - 1 and cell[i + wd] == AIR:
		active[i] = true

# sand/soil: fall, topple to angle of repose; soil wets in contact with water
func _step_grain(x: int, y: int, z: int, i: int, m: int) -> void:
	if y == 0:
		return
	if DENS[cell[i - wd]] < DENS[m]:
		_swap(i, i - wd)
		_wake_around(x, y, z)
		_wake_around(x, y - 1, z)
		return
	# topple down-diagonals; soil holds a steeper pile than sand
	var repose: float = 0.55 if m == SAND else 0.28
	if rng.randf() < repose:
		var r := rng.randi() & 3
		for k in range(4):
			var kk: int = (r + k) & 3
			var nx: int = x + DX[kk]
			var nz: int = z + DZ[kk]
			if nx >= 0 and nx < W and nz >= 0 and nz < D:
				var di: int = nx + nz * W + (y - 1) * wd
				if DENS[cell[di]] < DENS[m] and cell[nx + nz * W + y * wd] != BEDROCK:
					_swap(i, di)
					_wake_around(x, y, z)
					_wake_around(nx, y - 1, nz)
					return
	# erosion: soil touching flowing water loosens into sand
	if m == SOIL:
		if (cell[i + 1] == WATER or cell[i - 1] == WATER \
				or cell[i + W] == WATER or cell[i - W] == WATER \
				or (y + 1 < H and cell[i + wd] == WATER)) and rng.randf() < erode_prob:
			cell[i] = SAND
			active[i] = true

# wake the whole surface once so initial overhangs settle at startup
func prime() -> void:
	for z in range(D):
		for x in range(W):
			for y in range(1, H):
				var m := cell[idx(x, y, z)]
				if m == SAND or m == SOIL:
					active[idx(x, y, z)] = true
