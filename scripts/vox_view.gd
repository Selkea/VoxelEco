class_name VoxView
extends Node3D
## Renders a VoxWorld as face-culled meshes, chunked Noita-style: the world is
## split into 16x16-column chunks, each with its own terrain + water mesh, and
## only chunks the simulation flagged dirty get rebuilt. A settled landscape
## costs nothing to keep on screen.

const CHUNK := 16
# Y extent of the instance culling AABB: the full 256 m relief (RELIEF=5120) plus
# margin, so the vertical-tracking band is never culled wherever it rides.
const AABB_Y := 5400.0

var world: VoxWorld
var solid_mat: StandardMaterial3D
var water_mat: StandardMaterial3D
var solid_chunks: Array[MeshInstance3D] = []
var water_chunks: Array[MeshInstance3D] = []
var cw := 0
var cd := 0
# instanced path (GPU worlds): surface voxels as MultiMesh cubes, buffers
# written by the emit compute pass
var use_instances := false
var solid_mm: MultiMeshInstance3D
var water_mm: MultiMeshInstance3D

const COLOR := {
	VoxWorld.BEDROCK: Color("#3a3f47"),
	VoxWorld.STONE: Color("#8b8d94"),
	VoxWorld.SOIL: Color("#916a3e"),
	VoxWorld.MUD: Color("#1d1209"),
	VoxWorld.GRASS: Color("#2f5a1a"),
	VoxWorld.SAND: Color("#c4ae74"),
}
# 6 face directions: normal + the 4 corner offsets of the quad, wound so the
# front face points along the normal. Godot's front faces wind CLOCKWISE:
# originally all six were wound CCW (inside-out) and the world rendered as a
# coherent-looking inverted box — verified and fixed with the VOX_CUBETEST
# five-axis probe.
const FACES := [
	{ "n": Vector3(0, 1, 0), "v": [Vector3(1,1,0), Vector3(1,1,1), Vector3(0,1,1), Vector3(0,1,0)] },
	{ "n": Vector3(0, -1, 0), "v": [Vector3(0,0,1), Vector3(1,0,1), Vector3(1,0,0), Vector3(0,0,0)] },
	{ "n": Vector3(1, 0, 0), "v": [Vector3(1,0,1), Vector3(1,1,1), Vector3(1,1,0), Vector3(1,0,0)] },
	{ "n": Vector3(-1, 0, 0), "v": [Vector3(0,1,0), Vector3(0,1,1), Vector3(0,0,1), Vector3(0,0,0)] },
	{ "n": Vector3(0, 0, 1), "v": [Vector3(0,1,1), Vector3(1,1,1), Vector3(1,0,1), Vector3(0,0,1)] },
	{ "n": Vector3(0, 0, -1), "v": [Vector3(1,0,0), Vector3(1,1,0), Vector3(0,1,0), Vector3(0,0,0)] },
]

func _ready() -> void:
	solid_mat = StandardMaterial3D.new()
	solid_mat.vertex_color_use_as_albedo = true
	solid_mat.roughness = 0.95
	water_mat = StandardMaterial3D.new()
	# OPAQUE water. Transparent water shimmered: the emit writes instances in a
	# non-deterministic order each frame, and alpha blending is order-dependent,
	# so a still surface flickered. Opaque water is order-independent — coincident
	# faces of overlapping water cubes are the same colour, so which one wins the
	# depth test is invisible (the terrain uses the same overlapping cubes and
	# never flickers for exactly this reason).
	water_mat.albedo_color = Color(0.17, 0.39, 0.62)
	water_mat.roughness = 0.12
	water_mat.metallic = 0.15
	water_mat.specular = 0.6
	if use_instances:
		# cap-allocated: the emit compute pass writes these multimesh buffers
		# directly in VRAM; only visible_instance_count changes per frame
		solid_mm = _make_mm(solid_mat, world.solid_cap)
		water_mm = _make_mm(water_mat, world.water_cap)
	else:
		_alloc_chunks()

func _make_mm(mat: StandardMaterial3D, cap: int) -> MultiMeshInstance3D:
	var mmi := MultiMeshInstance3D.new()
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	var box := BoxMesh.new()
	box.size = Vector3.ONE
	mm.mesh = box
	mm.instance_count = cap
	mm.visible_instance_count = 0
	mmi.multimesh = mm
	mmi.material_override = mat
	# a huge custom AABB so instances written GPU-side are never culled away. Y
	# spans the full vertical relief (not just the band height): the vertical-
	# tracking band places instances anywhere in world-Y [0, RELIEF], so a band-
	# height Y bound would cull the terrain whenever the band rode up a mountain.
	mmi.custom_aabb = AABB(Vector3(-16, -16, -16),
			Vector3(world.W + 32, AABB_Y, world.D + 32))
	add_child(mmi)
	return mmi

func solid_buffer_rid() -> RID:
	return RenderingServer.multimesh_get_buffer_rd_rid(solid_mm.multimesh.get_rid())

func water_buffer_rid() -> RID:
	return RenderingServer.multimesh_get_buffer_rd_rid(water_mm.multimesh.get_rid())

## per-frame: the buffers are already written GPU-side; just set the counts
func set_visible_counts(ns: int, nw: int) -> void:
	solid_mm.multimesh.visible_instance_count = ns
	water_mm.multimesh.visible_instance_count = nw

## FLOATING ORIGIN: the emit writes voxels in the LOCAL frame [0, W) (relative to
## the window origin), so the culling AABB is a fixed local box — the whole scene
## is drawn near 0 and the camera is offset by the same origin. Kept as a hook so
## streaming can call it; the box no longer depends on the world origin.
func set_stream_origin(_ox: int, _oz: int) -> void:
	if not use_instances:
		return
	var a := AABB(Vector3(-16, -16, -16),
			Vector3(world.W + 32, AABB_Y, world.D + 32))
	solid_mm.custom_aabb = a
	water_mm.custom_aabb = a

func _alloc_chunks() -> void:
	for mi in solid_chunks + water_chunks:
		mi.queue_free()
	solid_chunks.clear()
	water_chunks.clear()
	cw = (world.W + CHUNK - 1) / CHUNK
	cd = (world.D + CHUNK - 1) / CHUNK
	for k in range(cw * cd):
		var s := MeshInstance3D.new()
		s.material_override = solid_mat
		add_child(s)
		solid_chunks.append(s)
		var wmi := MeshInstance3D.new()
		wmi.material_override = water_mat
		add_child(wmi)
		water_chunks.append(wmi)

## rebuild dirty chunks only; `dirty` empty or wrong-sized = rebuild all
func rebuild(dirty: PackedInt32Array = PackedInt32Array()) -> void:
	if world == null:
		return
	if cw != (world.W + CHUNK - 1) / CHUNK or cd != (world.D + CHUNK - 1) / CHUNK:
		_alloc_chunks()
	var all := dirty.size() != cw * cd
	for cz in range(cd):
		for cx in range(cw):
			var ci := cz * cw + cx
			if all or dirty[ci] != 0:
				solid_chunks[ci].mesh = _build_chunk(cx, cz, false)
				water_chunks[ci].mesh = _build_chunk(cx, cz, true)
	if OS.get_environment("VOX_MESHSTATS") != "":
		var sv := 0
		var wv := 0
		for k in range(cw * cd):
			if solid_chunks[k].mesh and solid_chunks[k].mesh.get_surface_count() > 0:
				sv += solid_chunks[k].mesh.surface_get_array_len(0)
			if water_chunks[k].mesh and water_chunks[k].mesh.get_surface_count() > 0:
				wv += water_chunks[k].mesh.surface_get_array_len(0)
		print("mesh stats: solid verts=%d  water verts=%d" % [sv, wv])

func _transparent_to(m: int, water_pass: bool) -> bool:
	if water_pass:
		# water is only visible at the water-air interface; faces against
		# opaque solids (pool floors/walls) can never be seen
		return m == VoxWorld.AIR
	return m == VoxWorld.AIR or m == VoxWorld.WATER

## does this cell block sky/light for ambient-occlusion purposes?
func _occ(x: int, y: int, z: int) -> bool:
	if not world.in_b(x, y, z):
		return false
	var m: int = world.cell[world.idx(x, y, z)]
	return m != VoxWorld.AIR and m != VoxWorld.WATER

func _build_chunk(cx: int, cz: int, water_pass: bool) -> ArrayMesh:
	var w := world
	var verts := PackedVector3Array()
	var norms := PackedVector3Array()
	var cols := PackedColorArray()
	var x0 := cx * CHUNK
	var z0 := cz * CHUNK
	var x1 := mini(x0 + CHUNK, w.W)
	var z1 := mini(z0 + CHUNK, w.D)
	for z in range(z0, z1):
		for x in range(x0, x1):
			for y in range(w.H):
				var i := w.idx(x, y, z)
				var m: int = w.cell[i]
				if water_pass:
					if m != VoxWorld.WATER:
						continue
				elif m == VoxWorld.AIR or m == VoxWorld.WATER:
					continue
				var base := Vector3(x, y, z)
				# vertex colors skip the material sRGB->linear conversion in
				# forward_plus, so convert here or everything renders bleached
				var col: Color = Color(0.24, 0.52, 0.82) if water_pass else COLOR[m]
				col = col.srgb_to_linear()
				if not water_pass:
					# per-voxel tint jitter breaks up flat single-colour regions
					var jit := 1.0 + (float((i * 2654435761) & 255) / 255.0 * 0.24 - 0.12)
					col = Color(col.r * jit, col.g * jit, col.b * jit)
				for f in FACES:
					var n: Vector3 = f["n"]
					var ni := Vector3i(int(n.x), int(n.y), int(n.z))
					var nx := x + ni.x
					var ny := y + ni.y
					var nz := z + ni.z
					var neighbour := VoxWorld.AIR
					if w.in_b(nx, ny, nz):
						neighbour = w.cell[w.idx(nx, ny, nz)]
					elif ny < 0:
						continue   # the world's underside is never visible
					# out of bounds otherwise counts as AIR: terrain and water
					# both draw their cut faces at the diorama boundary
					if not _transparent_to(neighbour, water_pass):
						continue
					var vv: Array = f["v"]
					var shade := 1.0 - (1.0 - maxf(n.y, 0.0)) * 0.16
					var c := Color(col.r * shade, col.g * shade, col.b * shade, col.a)
					var q := [base + vv[0], base + vv[1], base + vv[2], base + vv[3]]
					if water_pass:
						for t in [0, 1, 2, 0, 2, 3]:
							verts.append(q[t])
							norms.append(n)
							cols.append(c)
						continue
					# classic voxel vertex AO: each quad corner is darkened by
					# the solid neighbours around it in the face plane
					var t1 := Vector3i(1, 0, 0) if ni.x == 0 else Vector3i(0, 1, 0)
					var t2 := Vector3i(0, 0, 1) if ni.z == 0 else Vector3i(0, 1, 0)
					var ao := []
					for corner in vv:
						var s1: int = 1 if Vector3(corner).dot(Vector3(t1)) > 0.5 else -1
						var s2: int = 1 if Vector3(corner).dot(Vector3(t2)) > 0.5 else -1
						var o1 := _occ(nx + s1 * t1.x, ny + s1 * t1.y, nz + s1 * t1.z)
						var o2 := _occ(nx + s2 * t2.x, ny + s2 * t2.y, nz + s2 * t2.z)
						var oc := _occ(nx + s1 * t1.x + s2 * t2.x, ny + s1 * t1.y + s2 * t2.y,
								nz + s1 * t1.z + s2 * t2.z)
						var occ := 3 if (o1 and o2) else (int(o1) + int(o2) + int(oc))
						ao.append(1.0 - 0.2 * occ)
					# flip the quad diagonal toward the smoother AO gradient
					var order := [0, 1, 2, 0, 2, 3] if ao[0] + ao[2] >= ao[1] + ao[3] \
							else [1, 2, 3, 1, 3, 0]
					for t in order:
						verts.append(q[t])
						norms.append(n)
						cols.append(Color(c.r * ao[t], c.g * ao[t], c.b * ao[t], c.a))
	var mesh := ArrayMesh.new()
	if verts.is_empty():
		return mesh
	var arr := []
	arr.resize(Mesh.ARRAY_MAX)
	arr[Mesh.ARRAY_VERTEX] = verts
	arr[Mesh.ARRAY_NORMAL] = norms
	arr[Mesh.ARRAY_COLOR] = cols
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arr)
	return mesh
