import bpy
import mathutils
import random
import os
import json
import sys
from pathlib import Path

import argparse

# === ARGUMENT PARSER ===
parser = argparse.ArgumentParser(description="Given a blender dataset with transforms.json, render emissivities, reflection and self-emission of objects in the scene.")
parser.add_argument("transforms_json", type=str, help="Path to the transforms.json file.")

args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

STARDIS_ADDON_PATH = Path(__file__).parent.parent / "blender-stardis-exporter"
bpy.ops.preferences.addon_install(
	filepath=str(STARDIS_ADDON_PATH / "blender_stardis_exporter.py"),
	overwrite=True,
)

bpy.ops.preferences.addon_enable(module="blender_stardis_exporter")
bpy.ops.wm.save_userpref()

print(bpy.types.Object.stardis_object_properties)

# === LOAD FILES ===
if not os.path.exists(args.transforms_json):
	print(f"❌ Error: The specified transforms.json file does not exist: {args.transforms_json}")
	exit(1)

with open(args.transforms_json, 'r') as f:
	transforms = json.load(f)

# === CONFIG ===
TEMP_MAT_PREFIX = "Temp_UniqueColor_"
original_materials = {}
world_node_backup = {}
object_hidden = {}

COLORS = [
	(1.0, 0.0, 0.0, 1.0),  # Red
	(0.0, 1.0, 0.0, 1.0),  # Green
	(0.0, 0.0, 1.0, 1.0),  # Blue
	(1.0, 1.0, 0.0, 1.0),  # Yellow
	(1.0, 0.0, 1.0, 1.0),  # Magenta
	(0.0, 1.0, 1.0, 1.0),  # Cyan
	(0.5, 0.5, 0.5, 1.0),  # Gray
]

def generate_unique_color(index):
	# random.seed(index)
	# return (random.random(), random.random(), random.random(), 1)
	return COLORS[index % len(COLORS)]



def parse_emissivity_from_file(file_path):
	with open(file_path, 'r') as f:
		lines = f.readlines()
	
	emissivity_map = {}
	for line in lines:
		line = line.strip()
		if not line or line.startswith('#'):
			continue
	
		parts = line.split()

		if parts[0] == "TRAD": continue

		if parts[1].endswith("DIRICHLET"):
			name = parts[1][:-len("-DIRICHLET")]
			emiss = 0.5
		elif parts[1].endswith("ROBIN_SOLID"):
			name = parts[1][:-len("-ROBIN_SOLID")]
			emiss = float(parts[3])
		elif parts[1].endswith("ROBIN_FLUID"):
			name = parts[1][:-len("-ROBIN_FLUID")]
			emiss = float(parts[3])
		elif parts[1].endswith("SOLID"):
			name = parts[1][:-len("-SOLID")]
			emiss = 1.0
		else:
			raise ValueError(f"Unknown object type in emissivity file: {parts[1]}. {line}")

		emissivity_map[name] = emiss
	
	return emissivity_map


def get_object_emissivity(name, emissivity_map):
	"""Get the emissivity value for an object by its name."""
	if name in emissivity_map:
		return emissivity_map[name]
	else:
		raise ValueError(f"Object '{name}' not found in emissivity map. Please check the emissivity file. keys: {list(emissivity_map.keys())}")


def get_emissivity_value(obj):
	if obj.type == 'MESH' and len(obj.stardis_object_properties) > 0:
		# write the object properties to the model.txt file
		for prop in obj.stardis_object_properties:
			prop_name = f"{obj.name}_{prop.stardis_object_type}".replace(" ", "-")
			if prop.stardis_object_type == "SOLID":
				pass

			
			elif prop.stardis_object_type == "DIRICHLET":
				dirichlet = prop.dirichlet
				return 0.5

			elif prop.stardis_object_type in ("ROBIN_SOLID", "ROBIN_FLUID"):
				robin = prop.robin_solid
				return robin.emissivity

			else:
				print("Unknown object type: " + prop.stardis_object_type)
				continue

	return 1.0 


def backup_world_nodes():
	"""Serializes the World node tree into a plain dict."""
	global world_node_backup
	world = bpy.context.scene.world
	if not world or not world.use_nodes:
		world_node_backup = {}
		return

	node_tree = world.node_tree
	nodes_data = []
	for node in node_tree.nodes:
		node_info = {
			"name": node.name,
			"type": node.bl_idname,
			"location": node.location[:],
			"inputs": {}
		}

		for inp in node.inputs:
			if hasattr(inp, "default_value"):
				try:
					val = inp.default_value
					node_info["inputs"][inp.name] = list(val) if isinstance(val, (list, tuple)) else val
				except:
					continue  # Skip uncopyable inputs

		nodes_data.append(node_info)

	links_data = []
	for link in node_tree.links:
		links_data.append({
			"from_node": link.from_node.name,
			"from_socket": link.from_socket.name,
			"to_node": link.to_node.name,
			"to_socket": link.to_socket.name,
		})

	world_node_backup = {
		"nodes": nodes_data,
		"links": links_data,
	}

def replace_world_with_white_background():
	"""Overrides the World node tree with a constant white background."""
	world = bpy.context.scene.world
	tree = world.node_tree
	tree.nodes.clear()

	bg = tree.nodes.new(type='ShaderNodeBackground')
	bg.inputs[0].default_value = (1, 1, 1, 1)
	bg.inputs[1].default_value = 1.0

	output = tree.nodes.new(type='ShaderNodeOutputWorld')
	tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

def restore_world_nodes():
	"""Rebuilds the World node tree from the plain dict."""
	global world_node_backup
	if not world_node_backup:
		return

	world = bpy.context.scene.world
	tree = world.node_tree
	tree.nodes.clear()

	name_to_node = {}

	for node_info in world_node_backup["nodes"]:
		node = tree.nodes.new(type=node_info["type"])
		node.name = node_info["name"]
		node.location = node_info.get("location", (0, 0))
		name_to_node[node.name] = node

		for input_name, value in node_info.get("inputs", {}).items():
			if input_name in node.inputs:
				input_socket = node.inputs.get(input_name)
				if hasattr(input_socket, "default_value"):
					try:
						if isinstance(input_socket.default_value, float):
							input_socket.default_value = float(value)
						elif isinstance(input_socket.default_value, (int, bool)):
							input_socket.default_value = value
						elif isinstance(input_socket.default_value, (list, tuple)):
							input_socket.default_value = list(value)
						else:
							pass  # unhandled type
					except Exception as e:
						print(f"⚠️ Could not assign '{input_name}': {e}") 


	for link in world_node_backup["links"]:
		from_node = name_to_node.get(link["from_node"])
		to_node = name_to_node.get(link["to_node"])
		if from_node and to_node:
			try:
				tree.links.new(
					from_node.outputs[link["from_socket"]],
					to_node.inputs[link["to_socket"]]
				)
			except:
				continue

def setup_render_settings():
	scene = bpy.context.scene
	scene.render.engine = 'CYCLES'
	scene.cycles.device = 'GPU'
	scene.cycles.samples = 512

	scene.render.image_settings.file_format = 'PNG'
	scene.render.film_transparent = False

	backup_world_nodes()
	replace_world_with_white_background()

	# Disable indirect lighting and shadows
	scene.cycles.use_adaptive_sampling = False
	scene.cycles.use_denoising = False
	scene.cycles.caustics_reflective = False
	scene.cycles.caustics_refractive = False

	scene.view_settings.exposure = 0.0
	scene.view_settings.gamma = 1.0
	scene.view_settings.view_transform = 'Raw'
	scene.view_settings.look = 'None'

	for light in [obj for obj in bpy.data.objects if obj.type == 'LIGHT']:
		light.hide_render = True

def assign_temp_materials():
	heatmodel_file = Path(args.transforms_json).parent / "heatmodels" / f"model-00000.txt"
	emissivity_map = parse_emissivity_from_file(heatmodel_file)

	for i, obj in enumerate(bpy.data.objects):
		if obj.type == 'MESH':
			original_materials[obj.name] = [slot.material for slot in obj.material_slots]
			object_hidden[obj.name] = obj.hide_render

			# emissivity = get_emissivity_value(obj)
			emissivity = get_object_emissivity(obj.name, emissivity_map)

			unique_color = generate_unique_color(i)

			obj.hide_render = not obj.hide_render  # Ensure the object is visible for rendering
			color = (emissivity, emissivity, emissivity, 1.0)  # Grayscale color based on emissivity
			# color = unique_color
			print(f"Object: {obj.name}, Color: {color}, will be rendered? {not obj.hide_render}")


			mat = bpy.data.materials.new(name=f"{TEMP_MAT_PREFIX}{i}")
			mat.use_nodes = True
			nodes = mat.node_tree.nodes
			links = mat.node_tree.links
			nodes.clear()

			emission = nodes.new(type='ShaderNodeEmission')
			emission.inputs['Color'].default_value = color
			emission.inputs['Strength'].default_value = 1.0

			output = nodes.new(type='ShaderNodeOutputMaterial')
			links.new(emission.outputs['Emission'], output.inputs['Surface'])

			if not obj.material_slots:
				obj.data.materials.append(mat)
			else:
				for j in range(len(obj.material_slots)):
					obj.material_slots[j].material = mat

def restore_original_materials():
	for obj_name, materials in original_materials.items():
		obj = bpy.data.objects.get(obj_name)
		if obj and obj.type == 'MESH':
			obj.data.materials.clear()
			for mat in materials:
				if mat:
					obj.data.materials.append(mat)
	
	for obj_name, hidden in object_hidden.items():
		obj = bpy.data.objects.get(obj_name)
		if obj and obj.type == 'MESH':
			obj.hide_render = hidden

def delete_temp_materials():
	temp_mats = [mat for mat in bpy.data.materials if mat.name.startswith(TEMP_MAT_PREFIX)]
	for mat in temp_mats:
		bpy.data.materials.remove(mat, do_unlink=True)

# === MAIN EXECUTION ===
setup_render_settings()
assign_temp_materials()

for frame in transforms['frames']:
	frame_img_path = Path(args.transforms_json).parent / frame['heat_img_path']
	img_stem = frame_img_path.stem
	heat_img = bpy.data.images.load(str(frame_img_path))
	width, height = heat_img.size

	camera_matrix = mathutils.Matrix(frame["transform_matrix"])
	bpy.context.scene.camera.matrix_world = camera_matrix
	# bpy.context.scene.camera.angle_x = transforms["camera_angle_x"]
	bpy.context.scene.render.resolution_x = width
	bpy.context.scene.render.resolution_y = height


	emiss_output_path = Path(args.transforms_json).parent / "emissivity" / f"{img_stem}.png"
	emiss_output_path.parent.mkdir(parents=True, exist_ok=True)

	bpy.context.scene.render.filepath = str(emiss_output_path)
	bpy.context.scene.render.image_settings.file_format = 'PNG'

	bpy.ops.render.render(write_still=True)

	emiss_img = bpy.data.images.load(str(emiss_output_path))

	if tuple(heat_img.size) != tuple(emiss_img.size):
		print(f"❌ Error: Heat image and emissivity image sizes do not match for {img_stem}. Skipping this frame.")
		print(f"Heat image size: {tuple(heat_img.size)}, Emissivity image size: {tuple(emiss_img.size)}")
		exit(1)

	# width, height = heat_img.size

	# heat_pixels = list(heat_img.pixels)
	# emiss_pixels = list(emiss_img.pixels)
	
	# self_emission_pixels = [
	# 	emiss_pixels[i] * heat_pixels[i] for i in range(len(emiss_pixels))
	# ]
	# reflection_pixels = [
	# 	heat_pixels[i] - self_emission_pixels[i] if i % 4 != 3 else 1.0
	# 	for i in range(len(emiss_pixels))
	# ]

	# self_emission_output_path = Path(args.transforms_json).parent / "emission" / f"{img_stem}.exr"
	# self_emission_output_path.parent.mkdir(parents=True, exist_ok=True)

	# self_emission_img = bpy.data.images.new(
	# 	name=f"{img_stem}_self_emission",
	# 	width=width,
	# 	height=height,
	# 	alpha=True,
	# )
	# self_emission_img.pixels = self_emission_pixels
	# self_emission_img.file_format = "OPEN_EXR"
	# self_emission_img.filepath_raw = str(self_emission_output_path)
	# self_emission_img.save()

	# reflection_output_path = Path(args.transforms_json).parent / "reflection" / f"{img_stem}.exr"
	# reflection_output_path.parent.mkdir(parents=True, exist_ok=True)
	
	# reflection_img = bpy.data.images.new(
	# 	name=f"{img_stem}_reflection",
	# 	width=width,
	# 	height=height,
	# 	alpha=True,
	# )

	# reflection_img.pixels = reflection_pixels
	# reflection_img.file_format = "OPEN_EXR"
	# reflection_img.filepath_raw = str(reflection_output_path)
	# reflection_img.save()


restore_original_materials()
restore_world_nodes()
delete_temp_materials()

print(f"✅ Render complete.\n🖼️ 🧹 Cleaned up temporary data.")

