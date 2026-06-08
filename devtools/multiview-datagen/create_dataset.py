import bpy
from mathutils import Vector
import os
import sys
from pathlib import Path
import argparse
import importlib
import hashlib
import math
import numpy as np
import json
import copy
import subprocess
import shutil

STARDIS_EXECUTABLE = os.environ.get("STARDIS_EXECUTABLE", "stardis")
STARDIS_REFLECTIVE_EXECUTABLE = os.environ.get("STARDIS_REFLECTIVE_EXECUTABLE", "stardis")
DISABLE_HTPP = os.environ.get("DISABLE_HTPP", "0") == "1"
HTPP_EXECUTABLE = os.environ.get("HTPP_EXECUTABLE", "htpp")
PYTHON_EXECUTABLE = os.environ.get("PYTHON_EXECUTABLE", shutil.which("python"))
assert PYTHON_EXECUTABLE is not None, "Could not find python executable, please set PYTHON_EXECUTABLE environment variable"

# make a copy of the environment variables
envs = copy.deepcopy(os.environ)

# check if the python has matplotlib installed
ret = subprocess.run([PYTHON_EXECUTABLE, "-c", "import matplotlib"])
if ret.returncode != 0:
    print(f"Error: matplotlib is not installed. Please install it in the current python env: {PYTHON_EXECUTABLE}")
    sys.exit(1)

# check if stardis, and htpp is installed
if not shutil.which(STARDIS_EXECUTABLE):
    print(f"Error: {STARDIS_EXECUTABLE = } not found. Please install it and add it to your PATH.")
    sys.exit(1)

if (not DISABLE_HTPP) and (not shutil.which(HTPP_EXECUTABLE)):
    print(f"Error: {HTPP_EXECUTABLE = } not found. Please install it and add it to your PATH, or disable the ppm image generation by setting 'DISABLE_HTPP'")
    sys.exit(1)

# This is neccessary as this script is run within Blender's own environment
blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
   sys.path.append(os.path.dirname(blend_dir))

from util import util, EasyDict, convert_ht_to_png


def env_with_stardis_ld_library_path(stardis_executable):
    envs = copy.deepcopy(os.environ)
    original_ld_library_path = envs.get("LD_LIBRARY_PATH", "")
    stardis_ld_library_path = str(Path(stardis_executable).parents[1] / "lib")
    envs["LD_LIBRARY_PATH"] = f"{stardis_ld_library_path}:{original_ld_library_path}"
    return envs

def set_seed(identifier):
    """Get device indpenedent seed."""
    config_hash = hashlib.sha1(identifier.encode('UTF-8')).hexdigest()
    np.random.seed(int(config_hash[:7], 16))

def get_cam_name(i, min_chars=7):
    """Create formatted camera name from index."""
    format_str = '{:0' + str(min_chars) + 'd}'
    return 'cam_' + format_str.format(i)

def matrix2list(mat):
    """Create list of lists from blender Matrix type."""
    return list(map(list, list(mat)))

def render_views():
    # Get the location of this script to create files relative to it later on
    path_script = os.path.dirname(__file__)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Create dataset from materials.blend as specified in the config file.')
    parser.add_argument('config', help='Path to config file.')
    parser.add_argument('--target-path', help="Path to write the genreated data")
    parser.add_argument('--first-n', type=int, help="Only generate the first n datapoints")
    parser.add_argument('--skip-blender', action='store_true', help="Skip rendering RGB images")
    parser.add_argument('--skip-stardis', action='store_true', help="Skip rendering heat images")
    parser.add_argument('--skip-stardis-reflection', action='store_true', help="Skip rendering heat images")
    parser.add_argument('--restart', action='store_true', help="Restart the rendering process from the beginning")
    parser.add_argument('--subsets', nargs="+", action='extend', help="Only render subsets (accept multiple subset names), default will be all")

    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

    # Clip away .py ending if neccessary and replace / by .
    config_path = args.config[:-3] if args.config[-3:] == '.py' else args.config
    config_module = config_path.replace('/', '.')

    # Import config file
    config = EasyDict(importlib.import_module(config_module).config)

    if args.target_path:
        config.target_path = args.target_path
        print(f"Overriding target_path {config.target_path} -> {args.target_path}")
    
    if args.first_n:
        print(f"Only rendering first {args.first_n} samples, this is specified from the command line argument!")
        finish_first_n = False
    else:
        print("Rendering all samples...")

    # Create a folder for the dataset and save a copy of the configs
    dataset_dir = config.target_path  #os.path.join(path_script, config.target_path)
    os.makedirs(dataset_dir, exist_ok=True)

    path_config_save = os.path.join(dataset_dir, 'config.json')
    with open(path_config_save, 'w+') as config_file:
        json.dump(config, config_file, indent=4)

    # Configure Blender
    if 'resolution' in config:
        bpy.context.scene.render.resolution_x = config.resolution
        bpy.context.scene.render.resolution_y = config.resolution

    if 'samples' in config:
        bpy.context.scene.cycles.samples = config['samples']
        bpy.context.scene.eevee.taa_samples = config['samples'] # TODO: That is not set for some reason...

    image_settings = bpy.context.scene.render.image_settings
    image_settings.file_format = 'PNG'
    file_ending = '.png'
    if 'file_format' in config:
        if config['file_format'] == 'exr':
            image_settings.file_format = 'OPEN_EXR'
            image_settings.color_depth = '32'
            file_ending = '.exr'

    if 'ambient_light_strength' in config:
        bpy.data.worlds['World'].node_tree.nodes['Background'].inputs['Strength'].default_value = config.ambient_light_strength

    # Make sure to use the all GPUs available
    log_path = os.path.join(dataset_dir, 'info.log')
    with open(log_path, 'w+') as log_file:
        blender_preferences = bpy.context.preferences.addons['cycles'].preferences
        blender_preferences.compute_device_type = config['compute_device']
        for devices in blender_preferences.devices:
            for device in devices:
                if device.type == 'CPU':
                    device.use == False
                else:
                    device.use == True
                print("Device '{}' - {}: {}.".format(device.name, device.type, 'enabled' if device.use else 'disabled'))
                log_file.write("Device '{}' - {}: {}.\n".format(device.name, device.type, 'enabled' if device.use else 'disabled'))
        bpy.context.scene.cycles.device = 'GPU'

    # Get reference camera and add 'Cameras' collection
    cam_reference_name = config['cam_name'] if 'cam_name' in config else 'Camera'
    cam_reference = bpy.data.cameras[cam_reference_name]
    cam_collection = bpy.data.collections.new(name='Cameras')
    bpy.context.scene.collection.children.link(bpy.data.collections['Cameras'])

    # Create camera to render with from reference
    # Create and add to 'Cameras' collection
    cam_name = 'cam'
    cam = bpy.data.cameras.new(cam_name)
    cam_object = bpy.data.objects.new(cam_name, cam)
    cam_collection.objects.link(cam_object)

    # Copy properties of 'Camera' object in materials.blend
    cam.angle = config['angle'] if 'angle' in config else cam_reference.angle

    # Calculate the FOV in degrees (for perspective camera)
    fov_rad = 2 * math.atan(cam.sensor_width / (2 * cam.lens))  # Field of View in radians
    fov_deg = math.degrees(fov_rad)  # Convert to degrees

    # Set as camera to render with
    bpy.context.scene.camera = cam_object

    subset_map = {
        'subsets': {}
    }

    for subset in config.subsets:
        subset_map['subsets'][subset['name']] = {
            'name': subset['name'],
            'type': "heat-blender",
            'data': f"./{subset['name']}",
        }
    
    with open(os.path.join(dataset_dir, "subsets.json"), 'w') as subset_file:
        json.dump(subset_map, subset_file, indent=4)
        print("writing subset map to file at", os.path.join(dataset_dir, "subsets.json"))

    subsets_to_render = config.subsets if (args.subsets is None or len(args.subsets) == 0) else [s for s in config.subsets if s['name'] in args.subsets]

    # Create separate sets for training, validation and testing
    for subset in subsets_to_render:
        
        # Init distribution
        distribution = util.instantiate(subset['pose_dist_config'])

        # Init driver sampler
        driver_sampler = util.instantiate(subset['parameter_dist_config'])
        # light_sampler = util.instantiate(subset['light_dist_config'])

        # Init offset to split dataset creation among multiple machines
        offset = 0
        if 'offset' in config:
            offset = config['offset']

        # Check if pose file already exists, if so append, else create a new one
        path_transforms = os.path.join(dataset_dir, subset['name'], config.pose_file_prefix + subset['name'] + '.json')

        # if not making render, then only rewrite jsons, no need to start from the nearset render
        # if os.path.exists(path_transforms):
        #     with open(path_transforms) as pose_file:
        #         transforms = json.load(pose_file)
            
        #     if not args.restart:
        #         offset += len(transforms['frames'])

        #     distribution.sampler.idx = offset
        #     driver_sampler.idx = offset

        # else:
        if True:
            os.makedirs(os.path.join(dataset_dir, subset['name']), exist_ok=True)
            transforms = {}
            transforms['camera_angle_x'] = cam_reference.angle_x
            transforms['frames'] = []
            offset += 0

        path_dir = os.path.join(dataset_dir, subset['name'])

        # #### Thermal propertie set
        # collection_values = subset['collection_values']
        # for i, thermal_item in enumerate(config["collections"]):
        #     obj_name, obj_type, prop_name = thermal_item["object"], thermal_item["type"], thermal_item["property"]
        #     therm_obj = bpy.context.scene.objects[obj_name]
        #     print(bpy.types.Object.stardis_object_properties )
        #     print(bpy.data.objects["Top"].stardis_object_properties)
        #     for prop in therm_obj.stardis_object_properties:
        #         if prop.stardis_object_type == obj_type:
        #             setattr(prop, prop_name, collection_values[i])
        #             break

        # #### Finish

        # Get collection containing the objects to render
        object_collection = bpy.context.scene.collection.children[0]

        if args.first_n:
            print(f"Only rendering first {args.first_n} samples, this is specified from the command line argument!")

            if finish_first_n:
                print(f"already finished first {args.first_n}; skipping...")
                break

        # Create views
        i = 0
        n_samples = max(distribution.sampler.n, driver_sampler.sampler.n)
        print(f"Rendering {n_samples} samples for subset {subset['name']}.")

        while not (distribution.sampler.done() or driver_sampler.sampler.done() ):
            if args.first_n and (i >= args.first_n or finish_first_n):
                finish_first_n = True
                print(f"already finished first {args.first_n}; skipping the rest of this subset")
                break

            print('### RENDERING IMAGE {} / {} ###\n'.format(i + offset, n_samples))

            # Init the random sampler, platform independently
            set_seed(str(config.seed) + subset['name'] + str(i + offset))

            cam_name = get_cam_name(i + offset, math.ceil(np.log10(n_samples)))

            # Translate to a point given by the specified distribution on a sphere of radius 'cam_radius' and point to origin
            cam_object.location = subset['cam_radius'] * distribution()
            cam_direction = -cam_object.location
            cam_rot_quat = cam_direction.to_track_quat('-Z', 'Y')
            cam_object.rotation_euler = cam_rot_quat.to_euler()
            if 'cam_offset' in subset:
                cam_object.location += Vector(subset['cam_offset'])
            bpy.context.view_layer.update()

            # all the objects are in the scene
            objects = bpy.data.objects

            # use smooth shading
            # for f in obj.data.polygons:
            #     f.use_smooth = True

            # Sample driver setting
            param_sample = driver_sampler()
            print('DRIVERS: {}'.format(param_sample))

            # Dict to collect driver samples
            driver_params = {}

            # Set drivers according to sample
            idx = 0
            
            # Create folder for the object
            os.makedirs(path_dir, exist_ok=True)

            # Render object and save to file
            # if type(obj) == bpy.types.Object:
            #     obj.hide_render = False
            # else:
            #     obj.exclude = False
            rgb_filename = os.path.join("rgb", cam_name + file_ending)
            os.makedirs(os.path.join(path_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(path_dir, "heat"), exist_ok=True)
            os.makedirs(os.path.join(path_dir, "reflection"), exist_ok=True)

            bpy.context.scene.render.filepath = os.path.join(path_dir, rgb_filename)

            # Kick off the RGB renderer!!!
            if not args.skip_blender:
                rgb_rendered = os.path.exists(bpy.context.scene.render.filepath)
                if args.restart or (not rgb_rendered):
                    bpy.ops.render.render(write_still=True)

            # if type(obj) == bpy.types.Object:
            #     obj.hide_render = True
            # else:
            #     obj.exclude = True
        
            cam_lookat = cam_object.matrix_world.to_quaternion() @ Vector((0, 0, -1))
            cam_up = cam_object.matrix_world.to_quaternion() @ Vector((0, 1, 0))
            cam_right = cam_object.matrix_world.to_quaternion() @ Vector((1, 0, 0))
            cam_target = cam_object.location + cam_lookat

            # cam.lens_unit = "FOV"!!!
            stardis_samples = config['stardis_samples']
            stardis_out_ht = os.path.join(path_dir, "heat", cam_name + ".ht")
            stardis_out_png = os.path.join(path_dir, "heat", cam_name + ".png")

            reflective_stardis_out_ht = os.path.join(path_dir, "reflection", cam_name + ".ht")
            reflective_stardis_out_png = os.path.join(path_dir, "reflection", cam_name + ".png")

            v2t = lambda v: f"{v.x},{v.y},{v.z}"

            with open(os.path.join(config['template'], "model.txt")) as f:
                template = f.read()

            description = template.format(*param_sample)

            # copy the template file to the output folder
            shutil.copytree(config["template"], os.path.join(path_dir, f'heatmodels'), dirs_exist_ok=True)

            # populate the template with the parameters defined in the config
            with open(os.path.join(path_dir, "heatmodels", f'model-{i:05d}.txt'), 'w+') as f:
                f.write(description)

            cwd = os.getcwd()

            heat_fname = lambda p: os.path.join("..", "heat", os.path.split(p)[1])
            os.chdir(f"{path_dir}/heatmodels")

            stardis_cmd = " ".join((
                f"{STARDIS_EXECUTABLE} -V 3",
                f"-M model-{i:05d}.txt",
                f"-R spp={stardis_samples}:img={config['resolution']}x{config['resolution']}:fov={fov_deg}:pos={v2t(cam_object.location)}:tgt={v2t(cam_target)}:up={v2t(cam_up)}:file={heat_fname(stardis_out_ht)}"
            ))

            with open(f"stardis-command-{i:05d}.sh", "w") as f:
                print("#!/bin/bash", file=f)
                print(stardis_cmd, file=f)

            stardis_rendered = os.path.exists(heat_fname(stardis_out_ht)) and open(heat_fname(stardis_out_ht)).read().strip() != ""
            if (not args.skip_stardis) and (args.restart or (not stardis_rendered)):
                print("Running stardis:")
                print(stardis_cmd)

                proc = subprocess.run(stardis_cmd, shell=True, env=env_with_stardis_ld_library_path(STARDIS_EXECUTABLE))
                if proc.returncode != 0:
                    print("Stardis failed with return code", proc.returncode)

                convert_cmd = [
                    shutil.which("python"),
                    os.path.join(path_script, "ht2exr.py"),
                    "--low-temp", str(config.get("range_low", 260)),
                    "--high-temp", str(config.get("range_high", 330)),
                    "--cmap", config.get("colormap", "inferno"),
                    heat_fname(stardis_out_ht),
                    heat_fname(stardis_out_png),
                ]
                print("Running convert command:")
                print(" ".join(convert_cmd))
                proc = subprocess.run(convert_cmd)
                if proc.returncode != 0:
                    print("Convert failed with return code", proc.returncode)

            else:
                if os.path.exists(heat_fname(stardis_out_ht)):
                    print(f"already existed {stardis_out_ht} and length {len(open(heat_fname(stardis_out_ht)).read().strip())}")

            reflection_fname = lambda p: os.path.join("..", "reflection", os.path.split(p)[1])
                
            reflective_stardis_cmd = " ".join((
                f"{STARDIS_REFLECTIVE_EXECUTABLE} -V 3",
                f"-M model-{i:05d}.txt",
                f"-R spp={stardis_samples}:img={config['resolution']}x{config['resolution']}:fov={fov_deg}:pos={v2t(cam_object.location)}:tgt={v2t(cam_target)}:up={v2t(cam_up)}:file={reflection_fname(reflective_stardis_out_ht)}"
            ))


            stardis_rendered = os.path.exists(reflection_fname(reflective_stardis_out_ht)) and open(reflection_fname(reflective_stardis_out_ht)).read().strip() != ""
            if (not args.skip_stardis_reflection) and (args.restart or (not stardis_rendered)):
                print("Running reflective stardis:")
                print(reflective_stardis_cmd)

                proc = subprocess.run(reflective_stardis_cmd, shell=True, env=env_with_stardis_ld_library_path(STARDIS_REFLECTIVE_EXECUTABLE))
                if proc.returncode != 0:
                    print("Stardis failed with return code", proc.returncode)

                convert_cmd = [
                    shutil.which("python"),
                    os.path.join(path_script, "ht2exr.py"),
                    "--low-temp", str(config.get("range_low", 260)),
                    "--high-temp", str(config.get("range_high", 330)),
                    "--cmap", config.get("colormap", "inferno"),
                    reflection_fname(reflective_stardis_out_ht),
                    reflection_fname(reflective_stardis_out_png),
                ]
                print("Running convert command:")
                print(" ".join(convert_cmd))
                proc = subprocess.run(convert_cmd)
                if proc.returncode != 0:
                    print("Convert failed with return code", proc.returncode)

            else:
                if os.path.exists(reflection_fname(reflective_stardis_out_ht)):
                    print(f"already existed {reflective_stardis_out_ht} and length {len(open(reflection_fname(reflective_stardis_out_ht)).read().strip())}")
               

            os.chdir(cwd)
            
            # Add pose to dict
            transforms['frames'].append({
                'file_path': os.path.join(".", rgb_filename),
                'heat_img_path': os.path.join(".", os.path.relpath(stardis_out_png, start=path_dir)),
                #'rotation': 0, # Commented out as it isn't actually used in NeRF's implementation
                'transform_matrix': matrix2list(cam_object.matrix_world),
                'driver_parameters': driver_params
            })

            # Save pose dict to file at specified intervals
            if 'pose_file_save_interval' in config and (i + 1) % config['pose_file_save_interval'] == 0:
                open_mode = 'w' if args.restart else 'w+'
                with open(path_transforms, open_mode) as pose_file:
                    json.dump(transforms, pose_file, sort_keys=False, indent=4)

            i += 1


        # Write the remaining poses to file
        with open(path_transforms, 'w') as pose_file:
            print("writing transforms to file at", path_transforms)
            json.dump(transforms, pose_file, sort_keys=False, indent=4)
        
        for additional_copy in ["transforms_train.json", "transforms_val.json", "transforms_test.json"]:
            additional_copyname = os.path.join(os.path.dirname(path_transforms), additional_copy)
            if additional_copyname != path_transforms:
                shutil.copyfile(path_transforms, additional_copyname)
    

if __name__ == '__main__':
    render_views()