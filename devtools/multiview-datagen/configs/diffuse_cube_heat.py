import numpy as np
from .utils import unit_square_to_sphere, quaternion_from_between_two_vectors

from .utils import set_seed; set_seed(0)
GRID_X, GRID_Y = 8, 8

def make_olat_subset(index, grid_x, grid_y):
    # map a grid of points to the upper hemisphere
    radius = 6

    _, du = np.linspace(0, 1, grid_x, endpoint=True, retstep=True)
    # _, dv = np.linspace(0, 1, grid_y, endpoint=True, retstep=True)
    gu, du = np.linspace(0 + du / 2, 1 - du / 2, grid_x, endpoint=True, retstep=True)
    gv, dv = np.linspace(0, 1, grid_y, endpoint=False, retstep=True)

    lu, lv = np.stack(np.meshgrid(gu, gv), axis=-1).reshape(-1, 2).T
    # cam_u, cam_v = np.stack(np.meshgrid(gu, gv), axis=-1).reshape(-1, 2).T
    cam_u, cam_v = np.stack(np.meshgrid(gu, gv + dv / 2), axis=-1).reshape(-1, 2).T

    lx, ly, lz = unit_square_to_sphere(lu, lv)
    cam_x, cam_y, cam_z = unit_square_to_sphere(cam_u, cam_v)

    # print("cam_x, cam_y, cam_z")
    # print(np.column_stack((cam_x, cam_y, cam_z)))

    # print("lx, ly, lz")
    # print(np.column_stack((lx, ly, lz)))

    num_cam = cam_x.size

    init_light = np.array([0, 0, -1])

    target_light = np.array([-lx[index], -ly[index], -lz[index]])
    rot_angle, *rot_axis = quaternion_from_between_two_vectors(init_light, target_light)


    return {
        'name': f'olat_{index}',                                # Subset name
        'cam_radius': radius,                                # Camera radius, used to scale position with. Camera is set to point to the origin.
        'pose_dist_config': {                           # Distribution for the camera positions
            'module': 'distribution.Constant',   # In this case, the upper hemisphere. For more options see data/distributions.py
            'constants': [[cam_x[i], cam_y[i], cam_z[i]] for i in range(num_cam)]
        },
        'parameter_dist_config': {                      # Distribution for the material and light parameters
            'module': 'distribution.Constant',     # In this case, the upper hemisphere. For more options see data/distributions.py
            'constants': np.zeros((num_cam, 1)).tolist(),
        }
    }


def make_test_set():
    n = 60
    ts = np.linspace(0, np.pi * 2, n, endpoint=False)
    zs = (np.sin(4 * ts)) * 0.9
    rs = np.sqrt(1 - zs ** 2)
    xs = np.cos(ts) * rs
    ys = np.sin(ts) * rs

    init_light = np.array([0, 0, -1])
    target_lights = np.column_stack((-xs, -ys, -zs))
    rot_angle_axes = np.array([quaternion_from_between_two_vectors(init_light, target_light) for target_light in target_lights])
    assert rot_angle_axes.shape == (n, 4)

    return {
        'name': 'test',
        'cam_radius': 6,
        'pose_dist_config': {
            'module': 'distribution.Constant',
            'constants': [[np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3] for _ in range(n)]
        },
        'light_dist_config': {
            'module': 'distribution.Constant',
            'constants': rot_angle_axes.tolist(),
        },
        'parameter_dist_config': {
            'module': 'distribution.Constant',
            'constants': [[0] for _ in range(n)]
        }
    }

n = 20
temp_range = np.linspace(270, 310, n)
config = {
    'compute_device': 'ONEAPI',                              # Device used for ray tracing, can be 'OPTIX', 'CUDA' or 'CPU'
    'seed': 0,
    'subsets': [                                            # List of dataset subsets to create
        {
            'name': 'train',
            'cam_radius': 6,
            'pose_dist_config': {
                'module': 'distribution.Constant',
                'constants': [[np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3] for _ in range(n)]
            },
            'parameter_dist_config': {
                'module': 'distribution.Constant',
                'constants': [[temp_range[i], 0.9] for i in range(n)]
            }
        }
    ],
    'resolution': 128,
    'samples': 512,
    'stardis_samples': 64,
    # 'light': 'Directional',                                 # Light source
    'template': 'templates/cube',
    'pose_file_prefix': 'transforms_',                      # Prefix of the pose file
    'pose_file_save_interval': 10,                          # Number of generated samples after whicht to save poses
    'target_path': f'datasets/cube-bc'              # Path where the generated files are stored to
}