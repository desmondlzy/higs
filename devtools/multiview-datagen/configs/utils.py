import numpy as np
import hashlib 

def set_seed(identifier):
    """Get device indpenedent seed."""
    config_hash = hashlib.sha1(str(identifier).encode('UTF-8')).hexdigest()
    np.random.seed(int(config_hash[:7], 16))


def unit_square_to_sphere(u, v):
    z = 2 * u - 1
    r = np.sqrt(1 - z ** 2)
    phi = 2 * np.pi * v
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z

def quaternion_from_between_two_vectors(v1, v2):
    rot_axis = np.cross(v1, v2)
    norms = np.linalg.norm(rot_axis)

    if norms < 1e-8:
        return 0, np.array([0, 0, 0])

    rot_axis = -rot_axis / norms
    rot_angle = np.arccos(np.dot(v1, v2))

    return np.array([rot_angle, *rot_axis])
