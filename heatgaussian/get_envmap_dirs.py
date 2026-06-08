import numpy as np
import torch


# inverse the mapping from https://github.com/NVlabs/nvdiffrec/blob/dad3249af8ede96c7dd72c30328272117fabb710/render/light.py#L22
def get_envmap_dirs(res: tuple[int] = (256, 512)) -> tuple[torch.Tensor, torch.Tensor]:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0, 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0, 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )
    d_theta, d_phi = np.pi / res[0], 2 * np.pi / res[1]

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]

    # get solid angles
    solid_angles = ((costheta - torch.cos(gy * np.pi + d_theta)) * d_phi)[..., None]  # [H, W, 1]
    print(f"solid_angles_sum error: {solid_angles.sum() - 4 * np.pi}")

    return solid_angles, reflvec

