import torch
from gsplat.utils import normalized_quat_to_rotmat


def local_to_worlds_from_normalized_quats(
	quats: torch.Tensor,
) -> torch.Tensor:
	"""
	quats: torch.Tensor, (..., 4)

	using scalar-first quaternion convention
	"""
	# (..., 3, 3)
	assert torch.allclose(
		torch.linalg.norm(quats, dim=-1),
		torch.ones_like(quats[..., 0]),
	), f"{torch.linalg.norm(quats, dim=-1)=}"

	local_to_worlds = normalized_quat_to_rotmat(quats)
	return local_to_worlds
