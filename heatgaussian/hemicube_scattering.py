from functools import cache
from dataclasses import dataclass

import torch
from torch.nn.functional import softplus
from tqdm import tqdm

from .rasterize_2dgs_normal_filtering import rasterize_2dgs_normal_filtering
from .pairwise_unit_vectors import pairwise_unit_vectors
from .heat_gaussians import HeatGaussians
from .hemicube_camera_matrices import hemicube_camera_matrices_lightweight
from .spherical_gaussian_radiance_cache import SphericalGaussianRadianceCache
from .double_spherical_gaussian_radiance_cache import DoubleSphericalGaussianRadianceCache
from .spherical_harmonic_radiance_cache import SphericalHarmonicRadianceCache
from .canonical_camera_up_local	import canonical_camera_up_local
from .local_to_worlds_from_normalized_quats import local_to_worlds_from_normalized_quats
from .hemicube_keys import hemicube_keys
from .hemicube_solid_angles import hemicube_solid_angles
from .hemicube_cosines import hemicube_cosines
from .hemicube_xyzs import hemicube_xyzs
from .reflect import reflect


@dataclass
class HemicubeConstants:
	hemicube_resolution: int
	xyzs: torch.Tensor  # wi
	lobes_axes: torch.Tensor  # wi reflected about n
	solid_angles: torch.Tensor  # d omega
	cosines: torch.Tensor  # cos(theta) for each wi

@cache
def get_constants(reso):
	hc_in_xyzs =  { k: torch.tensor(v.copy()).cuda() for k, v in hemicube_xyzs(reso=reso, pad=True).items() }
	hc_solid_angles = { k: torch.tensor(v.copy()).cuda() for k, v in  hemicube_solid_angles(reso, pad=True).items() }
	hc_cosines = { k: torch.tensor(v.copy()).cuda() for k, v in hemicube_cosines(reso, pad=True).items() }

	zaxis = torch.tensor([[[0.0, 0.0, 1.0]]]).cuda()

	win_xyzs = torch.vstack([hc_in_xyzs[k].reshape(-1, 3) for k in hemicube_keys])
	solid_angles = torch.vstack([hc_solid_angles[k].reshape(-1, 1) for k in hemicube_keys])
	cosines = torch.vstack([hc_cosines[k].reshape(-1, 1) for k in hemicube_keys])

	# local space
	lobes_axes = reflect(win_xyzs, zaxis.view(1, 3))


	return HemicubeConstants(
		hemicube_resolution=reso,
		xyzs=win_xyzs,
		solid_angles=solid_angles,
		cosines=cosines,
		lobes_axes=lobes_axes,
	)



def hemicube_scattering(
	gaussians: HeatGaussians,
	radiance_cache: SphericalGaussianRadianceCache,
	eye: torch.Tensor,
	max_batch_size: int,
	near_plane: float,
	hemicube_resolution: int = 64,
	brdf_model: str = 'ggx',  # 'ggx' or 'mirror_diffuse'
):
	"""
	compute the scattering for each gaussian
	"""
	n_gaussians = gaussians.means.shape[0]
	scatters = torch.zeros_like(gaussians.temperatures)

	gaussian_to_worlds = local_to_worlds_from_normalized_quats(gaussians.quats / torch.norm(gaussians.quats, dim=1, keepdim=True))
	assert gaussian_to_worlds.shape == (n_gaussians, 3, 3)
	# world_to_gaussians = gaussian_to_worlds.clone().transpose(1, 2)
	ups_world = gaussian_to_worlds @ canonical_camera_up_local.to(gaussian_to_worlds.device)

	indices_groups = [
		torch.arange(s, min(s + max_batch_size, n_gaussians)) 
		for s in range(0, n_gaussians, max_batch_size) 
	]

	for indice_group in indices_groups:
		indice_group = indice_group.cuda()
		assert indice_group.shape[0] <= max_batch_size, f"indice_group size {indice_group.shape[0]} exceeds max_batch_size {max_batch_size}"
		scatter_at_indices, *_ = hemicube_scattering_for_indices(
			gaussians=gaussians,
			radiance_cache=radiance_cache,
			gaus_indices=indice_group,
			eyes=eye,
			hemicube_ups=ups_world[indice_group],
			near_plane=near_plane,
			hemicube_resolution=hemicube_resolution,
			brdf_model=brdf_model,
		)

		scatters[indice_group] = scatter_at_indices

	return scatters



def hemicube_scattering_for_indices(
	gaussians: HeatGaussians,
	radiance_cache: SphericalGaussianRadianceCache,
	gaus_indices: torch.Tensor,
	eyes: torch.Tensor,
	hemicube_ups: torch.Tensor,
	near_plane: float,
	hemicube_resolution: int,
	brdf_model: str,
):
	n_gaussians = gaussians.means.shape[0]
	batch_size = gaus_indices.shape[0]

	consts = get_constants(int(hemicube_resolution))

	# (batch_size, 3)
	if eyes.shape == (3,) or eyes.shape == (1, 3):
		camera_eyes = eyes.view(1, 3).expand(batch_size, 3)
	elif eyes.shape == (batch_size, 3):
		camera_eyes = eyes
	else:
		raise ValueError(f"eyes shape {eyes.shape} not supported")

	# (batch_size, 3)
	gaussians_to_cam = camera_eyes.view(batch_size, 3) - gaussians.means[gaus_indices]
	gaussians_to_cam = torch.nn.functional.normalize(gaussians_to_cam, dim=-1)

	assert torch.isfinite(gaussians_to_cam).all(), f"gaussians_to_cam not finite: {gaussians_to_cam}"
	assert gaussians_to_cam.shape == (batch_size, 3), f"gaussians_to_cam shape {gaussians_to_cam.shape}"

	# radiances_to_camera = radiance_cache.partial_evaluate(gaussians_to_cam, gaus_indices).reshape(batch_size, 3)

	# (batch_size, n_gaussians, 3)
	other_gaussians_to_hemicubes = pairwise_unit_vectors(
		froms=gaussians.means,
		tos=gaussians.means[gaus_indices],
	)
	assert torch.isfinite(other_gaussians_to_hemicubes).all()

	# if the norm of a vector is zero, it means the gaussian is itself, so we set it to [0 0 1] to avoid nan
	if type(radiance_cache) is SphericalHarmonicRadianceCache:
		other_gaussians_to_hemicubes = torch.where(
			other_gaussians_to_hemicubes.norm(dim=-1, keepdim=True) < 1e-6,
			torch.tensor([[0.0, 0.0, 1.0]], device=other_gaussians_to_hemicubes.device),
			other_gaussians_to_hemicubes,
		)

	radiances_to_hemicube = (
		radiance_cache(other_gaussians_to_hemicubes, gaussians.normals)
		+ softplus(gaussians.temperatures) * torch.sigmoid(gaussians.emissivities)
	)
	assert torch.isfinite(radiances_to_hemicube).all()
	# # radiances to itself is undefined
	# radiances_to_hemicube[gaus_index] = 0.0

	# flip normals and gaussians_to_cam different hemisphere in world space
	flip_signs = torch.sum(gaussians.normals[gaus_indices] * gaussians_to_cam, dim=-1, keepdim=True) < 0.0
	oriented_normals = torch.where(
		flip_signs,
		-gaussians.normals[gaus_indices],
		gaussians.normals[gaus_indices],
	)

	gaus_hc_viewmats, gaus_Ks = hemicube_camera_matrices_lightweight(
		eye=gaussians.means[gaus_indices],
		look_dir=-oriented_normals,  # the normals need to be flip to get the correct matrices, dont know why...
		up=hemicube_ups,
		resolution=hemicube_resolution,
	)

	# broadcast
	radiances_to_hemicube_batch = torch.tile(
		radiances_to_hemicube.view(batch_size, 1, n_gaussians, 3),
		(1, 5, 1, 1)
	).reshape(batch_size * 5, n_gaussians, 3)

	assert torch.isfinite(radiances_to_hemicube).all()
	assert torch.isfinite(gaus_hc_viewmats).all()
	hemicube_in_radiance, *_ = rasterize_2dgs_normal_filtering(
		means=gaussians.means,
		quats=gaussians.quats,
		scales=torch.exp(gaussians.scales),
		opacities=torch.sigmoid(gaussians.thermal_opacities).squeeze(),
		colors=radiances_to_hemicube_batch,
		normals_world=gaussians.normals,
		viewmats=gaus_hc_viewmats.reshape(batch_size * 5, 4, 4),
		Ks=gaus_Ks.reshape(batch_size * 5, 3, 3),
		width=hemicube_resolution,
		height=hemicube_resolution,
		normal_transparent=False,
		near_plane=near_plane,
	)
	assert torch.isfinite(hemicube_in_radiance).all()
	assert hemicube_in_radiance.shape == (batch_size * 5, hemicube_resolution, hemicube_resolution, 3)

	# (batch_size, n_pixels, 3)
	hemicube_tensor = hemicube_in_radiance.reshape(batch_size, -1, 3) * consts.solid_angles.unsqueeze(0) * consts.cosines.unsqueeze(0)

	# the out_dirs are in local space of the gaussians
	w2g = gaus_hc_viewmats[torch.arange(batch_size), 0, :3, :3]
	assert w2g.shape == (batch_size, 3, 3), f"w2g shape {w2g.shape} != expected {(batch_size, 3, 3)}"

	out_dirs = (w2g @ gaussians_to_cam.view(batch_size, 3, 1)).squeeze(-1)
	assert torch.isfinite(out_dirs).all()
	assert out_dirs.shape == (batch_size, 3)

	match brdf_model:
		case 'ndf_sg' | 'ggx':   # the 'ggx' is for compatibility with the old code
			# interactive graph: https://www.desmos.com/calculator/yldtkvljsh
			roughnesses = -torch.sigmoid(gaussians.specularities[gaus_indices]) + 1
			amps = 1.0

			# size: (batch_size, n_pixels)
			cosine_x_minus_1  = torch.sum(out_dirs.view(batch_size, 1, 3) * consts.lobes_axes.view(1, -1, 3), axis=-1) - 1
			assert cosine_x_minus_1.shape == (batch_size, hemicube_resolution * hemicube_resolution * 5), cosine_x_minus_1.shape

			brdf_vals = amps / torch.pi / roughnesses ** 2  * torch.exp((2 * cosine_x_minus_1) / (roughnesses ** 2))

			assert torch.isfinite(brdf_vals).all()

		case 'cook_torrance_ggx':
			brdf_vals = brdf_cook_torrance(
				reso=hemicube_resolution,
				batch_size=batch_size,
				gaussians=gaussians,
				gaus_indices=gaus_indices,
				out_dirs=out_dirs,
				ndf="ggx",
			)


		case 'cook_torrance_sg':
			brdf_vals = brdf_cook_torrance(
				reso=hemicube_resolution,
				batch_size=batch_size,
				gaussians=gaussians,
				gaus_indices=gaus_indices,
				out_dirs=out_dirs,
				ndf="sg",
			)


		case 'mirror_diffuse':
			brdf_vals = brdf_mirror_diffuse(
				reso=hemicube_resolution,
				batch_size=batch_size,
				gaussians=gaussians,
				gaus_indices=gaus_indices,
				w2g=w2g,
				gaussians_to_cam=gaussians_to_cam,
			)

		case 'lambertian':
			brdf_vals = brdf_lambertian(
				reso=hemicube_resolution,
				batch_size=batch_size,
				device=gaussians.specularities.device,
			)

		case _:
			raise ValueError(f"Unknown BRDF model: {brdf_model}")

	assert brdf_vals.shape == (batch_size, hemicube_resolution * hemicube_resolution * 5), f"brdf_vals shape {brdf_vals.shape} != expected {(batch_size, hemicube_resolution * hemicube_resolution)}"

	# size: (batch_size, n_pixels, n_channels)
	products = hemicube_tensor * brdf_vals.unsqueeze(-1)
	
	emissivity = torch.sigmoid(gaussians.emissivities[gaus_indices]).view(batch_size, 1)

	scatter = torch.sum(products, axis=-2) * (1 - emissivity)

	info = {
		"w2g": w2g,
		"gaussians_to_cam": gaussians_to_cam,
	}

	torch.cuda.empty_cache()

	return scatter.float(), info


def ndf_ggx(
	wm: torch.Tensor,
	roughness: torch.Tensor,
):
	cos2theta = torch.square(wm[..., 2])
	sin2theta = torch.clamp(1.0 - cos2theta, min=1e-6, max=1.0)
	tan2theta = sin2theta / cos2theta	
	cos4theta = torch.square(cos2theta)

	e = tan2theta / (roughness ** 2)

	D = 1 / (torch.pi * (roughness ** 2) * cos4theta * torch.square(1 + e))

	return D


def lambda_masking(
	w: torch.Tensor,  # [batch_size, n_pixels, 3]
	roughness: torch.Tensor,  # [batch_size, 1]
):
	# https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory#eq:microfacet-tr-lambda
	cos2theta = torch.square(w[..., 2])
	sin2theta = torch.clamp(1.0 - cos2theta, min=1e-6, max=1.0)
	tan2theta = sin2theta / cos2theta
	alpha2s = torch.square(roughness)

	return 0.5 * (torch.sqrt(1 + alpha2s * tan2theta) - 1.0)


def fresnel_dielectric(
	costheta_i: torch.Tensor,  # [batch_size, n_pixels]
	eta: float,
):
	# compute costheta_t using snell's law
	sin2theta_i = 1 - torch.clamp(torch.square(costheta_i), min=0.0, max=1.0)
	sin2theta_t = sin2theta_i / (eta ** 2)
	overflow = sin2theta_t >= 1.0
	costheta_t = torch.sqrt(torch.clamp(1.0 - sin2theta_t, min=0.0, max=1.0))

	r_parl = (eta * costheta_i - costheta_t) / (eta * costheta_i + costheta_t)
	r_perp = (costheta_i - eta * costheta_t) / (costheta_i + eta * costheta_t)
	avg_sqr = 0.5 * (torch.square(r_parl) + torch.square(r_perp))
	return torch.where(overflow, torch.ones_like(costheta_i), avg_sqr)


def fresnel_schlick(
	cosine: torch.Tensor,  # [batch_size, n_pixels]
	F0: float = 0.04,
):
	return F0 + (1 - F0) * torch.pow(1 - torch.clamp(cosine, min=0.0, max=1.0), 5)



def ndf_beckmann(
	wm: torch.Tensor,  # [batch_size, n_pixels, 3]
	roughness: torch.Tensor, # [batch_size, 1] 
):
	# normals are [0 0 1]
	cos2theta = torch.square(wm[..., 2])
	cos4theta = torch.square(cos2theta)
	tan2theta = torch.clamp(1.0 - cos2theta, min=1e-6, max=1.0) / cos2theta
	alpha2s = torch.square(roughness)
	return torch.exp(
		-torch.square(tan2theta) / alpha2s
	) / (torch.pi * alpha2s * cos4theta)



def brdf_lambertian(reso, batch_size, device="cuda"):
	consts = get_constants(reso)

	f_diffuse = torch.full((batch_size, reso * reso * 5), 1 / torch.pi, device=device)
	return f_diffuse


@torch.compile
def brdf_cook_torrance(reso, batch_size, gaussians, gaus_indices, out_dirs, ndf: str):
	"""
	ndf: normal distribution function, 'ggx' or 'sg'
	"""
	consts = get_constants(reso)

	assert torch.isfinite(out_dirs).all()
	assert out_dirs.shape == (batch_size, 3), f"out_dirs shape {out_dirs.shape} != expected {(batch_size, 3)}"

	diffuses = torch.sigmoid(gaussians.diffuses[gaus_indices]).view(batch_size, 1)  # [batch_size, 1]
	specularities = torch.sigmoid(gaussians.specularities[gaus_indices]).view(batch_size, 1)  # [batch_size, 1]

	roughnesses = -specularities + 1.0  # [batch_size, 1]

	# Lambertian diffuse
	f_diffuse = torch.full((batch_size, reso * reso * 5), 1 / torch.pi, device=gaussians.specularities.device)
	
	in_dirs = consts.xyzs

	mid_dirs = in_dirs.view(1, -1, 3) + out_dirs.view(batch_size, 1, 3)  # [batch_size, n_pixels, 3]
	mid_dirs_norms = mid_dirs.norm(dim=-1, keepdim=True)
	mid_dirs = torch.where(
		mid_dirs_norms < 1e-6,
		torch.tensor([[0.0, 0.0, 1.0]], device=mid_dirs.device),
		mid_dirs / mid_dirs_norms,
	)

	assert mid_dirs.shape == (batch_size, reso * reso * 5, 3), f"mid_dirs shape {mid_dirs.shape} != expected {(batch_size, reso * reso * 5, 3)}"


	#### Microfacet
	# D * G * F / (4 * cos(theta_i) * cos(theta_o))

	#### Fresnel: Schlick approximation
	F0 = 0.8
	# wo_dot_wm = torch.sum(out_dirs.view(batch_size, 1, 3) * mid_dirs, dim=-1)  # [batch_size, n_pixels]
	wi_dot_wm = torch.sum(in_dirs.view(1, -1, 3) * mid_dirs, dim=-1)  # [1, n_pixels]

	F = fresnel_schlick(
		cosine=torch.clamp(wi_dot_wm, min=0.0, max=1.0),
		F0=F0,
	)

	# F = fresnel_dielectric(
	# 	costheta_i=torch.clamp(wi_dot_wm, min=0.0, max=1.0),
	# 	eta=1.5,
	# )


	#### Distribution term
	match ndf:
		case "ggx":
			D = ndf_ggx(
				wm=mid_dirs,
				roughness=roughnesses.view(batch_size, 1),
			)
			assert torch.isfinite(D).all()
			assert D.shape == (batch_size, reso * reso * 5), f"D shape {D.shape} != expected {(batch_size, reso * reso * 5)}"

			# G shape: # [batch_size, n_pixels]

		case "sg":
			roughnesses = -torch.sigmoid(gaussians.specularities[gaus_indices]) + 1
			amps = 1.0

			# size: (batch_size, n_pixels)
			cosine_x_minus_1  = torch.sum(out_dirs.view(batch_size, 1, 3) * consts.lobes_axes.view(1, -1, 3), axis=-1) - 1
			assert cosine_x_minus_1.shape == (batch_size, reso * reso * 5), cosine_x_minus_1.shape

			D = amps / torch.pi / roughnesses ** 2  * torch.exp((2 * cosine_x_minus_1) / (roughnesses ** 2))
			assert torch.isfinite(D).all()


	# Geometric term
	lambda_out_dirs = lambda_masking(out_dirs.view(batch_size, 1, 3), roughnesses.view(batch_size, 1))
	lambda_in_dirs = lambda_masking(in_dirs.view(1, -1, 3), roughnesses.view(batch_size, 1))
	assert lambda_out_dirs.shape == (batch_size, 1), f"lambda_out_dirs shape {lambda_out_dirs.shape} != expected {(batch_size, 1)}"
	assert lambda_in_dirs.shape == (batch_size, reso * reso * 5), f"lambda_in_dirs shape {lambda_in_dirs.shape} != expected {(batch_size, reso * reso * 5)}"

	G = 1 / (
		+ lambda_out_dirs
		+ lambda_in_dirs
		+ 1 
	)

	diffuse_term = f_diffuse * diffuses
	assert torch.isfinite(diffuse_term).all(), "Diffuse term contains NaN or Inf"
	assert diffuse_term.shape == (batch_size, reso * reso * 5), f"diffuse_term shape {diffuse_term.shape} != expected {(batch_size, reso * reso * 5)}"

	# G = torch.clamp(G, min=1.0, max=1.0)
	# F = torch.clamp(F, min=1.0, max=1.0)

	nomi = D * G * F

	cosines_wo = out_dirs[..., 2]  # [batch_size, 1]
	assert torch.isfinite(nomi).all(), "Nomi term contains NaN or Inf"
	assert nomi.shape == (batch_size, reso * reso * 5), f"nomi shape {nomi.shape} != expected {(batch_size, reso * reso * 5)}"

	denom = torch.clamp(
		4 * consts.cosines.view(1, reso * reso * 5) * 
		cosines_wo.view(batch_size, 1), min=1e-6)

	assert denom.shape == (batch_size, reso * reso * 5), f"denom shape {denom.shape} != expected {(batch_size, reso * reso * 5)}"
	specular_term = (1 - diffuses) * nomi / denom
	assert torch.isfinite(specular_term).all(), "Specular term contains NaN or Inf"
	assert specular_term.shape == (batch_size, reso * reso * 5), f"specular_term shape {specular_term.shape} != expected {(batch_size, reso * reso * 5)}"

	return diffuse_term + specular_term



def brdf_mirror_diffuse(reso, batch_size, gaussians, gaus_indices, w2g, gaussians_to_cam):
	consts = get_constants(reso)
	out_dirs = (w2g @ gaussians_to_cam.view(batch_size, 3, 1)).squeeze(-1)
	assert torch.isfinite(out_dirs).all()
	assert out_dirs.shape == (batch_size, 3)

	f_diffuse = torch.full((batch_size, reso * reso * 5), 1 / torch.pi, device=gaussians.specularities.device)
	f_mirror = (
		consts.lobes_axes.view(1, -1, 3) * out_dirs.view(batch_size, 1, 3)).sum(dim=-1)

	specularities = torch.sigmoid(gaussians.specularities[gaus_indices]).view(batch_size, 1)
	f_values = f_diffuse + f_mirror * specularities.view(batch_size, 1)

	assert torch.isfinite(f_values).all()
	assert f_values.shape == (batch_size, reso * reso * 5), f"f_values shape {f_values.shape} != expected {(batch_size, reso * reso)}"

	return f_values

