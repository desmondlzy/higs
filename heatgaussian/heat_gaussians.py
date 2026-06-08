import torch

class HeatGaussians(torch.nn.Module):
	means: torch.nn.Parameter
	quats: torch.nn.Parameter
	scales: torch.nn.Parameter
	thermal_opacities: torch.nn.Parameter
	normals: torch.nn.Parameter
	rgb_colors: torch.nn.Parameter

	specularities: torch.nn.Parameter
	emissivities: torch.nn.Parameter
	temperatures: torch.nn.Parameter

	def __init__(
			self,
			means: torch.Tensor,
			quats: torch.Tensor,
			scales: torch.Tensor,
			thermal_opacities: torch.Tensor,
			normals: torch.Tensor,
			rgb_colors: torch.Tensor,
			specularities: torch.Tensor,
			emissivities: torch.Tensor,
			temperatures: torch.Tensor,
	):
		super().__init__()
		self.means = torch.nn.Parameter(means)
		self.quats = torch.nn.Parameter(quats)
		self.scales = torch.nn.Parameter(scales)
		self.thermal_opacities = torch.nn.Parameter(thermal_opacities)
		self.normals = torch.nn.Parameter(normals)
		self.rgb_colors = torch.nn.Parameter(rgb_colors)
		self.specularities = torch.nn.Parameter(specularities)
		self.emissivities = torch.nn.Parameter(emissivities)
		self.temperatures = torch.nn.Parameter(temperatures)



class HeatGaussianWithDiffuse(HeatGaussians):
	diffuses: torch.nn.Parameter

	def __init__(
			self,
			means: torch.Tensor,
			quats: torch.Tensor,
			scales: torch.Tensor,
			thermal_opacities: torch.Tensor,
			normals: torch.Tensor,
			rgb_colors: torch.Tensor,
			specularities: torch.Tensor,
			emissivities: torch.Tensor,
			temperatures: torch.Tensor,
			diffuses: torch.Tensor,
	):
		super().__init__(
			means=means,
			quats=quats,
			scales=scales,
			thermal_opacities=thermal_opacities,
			normals=normals,
			rgb_colors=rgb_colors,
			specularities=specularities,
			emissivities=emissivities,
			temperatures=temperatures,
		)
		self.diffuses = torch.nn.Parameter(diffuses)
