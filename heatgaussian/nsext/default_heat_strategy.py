from dataclasses import dataclass
from typing import Any, Dict, Union, Tuple

import torch
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import remove, duplicate

from .gs_ops import split


@dataclass
class DefaultHeatStrategy(DefaultStrategy):
	prune_thermal_opa: float = 0.01

	def check_sanity(self, params, optimizers):
		super().check_sanity(params, optimizers)
		
		for key in ["thermal_opacities"]:
			assert key in params, f"{key} is required in params but missing."


	@torch.no_grad()
	def _prune_gs(
		self,
		params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
		optimizers: Dict[str, torch.optim.Optimizer],
		state: Dict[str, Any],
		step: int,
	) -> int:
		# changed the mask
		is_prune = torch.logical_and(
			torch.sigmoid(params["opacities"].flatten()) < self.prune_opa,
			torch.sigmoid(params["thermal_opacities"].flatten()) < self.prune_thermal_opa,
		)

		# until the end of function is copied from the original code
		if step > self.reset_every:
			is_too_big = (
				torch.exp(params["scales"]).max(dim=-1).values
				> self.prune_scale3d * state["scene_scale"]
			)
			# The official code also implements sreen-size pruning but
			# it's actually not being used due to a bug:
			# https://github.com/graphdeco-inria/gaussian-splatting/issues/123
			# We implement it here for completeness but set `refine_scale2d_stop_iter`
			# to 0 by default to disable it.
			if step < self.refine_scale2d_stop_iter:
				is_too_big |= state["radii"] > self.prune_scale2d

			is_prune = is_prune | is_too_big

		n_prune = is_prune.sum().item()
		if n_prune > 0:
			remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

		return n_prune


	@torch.no_grad()
	def _grow_gs(
		self,
		params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
		optimizers: Dict[str, torch.optim.Optimizer],
		state: Dict[str, Any],
		step: int,
	) -> Tuple[int, int]:
		count = state["count"]
		grads = state["grad2d"] / count.clamp_min(1)
		device = grads.device

		is_grad_high = grads > self.grow_grad2d
		is_small = (
			torch.exp(params["scales"]).max(dim=-1).values
			<= self.grow_scale3d * state["scene_scale"]
		)
		is_dupli = is_grad_high & is_small
		n_dupli = is_dupli.sum().item()

		is_large = ~is_small
		is_split = is_grad_high & is_large
		if step < self.refine_scale2d_stop_iter:
			is_split |= state["radii"] > self.grow_scale2d
		n_split = is_split.sum().item()

		# first duplicate
		if n_dupli > 0:
			duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

		# new GSs added by duplication will not be split
		is_split = torch.cat(
			[
				is_split,
				torch.zeros(n_dupli, dtype=torch.bool, device=device),
			]
		)

		# then split
		if n_split > 0:
			split(
				params=params,
				optimizers=optimizers,
				state=state,
				mask=is_split,
				revised_opacity=self.revised_opacity,
			)
		return n_dupli, n_split