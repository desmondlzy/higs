#!/usr/bin/env python3
"""
Full pipeline script that runs both geometry and thermal reconstruction.

This script chains together:
1. Geometry reconstruction (training thermal 2DGS)
2. Thermal reconstruction (training material properties with radiosity)
"""

from dataclasses import dataclass
from typing import Literal
import tyro
from pathlib import Path

# Import the training functions
from geometry_reconstruction import GeometryReconstructionArgs, geometry_reconstruction
from thermal_reconstruction import ThermalReconstructionArgs, thermal_reconstruction

from heatgaussian.download_data import maybe_download_data


@dataclass
class TrainArgs:
    """Configuration for full reconstruction pipeline."""

    # Data parameters
    data_path: str = "data/higs/radiator"
    """Path to the dataset"""

    geometry_output_dir: str | None = None
    """Output folder name"""

    # Experiment parameters
    experiment_name: str = "higs"
    """Experiment name for geometry reconstruction"""

    # Near plane (used in both stages)
    near_plane: float = 0.3
    """Near clipping plane for rendering"""

    # Geometry reconstruction parameters
    max_gaussian_num: int = 100000
    """Maximum number of Gaussians"""

    random_scale: float = 2.0
    """Random initialization scale for Gaussians"""

    prune_opa: float = 0.01
    """Opacity threshold for pruning RGB Gaussians"""

    prune_thermal_opa: float = 0.01
    """Opacity threshold for pruning thermal Gaussians"""

    grow_scale3d: float = 0.02
    """3D scale threshold for growing Gaussians"""

    grow_grad2d: float = 0.0012
    """2D gradient threshold for growing Gaussians"""

    use_tonemapping: bool = True
    """Enable tonemapping for thermal rendering"""

    num_geometry_iters: int = 30000
    """Maximum number of iterations for geometry reconstruction"""

    # Thermal reconstruction parameters
    num_thermal_iters: int = 300000
    """Total number of training iterations for thermal reconstruction"""

    emissivity_inits: float = 0.5
    """Initial emissivity value"""

    specularity_inits: float = 0.4
    """Initial specularity value"""

    temperature_lr: float = 1e-3
    """Learning rate for temperature"""

    specularity_lr: float = 1e-2
    """Learning rate for specularity"""

    emissivities_lr: float = 1e-4
    """Learning rate for emissivities"""

    residue_weight: float = 1e1
    """Weight for radiosity residue loss"""

    brdf_model: Literal["cook_torrance_ggx", "cook_torrance_sg"] = "cook_torrance_ggx"
    """BRDF model: 'cook_torrance_ggx' or 'cook_torrance_sg'"""

    batch_size: int = 64
    """Batch size for radiosity computation"""

    image_scale_factor: float = 1.0
    """Downsample images before caching (e.g. 0.5 = half res, 0.25 = quarter res). Reduces RAM ~4x per halving."""

    viewer: bool = False
    """Enable the viser viewer during geometry reconstruction"""


def train():
    """Run the complete reconstruction pipeline."""
    args: TrainArgs = tyro.cli(TrainArgs)

    maybe_download_data(args.data_path)

    print("=" * 80)
    print("FULL RECONSTRUCTION PIPELINE")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Geometry output folder: {args.geometry_output_dir}")
    print(f"Near plane: {args.near_plane}")
    print("=" * 80)

    # ========================================================================
    # STAGE 1: Geometry Reconstruction
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: GEOMETRY RECONSTRUCTION")
    print("=" * 80)

    geometry_args = GeometryReconstructionArgs(
        data_path=args.data_path,
        output_dir=args.geometry_output_dir,
        experiment_name=args.experiment_name,
        max_gaussian_num=args.max_gaussian_num,
        dist_loss=True,
        random_scale=args.random_scale,
        prune_opa=args.prune_opa,
        prune_thermal_opa=args.prune_thermal_opa,
        grow_scale3d=args.grow_scale3d,
        grow_grad2d=args.grow_grad2d,
        use_tonemapping=args.use_tonemapping,
        auto_scale_poses=False,
        center_method="none",
        orientation_method="none",
        num_iters=args.num_geometry_iters,
        steps_per_save=2000,
        steps_per_eval_image=100,
        steps_per_eval_all_images=1000,
        image_scale_factor=args.image_scale_factor,
        viewer=args.viewer,
    )

    # Run geometry reconstruction
    geometry_output_dir = geometry_reconstruction(geometry_args)

    # ========================================================================
    # STAGE 2: Thermal Reconstruction
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: THERMAL RECONSTRUCTION")
    print("=" * 80)

    thermal_args = ThermalReconstructionArgs(
        geometry_output_path=str(geometry_output_dir),
        experiment_name=args.experiment_name,
        near_plane=args.near_plane,
        num_iters=args.num_thermal_iters,
        emissivity_inits=args.emissivity_inits,
        specularity_inits=args.specularity_inits,
        temperature_lr=args.temperature_lr,
        specularity_lr=args.specularity_lr,
        emissivities_lr=args.emissivities_lr,
        residue_weight=args.residue_weight,
        save_checkpoint_every=25000,
        brdf_model=args.brdf_model,
        batch_size=args.batch_size,
        save_scene_video=False,
    )

    # Run thermal reconstruction
    print(f"\nLoading geometry from: {geometry_output_dir}")
    thermal_output_dir = thermal_reconstruction(thermal_args)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"Geometry output: {geometry_output_dir}")
    print(f"Thermal output: {thermal_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    train()
