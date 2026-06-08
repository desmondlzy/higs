from pathlib import Path

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.cameras import Cameras
from heatgaussian.nsext.heatblender_dataparser import HeatDataparserOutputs

from PIL import Image  

import os; os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import imageio.v2 as imageio
import numpy as np
import torch


class HeatDataset(InputDataset):
    """A simple dataset class."""
    load_heatimages: bool = True

    def __init__(self, dataparser_outputs: HeatDataparserOutputs, scale_factor = 1):
        super().__init__(dataparser_outputs, scale_factor)
        self._dataparser_outputs = dataparser_outputs
        print(f"Parser outputs type: {type(dataparser_outputs)}")
        print(f"HeatDataset: {len(dataparser_outputs.heatimage_filenames)} heat images found. #reflections: {len(dataparser_outputs.reflection_filenames)}")

    def get_data(self, image_idx: int, image_type="float32"):
        data = super().get_data(image_idx, image_type)
        assert image_type == "float32", "Only float32 is supported for heat images."

        if self.load_heatimages:
            data["heat_image"] = self.get_heatimage_float32(image_idx)

        return data
        
    
    def get_emissivity_reflection_path(self, image_idx: int):
        rgb_filename = self._dataparser_outputs.image_filenames[image_idx]
        emissivity_map_path = rgb_filename.parents[1] / "emissivity" / rgb_filename.with_suffix(".png").name
        ref_reflection_path = rgb_filename.parents[1] / "reflection" / rgb_filename.with_suffix(".exr").name

        if emissivity_map_path.exists() and ref_reflection_path.exists():
            return emissivity_map_path, ref_reflection_path
        else:
            return None, None
        
    
    def get_reflection(self, image_idx: int):
        emissivity_map_path, ref_reflection_path = self.get_emissivity_reflection_path(image_idx)

        if emissivity_map_path is not None or ref_reflection_path is not None:
            ref_emissivity_map = cv2.imread(str(emissivity_map_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)[..., 0] / 255.0
            ref_reflection = cv2.imread(str(ref_reflection_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED) * (1 - ref_emissivity_map)

            return torch.tensor(ref_reflection, dtype=torch.float32, device="cuda").unsqueeze(-1)
        else:
            return None


    def get_heatimage_float32(self, image_idx: int):
        '''
        get the grayscale heat image as a float32 tensor: shape (H, W)
        '''
        heat_filename = self.heatimage_filenames[image_idx]

        if heat_filename.suffix == ".png":
            heat_image = Image.open(heat_filename)
            if self.scale_factor != 1.0:
                width, height = heat_image.size
                newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
                heat_image = heat_image.resize(newsize, resample=Image.Resampling.BILINEAR)
            heat_image = np.array(heat_image, dtype="uint8")
            assert heat_image.shape[2] in [3, 4], f"Heat image shape of {heat_image.shape} is incorrect."
            if len(heat_image.shape) == 2:
                heat_image = heat_image[:, :, None].repeat(3, axis=2)
            
            heat_image = heat_image.astype("float32") / 255.0

        elif heat_filename.suffix == ".exr":
            # heat_image = imageio.imread(heat_filename)
            heat_image = cv2.imread(heat_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
            if len(heat_image.shape) == 2:
                heat_image = heat_image[:, :, None].repeat(3, axis=2)

            assert len(heat_image.shape) == 3
            if self.scale_factor != 1.0:
                heat_image = cv2.resize(heat_image, (0, 0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)

        else:
            raise NotImplementedError(f"Unsupported heat image format {heat_filename.suffix}")

        
        assert len(heat_image.shape) == 3, f"Expected 3 channels in heat image, got {heat_image.shape}"

        heat_image = torch.from_numpy(heat_image)

        return heat_image[..., 0:1]


    @property
    def heatimage_filenames(self):
        return self._dataparser_outputs.heatimage_filenames


    @property
    def reflection_filenames(self):
        return self._dataparser_outputs.reflection_filenames


    def get_metadata(self, data):
        metadata = super().get_metadata(data)

        idx = data["image_idx"]
        camera: Cameras = self._dataparser_outputs.cameras[idx]
        K = camera.get_intrinsics_matrices()
        c2w_34 = camera.camera_to_worlds
        assert c2w_34.shape == (3, 4)
        c2w_44 = torch.eye(4, dtype=c2w_34.dtype, device=c2w_34.device)
        c2w_44[:3, :4] = c2w_34

        rgb_image_filename = self._dataparser_outputs.image_filenames[idx]

        metadata["image_filename"] = rgb_image_filename
        metadata["thermal_filename"] = self._dataparser_outputs.heatimage_filenames[idx]
        metadata["reflection_filename"] = self._dataparser_outputs.reflection_filenames[idx]
        metadata["image_id"] = idx
        metadata["K"] = K
        metadata["camtoworld"] = c2w_44

        # for multi_heat_dataparser
        if "subset_idx" in self._dataparser_outputs.metadata:
            metadata["subset_idx"] = self._dataparser_outputs.metadata['subset_idx'][idx].item()
            metadata["num_subsets"] = self._dataparser_outputs.metadata['num_subsets']
        else:
            metadata["subset_idx"] = 0
            metadata["num_subsets"] = 1

        return metadata
