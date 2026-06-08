import numpy as np
import torch
from matplotlib import pyplot as plt

from .hemicube_keys import hemicube_keys

def plot_hemicube(renders, background_value=0, only_half=False, **kwargs):
	if isinstance(renders, torch.Tensor):
		renders = {
			key: renders[i].detach().cpu().numpy() for i, key in enumerate(hemicube_keys)
		}
	if isinstance(renders, np.ndarray):
		renders = {
			key: renders[i] for i, key in enumerate(hemicube_keys)
		}

	reso = renders["front"].shape[0]

	is_side_face_square = renders["left"].shape[0] == renders["left"].shape[1]

	ndim = len(renders["front"].shape)
	multi = 3 if is_side_face_square else 2
	match ndim:
		case 2:
			img = np.zeros((reso * multi, reso * multi))
		case 3:
			channels = renders["front"].shape[2]
			img = np.zeros((reso * multi, reso * multi, channels))
		case _:
			raise ValueError(f"invalid number of dimensions; must be 2 or 3, but got {ndim}")

	img.fill(background_value)
	
	_to_numpy = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

	half = reso // 2 if not is_side_face_square else reso 
	img[half:half + reso, half: half + reso] = _to_numpy(renders["front"])
	img[half:reso + half, :half] = _to_numpy(renders["left"])
	img[:half, half:half + reso] = _to_numpy(renders["up"])
	img[half + reso:, half:half + reso] = _to_numpy(renders["down"])
	img[half:half + reso, half + reso:] = _to_numpy(renders["right"])

	if is_side_face_square and only_half:
		img = img[reso // 2 : 2 * reso + reso // 2, reso // 2 :  2 * reso + reso // 2]


	plt.imshow(img, **kwargs)
	plt.axis('off')

	return img
