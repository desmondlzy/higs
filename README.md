# HiGS

Code release for _HiGS: Inverse Radiative Transport for Infrared Scenes with Gaussian Primitives_ (SIGGRAPH 2025).
See our [project page](https://desmondlzy.me/publications/higs/) for more information.

## Installation

Running our code requires an NVIDIA GPU and compiling some CUDA kernels. 
We test our code on the following configuration:

| Component | Version |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| GPU | RTX 6000 Ada (48 GB) |
| CUDA toolkit | 12.4 |
| Python | 3.12 |

If you use a different CUDA version, change the `--index-url` line in `install.sh` to the
matching PyTorch wheel index (e.g. `cu121`).

**Clone with submodules**

```bash
git clone --recursive <repo-url>
cd higs
```

**Run the install script** 

`install.sh` manages all you need to install the necessary dependencies and the nerfstudio extensions

```bash
bash install.sh
conda activate higs
```

Then, you can verify the install by running the following commands and check for errors 

```bash
ns-train heat-2dgs -h | grep higs-heat
```

## Prepare the data

Our dataset is hosted on [Huggingface](https://huggingface.co/datasets/desmondlzy/higs-data).
The data will be **automatically downloaded** into the `data/` directory on the first run of `train.py`.

To use your own data, format it as ours and add the necessary metadata.
A conversion script we wrote for another dataset can be found in `scripts/process_wolfenschiessen.py` for your reference (though it's likely not directly applicable to your data).


## Training

```bash
python entries/train.py --data-path data/higs/radiator --geometry-output-dir <dir>
```

## Export 2DGS PLY files (from the geometry reconstruction step)
```bash
python entries/export_ply.py --geometry-output-dir <dir>
```

## Evaluate re-heating 

Move the emitters to new positions and recompute the radiation.
```bash
python entries/eval_reposition.py --thermal-output-dir <dir>
```

Change the temperature of the emitters and recompute the radiation.
```bash
python entries/eval_temperature.py --thermal-output-dir <dir>
```


## Cite us

If you find our work useful, please consider citing us using the following bibtex entry.

```bibtex
@inproceedings{zhenyuan2025higs,
  title = {{Inverse Radiative Transport for Infrared Scenes with Gaussian Primitives}},
  booktitle = {ACM} {SIGGRAPH Asia} 2025 {Conference} {Proceedings},
  author = {Zhenyuan, Liu and Seshadri, Bharath and Kopanas, George and Bickel, Bernd},
  year = {2025},
  month = dec,
  doi = {10.1145/3757377.3763938},
}
```