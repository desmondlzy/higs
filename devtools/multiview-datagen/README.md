# Multiview Datagen

The code is built on top of the data generation pipeline of [NeRF-tex](https://github.com/hbaatz/nerf-tex).

## Requirements

- Blender 4.0+
- Stardis (available from the command line)

## Usage

To begin, open `create_dataset.sh` and change the blender path to your blender executable path.

Then run it:
```bash
bash create_dataset.sh configs/twocubes.blend configs/diffuse_twocubes.py
```

If no error, this script generates RGB rendering of the scene specified in `twocubes.blend` and thermal rendering specified in `template/twocubes/model.txt`. In `configs/diffuse_twocubes.py`, you can specify many rendering related parameters such as spp (`stardis_samples` for thermal and `samples` for RGB), the temperature range for generate thermal images (`range_low` and `range_high`); size of the images (`resolution`)
