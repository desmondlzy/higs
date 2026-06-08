#!/bin/sh
# Create (render) the dataset from the source .blend file. This takes a while.

# blender $1 --background -noaudio --python create_dataset.py -- $2 

# ${@:2} is the rest of the arguments passed to this script
# source: https://stackoverflow.com/questions/9057387/process-all-arguments-except-the-first-one-in-a-bash-script

blender $1 --background -noaudio --python create_dataset.py -- ${@:2}