#!/bin/bash

echo "Visualizing activation plateaus..."

for noise in "0.00"; do
    uv run vis_plateaus/visualize_plateaus.py --model_type toy_resnet --data_type class_spiral --model_path "checkpoints/class_spiral/ResNetMLP_epoch500/noise-${noise}-multi_seed_5_runs" --multi_seed
done
