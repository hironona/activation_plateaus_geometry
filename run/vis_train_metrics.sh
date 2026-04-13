#!/bin/bash

echo "Visualizing training metrics..."

uv run train/plot_metrics.py --checkpoint_dir "checkpoints/class_spiral/ResNetMLP_epoch500/noise-0.075-multi_seed_5_runs"