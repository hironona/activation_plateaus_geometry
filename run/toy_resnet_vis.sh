#!/bin/bash

# Partial experiment script for activation plateau analyses

echo "Running activation extraction (normal)..."
uv run ./vis_plots/interpolate_and_record_activations.py --model_type toy_resnet --data_type class_spiral

echo "Generating plots..."
uv run ./vis_plots/step_sizes_plots.py --model_type toy_resnet --data_type class_spiral --multi_seed
uv run ./vis_plots/relative_distances_layerwise_plots.py --model_type toy_resnet --data_type class_spiral --multi_seed
uv run ./vis_plots/relative_distances_layerwise_plots.py --model_type toy_resnet --data_type class_spiral --multi_seed --single_layer 9
uv run ./vis_plots/relative_distances_logits_plots.py --model_type toy_resnet --data_type class_spiral --multi_seed

uv run ./vis_plots/spline_plots.py --model_type toy_resnet --data_type class_spiral --multi_seed

uv run ./vis_plots/jacobians_layerwise.py --model_type toy_resnet --data_type class_spiral --multi_seed

echo "All plots completed! Check: ./plots/"
