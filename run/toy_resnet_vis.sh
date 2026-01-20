#!/bin/bash

# Partial experiment script for activation plateau analyses

echo "Running activation extraction (normal)..."
python ./scripts/interpolate_and_record_activations.py --model_type toy_resnet --data_type class_spiral

echo "Generating plots..."
python ./scripts/step_sizes_plots.py --model_type toy_resnet --data_type class_spiral
python ./scripts/relative_distances_layerwise_plots.py --model_type toy_resnet --data_type class_spiral
python ./scripts/relative_distances_logits_plots.py --model_type toy_resnet --data_type class_spiral
python ./scripts/spline_plots.py --model_type toy_resnet --data_type class_spiral

# python ./scripts/jacobians_layerwise.py --model_type toy_resnet --data_type class_spiral
# python ./scripts/jacobians_full_residual.py --model_type toy_resnet --data_type class_spiral

echo "All plots completed! Check: ./plots/"
