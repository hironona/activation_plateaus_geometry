#!/bin/bash

# Partial experiment script for activation plateau analyses

# echo "Running activation extraction (normal)..."
# python ./scripts/interpolate_and_record_activations.py --model_type vit --data_type image

echo "Generating plots..."
python ./scripts/step_sizes_plots.py --model_type vit --data_type image
python ./scripts/relative_distances_layerwise_plots.py --model_type vit --data_type image
python ./scripts/relative_distances_logits_plots.py --model_type vit --data_type image

echo "All plots completed! Check: ./plots/"
