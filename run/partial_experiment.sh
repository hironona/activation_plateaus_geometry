#!/bin/bash

# Partial experiment script for activation plateau analyses

echo "Running activation extraction (normal)..."
python ./scripts/interpolate_and_record_activations.py --model_type hooked_transformer --data_type text --interpolate_only_first_layer

echo "Generating plots..."
python ./scripts/step_sizes_plots.py --model_type hooked_transformer --data_type text
python ./scripts/relative_distances_layerwise_plots.py --model_type hooked_transformer --data_type text --interpolate_only_first_layer
python ./scripts/relative_distances_logits_plots.py --model_type hooked_transformer --data_type text

echo "All plots completed! Check: ./plots/"
