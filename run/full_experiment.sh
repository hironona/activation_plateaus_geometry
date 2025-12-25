#!/bin/bash

# Full experiment script for activation plateau analyses

echo "Running activation extraction (normal)..."
python ./scripts/interpolate_and_record_activations.py --model_type hooked_transformer --data_type text

echo "Running activation extraction (attention frozen)..."
python ./scripts/interpolate_and_record_activations.py --model_type hooked_transformer --data_type text --freeze_attention

echo "Running activation extraction (MLP frozen)..."
python ./scripts/interpolate_and_record_activations.py --model_type hooked_transformer --data_type text --freeze_mlp

echo "Generating plots..."
python ./scripts/spline_plots.py --model_type hooked_transformer --data_type text
python ./scripts/step_sizes_plots.py --model_type hooked_transformer --data_type text
python ./scripts/relative_distances_layerwise_plots.py --model_type hooked_transformer --data_type text
python ./scripts/relative_distances_logits_plots.py --model_type hooked_transformer --data_type text
python ./scripts/jacobians_attention.py --model_type hooked_transformer --data_type text
python ./scripts/jacobians_mlp.py --model_type hooked_transformer --data_type text
python ./scripts/jacobians_layerwise.py --model_type hooked_transformer --data_type text
python ./scripts/jacobians_full_residual.py --model_type hooked_transformer --data_type text

echo "All plots completed! Check: ./plots/"
