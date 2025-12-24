#!/bin/bash

echo "Generating plots..."
# python ./scripts/spline_plots.py
python ./scripts/step_sizes_plots.py --model_type hooked_transformer --data_type text
python ./scripts/relative_distances_layerwise_plots.py --data_type text --interpolate_only_first_layer
python ./scripts/relative_distances_logits_plots.py --data_type text
# python ./scripts/jacobians_attention.py --data_type text
# python ./scripts/jacobians_mlp.py --data_type text
# python ./scripts/jacobians_full_residual.py --data_type text

echo "All plots completed! Check: ./plots/"
