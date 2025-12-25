#!/bin/bash

echo "Generating plots..."
python ./scripts/step_sizes_plots.py --model_type vit --data_type image
python ./scripts/relative_distances_layerwise_plots.py --model_type vit --data_type image --interpolate_only_first_layer
python ./scripts/relative_distances_logits_plots.py --model_type vit --data_type image
echo "All plots completed! Check: ./plots/"
