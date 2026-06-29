#!/bin/bash

# Sweep target_layer_idx from 0..9 and 'logits' to visualize how sensitive
# regions evolve over layers. Source layer is fixed by vis_plateaus/config.yaml
# (currently -2, the input space).

MODEL_PATH="checkpoints/class_spiral/ResNetMLP_epoch500/noise-0.05-multi_seed_5_runs"

echo "Sweeping target layers for plateau visualization..."

for target in 0 1 2 3 4 5 6 7 8 9 logits; do
    echo "=== target_layer_idx=${target} ==="
    uv run vis_plateaus/visualize_plateaus.py \
        --model_type toy_resnet \
        --data_type class_spiral \
        --model_path "${MODEL_PATH}" \
        --target_layer_idx "${target}" \
        --multi_seed
done
