#!/usr/bin/env python3
"""
Logits Relative Distances Analysis Script

Generates logits relative distance plots for all freezing variants:
normal, freeze_attn, and freeze_mlp.
"""

import argparse
import torch
import os
import sys
sys.path.append('./scripts')
from utils import load_activations, load_config, compute_relative_distances, construct_filepath, generate_interpolation_results_plot

config = load_config()
N_STEPS = config['n_steps']

FREEZING_VARIANTS = [
    ("", "Normal"),
    ("_freeze_attn", "Attention Frozen"),
    ("_freeze_mlp", "MLP Frozen")
]


def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--model_type', type=str, choices=['hooked_transformer', 'vit', 'resnet', 'toy_resnet'], required=True, help='Type of model to use (hooked_transformer or vit or resnet or toy_resnet)')
    parser.add_argument('--data_type', type=str, choices=['text', 'image', 'class_spiral'], required=True, help='Type of data type to use (text or image or class_spiral)')

    args = parser.parse_args()

    MODEL_NAME = config['model_names'][args.model_type]

    if args.data_type == 'image':
        SHARED_ID = config['image']['shared_image_id']
        PAIRS_IDS = config['image']['pairs_ids']
    elif args.data_type == 'text':
        SHARED_ID = config['text']['shared_context']
        PAIRS_IDS = config['text']['token_pairs']
    elif args.data_type == 'class_spiral':
        SHARED_ID = ''
        PAIRS_IDS = [[f'{num}' for num in pair] for pair in config['class_spiral']['pairs']]

    # Assumption: vision models need to deal with low-level features in early layers, thus interpolating at layer 0 does not show clear plateaus
    if args.model_type in ['vit']:
        layer_to_interpolate = 3
    elif args.model_type in ['resnet']:
        layer_to_interpolate = 1
    else:
        layer_to_interpolate = 0

    print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")

    os.makedirs(f"./plots/{MODEL_NAME}", exist_ok=True)

    for freeze_suffix, variant_name in FREEZING_VARIANTS:
        # Check if data exists for this variant
        if not os.path.exists(construct_filepath(MODEL_NAME, SHARED_ID, layer_to_interpolate, PAIRS_IDS[0], N_STEPS, freeze_suffix)):
            print(f"\nSkipping {variant_name} (data not found)")
            continue

        print(f"\nProcessing {variant_name}...")

        # Load logits and compute relative distances
        plot_data = {}
        for pair_ids in PAIRS_IDS:
            logits = load_activations(MODEL_NAME, SHARED_ID, layer_to_interpolate, pair_ids, N_STEPS, freeze_suffix)['logits']
            pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
            plot_data[pair_name] = torch.tensor(compute_relative_distances(logits))

        # Generate plot
        output_path = f"./plots/{MODEL_NAME}/relative_distances_logits{freeze_suffix}_layer{layer_to_interpolate}_interpolation.png"
        generate_interpolation_results_plot(
            data_dict=plot_data,
            suptitle=f"Logits Relative Distances ({variant_name}), Interpolated at Layer {layer_to_interpolate}",
            ylabel="Relative Distance to A (0) vs B (1)",
            output_path=output_path,
            n_steps=N_STEPS,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
            alpha_range=[0, 1],
            skip_interpolation_layer=False
        )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()