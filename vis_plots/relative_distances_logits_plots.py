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
sys.path.append('./vis_plots')
from utils import load_activations, load_config, compute_relative_distances, construct_filepath, generate_interpolation_results_plot, get_model_name, get_model_names, aggregate_metric_data

config = load_config()
N_STEPS = config['n_steps']

FREEZING_VARIANTS = [
    ("", "Normal"),
    ("_freeze_attn", "Attention Frozen"),
    ("_freeze_mlp", "MLP Frozen")
]


def compute_logits_data(model_name, shared_id, pairs_ids, layer_to_interpolate, n_steps, freeze_suffix=""):
    """Compute logits relative distances plot_data for a single model checkpoint."""
    plot_data = {}
    for pair_ids in pairs_ids:
        logits = load_activations(model_name, shared_id, layer_to_interpolate, pair_ids, n_steps, freeze_suffix)['logits']
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        plot_data[pair_name] = torch.tensor(compute_relative_distances(logits))
    return plot_data


def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--model_type', type=str, choices=['hooked_transformer', 'vit', 'resnet', 'toy_resnet'], required=True, help='Type of model to use (hooked_transformer or vit or resnet or toy_resnet)')
    parser.add_argument('--data_type', type=str, choices=['text', 'image', 'class_spiral'], required=True, help='Type of data type to use (text or image or class_spiral)')
    parser.add_argument('--multi_seed', action='store_true', help='Aggregate across multiple seeds (toy_resnet only)')

    args = parser.parse_args()

    MODEL_NAME = get_model_name(config, args.model_type)

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
    elif args.model_type in ['toy_resnet']:
        layer_to_interpolate = config['layer_to_interpolate_toy_resnet']
    else:
        layer_to_interpolate = 0

    if args.multi_seed:
        assert args.model_type == 'toy_resnet', "--multi_seed is only supported for toy_resnet"
        model_names = get_model_names(config, args.model_type)
        print(f"Multi-seed mode: {len(model_names)} seeds | Steps: {N_STEPS}")
        output_dir = f"./plots/{MODEL_NAME}"

        for freeze_suffix, variant_name in FREEZING_VARIANTS:
            # Check if data exists for first seed
            if not os.path.exists(construct_filepath(model_names[0], SHARED_ID, layer_to_interpolate, PAIRS_IDS[0], N_STEPS, freeze_suffix)):
                print(f"\nSkipping {variant_name} (data not found)")
                continue

            print(f"\nProcessing {variant_name}...")

            all_seeds_data = []
            for idx, mn in enumerate(model_names):
                print(f"  Seed {idx + 1}/{len(model_names)}: {mn}")
                all_seeds_data.append(
                    compute_logits_data(mn, SHARED_ID, PAIRS_IDS, layer_to_interpolate, N_STEPS, freeze_suffix)
                )

            mean_data, std_data = aggregate_metric_data(all_seeds_data)

            output_path = f"{output_dir}/relative_distances_logits{freeze_suffix}_layer{layer_to_interpolate}_interpolation.png"
            generate_interpolation_results_plot(
                data_dict=mean_data,
                suptitle=f"Logits Relative Distances ({variant_name}), Interpolated at Layer {layer_to_interpolate}",
                ylabel="Relative Distance to A (0) vs B (1)",
                output_path=output_path,
                n_steps=N_STEPS,
                shared_id=SHARED_ID,
                pairs_ids=PAIRS_IDS,
                alpha_range=[0, 1],
                skip_interpolation_layer=False,
                std_dict=std_data
            )
    else:
        print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")

        os.makedirs(f"./plots/{MODEL_NAME}", exist_ok=True)

        for freeze_suffix, variant_name in FREEZING_VARIANTS:
            # Check if data exists for this variant
            if not os.path.exists(construct_filepath(MODEL_NAME, SHARED_ID, layer_to_interpolate, PAIRS_IDS[0], N_STEPS, freeze_suffix)):
                print(f"\nSkipping {variant_name} (data not found)")
                continue

            print(f"\nProcessing {variant_name}...")

            plot_data = compute_logits_data(MODEL_NAME, SHARED_ID, PAIRS_IDS, layer_to_interpolate, N_STEPS, freeze_suffix)

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