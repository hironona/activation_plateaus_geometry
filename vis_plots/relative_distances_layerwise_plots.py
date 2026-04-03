#!/usr/bin/env python3
"""
Layerwise Relative Distances Analysis Script

Generates three types of relative distance plots:
1. Interpolate in layer 0, record in all subsequent layers
2. Interpolate in each layer, record in the last layer only
3. Interpolate in layer i, record in layer i+N for various N values
"""

import argparse
import torch
import os
from typing import Dict, List
import sys
sys.path.append('./vis_plots')
from utils import load_activations, load_config, get_n_layers, compute_relative_distances, generate_interpolation_results_plot, construct_filepath, get_model_name, get_model_names, aggregate_metric_data

config = load_config()
N_STEPS = config['n_steps']


def variant1_compute(shared_id, pairs_ids, model_name, layer_to_interpolate):
    """Variant 1: Compute data - Interpolate in layer 0, record in all subsequent layers."""
    plot_data = {}
    for pair_ids in pairs_ids:
        activations = load_activations(model_name, shared_id, layer_to_interpolate, pair_ids, N_STEPS)
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        layer_dict = {}
        for key, layer_activations in activations.items():
            if key.startswith('layer') and key.endswith('_resid_post'):
                layer_idx = int(key.split('_')[0].replace('layer', ''))
                layer_dict[f"Layer {layer_idx}"] = torch.tensor(compute_relative_distances(layer_activations))
        plot_data[pair_name] = layer_dict
    return plot_data


def variant1_plot(plot_data, shared_id, pairs_ids, output_dir, layer_to_interpolate, std_dict=None):
    """Variant 1: Plot - Interpolate in layer 0, record in all subsequent layers."""
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle=f"Relative Distances (Interpolate in Layer {layer_to_interpolate})",
        ylabel="Relative Distance to Token A (0) vs Token B (1)",
        output_path=f"{output_dir}/relative_distances_layerwise_layer{layer_to_interpolate}_interpolation.png",
        n_steps=N_STEPS,
        shared_id=shared_id,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        skip_interpolation_layer=True,
        std_dict=std_dict
    )


def variant1_interpolate_layer0(shared_id, pairs_ids, model_name, layer_to_interpolate):
    """Variant 1: Interpolate in layer 0, record in all subsequent layers."""
    print(f"\nVariant 1: Interpolate in layer {layer_to_interpolate}, record in all layers")
    plot_data = variant1_compute(shared_id, pairs_ids, model_name, layer_to_interpolate)
    variant1_plot(plot_data, shared_id, pairs_ids, f"./plots/{model_name}", layer_to_interpolate)


def variant2_compute(n_layers, shared_id, pairs_ids, model_name):
    """Variant 2: Compute data - Interpolate in each layer, record in last layer only."""
    last_layer_key = f'layer{n_layers - 1}_resid_post'
    plot_data = {}
    for pair_idx, pair_ids in enumerate(pairs_ids):
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        layer_dict = {}
        for interpolation_layer in range(n_layers - 1):
            activations = load_activations(model_name, shared_id, interpolation_layer, pair_ids, N_STEPS)
            distances = compute_relative_distances(activations[last_layer_key])
            layer_dict[f"Layer {interpolation_layer}"] = torch.tensor(distances)
        plot_data[pair_name] = layer_dict
    return plot_data


def variant2_plot(plot_data, shared_id, pairs_ids, output_dir, std_dict=None):
    """Variant 2: Plot - Interpolate in each layer, record in last layer only."""
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle="Relative Distances (Record in Last Layer)",
        ylabel="Relative Distance to Token A (0) vs Token B (1)",
        output_path=f"{output_dir}/relative_distances_layerwise_last_layer_recording.png",
        n_steps=N_STEPS,
        shared_id=shared_id,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        skip_interpolation_layer=True,
        std_dict=std_dict
    )


def variant2_record_last_layer(n_layers: int, shared_id, pairs_ids, model_name):
    """Variant 2: Interpolate in each layer, record in last layer only."""
    print("\nVariant 2: Interpolate in each layer, record in last layer")
    plot_data = variant2_compute(n_layers, shared_id, pairs_ids, model_name)
    variant2_plot(plot_data, shared_id, pairs_ids, f"./plots/{model_name}")


def variant3_compute(n_layers, shared_id, pairs_ids, model_name, N):
    """Variant 3: Compute data - Interpolate in layer i, record in layer i+N."""
    plot_data = {}
    for pair_ids in pairs_ids:
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        layer_dict = {}
        for interpolation_layer in range(n_layers - N):
            activations = load_activations(model_name, shared_id, interpolation_layer, pair_ids, N_STEPS)
            target_layer_key = f'layer{interpolation_layer + N}_resid_post'
            distances = compute_relative_distances(activations[target_layer_key])
            layer_dict[f"Layer {interpolation_layer}"] = torch.tensor(distances)
        plot_data[pair_name] = layer_dict
    return plot_data


def variant3_plot(plot_data, shared_id, pairs_ids, output_dir, N, std_dict=None):
    """Variant 3: Plot - Interpolate in layer i, record in layer i+N."""
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle=f"Relative Distances (N={N})",
        ylabel="Relative Distance to Token A (0) vs Token B (1)",
        output_path=f"{output_dir}/relative_distances_layerwise_N{N}.png",
        n_steps=N_STEPS,
        shared_id=shared_id,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        skip_interpolation_layer=True,
        std_dict=std_dict
    )


def single_layer_compute(shared_id, pairs_ids, model_name, layer_to_interpolate, target_layer):
    """Compute relative distances for a single target layer only.

    Reuses variant1_compute and filters to target_layer, returning single-tensor format.
    """
    full_data = variant1_compute(shared_id, pairs_ids, model_name, layer_to_interpolate)
    target_key = f"Layer {target_layer}"
    filtered = {}
    for pair_name, layer_dict in full_data.items():
        if target_key not in layer_dict:
            raise KeyError(f"Layer {target_layer} not found in activations. Available: {list(layer_dict.keys())}")
        filtered[pair_name] = layer_dict[target_key]
    return filtered


def single_layer_plot(plot_data, shared_id, pairs_ids, output_dir, layer_to_interpolate, target_layer, std_dict=None):
    """Plot relative distances for a single target layer."""
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle=f"Relative Distances — Layer {target_layer} (Interpolate in Layer {layer_to_interpolate})",
        ylabel="Relative Distance to Token A (0) vs Token B (1)",
        output_path=f"{output_dir}/relative_distances_layer{target_layer}_interpolate{layer_to_interpolate}.png",
        n_steps=N_STEPS,
        shared_id=shared_id,
        pairs_ids=pairs_ids,
        alpha_range=[0, 1],
        std_dict=std_dict
    )


def variant3_record_layer_plus_n(n_layers: int, shared_id, pairs_ids, model_name: str):
    """Variant 3: Interpolate in layer i, record in layer i+N for various N."""
    print("\nVariant 3: Interpolate in layer i, record in layer i+N")

    for N in [1, 4, 8, 16, 24]:
        print(f"  Processing N={N}...")
        plot_data = variant3_compute(n_layers, shared_id, pairs_ids, model_name, N)
        variant3_plot(plot_data, shared_id, pairs_ids, f"./plots/{model_name}", N)


def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--model_type', type=str, choices=['hooked_transformer', 'vit', 'resnet', 'toy_resnet'], required=True, help='Type of model to use (hooked_transformer or vit or resnet or toy_resnet)')
    parser.add_argument('--data_type', type=str, choices=['text', 'image', 'class_spiral'], required=True, help='Type of data type to use (text or image or class_spiral)')
    parser.add_argument('--interpolate_only_first_layer', action='store_true', help='Only interpolate at the first layer for less data')
    parser.add_argument('--multi_seed', action='store_true', help='Aggregate across multiple seeds (toy_resnet only)')
    parser.add_argument('--single_layer', type=int, default=None, help='Plot relative distances for a single target layer only')

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

        if args.single_layer is not None:
            print(f"\nSingle layer mode: Layer {args.single_layer}, interpolate in layer {layer_to_interpolate}")
            all_sl = []
            for idx, mn in enumerate(model_names):
                print(f"  Seed {idx + 1}/{len(model_names)}: {mn}")
                all_sl.append(single_layer_compute(SHARED_ID, PAIRS_IDS, mn, layer_to_interpolate, args.single_layer))
            mean_sl, std_sl = aggregate_metric_data(all_sl)
            os.makedirs(output_dir, exist_ok=True)
            single_layer_plot(mean_sl, SHARED_ID, PAIRS_IDS, output_dir, layer_to_interpolate, args.single_layer, std_dict=std_sl)
            print(f"\nPlot saved to: {output_dir}/")
            print("\n=== Complete ===")
            return

        # Variant 1
        print(f"\nVariant 1: Interpolate in layer {layer_to_interpolate}, record in all layers")
        all_v1 = []
        for idx, mn in enumerate(model_names):
            print(f"  Seed {idx + 1}/{len(model_names)}: {mn}")
            all_v1.append(variant1_compute(SHARED_ID, PAIRS_IDS, mn, layer_to_interpolate))
        mean_v1, std_v1 = aggregate_metric_data(all_v1)
        variant1_plot(mean_v1, SHARED_ID, PAIRS_IDS, output_dir, layer_to_interpolate, std_dict=std_v1)

        if not args.interpolate_only_first_layer:
            # Get n_layers from first seed
            activations = load_activations(model_names[0], SHARED_ID, layer_to_interpolate, PAIRS_IDS[0], N_STEPS)
            n_layers = get_n_layers(activations)

            # Variant 2
            print("\nVariant 2: Interpolate in each layer, record in last layer")
            all_v2 = []
            for idx, mn in enumerate(model_names):
                print(f"  Seed {idx + 1}/{len(model_names)}: {mn}")
                all_v2.append(variant2_compute(n_layers, SHARED_ID, PAIRS_IDS, mn))
            mean_v2, std_v2 = aggregate_metric_data(all_v2)
            variant2_plot(mean_v2, SHARED_ID, PAIRS_IDS, output_dir, std_dict=std_v2)

            # Variant 3
            print("\nVariant 3: Interpolate in layer i, record in layer i+N")
            for N in [1, 4, 8, 16, 24]:
                print(f"  Processing N={N}...")
                all_v3 = []
                for mn in model_names:
                    all_v3.append(variant3_compute(n_layers, SHARED_ID, PAIRS_IDS, mn, N))
                mean_v3, std_v3 = aggregate_metric_data(all_v3)
                variant3_plot(mean_v3, SHARED_ID, PAIRS_IDS, output_dir, N, std_dict=std_v3)
    else:
        print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")

        # Check if data exists
        if not os.path.exists(construct_filepath(MODEL_NAME, SHARED_ID, layer_to_interpolate, PAIRS_IDS[0], N_STEPS)):
            print("Data not found, skipping")
            return

        os.makedirs(f"./plots/{MODEL_NAME}", exist_ok=True)

        if args.single_layer is not None:
            print(f"\nSingle layer mode: Layer {args.single_layer}, interpolate in layer {layer_to_interpolate}")
            plot_data = single_layer_compute(SHARED_ID, PAIRS_IDS, MODEL_NAME, layer_to_interpolate, args.single_layer)
            single_layer_plot(plot_data, SHARED_ID, PAIRS_IDS, f"./plots/{MODEL_NAME}", layer_to_interpolate, args.single_layer)
            print("\n=== Complete ===")
            return

        activations = load_activations(MODEL_NAME, SHARED_ID, layer_to_interpolate, PAIRS_IDS[0], N_STEPS)
        n_layers = get_n_layers(activations)
        print(f"Model has {n_layers} layers")

        variant1_interpolate_layer0(SHARED_ID, PAIRS_IDS, MODEL_NAME, layer_to_interpolate)

        if not args.interpolate_only_first_layer:
            variant2_record_last_layer(n_layers, SHARED_ID, PAIRS_IDS, MODEL_NAME)
            variant3_record_layer_plus_n(n_layers, SHARED_ID, PAIRS_IDS, MODEL_NAME)

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
