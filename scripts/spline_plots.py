#!/usr/bin/env python3
"""
Spline Analysis Script

This script computes Hamming distances of spline codes normalized by step size.
For each layer and step, compute Hamming distance between consecutive spline codes (mlp_post > 0),
normalized by the L2 step size in resid_mid space.
"""

import argparse
import os
import sys
import torch
from tqdm import tqdm

sys.path.append('./scripts')
from utils import load_activations, load_config, generate_interpolation_results_plot, get_model_name, get_model_names, aggregate_metric_data

config = load_config()
N_STEPS = config['n_steps']


def compute_normalized_hamming_distances_hooked_transformer(mlp_post_activations, resid_mid_activations):
    """Compute Hamming distances of spline codes normalized by resid_mid step sizes."""
    last_token_mlp = mlp_post_activations[:, -1, :]  # [n_steps, d_mlp]
    spline_codes = [(last_token_mlp[step] > 0).float() for step in range(last_token_mlp.shape[0])]
    normalized_distances = []
    for step in range(len(spline_codes) - 1):
        hamming_dist = torch.sum(spline_codes[step] != spline_codes[step + 1]).item()
        step_size = torch.norm(resid_mid_activations[step + 1, -1, :] - resid_mid_activations[step, -1, :]).item()
        normalized_distances.append(hamming_dist / step_size if step_size > 0 else 0.0)
    return normalized_distances

def compute_normalized_hamming_distances_vit(mlp_post_activations, resid_mid_activations):
    """Compute Hamming distances of spline codes normalized by resid_mid step sizes."""
    last_token_mlp = mlp_post_activations[:, 0, :]  # [n_steps, d_mlp]
    spline_codes = [(last_token_mlp[step] > 0).float() for step in range(last_token_mlp.shape[0])]
    normalized_distances = []
    for step in range(len(spline_codes) - 1):
        hamming_dist = torch.sum(spline_codes[step] != spline_codes[step + 1]).item()
        step_size = torch.norm(resid_mid_activations[step + 1, 0, :] - resid_mid_activations[step, 0, :]).item()
        normalized_distances.append(hamming_dist / step_size if step_size > 0 else 0.0)
    return normalized_distances

def compute_normalized_hamming_distances_toy_resnet(mlp_post_activations, resid_mid_activations):
    """Compute Hamming distances of spline codes normalized by resid_mid step sizes."""
    last_token_mlp = mlp_post_activations  # [n_steps, d_mlp]
    spline_codes = [(last_token_mlp[step] > 0).float() for step in range(last_token_mlp.shape[0])]
    normalized_distances = []
    for step in range(len(spline_codes) - 1):
        hamming_dist = torch.sum(spline_codes[step] != spline_codes[step + 1]).item()
        step_size = torch.norm(resid_mid_activations[step + 1] - resid_mid_activations[step]).item()
        normalized_distances.append(hamming_dist / step_size if step_size > 0 else 0.0)
    return normalized_distances


def get_layer_ids(model_type):
    """Get layer ID templates and compute function for the model type."""
    if model_type == 'hooked_transformer':
        return 'layer{}_resid_mid', 'layer{}_mlp_post', compute_normalized_hamming_distances_hooked_transformer
    elif model_type == 'vit':
        return 'layer{}_resid_mid', 'layer{}_mlp_post', compute_normalized_hamming_distances_vit
    elif model_type == 'toy_resnet':
        return 'layer{}_resid_post', 'layer{}_mlp_out', compute_normalized_hamming_distances_toy_resnet
    else:
        raise ValueError(f"Currently not supported: {model_type}")


def compute_spline_plot_data(model_name, model_type, shared_id, pairs_ids, n_steps):
    """Compute spline hamming distance plot_data for a single model checkpoint."""
    resid_layer_id, mlp_post_layer_id, compute_fn = get_layer_ids(model_type)

    all_activations = [load_activations(model_name, shared_id, 0, pair_ids, n_steps) for pair_ids in pairs_ids]
    n_layers = len([k for k in all_activations[0].keys() if k.startswith('layer') and (k.endswith('_mlp_post') or k.endswith('_mlp_out'))])

    plot_data = {}
    for pair_idx, activations in enumerate(all_activations):
        layer_data = {}
        for layer_idx in tqdm(range(n_layers), desc=f"Pair {pair_idx + 1}/{len(all_activations)}"):
            mlp_post_layer = activations[mlp_post_layer_id.format(layer_idx)]
            resid_mid_layer = activations[resid_layer_id.format(layer_idx)]
            distances = compute_fn(mlp_post_layer, resid_mid_layer)
            if distances:
                layer_data[f"Layer {layer_idx}"] = torch.tensor(distances)

        pair_name = f"{pairs_ids[pair_idx][0]}_{pairs_ids[pair_idx][1]}"
        plot_data[pair_name] = layer_data
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

    if args.multi_seed:
        assert args.model_type == 'toy_resnet', "--multi_seed is only supported for toy_resnet"
        model_names = get_model_names(config, args.model_type)
        print(f"Multi-seed mode: {len(model_names)} seeds | Steps: {N_STEPS}")

        all_seeds_data = []
        for idx, mn in enumerate(model_names):
            print(f"  Seed {idx + 1}/{len(model_names)}: {mn}")
            all_seeds_data.append(
                compute_spline_plot_data(mn, args.model_type, SHARED_ID, PAIRS_IDS, N_STEPS)
            )

        mean_data, std_data = aggregate_metric_data(all_seeds_data)

        output_path = f"./plots/{MODEL_NAME}/spline_hamming_distances.png"
        generate_interpolation_results_plot(
            data_dict=mean_data,
            suptitle="Normalized Hamming Distances of Spline Codes",
            ylabel="Normalized Hamming Distance",
            output_path=output_path,
            n_steps=N_STEPS - 1,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
            alpha_range=[0, 1],
            skip_interpolation_layer=False,
            std_dict=std_data,
            y_floor=0
        )
        print(f"\nPlot saved to: {output_path}")
    else:
        print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")

        plot_data = compute_spline_plot_data(MODEL_NAME, args.model_type, SHARED_ID, PAIRS_IDS, N_STEPS)

        output_path = f"./plots/{MODEL_NAME}/spline_hamming_distances.png"
        generate_interpolation_results_plot(
            data_dict=plot_data,
            suptitle="Normalized Hamming Distances of Spline Codes",
            ylabel="Normalized Hamming Distance",
            output_path=output_path,
            n_steps=N_STEPS - 1,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
            alpha_range=[0, 1],
            skip_interpolation_layer=False
        )
        print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    main()
