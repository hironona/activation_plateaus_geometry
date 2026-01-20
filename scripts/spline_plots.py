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
from utils import load_activations, load_config, generate_interpolation_results_plot

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

    print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")
    
    # Load activations for all token pairs
    all_activations = [load_activations(MODEL_NAME, SHARED_ID, 0, pair_ids, N_STEPS) for pair_ids in PAIRS_IDS]

    if args.model_type == 'hooked_transformer':
        resid_layer_id = 'layer{}_resid_mid'
        mlp_post_layer_id = 'layer{}_mlp_post'
        compute_normalized_hamming_distances = compute_normalized_hamming_distances_hooked_transformer
    elif args.model_type == 'vit':
        resid_layer_id = 'layer{}_resid_mid'
        mlp_post_layer_id = 'layer{}_mlp_post'
        compute_normalized_hamming_distances = compute_normalized_hamming_distances_vit
    elif args.model_type == 'toy_resnet':
        resid_layer_id = 'layer{}_resid_post'
        mlp_post_layer_id = 'layer{}_mlp_out'
        compute_normalized_hamming_distances = compute_normalized_hamming_distances_toy_resnet
    else:
        raise ValueError(f"Currently not supported: {args.model_type}")

    # Get number of layers
    n_layers = len([k for k in all_activations[0].keys() if k.startswith('layer') and (k.endswith('_mlp_post') or k.endswith('_mlp_out'))])

    # Compute hamming distances for each pair
    plot_data = {}
    for pair_idx, activations in enumerate(all_activations):
        layer_data = {}
        for layer_idx in tqdm(range(n_layers), desc=f"Pair {pair_idx + 1}/{len(all_activations)}"):
            mlp_post_layer = activations[mlp_post_layer_id.format(layer_idx)]
            resid_mid_layer = activations[resid_layer_id.format(layer_idx)]
            distances = compute_normalized_hamming_distances(mlp_post_layer, resid_mid_layer)
            if distances:
                layer_data[f"Layer {layer_idx}"] = torch.tensor(distances)

        pair_name = f"{PAIRS_IDS[pair_idx][0]}_{PAIRS_IDS[pair_idx][1]}"
        plot_data[pair_name] = layer_data

    # Generate plot
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
