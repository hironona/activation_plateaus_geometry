#!/usr/bin/env python3
"""
Step Sizes Analysis Script

This script generates step size plots (L2 norm of differences between consecutive steps) 
for all variants: normal, freeze_attn, and freeze_mlp.
"""

import argparse
import torch
import os
import sys
from typing import Dict, List
sys.path.append('./scripts')
from utils import load_config, load_activations, generate_interpolation_results_plot

config = load_config()
N_STEPS = config['n_steps']


def compute_step_sizes_hooked_transformer(activations: Dict[str, torch.Tensor], hook_name: str) -> Dict[str, torch.Tensor]:
    """Compute step sizes (L2 norm of differences between consecutive steps) for each layer."""
    step_sizes = {}

    for key, layer_activations in activations.items():
        if key.startswith('layer') and key.endswith(f'_{hook_name}'):
            last_token = layer_activations[:, -1, :]  # [n_steps, hidden_dim]
            step_diffs = last_token[1:] - last_token[:-1]  # [n_steps-1, hidden_dim]
            step_norms = torch.norm(step_diffs, p=2, dim=1)  # [n_steps-1]

            layer_idx = int(key.split('_')[0].replace('layer', ''))
            step_sizes[f"Layer {layer_idx}"] = step_norms

    return step_sizes

def compute_step_sizes_vit(activations: Dict[str, torch.Tensor], hook_name: str) -> Dict[str, torch.Tensor]:
    """Compute step sizes (L2 norm of differences between consecutive steps) for each layer."""
    step_sizes = {}

    for key, layer_activations in activations.items():
        if key.startswith('layer') and key.endswith(f'_{hook_name}'):
            cls_token = layer_activations[:, 0, :]  # [n_steps, hidden_dim]
            step_diffs = cls_token[1:] - cls_token[:-1]  # [n_steps-1, hidden_dim]
            step_norms = torch.norm(step_diffs, p=2, dim=1)  # [n_steps-1]

            layer_idx = int(key.split('_')[0].replace('layer', ''))
            step_sizes[f"Layer {layer_idx}"] = step_norms
    return step_sizes


def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--model_type', type=str, choices=['hooked_transformer', 'vit', 'resnet'], required=True, help='Type of model to use (hooked_transformer or vit)')
    parser.add_argument('--data_type', type=str, choices=['text', 'image'], required=True, help='Type of data type to use (text or image)')

    args = parser.parse_args()

    MODEL_NAME = config['model_names'][args.model_type]

    if args.data_type == 'image':
        SHARED_ID = config['image']['shared_image_id']
        PAIRS_IDS = config['image']['pairs_ids']
    elif args.data_type == 'text':
        SHARED_ID = config['text']['shared_context']
        PAIRS_IDS = config['text']['token_pairs']

    # Assumption: vision models need to deal with low-level features in early layers, thus interpolating at layer 0 does not show clear plateaus
    if args.model_type in ['vit']:
        layer_to_interpolate = 3
    elif args.model_type in ['resnet']:
        layer_to_interpolate = 1
    else:
        layer_to_interpolate = 0

    print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")
    
    # Load activations and compute step sizes for each pair
    plot_data = {}
    for pair_ids in PAIRS_IDS:
        activations = load_activations(MODEL_NAME, SHARED_ID, layer_to_interpolate, pair_ids, N_STEPS)
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        if args.model_type == 'hooked_transformer':
            plot_data[pair_name] = compute_step_sizes_hooked_transformer(activations, 'resid_post')
        elif args.model_type == 'vit':
            plot_data[pair_name] = compute_step_sizes_vit(activations, 'resid_post')

    # Generate plot
    output_path = f"./plots/{MODEL_NAME}/step_sizes_resid_post_layer{layer_to_interpolate}_interpolation.png"
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle=f"Resid Post Step Sizes (Interpolated at Layer {layer_to_interpolate})",
        ylabel="L2 Norm of Step Difference",
        output_path=output_path,
        n_steps=N_STEPS - 1,
        shared_id=SHARED_ID,
        pairs_ids=PAIRS_IDS,
        alpha_range=[0, 1]
    )
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()
