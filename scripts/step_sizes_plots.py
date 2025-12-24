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
MODEL_NAME = config['model_name']
N_STEPS = config['n_steps']


def compute_step_sizes(activations: Dict[str, torch.Tensor], hook_name: str) -> Dict[str, torch.Tensor]:
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


def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--data_type', type=str, choices=['image', 'text'], required=True, help='Type of data/model to use (image or text)')
    args = parser.parse_args()

    if args.data_type == 'image':
        SHARED_ID = config['image']['shared_image_id']
        PAIRS_IDS = config['image']['pairs_ids']
    elif args.data_type == 'text':
        SHARED_ID = config['text']['shared_context']
        PAIRS_IDS = config['text']['token_pairs']

    print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")
    
    # Load activations and compute step sizes for each pair
    plot_data = {}
    for pair_ids in PAIRS_IDS:
        activations = load_activations(MODEL_NAME, SHARED_ID, 0, pair_ids, N_STEPS)
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        plot_data[pair_name] = compute_step_sizes(activations, 'resid_post')

    # Generate plot
    output_path = f"./plots/{MODEL_NAME}/step_sizes_resid_post.png"
    generate_interpolation_results_plot(
        data_dict=plot_data,
        suptitle="Resid Post Step Sizes",
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
