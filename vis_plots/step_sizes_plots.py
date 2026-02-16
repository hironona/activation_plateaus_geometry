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
sys.path.append('./vis_plots')
from utils import load_config, load_activations, generate_interpolation_results_plot, get_model_name, get_model_names, aggregate_metric_data

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

def compute_step_sizes_mlp(activations: Dict[str, torch.Tensor], hook_name: str) -> Dict[str, torch.Tensor]:
    """Compute step sizes (L2 norm of differences between consecutive steps) for each layer."""
    step_sizes = {}

    for key, layer_activations in activations.items():
        if key.startswith('layer') and key.endswith(f'_{hook_name}'):
            step_diffs = layer_activations[1:] - layer_activations[:-1]  # [n_steps-1, hidden_dim]
            step_norms = torch.norm(step_diffs, p=2, dim=1)  # [n_steps-1]

            layer_idx = int(key.split('_')[0].replace('layer', ''))
            step_sizes[f"Layer {layer_idx}"] = step_norms
    return step_sizes

def compute_step_sizes_for_model(model_name, model_type, shared_id, pairs_ids, layer_to_interpolate, n_steps):
    """Compute step sizes plot_data for a single model checkpoint."""
    plot_data = {}
    for pair_ids in pairs_ids:
        activations = load_activations(model_name, shared_id, layer_to_interpolate, pair_ids, n_steps)
        pair_name = f"{pair_ids[0]}_{pair_ids[1]}"
        if model_type == 'hooked_transformer':
            plot_data[pair_name] = compute_step_sizes_hooked_transformer(activations, 'resid_post')
        elif model_type == 'vit':
            plot_data[pair_name] = compute_step_sizes_vit(activations, 'resid_post')
        elif model_type == 'toy_resnet':
            plot_data[pair_name] = compute_step_sizes_mlp(activations, 'resid_post')
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

        all_seeds_data = []
        for idx, mn in enumerate(model_names):
            print(f"  Seed {idx + 1}/{len(model_names)}: {mn}")
            all_seeds_data.append(
                compute_step_sizes_for_model(mn, args.model_type, SHARED_ID, PAIRS_IDS, layer_to_interpolate, N_STEPS)
            )

        mean_data, std_data = aggregate_metric_data(all_seeds_data)

        output_path = f"./plots/{MODEL_NAME}/step_sizes_resid_post_layer{layer_to_interpolate}_interpolation.png"
        generate_interpolation_results_plot(
            data_dict=mean_data,
            suptitle=f"Resid Post Step Sizes (Interpolated at Layer {layer_to_interpolate})",
            ylabel="L2 Norm of Step Difference",
            output_path=output_path,
            n_steps=N_STEPS - 1,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
            alpha_range=[0, 1],
            std_dict=std_data,
            y_floor=0
        )
        print(f"\nPlot saved to: {output_path}")
    else:
        print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")

        plot_data = compute_step_sizes_for_model(MODEL_NAME, args.model_type, SHARED_ID, PAIRS_IDS, layer_to_interpolate, N_STEPS)

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
