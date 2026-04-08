#!/usr/bin/env python3
"""
Full Residual Jacobian Analysis: Compute and plot ∂(layer_N resid_post) / ∂(layer_0 resid_post).
"""

#TODO: Add option for ViT models

import torch
import os
from tqdm import tqdm
import sys
import argparse
sys.path.append('./vis_plots')
from utils import load_model, load_config, load_activations, generate_interpolation_results_plot, get_n_layers_from_model, load_model_from_checkpoint, get_model_name, get_model_names, aggregate_metric_data, format_pair_ids_for_subdirectory

config = load_config()
N_STEPS = config['n_steps']


def compute_jacobian_full_residual(model, resid_post_interpolated: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute jacobian from layer 0 to last layer (HookedTransformer).
    Only computes jacobian for last token position wrt last token position.

    Args:
        resid_post_interpolated: [n_steps, seq_len, hidden_dim]

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    n_steps = resid_post_interpolated.shape[0]
    jacobians = []

    def forward_through_all_layers(last_token_resid):
        """Forward from layer 0 through all layers, last token only."""
        context = resid_post_interpolated[step_idx, :-1].detach().to(device)
        full_resid = torch.cat([context, last_token_resid.unsqueeze(0)], dim=0)

        activation = full_resid.unsqueeze(0)
        n_layers = get_n_layers_from_model(model)
        for layer_idx in range(n_layers):
            activation = model.blocks[layer_idx](activation)
        return activation[0, -1, :]

    for step_idx in range(n_steps):
        last_token = resid_post_interpolated[step_idx, -1, :].to(device)

        # Compute jacobian with chunking to reduce memory
        jac = torch.func.jacrev(forward_through_all_layers, chunk_size=128)(last_token)

        jacobians.append(jac.detach().cpu())
        del jac, last_token
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_jacobian_full_residual_toy(model, resid_post_layer0: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute jacobian from layer 0 to last layer (toy ResNet).
    No sequence dimension.

    Args:
        resid_post_layer0: [n_steps, hidden_dim]

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    n_steps = resid_post_layer0.shape[0]
    n_layers = len(model.blocks)
    jacobians = []

    def forward_through_all_layers(resid):
        activation = resid.unsqueeze(0)
        for layer_idx in range(n_layers):
            activation = model.blocks[layer_idx](activation)
        return activation.squeeze(0)

    for step_idx in tqdm(range(n_steps), desc="Steps", leave=False):
        resid = resid_post_layer0[step_idx].to(device)
        jac = torch.func.jacrev(forward_through_all_layers)(resid)
        jacobians.append(jac.detach().cpu())
        del jac, resid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_full_residual_norms_for_model(model_name, model_type, shared_id, pairs_ids, layer_to_interpolate, n_steps):
    """Compute full residual jacobian norms for a single model checkpoint.

    Returns:
        {pair_key: norms_tensor} — single-line format for aggregate_metric_data.
    """
    if model_type == 'toy_resnet':
        model, _, _ = load_model_from_checkpoint(model_name)
    else:
        model = load_model(model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    jacobian_norms = {}

    for pair_ids in tqdm(pairs_ids, desc="Pairs", leave=False):
        activations = load_activations(model_name, shared_id, layer_to_interpolate, pair_ids, n_steps)
        resid_post_layer0 = activations['layer0_resid_post']

        if model_type == 'toy_resnet':
            jacobians = compute_jacobian_full_residual_toy(model, resid_post_layer0, device)
        else:
            jacobians = compute_jacobian_full_residual(model, resid_post_layer0, device)

        norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)

        pair_key = f"{pair_ids[0]}_{pair_ids[1]}"
        jacobian_norms[pair_key] = norms

        del jacobians, resid_post_layer0, activations
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return jacobian_norms


def main():
    parser = argparse.ArgumentParser(description='Interpolate activations between token pairs')
    parser.add_argument('--model_type', type=str, choices=['hooked_transformer', 'vit', 'resnet', 'toy_resnet'], required=True, help='Type of model to use (hooked_transformer or vit or resnet or toy_resnet)')
    parser.add_argument('--data_type', type=str, choices=['text', 'image', 'class_spiral'], required=True, help='Type of data type to use (text or image or class_spiral)')
    parser.add_argument('--multi_seed', action='store_true', help='Aggregate across multiple seeds (toy_resnet only)')

    args = parser.parse_args()

    if args.data_type == 'image':
        SHARED_ID = config['image']['shared_image_id']
        PAIRS_IDS = config['image']['pairs_ids']
    elif args.data_type == 'text':
        SHARED_ID = config['text']['shared_context']
        PAIRS_IDS = config['text']['token_pairs']
    elif args.data_type == 'class_spiral':
        SHARED_ID = ''
        PAIRS_IDS = [[f'{num}' for num in pair] for pair in config['class_spiral']['pairs']]

    # Check if this is a ViT model (jacobian computation not yet supported)
    if args.model_type == 'vit':
        print("ERROR: Jacobian computation for ViT models is not yet implemented.")
        return

    MODEL_NAME = get_model_name(config, args.model_type)

    if args.model_type in ['toy_resnet']:
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
                compute_full_residual_norms_for_model(mn, args.model_type, SHARED_ID, PAIRS_IDS, layer_to_interpolate, N_STEPS)
            )

        mean_data, std_data = aggregate_metric_data(all_seeds_data)

        pairs_subdir = format_pair_ids_for_subdirectory(PAIRS_IDS)
        output_path = f"./plots/{MODEL_NAME}/{pairs_subdir}/jacobians_full_residual_norms.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generate_interpolation_results_plot(
            data_dict=mean_data,
            suptitle="Jacobian Norms: Full Residual Stream (Layer 0 → Last Layer)",
            ylabel="Frobenius Norm",
            output_path=output_path,
            n_steps=N_STEPS,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
            std_dict=std_data,
            y_floor=0
        )
        print(f"\nPlot saved to: {output_path}")
    else:
        # Single model
        if args.model_type == 'toy_resnet':
            MODEL_NAME = get_model_names(config, args.model_type)[0]

        print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")

        jacobian_norms = compute_full_residual_norms_for_model(
            MODEL_NAME, args.model_type, SHARED_ID, PAIRS_IDS, layer_to_interpolate, N_STEPS
        )

        pairs_subdir = format_pair_ids_for_subdirectory(PAIRS_IDS)
        output_path = f"./plots/{MODEL_NAME}/{pairs_subdir}/jacobians_full_residual_norms.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generate_interpolation_results_plot(
            data_dict=jacobian_norms,
            suptitle="Jacobian Norms: Full Residual Stream (Layer 0 → Last Layer)",
            ylabel="Frobenius Norm",
            output_path=output_path,
            n_steps=N_STEPS,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
        )
        print(f"\nPlot saved to: {output_path}")

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
