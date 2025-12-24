#!/usr/bin/env python3
"""
Layerwise Residual Jacobian Analysis: Compute and plot ∂(layer_i+1 resid_post) / ∂(layer_i resid_post) for each layer.
"""

import argparse
import torch
import os
from tqdm import tqdm
import sys
sys.path.append('./scripts')
from utils import load_model, load_config, load_activations, generate_interpolation_results_plot

config = load_config()
N_STEPS = config['n_steps']


def compute_jacobian_layerwise(model, resid_post_interpolated: torch.Tensor, layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian for single layer: ∂(layer_i+1 resid_post) / ∂(layer_i resid_post).
    Only computes jacobian for last token position wrt last token position.

    Args:
        resid_post_interpolated: [n_steps, seq_len, hidden_dim]
        layer_idx: Which layer to compute jacobian for
        activations: Dict with recorded activations for validation

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    n_steps = resid_post_interpolated.shape[0]
    jacobians = []

    def forward_single_layer(last_token_resid):
        """Forward through single layer, last token only."""
        context = resid_post_interpolated[step_idx, :-1].detach().to(device)
        full_resid = torch.cat([context, last_token_resid.unsqueeze(0)], dim=0)

        activation = full_resid.unsqueeze(0)
        activation = model.blocks[layer_idx + 1](activation)
        return activation[0, -1, :]

    for step_idx in range(n_steps):
        last_token = resid_post_interpolated[step_idx, -1, :].to(device)

        # Compute jacobian with chunking to reduce memory
        jac = torch.func.jacrev(forward_single_layer, chunk_size=128)(last_token)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_single_layer(last_token)
            recorded_out = activations[f'layer{layer_idx+1}_resid_post'][step_idx, -1, :].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"Layerwise validation failed: layer {layer_idx}→{layer_idx+1}, step {step_idx}, diff {diff}"

        jacobians.append(jac.detach().cpu())
        del jac, last_token
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


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
        
    print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")

    model = load_model(MODEL_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_layers = model.cfg.n_layers
    print(f"Loaded {n_layers}-layer model on {device}")

    os.makedirs(f"./plots/{MODEL_NAME}", exist_ok=True)

    print("\n=== Layerwise Residual Jacobians ===")

    jacobian_norms_by_pair = {}
    jacobians_by_pair = {}

    for pair_ids in PAIRS_IDS:
        pair_key = f"{pair_ids[0]}_{pair_ids[1]}"
        jacobian_norms_by_pair[pair_key] = {}
        jacobians_by_pair[pair_key] = []

        activations = load_activations(MODEL_NAME, SHARED_ID, 0, pair_ids, N_STEPS)

        for layer_idx in tqdm(range(1, n_layers - 1), desc=f"Computing Layerwise Jacobians for {pair_key}"):
            resid_post = activations[f'layer{layer_idx}_resid_post']
            jacobians = compute_jacobian_layerwise(model, resid_post, layer_idx, device, activations)

            norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)
            jacobian_norms_by_pair[pair_key][layer_idx] = norms
            jacobians_by_pair[pair_key].append(jacobians)

            del jacobians, resid_post
            torch.cuda.empty_cache()

        del activations

    # Plot layerwise norms
    data_dict = {}
    for pair_key, layer_norms in jacobian_norms_by_pair.items():
        data_dict[pair_key] = {f"Layer {idx}→{idx+1}": norms for idx, norms in layer_norms.items()}

    generate_interpolation_results_plot(
        data_dict=data_dict,
        suptitle="Jacobian Norms: Layerwise Residual (Layer i → Layer i+1)",
        ylabel="Frobenius Norm",
        output_path=f"./plots/{MODEL_NAME}/jacobians_layerwise_norms.png",
        n_steps=N_STEPS,
        shared_id=SHARED_ID,
        pairs_ids=PAIRS_IDS,
        skip_interpolation_layer=False
    )

    # Compute and plot jacobian products
    product_norms = {}
    for pair_key, jacs_list in jacobians_by_pair.items():
        # Multiply jacobians across layers for each step
        product = jacs_list[0]  # Start with first layer
        for jac in jacs_list[1:]:
            product = torch.bmm(jac, product)  # [n_steps, hidden_dim, hidden_dim]

        norms = torch.norm(product.view(product.shape[0], -1), dim=1)
        product_norms[pair_key] = norms

    generate_interpolation_results_plot(
        data_dict=product_norms,
        suptitle="Jacobian Products: Layerwise Residual (Full Chain)",
        ylabel="Frobenius Norm",
        output_path=f"./plots/{MODEL_NAME}/jacobians_layerwise_product.png",
        n_steps=N_STEPS,
        shared_id=SHARED_ID,
        pairs_ids=PAIRS_IDS
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
