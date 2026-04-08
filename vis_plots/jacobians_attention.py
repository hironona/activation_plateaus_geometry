#!/usr/bin/env python3
"""
Attention Jacobian Analysis: Compute and plot ∂(attn_out) / ∂(resid_pre) for each layer.
"""

#TODO: Add option for ViT models

import torch
import os
from tqdm import tqdm
import sys
import argparse
sys.path.append('./vis_plots')
from utils import load_model, load_config, load_activations, generate_interpolation_results_plot, get_n_layers_from_model, format_pair_ids_for_subdirectory

config = load_config()
N_STEPS = config['n_steps']


def compute_jacobian_attention(model, resid_pre_interpolated: torch.Tensor, layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian for attention: ∂(attn_out) / ∂(resid_pre).
    Only computes jacobian for last token position wrt last token position.

    Args:
        resid_pre_interpolated: [n_steps, seq_len, hidden_dim]
        layer_idx: Which layer to compute jacobian for
        activations: Dict with recorded activations for validation

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    n_steps = resid_pre_interpolated.shape[0]
    jacobians = []

    def forward_attention(last_token_resid):
        """Forward through ln1 + attention, last token only."""
        context = resid_pre_interpolated[step_idx, :-1].detach().to(device)
        full_resid = torch.cat([context, last_token_resid.unsqueeze(0)], dim=0)

        block = model.blocks[layer_idx]
        resid_with_batch = full_resid.unsqueeze(0)
        ln_out = block.ln1(resid_with_batch)
        attn_out = block.attn(ln_out, ln_out, ln_out)
        return attn_out[0, -1, :]

    for step_idx in range(n_steps):
        last_token = resid_pre_interpolated[step_idx, -1, :].to(device)

        # Compute jacobian with chunking to reduce memory
        jac = torch.func.jacrev(forward_attention, chunk_size=128)(last_token)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_attention(last_token)
            recorded_out = activations[f'layer{layer_idx}_attn_out'][step_idx, -1, :].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"Attention validation failed: layer {layer_idx}, step {step_idx}, diff {diff}"

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
    
    # Check if this is a ViT model (jacobian computation not yet supported)
    if args.model_type == 'vit':
        print("ERROR: Jacobian computation for ViT models is not yet implemented.")
        print("This script currently only supports HookedTransformer models.")
        return
    
    n_layers = get_n_layers_from_model(model)
    print(f"Loaded {n_layers}-layer model on {device}")

    os.makedirs(f"./plots/{MODEL_NAME}", exist_ok=True)

    print("\n=== Attention Jacobians ===")

    jacobian_norms_by_pair = {}
    jacobians_by_pair = {}

    for pair_ids in PAIRS_IDS:
        pair_key = f"{pair_ids[0]}_{pair_ids[1]}"
        jacobian_norms_by_pair[pair_key] = {}
        jacobians_by_pair[pair_key] = []

        activations = load_activations(MODEL_NAME, SHARED_ID, 0, pair_ids, N_STEPS)

        for layer_idx in tqdm(range(1, n_layers), desc=f"Computing Attention Jacobians for {pair_key}"):
            resid_pre = activations[f'layer{layer_idx-1}_resid_post']
            jacobians = compute_jacobian_attention(model, resid_pre, layer_idx, device, activations)

            norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)
            jacobian_norms_by_pair[pair_key][layer_idx] = norms
            jacobians_by_pair[pair_key].append(jacobians)

            del jacobians, resid_pre
            torch.cuda.empty_cache()

        del activations

    # Plot layerwise norms
    data_dict = {}
    for pair_key, layer_norms in jacobian_norms_by_pair.items():
        data_dict[pair_key] = {f"Layer {idx}": norms for idx, norms in layer_norms.items()}

    pairs_subdir = format_pair_ids_for_subdirectory(PAIRS_IDS)
    output_dir = f"./plots/{MODEL_NAME}/{pairs_subdir}"
    os.makedirs(output_dir, exist_ok=True)
    generate_interpolation_results_plot(
        data_dict=data_dict,
        suptitle="Jacobian Norms: Attention (∂attn_out / ∂resid_pre)",
        ylabel="Frobenius Norm",
        output_path=f"{output_dir}/jacobians_attention_norms.png",
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
        suptitle="Jacobian Products: Attention (Full Chain)",
        ylabel="Frobenius Norm",
        output_path=f"{output_dir}/jacobians_attention_product.png",
        n_steps=N_STEPS,
        shared_id=SHARED_ID,
        pairs_ids=PAIRS_IDS
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
