#!/usr/bin/env python3
"""
Layerwise Residual Jacobian Analysis: Compute and plot ∂(layer_i+1 resid_post) / ∂(layer_i resid_post) for each layer.
"""

import argparse
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import sys
sys.path.append('./vis_plots')
from utils import load_model, load_config, load_activations, generate_interpolation_results_plot, get_n_layers_from_model, load_model_from_checkpoint, get_model_name, get_model_names, aggregate_metric_data, format_pair_ids_for_subdirectory

config = load_config()
N_STEPS = config['n_steps']


def compute_jacobian_layerwise(model, resid_post_interpolated: torch.Tensor, layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian for single layer: ∂(layer_i+1 resid_post) / ∂(layer_i resid_post).
    Only computes jacobian for last token position wrt last token position (HookedTransformer).

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


def compute_jacobian_layerwise_toy(model, resid_post_interpolated: torch.Tensor, layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian for single block: ∂(layer_i+1 resid_post) / ∂(layer_i resid_post).
    Toy ResNet: no sequence dimension.

    Args:
        resid_post_interpolated: [n_steps, hidden_dim]
        layer_idx: Which layer's resid_post is the input

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    n_steps = resid_post_interpolated.shape[0]
    jacobians = []

    def forward_single_layer(resid):
        activation = resid.unsqueeze(0)
        activation = model.blocks[layer_idx + 1](activation)
        return activation.squeeze(0)

    for step_idx in range(n_steps):
        resid = resid_post_interpolated[step_idx].to(device)
        jac = torch.func.jacrev(forward_single_layer)(resid)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_single_layer(resid)
            recorded_out = activations[f'layer{layer_idx+1}_resid_post'][step_idx].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"Layerwise validation failed: layer {layer_idx}→{layer_idx+1}, step {step_idx}, diff {diff}"

        jacobians.append(jac.detach().cpu())
        del jac, resid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_jacobian_embed_toy(model, input_data: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute Jacobian of the embedding layer (input_layer) for toy ResNet.

    Args:
        model: Toy ResNet model
        input_data: [n_steps, input_dim]
        device: target device

    Returns:
        Jacobian: [n_steps, hidden_dim, input_dim]  (or identity-like if no input_layer)
    """
    n_steps = input_data.shape[0]

    if not hasattr(model, 'input_layer'):
        # ResNetMLPSkeleton: identity embedding, norm is 1.0
        return None

    def forward_embed(x):
        return model.input_layer(x)

    jac_fn = torch.func.jacrev(forward_embed)
    jacobians = []
    for step_idx in range(n_steps):
        x = input_data[step_idx].to(device)
        jac = jac_fn(x)
        jacobians.append(jac.detach().cpu())
        del jac, x
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_jacobian_unembed_toy(model, last_block_data: torch.Tensor, device: str) -> torch.Tensor:
    """
    Compute Jacobian of the unembedding layers (final_norm + relu + output_layer) for toy ResNet.

    Args:
        model: Toy ResNet model
        last_block_data: [n_steps, hidden_dim]
        device: target device

    Returns:
        Jacobian: [n_steps, output_dim, hidden_dim]
    """
    n_steps = last_block_data.shape[0]

    def forward_unembed(resid):
        x = model.final_norm(resid.unsqueeze(0))
        x = F.relu(x)
        x = model.output_layer(x)
        return x.squeeze(0)

    jac_fn = torch.func.jacrev(forward_unembed)
    jacobians = []
    for step_idx in range(n_steps):
        x = last_block_data[step_idx].to(device)
        jac = jac_fn(x)
        jacobians.append(jac.detach().cpu())
        del jac, x
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_layerwise_data_for_model(model_name, model_type, shared_id, pairs_ids, layer_to_interpolate, n_steps):
    """Compute layerwise jacobian norms and product norms for a single model checkpoint.

    For toy_resnet, includes embedding (Input→Embed) and unembedding (LastBlock→Logits)
    Jacobians. The product norm is:
        ||J_embed||_F × ||J_{n-1} · ... · J_0||_F × ||J_unembed||_F

    Returns:
        (layerwise_norms_data, product_norms_data)
        layerwise_norms_data: {pair_key: {f"Layer {i}→{i+1}": tensor}} — multi-layer format
        product_norms_data: {pair_key: tensor} — single-line format
    """
    if model_type == 'toy_resnet':
        model, _, _ = load_model_from_checkpoint(model_name)
        n_layers = len(model.blocks)
    else:
        model = load_model(model_name)
        n_layers = get_n_layers_from_model(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    layerwise_norms_data = {}
    product_norms_data = {}

    for pair_ids in pairs_ids:
        pair_key = f"{pair_ids[0]}_{pair_ids[1]}"
        layer_norms = {}
        jacs_list = []

        activations = load_activations(model_name, shared_id, layer_to_interpolate, pair_ids, n_steps)

        # --- Embedding Jacobian (toy_resnet only) ---
        embed_jac = None
        if model_type == 'toy_resnet':
            input_data = activations.get('layer-2_resid_post')
            if input_data is not None:
                embed_jac = compute_jacobian_embed_toy(model, input_data, device)
                if embed_jac is not None:
                    embed_norm = torch.norm(embed_jac.view(embed_jac.shape[0], -1), dim=1)
                    layer_norms["Input→Embed"] = embed_norm

        # --- Hidden layer Jacobians ---
        for layer_idx in tqdm(range(0, n_layers - 1), desc=f"Layerwise Jacobians {pair_key}", leave=False):
            resid_post = activations[f'layer{layer_idx}_resid_post']

            if model_type == 'toy_resnet':
                jacobians = compute_jacobian_layerwise_toy(model, resid_post, layer_idx, device, activations)
            else:
                jacobians = compute_jacobian_layerwise(model, resid_post, layer_idx, device, activations)

            norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)
            layer_norms[f"Layer {layer_idx}→{layer_idx+1}"] = norms
            jacs_list.append(jacobians)

            del resid_post
            torch.cuda.empty_cache()

        # --- Unembedding Jacobian (toy_resnet only) ---
        unembed_jac = None
        if model_type == 'toy_resnet':
            last_block_data = activations.get(f'layer{n_layers - 1}_resid_post')
            if last_block_data is not None:
                unembed_jac = compute_jacobian_unembed_toy(model, last_block_data, device)
                if unembed_jac is not None:
                    unembed_norm = torch.norm(unembed_jac.view(unembed_jac.shape[0], -1), dim=1)
                    layer_norms["LastBlock→Logits"] = unembed_norm

        layerwise_norms_data[pair_key] = layer_norms

        # Compute product norms:
        # ||J_unembed · J_{n-1} · ... · J_0 · J_embed||_F
        product = jacs_list[0]
        for jac in jacs_list[1:]:
            product = torch.bmm(jac, product)

        # Include embedding in the product if available
        if embed_jac is not None:
            product = torch.bmm(product, embed_jac)

        # Include unembedding in the product if available
        if unembed_jac is not None:
            product = torch.bmm(unembed_jac, product)

        product_norm = torch.norm(product.view(product.shape[0], -1), dim=1)
        product_norms_data[pair_key] = product_norm

        del activations, jacs_list, product
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return layerwise_norms_data, product_norms_data


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

        all_layerwise = []
        all_products = []
        for idx, mn in enumerate(model_names):
            print(f"  Seed {idx + 1}/{len(model_names)}: {mn}")
            lw, prod = compute_layerwise_data_for_model(mn, args.model_type, SHARED_ID, PAIRS_IDS, layer_to_interpolate, N_STEPS)
            all_layerwise.append(lw)
            all_products.append(prod)

        mean_lw, std_lw = aggregate_metric_data(all_layerwise)
        mean_prod, std_prod = aggregate_metric_data(all_products)

        output_dir = f"./plots/{MODEL_NAME}"
        pairs_subdir = format_pair_ids_for_subdirectory(PAIRS_IDS)
        plot_output_dir = f"{output_dir}/{pairs_subdir}"
        os.makedirs(plot_output_dir, exist_ok=True)

        generate_interpolation_results_plot(
            data_dict=mean_lw,
            suptitle="Jacobian Norms: Layerwise Residual (Layer i → Layer i+1)",
            ylabel="Frobenius Norm",
            output_path=f"{plot_output_dir}/jacobians_layerwise_norms.png",
            n_steps=N_STEPS,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
            skip_interpolation_layer=False,
            std_dict=std_lw,
            y_floor=0
        )

        generate_interpolation_results_plot(
            data_dict=mean_prod,
            suptitle="Jacobian Products: Layerwise Residual (Full Chain)",
            ylabel="Frobenius Norm",
            output_path=f"{plot_output_dir}/jacobians_layerwise_product.png",
            n_steps=N_STEPS,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
            std_dict=std_prod,
            y_floor=0
        )

        print(f"\nPlots saved to: {output_dir}/")
    else:
        # Single model
        if args.model_type == 'toy_resnet':
            MODEL_NAME = get_model_names(config, args.model_type)[0]

        print(f"Model: {MODEL_NAME} | Steps: {N_STEPS}")

        layerwise_norms_data, product_norms_data = compute_layerwise_data_for_model(
            MODEL_NAME, args.model_type, SHARED_ID, PAIRS_IDS, layer_to_interpolate, N_STEPS
        )

        output_dir = f"./plots/{MODEL_NAME}"
        pairs_subdir = format_pair_ids_for_subdirectory(PAIRS_IDS)
        plot_output_dir = f"{output_dir}/{pairs_subdir}"
        os.makedirs(plot_output_dir, exist_ok=True)

        generate_interpolation_results_plot(
            data_dict=layerwise_norms_data,
            suptitle="Jacobian Norms: Layerwise Residual (Layer i → Layer i+1)",
            ylabel="Frobenius Norm",
            output_path=f"{plot_output_dir}/jacobians_layerwise_norms.png",
            n_steps=N_STEPS,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
            skip_interpolation_layer=False
        )

        generate_interpolation_results_plot(
            data_dict=product_norms_data,
            suptitle="Jacobian Products: Layerwise Residual (Full Chain)",
            ylabel="Frobenius Norm",
            output_path=f"{plot_output_dir}/jacobians_layerwise_product.png",
            n_steps=N_STEPS,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS
        )

        print(f"\nPlots saved to: {output_dir}/")

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
