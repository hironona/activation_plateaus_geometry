#!/usr/bin/env python3
"""
MLP Jacobian Analysis: Compute and plot ∂(mlp_out) / ∂(resid_mid) for each layer.
For toy ResNet (no attention), computes ∂(mlp_out) / ∂(resid_pre) instead.
"""

#TODO: Add option for ViT models

import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import sys
import argparse
sys.path.append('./vis_plots')
from utils import load_model, load_config, load_activations, generate_interpolation_results_plot, get_n_layers_from_model, load_model_from_checkpoint, get_model_name, get_model_names, aggregate_metric_data

config = load_config()
N_STEPS = config['n_steps']


def compute_jacobian_mlp(model, resid_mid_interpolated: torch.Tensor, layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian for MLP: ∂(mlp_out) / ∂(resid_mid) (HookedTransformer).

    Args:
        resid_mid_interpolated: [n_steps, hidden_dim] (last token only)
        layer_idx: Which layer to compute jacobian for
        activations: Dict with recorded activations for validation

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    def forward_mlp(resid_mid):
        """Forward through ln2 + MLP."""
        block = model.blocks[layer_idx]
        resid_with_batch = resid_mid.unsqueeze(0).unsqueeze(0)
        mlp_out = block.mlp(block.ln2(resid_with_batch))
        return mlp_out.squeeze(0).squeeze(0)

    n_steps = resid_mid_interpolated.shape[0]
    jacobians = []

    for step_idx in range(n_steps):
        resid_mid = resid_mid_interpolated[step_idx].to(device)

        # Compute jacobian with chunking to reduce memory
        jac = torch.func.jacrev(forward_mlp, chunk_size=128)(resid_mid)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_mlp(resid_mid)
            recorded_out = activations[f'layer{layer_idx}_mlp_out'][step_idx, -1, :].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"MLP validation failed: layer {layer_idx}, step {step_idx}, diff {diff}"

        jacobians.append(jac.detach().cpu())
        del jac, resid_mid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_jacobian_mlp_toy(model, resid_pre: torch.Tensor, layer_idx: int, device: str, activations: dict) -> torch.Tensor:
    """
    Compute jacobian for MLP: ∂(mlp_out) / ∂(resid_pre) (toy ResNet).
    The MLP is the entire non-residual path: ln1 → relu → fc1 → dropout → ln2 → relu → fc2.

    Args:
        resid_pre: [n_steps, hidden_dim]
        layer_idx: Which block to compute jacobian for

    Returns:
        Jacobian: [n_steps, hidden_dim, hidden_dim]
    """
    def forward_mlp(resid):
        block = model.blocks[layer_idx]
        out = block.ln1(resid.unsqueeze(0))
        out = F.relu(out)
        out = block.fc1(out)
        out = block.dropout(out)
        out = block.ln2(out)
        out = F.relu(out)
        out = block.fc2(out)
        return out.squeeze(0)

    n_steps = resid_pre.shape[0]
    jacobians = []

    for step_idx in range(n_steps):
        resid = resid_pre[step_idx].to(device)
        jac = torch.func.jacrev(forward_mlp)(resid)

        # Validate against recorded activations
        with torch.no_grad():
            computed_out = forward_mlp(resid)
            recorded_out = activations[f'layer{layer_idx}_mlp_out'][step_idx].to(device)
            diff = torch.norm(computed_out - recorded_out).item()
            assert diff < 1e-4, f"MLP validation failed: layer {layer_idx}, step {step_idx}, diff {diff}"

        jacobians.append(jac.detach().cpu())
        del jac, resid
        torch.cuda.empty_cache()

    return torch.stack(jacobians)


def compute_mlp_data_for_model(model_name, model_type, shared_id, pairs_ids, layer_to_interpolate, n_steps):
    """Compute MLP jacobian norms and product norms for a single model checkpoint.

    Returns:
        (layerwise_norms_data, product_norms_data)
        layerwise_norms_data: {pair_key: {f"Layer {i}": tensor}} — multi-layer format
        product_norms_data: {pair_key: tensor} — single-line format
    """
    if model_type == 'toy_resnet':
        model, _ = load_model_from_checkpoint(model_name)
        n_layers = len(model.blocks)
        layer_range = range(0, n_layers)
    else:
        model = load_model(model_name)
        n_layers = get_n_layers_from_model(model)
        layer_range = range(1, n_layers)

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

        for layer_idx in tqdm(layer_range, desc=f"MLP Jacobians {pair_key}", leave=False):
            if model_type == 'toy_resnet':
                if layer_idx == 0:
                    resid_input = activations['layer-1_resid_post']
                else:
                    resid_input = activations[f'layer{layer_idx-1}_resid_post']
                jacobians = compute_jacobian_mlp_toy(model, resid_input, layer_idx, device, activations)
            else:
                resid_mid = activations[f'layer{layer_idx}_resid_mid'][:, -1, :]
                jacobians = compute_jacobian_mlp(model, resid_mid, layer_idx, device, activations)

            norms = torch.norm(jacobians.view(jacobians.shape[0], -1), dim=1)
            layer_norms[f"Layer {layer_idx}"] = norms
            jacs_list.append(jacobians)

            del jacobians
            torch.cuda.empty_cache()

        layerwise_norms_data[pair_key] = layer_norms

        # Compute product norms
        product = jacs_list[0]
        for jac in jacs_list[1:]:
            product = torch.bmm(jac, product)
        product_norms_data[pair_key] = torch.norm(product.view(product.shape[0], -1), dim=1)

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
            lw, prod = compute_mlp_data_for_model(mn, args.model_type, SHARED_ID, PAIRS_IDS, layer_to_interpolate, N_STEPS)
            all_layerwise.append(lw)
            all_products.append(prod)

        mean_lw, std_lw = aggregate_metric_data(all_layerwise)
        mean_prod, std_prod = aggregate_metric_data(all_products)

        output_dir = f"./plots/{MODEL_NAME}"
        os.makedirs(output_dir, exist_ok=True)

        generate_interpolation_results_plot(
            data_dict=mean_lw,
            suptitle="Jacobian Norms: MLP (∂mlp_out / ∂resid_pre)",
            ylabel="Frobenius Norm",
            output_path=f"{output_dir}/jacobians_mlp_norms.png",
            n_steps=N_STEPS,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
            skip_interpolation_layer=False,
            std_dict=std_lw,
            y_floor=0
        )

        generate_interpolation_results_plot(
            data_dict=mean_prod,
            suptitle="Jacobian Products: MLP (Full Chain)",
            ylabel="Frobenius Norm",
            output_path=f"{output_dir}/jacobians_mlp_product.png",
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

        layerwise_norms_data, product_norms_data = compute_mlp_data_for_model(
            MODEL_NAME, args.model_type, SHARED_ID, PAIRS_IDS, layer_to_interpolate, N_STEPS
        )

        output_dir = f"./plots/{MODEL_NAME}"
        os.makedirs(output_dir, exist_ok=True)

        generate_interpolation_results_plot(
            data_dict=layerwise_norms_data,
            suptitle="Jacobian Norms: MLP (∂mlp_out / ∂resid_pre)",
            ylabel="Frobenius Norm",
            output_path=f"{output_dir}/jacobians_mlp_norms.png",
            n_steps=N_STEPS,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS,
            skip_interpolation_layer=False
        )

        generate_interpolation_results_plot(
            data_dict=product_norms_data,
            suptitle="Jacobian Products: MLP (Full Chain)",
            ylabel="Frobenius Norm",
            output_path=f"{output_dir}/jacobians_mlp_product.png",
            n_steps=N_STEPS,
            shared_id=SHARED_ID,
            pairs_ids=PAIRS_IDS
        )

        print(f"\nPlots saved to: {output_dir}/")

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
