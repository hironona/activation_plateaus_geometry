#!/usr/bin/env python3
"""
Visualize activation plateaus by plotting data points colored by metrics
(L2 norm, Jacobian determinant, layerwise product of Jacobian determinants).
"""

from typing import Union, Dict, Any, List
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

import sys
sys.path.append('./train')
from utils import load_config, load_model, load_model_from_checkpoint, construct_filepath, get_n_layers_from_model, get_model_names, get_model_name, LAYER_ARGUMENT_IDX_MAPPING
from compute_metrics import l2_norm_metric, jacobian_determinant_metric, jacobian_determinant_layerwise_prod_metric, jacobian_norm_metric, jacobian_norm_layerwise_prod_metric
from model import ResNetMLP, ResNetMLPSkeleton

config = load_config("vis_plateaus/config.yaml")

METRIC_OPTIONS = ['l2_norm', 'jacobian_determinant_full', 'jacobian_determinant_layerwise_prod', 'jacobian_norm_full', 'jacobian_norm_layerwise_prod']

LAYER_ALIASES = {'embedding': 'embed'}


def collect_activations_resid_post(model: Union[ResNetMLP, ResNetMLPSkeleton], data, source_layer_idx: int, target_layer_idx: int, device) -> Dict[str, torch.Tensor]:

    activations = {}

    def create_hook(layer_idx):
        def hook(module, input, output):
            activations[f'layer{layer_idx}_resid_post'] = output.cpu().clone()
        return hook

    layers = list(model.blocks) + [model.hook_input, model.hook_embed]
    hooks = []

    if source_layer_idx >= -1:  # collect activations at the input space of the source layer
        hooks.append(layers[source_layer_idx - 1].register_forward_hook(create_hook(source_layer_idx - 1)))

    for layer_idx in range(source_layer_idx, target_layer_idx + 1):
        hooks.append(layers[layer_idx].register_forward_hook(create_hook(layer_idx)))

    with torch.no_grad():
        logits = model(data)

    activations['logits'] = logits.cpu().clone()

    for hook in hooks:
        hook.remove()

    return activations

def generate_data_around_point(point: torch.Tensor, n_points: int, radius: float) -> torch.Tensor:
    """
    Generate n_points uniformly distributed around a point in the same dimension.
    Args:
        point: [dim]
        n_points: number of points to generate
        radius: maximum distance from the original point
    Returns:
        Tensor of shape [n_points, dim]
    """
    dim = point.shape[0]
    random_directions = torch.randn(n_points, dim)
    random_directions /= torch.max(torch.norm(random_directions, dim=1))   # Scale it to max 1
    random_directions /= torch.norm(random_directions, dim=1, keepdim=True)  # Normalize to unit vectors
    random_distances = torch.rand(n_points) * radius  # Random distances from the original point
    perturbations = random_directions * random_distances.unsqueeze(1)  # Scale by distances
    new_points = point.unsqueeze(0) + perturbations  # Shift by the original point
    new_points = torch.clamp(new_points, -1.0, 1.0)
    new_points = point.unsqueeze(0) + random_directions * radius  # Scale by radius and shift by the original point
    
    return new_points

def generate_uniform_data(n_points: int, radius: float) -> torch.Tensor:
    """
    Generate n_points uniformly distributed around a point in the same dimension.
    Args:
        n_points: number of points to generate
        radius: maximum distance from the original point
    Returns:
        Tensor of shape [n_points, dim]
    """
    new_points = [np.linspace([x1, -radius], [x1, radius], int(math.sqrt(n_points))) for x1 in np.linspace(-radius, radius, int(math.sqrt(n_points)))]
    new_points = torch.tensor(np.concatenate(new_points, axis=0), dtype=torch.float32)
    
    return new_points

def visualize_plateau(
    data: torch.Tensor,
    metric_values: torch.Tensor,
    metric_name: str,
    title: str,
    output_path: str,
    radius: float,
    reference_point: torch.Tensor = None,
):
    """
    Render the metric over the 2D input grid as a heatmap.

    Args:
        data: (n_points, 2) — input grid points produced by `generate_uniform_data`.
        metric_values: (n_points,) — metric value per point.
        metric_name: Colorbar label.
        title: Plot title.
        output_path: Where to save the plot.
        radius: Half-extent of the input grid.
        reference_point: (2,) — reference point in input space (for marking).
    """
    grid_size = int(math.sqrt(data.shape[0]))
    assert grid_size * grid_size == data.shape[0], "data must be a square grid"

    # `generate_uniform_data` builds points with x1 varying along the outer loop and x2 along
    # the inner loop, so reshape gives grid[i, j] for (x1[i], x2[j]). Transpose so axis 0 is x2
    # (vertical) and axis 1 is x1 (horizontal) — matches imshow's (row=y, col=x) convention.
    grid = metric_values.numpy().reshape(grid_size, grid_size).T

    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    im = ax.imshow(
        grid, origin='lower', extent=(-radius, radius, -radius, radius),
        cmap='coolwarm', aspect='equal', interpolation='nearest',
    )
    plt.colorbar(im, ax=ax, label=metric_name, pad=0.02)

    if reference_point is not None and metric_name.startswith('l2_norm'):
        ref = reference_point.numpy().squeeze()
        ax.scatter(ref[0], ref[1], c='black', marker='x', s=100, zorder=3, linewidths=2)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

def compute_metric_for_checkpoint(
    checkpoint_path: str,
    data: torch.Tensor,
    reference_point: torch.Tensor,
    metric: str,
    source_layer_idx,
    target_layer_idx,
    device: str,
) -> dict:
    """
    Load a single checkpoint and compute the metric for the given data.

    Returns:
        Dict with 'metric_values' (n_points,), 'source_activations' (n_points, d_source),
        'task_config', 'ref_source_act', and optionally 'predicted_class'.
    """
    model, model_config, full_config = load_model_from_checkpoint(checkpoint_path)
    task_config = full_config['task']
    model = model.to(device)
    n_blocks = len(model.blocks)
    target_is_logits = (target_layer_idx == 'logits')
    effective_target = n_blocks - 1 if target_is_logits else target_layer_idx

    activations = collect_activations_resid_post(
        model, data.to(device),
        source_layer_idx=source_layer_idx,
        target_layer_idx=effective_target,
        device=device,
    )

    if metric == 'l2_norm':
        point_activations = collect_activations_resid_post(
            model,
            reference_point.to(device).unsqueeze(0),
            source_layer_idx=source_layer_idx,
            target_layer_idx=effective_target,
            device=device
        )
        results = l2_norm_metric(model, point_activations, activations, source_layer_idx, target_layer_idx, device)
    elif metric == 'jacobian_norm_full':
        results = jacobian_norm_metric(model, activations, source_layer_idx, target_layer_idx, device)
    elif metric == 'jacobian_norm_layerwise_prod':
        results = jacobian_norm_layerwise_prod_metric(model, activations, source_layer_idx, target_layer_idx, device)
    elif metric == 'jacobian_determinant_full':
        results = jacobian_determinant_metric(model, activations, source_layer_idx, target_layer_idx, device)
    elif metric == 'jacobian_determinant_layerwise_prod':
        results = jacobian_determinant_layerwise_prod_metric(model, activations, source_layer_idx, target_layer_idx, device)
    else:
        raise ValueError(f"Metric {metric} not supported. Choose from {METRIC_OPTIONS}")

    # source_activations = activations[f'layer{source_layer_idx-1}_resid_post']
    source_activations = activations[f'layer{source_layer_idx}_resid_post']

    # Reference point in source space
    if source_layer_idx <= -2:
        ref_source_act = reference_point
    else:
        ref_acts = collect_activations_resid_post(
            model, reference_point.to(device).unsqueeze(0),
            source_layer_idx=-2, target_layer_idx=source_layer_idx,
            device=device,
        )
        # ref_source_act = ref_acts[f'layer{source_layer_idx-1}_resid_post'].squeeze(0)
        ref_source_act = ref_acts[f'layer{source_layer_idx}_resid_post'].squeeze(0)


    out = {
        'metric_values': results['metric_values'],
        'source_activations': source_activations,
        'ref_source_act': ref_source_act,
        'task_config': task_config,
        'model': model,
    }
    if 'predicted_class' in results:
        out['predicted_class'] = results['predicted_class']

    return out


def main():
    parser = argparse.ArgumentParser(description='Visualize activation plateaus')
    parser.add_argument('--model_type', type=str, choices=['hooked_transformer', 'vit', 'resnet', 'toy_resnet'], required=True, help='Type of model to use')
    parser.add_argument('--data_type', type=str, choices=['text', 'image', 'class_spiral'], required=True, help='Type of data to use')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint or directory')
    parser.add_argument('--multi_seed', action='store_true', help='Average metric across all seed checkpoints (toy_resnet only)')
    parser.add_argument('--target_layer_idx', type=str, default=None, help="Override target_layer_idx from config (e.g. '0', '9', 'logits')")

    args = parser.parse_args()

    model_type = args.model_type

    # Load config values
    if args.model_path:
        config['model_names'][model_type] = args.model_path
    model_paths = get_model_names(config, model_type)
    MODEL_NAME  = get_model_name(config, model_type)

    N_POINTS = config['n_points']
    RADIUS = config['radius']
    REFERENCE_POINT = torch.tensor(config['reference_point'])
    METRIC = config['metric']
    SOURCE_LAYER_IDX_RAW = config['source_layer_idx']
    TARGET_LAYER_IDX_RAW = args.target_layer_idx if args.target_layer_idx is not None else config['target_layer_idx']
    N_PCA_COMPONENTS = config.get('n_pca_components', 3)
    LOG_SCALE = config.get('log_scale', False)

    # Resolve layer indices (handle string aliases like 'embedding' -> 'embed')
    def resolve_layer_idx(raw_value):
        # if raw_value == 'input' or raw_value == -2:
        #     raise ValueError("Source layer cannot be 'input' or -2 because the input space is not a part of the model layers.")
        try:
            return int(raw_value)
        except (ValueError, TypeError):
            key = LAYER_ALIASES.get(raw_value, raw_value)
            return LAYER_ARGUMENT_IDX_MAPPING[model_type][key]

    source_layer_idx = resolve_layer_idx(SOURCE_LAYER_IDX_RAW)
    target_layer_idx = resolve_layer_idx(TARGET_LAYER_IDX_RAW)
    target_is_logits = (target_layer_idx == 'logits')

    if METRIC not in METRIC_OPTIONS:
        raise ValueError(f"Metric {METRIC} not supported. Choose from {METRIC_OPTIONS}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_type != 'toy_resnet':
        raise ValueError(f"Model type {model_type} currently not supported")

    # Generate data once (shared across all seeds)
    # data = generate_data_around_point(REFERENCE_POINT, N_POINTS, RADIUS)
    data = generate_uniform_data(N_POINTS, RADIUS)

    # Create directory
    output_dir = f"plots/{MODEL_NAME}/plateaus"

    if args.multi_seed:
        assert model_type == 'toy_resnet', "--multi_seed is only supported for toy_resnet"
        print(f"Multi-seed mode: {len(model_paths)} seeds | METRIC: {METRIC}")

        all_metric_values = []
        first_seed_result = None

        for idx, checkpoint_path in enumerate(model_paths):
            print(f"  Seed {idx + 1}/{len(model_paths)}: {checkpoint_path}")
            result = compute_metric_for_checkpoint(
                checkpoint_path, data, REFERENCE_POINT, METRIC,
                source_layer_idx, target_layer_idx, device,
            )
            all_metric_values.append(result['metric_values'])
            if first_seed_result is None:
                first_seed_result = result

        # Average metric values across seeds
        stacked = torch.stack(all_metric_values)  # (n_seeds, n_points)
        mean_metric = stacked.mean(dim=0)          # (n_points,)
        n_seeds = len(model_paths)

        task_config = first_seed_result['task_config']
        metric_values = mean_metric

        # Title
        target_label = TARGET_LAYER_IDX_RAW if target_is_logits else f"layer {TARGET_LAYER_IDX_RAW}"
        source_label = SOURCE_LAYER_IDX_RAW if isinstance(SOURCE_LAYER_IDX_RAW, str) else f"layer {SOURCE_LAYER_IDX_RAW}"
        title = f"Plateau Visualization: {METRIC} (mean over {n_seeds} seeds, log scale: {LOG_SCALE})\nSource: {source_label}, Target: {target_label}"

        if METRIC == 'l2_norm' and 'predicted_class' in first_seed_result:
            title += f"\nRef point {REFERENCE_POINT.tolist()} — predicted class: {first_seed_result['predicted_class']}"

        output_path = f"{output_dir}/{METRIC}_src-{SOURCE_LAYER_IDX_RAW}_tgt-{TARGET_LAYER_IDX_RAW}.png"

    else:
        MODEL_NAME = model_paths[0]
        print(f"Model: {MODEL_NAME} | N_POINTS: {N_POINTS} | REFERENCE_POINT: {REFERENCE_POINT} | RADIUS: {RADIUS} | METRIC: {METRIC}")
        print(f"Source layer: {SOURCE_LAYER_IDX_RAW} ({source_layer_idx}) | Target layer: {TARGET_LAYER_IDX_RAW} ({target_layer_idx})")

        result = compute_metric_for_checkpoint(
            MODEL_NAME, data, REFERENCE_POINT, METRIC,
            source_layer_idx, target_layer_idx, device,
        )

        task_config = result['task_config']
        metric_values = result['metric_values']

        # Title
        target_label = TARGET_LAYER_IDX_RAW if target_is_logits else f"layer {TARGET_LAYER_IDX_RAW}"
        source_label = SOURCE_LAYER_IDX_RAW if isinstance(SOURCE_LAYER_IDX_RAW, str) else f"layer {SOURCE_LAYER_IDX_RAW}"
        title = f"Plateau Visualization: {METRIC} (log scale: {LOG_SCALE})\nSource: {source_label}, Target: {target_label}, Radius: {RADIUS}"
        if METRIC == 'l2_norm' and 'predicted_class' in result:
            title += f"\nRef point {REFERENCE_POINT.tolist()} — predicted class: {result['predicted_class']}"

        output_path = f"{output_dir}/{METRIC}_src-{SOURCE_LAYER_IDX_RAW}_tgt-{TARGET_LAYER_IDX_RAW}.png"

    if LOG_SCALE:
        metric_values = torch.log(metric_values + 1e-8)  # Add small constant to avoid log(0)
        METRIC += " (log scale)"

    print(f"Loaded model on {device} (task config: {task_config})")

    visualize_plateau(
        data=data,
        metric_values=metric_values,
        metric_name=METRIC,
        title=title,
        output_path=output_path,
        radius=RADIUS,
        reference_point=REFERENCE_POINT,
    )

    print("\n=== Complete ===")

if __name__ == "__main__":
    main()
