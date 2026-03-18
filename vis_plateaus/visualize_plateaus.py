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
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from tqdm import tqdm
import math

import sys
sys.path.append('./train')
from utils import load_config, load_model, load_model_from_checkpoint, construct_filepath, get_n_layers_from_model, get_model_names, get_model_name, LAYER_ARGUMENT_IDX_MAPPING
from compute_metrics import l2_norm_metric, jacobian_determinant_metric, jacobian_determinant_layerwise_prod_metric, jacobian_norm_metric, jacobian_norm_layerwise_prod_metric
from model import ResNetMLP, ResNetMLPSkeleton
from data import ToyDataset

config = load_config("vis_plateaus/config.yaml")

METRIC_OPTIONS = ['l2_norm', 'jacobian_determinant_full', 'jacobian_determinant_layerwise_prod', 'jacobian_norm_full', 'jacobian_norm_layerwise_prod']

LAYER_ALIASES = {'embedding': 'embed'}

# Muted/pastel colors for background class overlay (up to 10 classes)
CLASS_COLORS = [
    '#F5E6A0',  # pastel yellow
    '#A8D5BA',  # pastel green
    '#F5C6A0',  # pastel orange
    '#B5C8E8',  # pastel blue
    '#E8B5D3',  # pastel pink
    '#D4D4AA',  # pastel olive
    '#C5A8D5',  # pastel purple
    '#A8D5D5',  # pastel teal
    '#D5A8A8',  # pastel red
    '#A8C5A8',  # pastel sage
]


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
    source_activations: torch.Tensor,
    metric_values: torch.Tensor,
    metric_name: str,
    title: str,
    output_path: str,
    n_pca_components: int = 3,
    reference_point_source_act: torch.Tensor = None,
    model: Union[ResNetMLP, ResNetMLPSkeleton] = None,
    source_layer_idx: int = None,
    device: str = 'cpu',
    task_config: dict = None,
):
    """
    Create a scatter plot of data points in source-layer space, colored by metric value,
    with a faint background overlay of the training dataset transformed to the same space.
    When the source dimension exceeds n_pca_components, PCA is used to reduce dimensionality.

    Args:
        source_activations: (n_points, d_source) — activations at source layer for sampled points.
        metric_values: (n_points,) — computed metric value per point.
        metric_name: Label for the colorbar.
        title: Plot title.
        output_path: Where to save the plot.
        n_pca_components: Number of PCA components to reduce to (2 or 3).
        reference_point_source_act: (d_source,) — reference point in source space (for marking).
        model: Model used to transform background data to source space.
        source_layer_idx: Source layer index for transforming background data.
        device: Device string.
        task_config: Task config dict from checkpoint (keys: name, num_classes, noise_std, distribution, etc.).
    """
    d_source = source_activations.shape[1]
    use_pca = d_source > n_pca_components
    n_plot_dims = n_pca_components if use_pca else d_source
    use_3d = n_plot_dims >= 3


    # --- Create figure ---
    fig = plt.figure(figsize=(9, 8), dpi=300)
    if use_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    # --- PCA fitting (on source activations) using sklearn ---
    pca = None
    if use_pca:
        print(f"Source dimension {d_source} exceeds {n_pca_components}, applying PCA for visualization...")
        pca = PCA(n_components=n_plot_dims)
        src_np = pca.fit_transform(source_activations.numpy())
    else:
        src_np = source_activations.numpy()


    # --- Resolve task config for background overlay ---
    num_classes = 2
    noise_std = 0.0
    distribution = 'uniform'
    task_name = 'class_spiral'
    if task_config is not None:
        num_classes = task_config.get('num_classes', 2)
        noise_std = task_config.get('noise_std', 0.0)
        # distribution = task_config.get('distribution', 'uniform')
        task_name = task_config.get('name', 'class_spiral')

    # --- Background: dataset overlay in source layer space ---
    bg_legend_handles = []
    if model is not None and source_layer_idx is not None:
        bg_dataset = ToyDataset(
            task_name=task_name, num_samples=10,
            num_classes=num_classes, noise_std=noise_std,
            distribution=distribution, seed=42
        )
        bg_X = bg_dataset.data   # (2500, 2)
        bg_y = bg_dataset.targets

        if source_layer_idx <= -2:
            bg_source_np = bg_X.numpy()
        else:
            bg_acts = collect_activations_resid_post(
                model, bg_X.to(device),
                source_layer_idx=-2, target_layer_idx=source_layer_idx, device=device
            )
            bg_source_np = bg_acts[f'layer{source_layer_idx-1}_resid_post'].numpy()

        if use_pca:
            bg_source_np = pca.transform(bg_source_np)

        # Get class labels
        if bg_y.dim() > 1 and bg_y.shape[-1] > 1:
            # One-hot encoded (num_classes > 2)
            bg_labels = bg_y.argmax(dim=1).numpy()
        else:
            # Scalar targets: shape (n,) or (n, 1) for binary classification
            bg_labels = bg_y.squeeze().numpy()
            # Binarize: threshold at 0.5 for sigmoid-style outputs
            bg_labels = (bg_labels >= 0.5).astype(int)

        unique_classes = sorted(set(bg_labels.tolist()))

        # Plot each class with a distinct pastel color
        for cls_idx in unique_classes:
            mask = bg_labels == cls_idx
            color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
            if use_3d:
                ax.scatter(bg_source_np[mask, 0], bg_source_np[mask, 1], bg_source_np[mask, 2],
                           c=color, s=8, alpha=0.5, zorder=1, marker='o')
            else:
                ax.scatter(bg_source_np[mask, 0], bg_source_np[mask, 1],
                           c=color, s=8, alpha=0.5, zorder=1, marker='o')

            bg_legend_handles.append(
                mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                              markersize=5, label=f'Class {cls_idx}')
            )

    # --- Main scatter: source activations colored by metric ---
    vals = metric_values.numpy()

    if use_3d:
        scatter = ax.scatter(src_np[:, 0], src_np[:, 1], src_np[:, 2], c=vals, cmap='coolwarm',
                             s=10, alpha=0.7, zorder=2, edgecolors='none')
    else:
        scatter = ax.scatter(src_np[:, 0], src_np[:, 1], c=vals, cmap='coolwarm',
                             s=10, alpha=0.7, zorder=2, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label=metric_name, shrink=0.7 if use_3d else 1.0, pad=0.1)

    # --- Mark reference point ---
    if reference_point_source_act is not None and metric_name == 'l2_norm':
        ref = reference_point_source_act.numpy()
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        if use_pca:
            ref = pca.transform(ref)
        ref = ref.squeeze()

        if use_3d:
            ax.scatter([ref[0]], [ref[1]], [ref[2]], c='black', marker='x', s=100, zorder=3, linewidths=2)
        else:
            ax.scatter(ref[0], ref[1], c='black', marker='x', s=100, zorder=3, linewidths=2)

    # --- Labels and Limits ---
    ax.set_title(title, fontsize=12)
    if use_pca:
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        if use_3d:
            ax.set_zlabel('PC 3')
    else:
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        if use_3d:
            ax.set_zlabel('Dim 3')
    
    if d_source == 2: # If the source space is 2D, assume it is the input space.
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        if use_3d:
            ax.set_zlim(-1, 1)

    # --- Class legend (discrete, inside plot) ---
    if bg_legend_handles:
        ax.legend(handles=bg_legend_handles, title='Training data',
                  loc='lower right',
                  fontsize=8, title_fontsize=9, framealpha=0.8)

    ax.grid(True, linestyle='--', alpha=0.3)
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
    TARGET_LAYER_IDX_RAW = config['target_layer_idx']
    N_PCA_COMPONENTS = config.get('n_pca_components', 3)
    LOG_SCALE = config.get('log_scale', False)

    # Resolve layer indices (handle string aliases like 'embedding' -> 'embed')
    def resolve_layer_idx(raw_value):
        if raw_value == 'input' or raw_value == -2:
            raise ValueError("Source layer cannot be 'input' or -2 because the input space is not a part of the model layers.")
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

        # Use first seed's spatial layout for plotting
        source_activations = first_seed_result['source_activations']
        ref_source_act = first_seed_result['ref_source_act']
        task_config = first_seed_result['task_config']
        model = first_seed_result['model']

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

        source_activations = result['source_activations']
        ref_source_act = result['ref_source_act']
        task_config = result['task_config']
        model = result['model']
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
        source_activations=source_activations,
        metric_values=metric_values,
        metric_name=METRIC,
        title=title,
        output_path=output_path,
        n_pca_components=N_PCA_COMPONENTS,
        reference_point_source_act=ref_source_act,
        model=model,
        source_layer_idx=source_layer_idx,
        device=device,
        task_config=task_config,
    )

    print("\n=== Complete ===")

if __name__ == "__main__":
    main()
