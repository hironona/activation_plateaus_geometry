#!/usr/bin/env python3
"""
Visualize the pre-image of open balls in target space, drawn in source/input space.

For each distance level r, the pre-image set {x : ||f(x) - f(ref)||_2 < r}
(where f maps input to target-layer activations) is shown as nested shaded regions
in source/input space, with contour boundary lines at each level.

Plot modes:
  - source_layer_idx == -2 (input space): regular contourf on the 2D grid
  - d_source == 2, not input space: tricontourf on source activations
  - d_source > 2: PCA + scatter colored by L2 distance level
"""

from typing import Union
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
from sklearn.decomposition import PCA
import math

import sys
sys.path.append('./train')
from utils import load_config, load_model_from_checkpoint, get_model_names, get_model_name, LAYER_ARGUMENT_IDX_MAPPING
from visualize_plateaus import collect_activations_resid_post, generate_uniform_data
from model import ResNetMLP, ResNetMLPSkeleton

LAYER_ALIASES = {'embedding': 'embed'}
SUB_REF_COLOR = '#CD5C5C'  # indianred (muted red)

config = load_config("vis_plateaus/contour_config.yaml")


def compute_for_checkpoint(
    checkpoint_path: str,
    data: torch.Tensor,
    reference_point: torch.Tensor,
    source_layer_idx: int,
    target_layer_idx,
    device: str,
    sub_reference_points: torch.Tensor = None,
) -> dict:
    """
    Load checkpoint and compute:
      - L2 distances in target space from the reference point's target activation
      - Source activations (input data itself when source_layer_idx == -2)
      - Reference source activation (for marking in plots)
      - Sub-reference source activations (optional)

    Returns dict with keys:
        l2_distances: (n_points,)
        source_activations: (n_points, d_source)
        ref_source_act: (d_source,)
        sub_ref_source_acts: (n_sub, d_source) or None
        task_config: dict
        model: loaded model
    """
    model, model_config, full_config = load_model_from_checkpoint(checkpoint_path)
    task_config = full_config['task']
    model = model.to(device)
    n_blocks = len(model.blocks)
    target_is_logits = (target_layer_idx == 'logits')
    effective_target = n_blocks - 1 if target_is_logits else target_layer_idx

    # collect_activations_resid_post requires source_layer_idx >= -1;
    # for input space (source_layer_idx == -2) we use -1 as the effective source
    effective_source = max(source_layer_idx, -1)

    # --- Collect activations for all grid points ---
    acts = collect_activations_resid_post(
        model, data.to(device),
        source_layer_idx=effective_source,
        target_layer_idx=effective_target,
        device=device,
    )

    # Source activations
    if source_layer_idx == -2:
        source_acts = data.clone()  # (n_points, 2), raw input
    else:
        # TODO: Sanity check: if source_layer_idx == -1 (embedding), should we use the post-embedding activations?
        # source_acts = acts[f'layer{source_layer_idx - 1}_resid_post']
        source_acts = acts[f'layer{source_layer_idx}_resid_post']

    # Target activations
    target_acts = (acts['logits'] if target_is_logits
                   else acts[f'layer{effective_target}_resid_post'])

    # --- Collect activations for the reference point ---
    ref_acts = collect_activations_resid_post(
        model, reference_point.to(device).unsqueeze(0),
        source_layer_idx=effective_source,
        target_layer_idx=effective_target,
        device=device,
    )

    ref_target = (ref_acts['logits'].squeeze(0) if target_is_logits
                  else ref_acts[f'layer{effective_target}_resid_post'].squeeze(0))

    # Reference point in source space (for marking)
    if source_layer_idx == -2:
        ref_source_act = reference_point.clone().cpu()
    else:
        # ref_source_act = ref_acts[f'layer{source_layer_idx - 1}_resid_post'].squeeze(0).cpu()
        ref_source_act = ref_acts[f'layer{source_layer_idx}_resid_post'].squeeze(0).cpu()

    # --- Sub-reference points source activations ---
    sub_ref_source_acts = None
    if sub_reference_points is not None:
        if source_layer_idx == -2:
            sub_ref_source_acts = sub_reference_points.clone().cpu()
        else:
            sub_acts = collect_activations_resid_post(
                model, sub_reference_points.to(device),
                source_layer_idx=effective_source,
                target_layer_idx=effective_target,
                device=device,
            )
            sub_ref_source_acts = sub_acts[f'layer{source_layer_idx}_resid_post'].cpu()

    # --- L2 distances in target space ---
    l2_dist = torch.norm(
        target_acts.float() - ref_target.float().unsqueeze(0), dim=1
    ).cpu()

    return {
        'l2_distances': l2_dist,
        'source_activations': source_acts.cpu(),
        'ref_source_act': ref_source_act,
        'sub_ref_source_acts': sub_ref_source_acts,
        'task_config': task_config,
        'model': model,
    }


def _compute_levels(vals: np.ndarray, n_levels: int, spacing: str) -> np.ndarray:
    z_min, z_max = float(vals.min()), float(vals.max())
    if spacing == "log":
        return np.logspace(
            np.log10(max(z_min, 1e-8)), np.log10(max(z_max, 1e-7)), n_levels
        )
    return np.linspace(z_min, z_max, n_levels)


def _draw_reference_annotations(ax, ref_source_act, sub_ref_source_acts,
                                reference_point, sub_reference_points,
                                pca=None, is_3d=False):
    """
    Draw reference marker, sub-reference markers, dashed lines, and coordinate labels.

    ref_source_act / sub_ref_source_acts are in source space (may need PCA transform).
    reference_point / sub_reference_points are original input-space coords (for labels).
    """
    if ref_source_act is None:
        return

    # Compute reference plot position
    if pca is not None:
        ref_pos = pca.transform(ref_source_act.numpy().reshape(1, -1)).squeeze()
    else:
        ref_pos = ref_source_act.numpy()

    # Reference marker
    if is_3d:
        ax.scatter([ref_pos[0]], [ref_pos[1]], [ref_pos[2]],
                   c='red', marker='x', s=150, zorder=5, linewidths=2.5)
    else:
        ax.scatter(ref_pos[0], ref_pos[1], c='red', marker='x', s=150,
                   zorder=5, linewidths=2.5)

    # Reference coordinate label
    if reference_point is not None and not is_3d:
        coord_str = f"[{reference_point[0]:.2f}, {reference_point[1]:.2f}]"
        ax.annotate(coord_str, (ref_pos[0], ref_pos[1]),
                    textcoords='offset points', xytext=(8, 8),
                    fontsize=7, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7,
                              edgecolor='none'),
                    zorder=6)

    # Sub-reference points
    if sub_ref_source_acts is None or sub_reference_points is None:
        return

    # Compute sub-reference plot positions
    if pca is not None:
        sub_positions = pca.transform(sub_ref_source_acts.numpy())
    else:
        sub_positions = sub_ref_source_acts.numpy()

    for i in range(len(sub_reference_points)):
        sp = sub_positions[i]
        sub_input = sub_reference_points[i]

        # Dashed line from reference to sub-reference
        if is_3d:
            ax.plot([ref_pos[0], sp[0]], [ref_pos[1], sp[1]], [ref_pos[2], sp[2]],
                    color=SUB_REF_COLOR, linestyle='--', linewidth=1.2, alpha=0.8,
                    zorder=4)
            ax.scatter([sp[0]], [sp[1]], [sp[2]],
                       c=SUB_REF_COLOR, marker='o', s=100, zorder=5,
                       edgecolors='darkred', linewidths=1)
        else:
            ax.plot([ref_pos[0], sp[0]], [ref_pos[1], sp[1]],
                    color=SUB_REF_COLOR, linestyle='--', linewidth=1.2, alpha=0.8,
                    zorder=4)
            ax.scatter(sp[0], sp[1],
                       c=SUB_REF_COLOR, marker='o', s=100, zorder=5,
                       edgecolors='darkred', linewidths=1)

            # Coordinate label
            coord_str = f"[{sub_input[0]:.2f}, {sub_input[1]:.2f}]"
            ax.annotate(coord_str, (sp[0], sp[1]),
                        textcoords='offset points', xytext=(8, -12),
                        fontsize=7, color=SUB_REF_COLOR, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7,
                                  edgecolor='none'),
                        zorder=6)


def visualize_open_ball_preimage(
    source_activations: torch.Tensor,
    l2_distances: torch.Tensor,
    grid_size: int,
    radius: float,
    title: str,
    output_path: str,
    n_levels: int = 15,
    level_spacing: str = "even",
    n_pca_components: int = 3,
    ref_source_act: torch.Tensor = None,
    reference_point: torch.Tensor = None,
    sub_ref_source_acts: torch.Tensor = None,
    sub_reference_points: torch.Tensor = None,
    source_layer_idx: int = None,
):
    """
    Visualize the pre-image of open balls.

    For each distance level r, the set {x : l2_dist(x) < r} is shown as a
    shaded region in source/input space; contour lines mark the boundaries.

    Plot modes (selected automatically):
      - source_layer_idx == -2: regular contourf on the 2D input grid
      - d_source == 2, not input space: tricontourf
      - d_source > 2: PCA scatter colored by L2 distance
    """
    d_source = source_activations.shape[1]
    vals = l2_distances.numpy()
    levels = _compute_levels(vals, n_levels, level_spacing)
    # BoundaryNorm gives each level band an equal colormap slice,
    # so regions are distinguishable even with log-spaced (very close) levels.
    norm = mcolors.BoundaryNorm(levels, ncolors=plt.cm.Blues_r.N)

    is_input_space = (source_layer_idx == -2)
    use_regular = is_input_space
    use_tri = (d_source == 2 and not is_input_space)
    use_pca = (d_source > 2)

    pca_obj = None
    is_3d_plot = False

    # =========================================================
    # PATH 1: Regular contourf on 2D input grid
    # =========================================================
    if use_regular:
        fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

        x_coords = np.linspace(-radius, radius, grid_size)
        y_coords = np.linspace(-radius, radius, grid_size)
        X, Y = np.meshgrid(x_coords, y_coords)
        # generate_uniform_data: outer loop x1, inner loop x2
        # index i*grid_size + j → (x_coords[i], y_coords[j])
        # contourf expects Z[y_idx, x_idx] → transpose
        Z = vals.reshape(grid_size, grid_size).T

        cf = ax.contourf(X, Y, Z, levels=levels, cmap='Blues_r', norm=norm,
                         alpha=0.75, zorder=2, extend='both')
        cs = ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.7,
                        alpha=0.6, zorder=3)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.3g')

        plt.colorbar(cf, ax=ax, label='L2 distance in target space',
                     shrink=1.0, pad=0.02)

        ax.set_xlabel('Input Dim 1')
        ax.set_ylabel('Input Dim 2')
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)

    # =========================================================
    # PATH 2: Triangulated contourf on 2D source activations
    # =========================================================
    elif use_tri:
        fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

        src_np = source_activations.numpy()
        triang = mtri.Triangulation(src_np[:, 0], src_np[:, 1])

        cf = ax.tricontourf(triang, vals, levels=levels, cmap='Blues_r', norm=norm,
                            alpha=0.75, zorder=2, extend='both')
        cs = ax.tricontour(triang, vals, levels=levels, colors='k',
                           linewidths=0.7, alpha=0.6, zorder=3)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.3g')

        plt.colorbar(cf, ax=ax, label='L2 distance in target space',
                     shrink=1.0, pad=0.02)

        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

    # =========================================================
    # PATH 3: PCA projection
    #   - 2D PCA: tricontourf + tricontour (same as Path 2)
    #   - 3D PCA: scatter colored by L2 distance level
    # =========================================================
    elif use_pca:
        n_plot_dims = min(n_pca_components, d_source)
        is_3d_plot = n_plot_dims >= 3

        pca_obj = PCA(n_components=n_plot_dims)
        src_np = pca_obj.fit_transform(source_activations.numpy())

        if not is_3d_plot:
            # 2D PCA — use triangulated contourf, identical to Path 2
            fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

            triang = mtri.Triangulation(src_np[:, 0], src_np[:, 1])
            cf = ax.tricontourf(triang, vals, levels=levels, cmap='Blues_r', norm=norm,
                                alpha=0.75, zorder=2, extend='both')
            cs = ax.tricontour(triang, vals, levels=levels, colors='k',
                               linewidths=0.7, alpha=0.6, zorder=3)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%.3g')

            plt.colorbar(cf, ax=ax, label='L2 distance in target space',
                         shrink=1.0, pad=0.02)

        else:
            # 3D PCA — scatter only (tricontour is 2D only)
            fig = plt.figure(figsize=(9, 8), dpi=300)
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(src_np[:, 0], src_np[:, 1], src_np[:, 2],
                                 c=vals, cmap='Blues_r', norm=norm,
                                 s=10, alpha=0.7, zorder=2, edgecolors='none')

            plt.colorbar(scatter, ax=ax, label='L2 distance in target space',
                         shrink=0.7, pad=0.1)

            ax.set_zlabel('PC 3')

        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        if is_3d_plot:
            ax.set_zlabel('PC 3')

    # --- Reference & sub-reference annotations (all paths) ---
    _draw_reference_annotations(
        ax, ref_source_act, sub_ref_source_acts,
        reference_point, sub_reference_points,
        pca=pca_obj, is_3d=is_3d_plot,
    )

    # --- Common ---
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize pre-image of open balls in target space'
    )
    parser.add_argument('--model_type', type=str, choices=['toy_resnet'], required=True)
    parser.add_argument('--data_type', type=str, choices=['class_spiral'], required=True)
    parser.add_argument('--model_path', type=str,
                        help='Path to checkpoint file or directory')
    parser.add_argument('--multi_seed', action='store_true',
                        help='Average L2 distances across all seed checkpoints')
    args = parser.parse_args()

    model_type = args.model_type

    if args.model_path:
        config['model_names'][model_type] = args.model_path
    model_paths = get_model_names(config, model_type)
    MODEL_NAME = get_model_name(config, model_type)

    N_POINTS      = config['n_points']
    RADIUS        = config['radius']
    REFERENCE_POINT = torch.tensor(config['reference_point'], dtype=torch.float32)
    SOURCE_LAYER_IDX_RAW = config['source_layer_idx']
    TARGET_LAYER_IDX_RAW = config['target_layer_idx']
    N_PCA         = config.get('n_pca_components', 3)
    N_LEVELS      = config.get('n_contour_levels', 15)
    LEVEL_SPACING = config.get('contour_level_spacing', 'even')

    # Sub-reference points (None if not set or null)
    raw_sub_refs = config.get('sub_reference_points', None)
    SUB_REFERENCE_POINTS = None
    if raw_sub_refs is not None:
        SUB_REFERENCE_POINTS = torch.tensor(raw_sub_refs, dtype=torch.float32)

    def resolve_layer_idx(raw_value):
        try:
            return int(raw_value)          # handles -2, -1, 0, 1, ... directly
        except (ValueError, TypeError):
            key = LAYER_ALIASES.get(raw_value, raw_value)
            return LAYER_ARGUMENT_IDX_MAPPING[model_type][key]  # 'logits' → 'logits'

    source_layer_idx = resolve_layer_idx(SOURCE_LAYER_IDX_RAW)
    target_layer_idx = resolve_layer_idx(TARGET_LAYER_IDX_RAW)  # int or 'logits'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = generate_uniform_data(N_POINTS, RADIUS)
    grid_size = int(math.sqrt(N_POINTS))

    output_dir = f"plots/{MODEL_NAME}/contours"

    src_label = 'input' if source_layer_idx == -2 else f"layer {SOURCE_LAYER_IDX_RAW}"
    tgt_label = TARGET_LAYER_IDX_RAW

    if args.multi_seed:
        print(f"Multi-seed mode: {len(model_paths)} seeds")
        all_l2 = []
        first_result = None

        for idx, ckpt in enumerate(model_paths):
            print(f"  Seed {idx + 1}/{len(model_paths)}: {ckpt}")
            result = compute_for_checkpoint(
                ckpt, data, REFERENCE_POINT, source_layer_idx, target_layer_idx, device,
                sub_reference_points=SUB_REFERENCE_POINTS,
            )
            all_l2.append(result['l2_distances'])
            if first_result is None:
                first_result = result

        l2_distances = torch.stack(all_l2).mean(dim=0)   # (n_points,)
        n_seeds = len(model_paths)
        source_activations = first_result['source_activations']
        ref_source_act     = first_result['ref_source_act']
        sub_ref_source_acts = first_result['sub_ref_source_acts']

        title = (
            f"Pre-image of Open Balls (mean over {n_seeds} seeds)\n"
            f"Source: {src_label}, Target: {tgt_label} | "
            f"Ref: {REFERENCE_POINT.tolist()} | "
            f"Spacing: {LEVEL_SPACING}"
        )

    else:
        ckpt = model_paths[0]
        print(f"Checkpoint: {ckpt}")
        result = compute_for_checkpoint(
            ckpt, data, REFERENCE_POINT, source_layer_idx, target_layer_idx, device,
            sub_reference_points=SUB_REFERENCE_POINTS,
        )
        l2_distances        = result['l2_distances']
        source_activations  = result['source_activations']
        ref_source_act      = result['ref_source_act']
        sub_ref_source_acts = result['sub_ref_source_acts']

        title = (
            f"Pre-image of Open Balls\n"
            f"Source: {src_label}, Target: {tgt_label} | "
            f"Ref: {REFERENCE_POINT.tolist()} | "
            f"Spacing: {LEVEL_SPACING}"
        )

    print(f"L2 distance range: [{l2_distances.min():.4f}, {l2_distances.max():.4f}]")

    output_path = (f"{output_dir}/"
                   f"contours_src_{SOURCE_LAYER_IDX_RAW}_tgt_{TARGET_LAYER_IDX_RAW}_ref_{REFERENCE_POINT.tolist()}_spacing_{LEVEL_SPACING}.png")

    visualize_open_ball_preimage(
        source_activations=source_activations,
        l2_distances=l2_distances,
        grid_size=grid_size,
        radius=RADIUS,
        title=title,
        output_path=output_path,
        n_levels=N_LEVELS,
        level_spacing=LEVEL_SPACING,
        n_pca_components=N_PCA,
        ref_source_act=ref_source_act,
        reference_point=REFERENCE_POINT,
        sub_ref_source_acts=sub_ref_source_acts,
        sub_reference_points=SUB_REFERENCE_POINTS,
        source_layer_idx=source_layer_idx,
    )

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
