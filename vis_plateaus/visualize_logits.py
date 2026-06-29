#!/usr/bin/env python3
"""
Visualize the scalar logit of a binary-classification toy ResNet over a 2D
input grid. Averages over seed checkpoints listed in vis_plateaus/config.yaml.
"""

import os
import math
import argparse
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append('./train')
sys.path.append('./vis_plots')

from utils import load_config, load_model_from_checkpoint, get_model_names, get_model_name
from data import ToyDataset
from visualize_plateaus import generate_uniform_data


def compute_logit_grid(checkpoint_path, data, device):
    model, _model_cfg, full_cfg = load_model_from_checkpoint(checkpoint_path)
    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(data.to(device)).cpu()
    return logits, full_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None,
                        help='Override model_names.toy_resnet from vis_plateaus/config.yaml')
    parser.add_argument('--n_points', type=int, default=None,
                        help='Override n_points (square grid resolution = sqrt(n_points))')
    parser.add_argument('--radius', type=float, default=None,
                        help='Override radius (half-extent of grid)')
    parser.add_argument('--no_overlay', action='store_true',
                        help='Skip dataset scatter overlay')
    args = parser.parse_args()

    cfg = load_config('vis_plateaus/config.yaml')
    if args.model_path:
        cfg['model_names']['toy_resnet'] = args.model_path

    n_points = args.n_points or cfg['n_points']
    radius = args.radius or cfg['radius']

    model_paths = get_model_names(cfg, 'toy_resnet')
    model_name = get_model_name(cfg, 'toy_resnet')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Model: {model_name}")
    print(f"Seeds: {len(model_paths)} | grid: {n_points} pts | radius: {radius}")

    data = generate_uniform_data(n_points, radius)

    all_logits = []
    task_cfg = None
    for i, ckpt in enumerate(model_paths):
        print(f"  seed {i+1}/{len(model_paths)}: {ckpt}")
        logits, full_cfg = compute_logit_grid(ckpt, data, device)
        if logits.shape[-1] != 1:
            raise ValueError(
                f"Expected scalar logits (output_dim=1) for binary classification, "
                f"got shape {tuple(logits.shape)}."
            )
        all_logits.append(logits.squeeze(-1))
        task_cfg = full_cfg['task']

    stacked = torch.stack(all_logits, dim=0)        # (n_seeds, n_points)
    mean_logit = stacked.mean(dim=0).numpy()
    std_logit = stacked.std(dim=0).numpy()

    grid_size = int(math.sqrt(n_points))
    # generate_uniform_data: outer loop over x1, inner over x2 -> reshape(grid, grid).T so
    # axis 0 is x2 (vertical) and axis 1 is x1 (horizontal). Matches imshow(row=y, col=x).
    mean_grid = mean_logit.reshape(grid_size, grid_size).T
    std_grid = std_logit.reshape(grid_size, grid_size).T

    # Symmetric colormap around 0 so the decision boundary (logit=0) is visible.
    vmax = float(np.max(np.abs(mean_grid)))
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=200)

    # --- left: mean logit ---
    ax = axes[0]
    im = ax.imshow(
        mean_grid, origin='lower', extent=(-radius, radius, -radius, radius),
        cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest',
    )
    ax.contour(
        mean_grid, levels=[0.0], origin='lower',
        extent=(-radius, radius, -radius, radius),
        colors='black', linewidths=1.2,
    )
    plt.colorbar(im, ax=ax, label='mean logit', pad=0.02)

    # Overlay training dataset (binary spiral)
    if not args.no_overlay and task_cfg is not None:
        try:
            ds = ToyDataset(
                task_name=task_cfg.get('name', 'class_spiral'),
                num_samples=task_cfg.get('num_samples', 5000),
                noise_std=task_cfg.get('noise_std', 0.05),
                num_classes=task_cfg.get('num_classes', 2),
                distribution=task_cfg.get('distribution', 'uniform'),
                seed=42,
            )
            X = ds.data
            y = ds.targets
            X = X.numpy() if torch.is_tensor(X) else np.asarray(X)
            y = y.numpy() if torch.is_tensor(y) else np.asarray(y)
            y = y.squeeze()
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
                       s=2, alpha=0.25, edgecolors='none')
        except Exception as e:
            print(f"  (skipped dataset overlay: {e})")

    ax.set_title(f"Mean logit over {len(model_paths)} seeds\n"
                 f"noise_std={task_cfg.get('noise_std') if task_cfg else '?'} "
                 f"| black contour = decision boundary (logit=0)")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)

    # --- right: across-seed std (disagreement) ---
    ax = axes[1]
    im = ax.imshow(
        std_grid, origin='lower', extent=(-radius, radius, -radius, radius),
        cmap='magma', aspect='equal', interpolation='nearest',
    )
    plt.colorbar(im, ax=ax, label='std of logit across seeds', pad=0.02)
    ax.set_title('Across-seed std of logit')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)

    plt.tight_layout()

    out_dir = f"plots/{model_name}/plateaus"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/logit_grid.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")
    print(f"logit range: [{mean_grid.min():.3f}, {mean_grid.max():.3f}]")


if __name__ == '__main__':
    main()
