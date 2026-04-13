"""Plot training curves from saved metrics.json files.

Usage:
    python train/plot_metrics.py --checkpoint_dir checkpoints/class_spiral/ResNetMLP/noise-0.1/20240101_120000
    python train/plot_metrics.py --checkpoint_dir checkpoints/class_spiral/ResNetMLP/noise-0.1-multi_seed_5_runs  # aggregates all seed runs
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def load_metrics(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_single_run(metrics, save_path):
    """Plot training curves for a single run."""
    epochs = np.arange(1, len(metrics['train_loss']) + 1)
    has_acc = 'train_acc' in metrics

    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(12 if has_acc else 6, 4))
    if not has_acc:
        axes = [axes]

    # Loss plot
    ax = axes[0]
    ax.plot(epochs, metrics['train_loss'], label='Train Loss')
    ax.plot(epochs, metrics['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    if has_acc:
        ax = axes[1]
        ax.plot(epochs, metrics['train_acc'], label='Train Acc')
        ax.plot(epochs, metrics['val_acc'], label='Val Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_multi_seed(all_metrics, save_path):
    """Plot training curves aggregated across multiple seeds with confidence intervals."""
    has_acc = 'train_acc' in all_metrics[0]
    n_epochs = len(all_metrics[0]['train_loss'])
    epochs = np.arange(1, n_epochs + 1)

    # Stack metrics across seeds
    train_losses = np.array([m['train_loss'] for m in all_metrics])
    val_losses = np.array([m['val_loss'] for m in all_metrics])

    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(12 if has_acc else 6, 4))
    if not has_acc:
        axes = [axes]

    n_seeds = len(all_metrics)
    ci_factor = 1.96 / np.sqrt(n_seeds)

    # Loss plot
    ax = axes[0]
    mean_tl, std_tl = train_losses.mean(axis=0), train_losses.std(axis=0)
    mean_vl, std_vl = val_losses.mean(axis=0), val_losses.std(axis=0)
    ci_tl, ci_vl = std_tl * ci_factor, std_vl * ci_factor
    ax.plot(epochs, mean_tl, label='Train Loss')
    ax.fill_between(epochs, mean_tl - ci_tl, mean_tl + ci_tl, alpha=0.2)
    ax.plot(epochs, mean_vl, label='Val Loss')
    ax.fill_between(epochs, mean_vl - ci_vl, mean_vl + ci_vl, alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss (n={n_seeds} seeds, 95% CI)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    if has_acc:
        train_accs = np.array([m['train_acc'] for m in all_metrics])
        val_accs = np.array([m['val_acc'] for m in all_metrics])
        mean_ta, std_ta = train_accs.mean(axis=0), train_accs.std(axis=0)
        mean_va, std_va = val_accs.mean(axis=0), val_accs.std(axis=0)
        ci_ta, ci_va = std_ta * ci_factor, std_va * ci_factor

        ax = axes[1]
        ax.plot(epochs, mean_ta, label='Train Acc')
        ax.fill_between(epochs, mean_ta - ci_ta, mean_ta + ci_ta, alpha=0.2)
        ax.plot(epochs, mean_va, label='Val Acc')
        ax.fill_between(epochs, mean_va - ci_va, mean_va + ci_va, alpha=0.2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy (n={n_seeds} seeds, 95% CI)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint dir (single run) or parent dir (multi-seed)")
    args = parser.parse_args()

    metrics_file = os.path.join(args.checkpoint_dir, "metrics.json")

    if os.path.isfile(metrics_file):
        # Single run
        metrics = load_metrics(metrics_file)
        save_path = os.path.join("plots", os.path.relpath(args.checkpoint_dir), "training_curves.png")
        plot_single_run(metrics, save_path)
    else:
        # Multi-seed: look for timestamp subdirs containing metrics.json
        all_metrics = []
        for subdir in sorted(os.listdir(args.checkpoint_dir)):
            mf = os.path.join(args.checkpoint_dir, subdir, "metrics.json")
            if os.path.isfile(mf):
                all_metrics.append(load_metrics(mf))

        if not all_metrics:
            print(f"No metrics.json found in {args.checkpoint_dir}")
            return

        if len(all_metrics) == 1:
            save_path = os.path.join("plots", os.path.relpath(args.checkpoint_dir), "training_curves.png")
            plot_single_run(all_metrics[0], save_path)
        else:
            save_path = os.path.join("plots", os.path.relpath(args.checkpoint_dir), "training_curves.png")
            plot_multi_seed(all_metrics, save_path)


if __name__ == "__main__":
    main()
