# Activation Plateau Geometry

> [!IMPORTANT]
> **Work in Progress**: This project is actively under development on the `dev` branch.

Research project analyzing the geometry of activation plateaus and residual stream dynamics in ResNet-like MLPs trained on toy 2D tasks. Investigates how activation representations evolve through residual blocks via interpolation, Jacobian analysis, and geometric visualization of plateau structures.

Forked from [MShinkle/activation_plateau_mechanisms](https://github.com/MShinkle/activation_plateau_mechanisms).

## Pipeline Overview

The project follows a four-phase workflow, focused on toy ResNet models trained on 2D classification tasks:

1. **Train** (`train/`) — Train `ResNetMLP` (or `ResNetMLPSkeleton`) on synthetic tasks (e.g., spiral classification). Supports multi-seed runs.
2. **Record activations** (`vis_plots/interpolate_and_record_activations.py`) — Interpolate between input pairs (simple linear interpolation), run through the model, and record residual stream activations at every layer. Saves `.pt` files to `activations/`.
3. **Plot interpolation metrics** (`vis_plots/*_plots.py`, `vis_plots/jacobians_*.py`) — Load recorded activations and plot step sizes, relative distances, spline (Hamming) distances, and Jacobian norms across layers. All plotting scripts support `--multi_seed` for aggregation with confidence intervals.
4. **Visualize plateau geometry** (`vis_plateaus/`) — Sample points in input/activation space, color by a metric (L2 norm, Jacobian norm/determinant), and plot directly. Includes contour visualization of open-ball pre-images.

## Quick Start

```bash
# 1. Install
uv sync

# 2. Train (multi-seed)
python train/train.py --config train/config.yaml --multi_seed

# 3. Plot training curves
python train/plot_metrics.py --checkpoint_dir checkpoints/class_spiral/ResNetMLP/<config_name>

# 4. Record activations & generate metric plots
bash run/toy_resnet_vis.sh

# 5. Visualize plateau geometry (loops over noise levels)
bash run/vis_plateaus.sh

# 6. Visualize contour pre-images
bash run/vis_contours.sh
```

All scripts must be run from the project root.

## Project Structure

```
.
├── train/                  # Training pipeline
│   ├── train.py           # Training entry point (--dry_run, --multi_seed)
│   ├── model.py           # ResNetMLP, ResNetMLPSkeleton architectures
│   ├── data.py            # ToyDataset: class_spiral, reg_sine_wave
│   ├── plot_metrics.py    # Plot training curves (single or multi-seed)
│   ├── upload_hf.py       # Upload checkpoints to HuggingFace Hub
│   └── config.yaml        # Training config
├── vis_plots/              # Interpolation-based analysis
│   ├── interpolate_and_record_activations.py  # Record activations along interpolation paths
│   ├── step_sizes_plots.py                    # L2 step sizes per layer
│   ├── relative_distances_layerwise_plots.py  # Relative distances (3 variants + --single_layer)
│   ├── relative_distances_logits_plots.py     # Relative distances in logit space
│   ├── spline_plots.py                        # Hamming distance of spline codes / step size
│   ├── jacobians_layerwise.py                 # ∂(layer i+1) / ∂(layer i) norms
│   ├── jacobians_full_residual.py             # ∂(last layer) / ∂(layer 0) norms
│   ├── jacobians_attention.py                 # Attention Jacobians (GPT-2 only)
│   ├── jacobians_mlp.py                       # MLP Jacobians (GPT-2 only)
│   ├── utils.py           # Shared utilities (model loading, interpolation, metrics, plotting)
│   └── config.yaml        # Analysis config (model paths, interpolation pairs, n_steps)
├── vis_plateaus/           # Direct plateau geometry visualization (toy ResNet only)
│   ├── visualize_plateaus.py  # Scatter plot colored by metric (L2, Jacobian norm/det)
│   ├── visualize_contours.py  # Contour plot of open-ball pre-images in source space
│   ├── compute_metrics.py     # Jacobian computation (full, layerwise, to-logits, input-to-embed)
│   ├── utils.py               # Shared utilities for this pipeline
│   ├── config.yaml            # Plateau vis settings
│   └── contour_config.yaml   # Contour vis settings (levels, spacing, sub-reference points)
├── run/                    # Shell scripts for running experiments
├── activations/            # Recorded activation .pt files (intermediate)
├── checkpoints/            # Trained model weights
├── plots/                  # Generated figures
└── images/                 # Input images (for ViT/ResNet experiments)
```

## Configuration

- **`train/config.yaml`**: Model architecture (`hidden_dim`, `resblock_width`, `num_blocks`), training settings (`lr`, `batch_size`, `epochs`, `lr_scheduler`), task (`class_spiral`/`reg_sine_wave`, `noise_std`, `num_classes`, `distribution`), multi-seed `n_runs`, and `checkpoint_name`.
- **`vis_plots/config.yaml`**: `n_steps` (interpolation resolution), `model_names` (checkpoint paths per model type), `layer_to_interpolate_toy_resnet` (hook layer index: `-2` = input, `-1` = embed, `0+` = residual blocks), input pairs for each data type.
- **`vis_plateaus/config.yaml`**: `metric` (one of `l2_norm`, `jacobian_norm_full`, `jacobian_norm_layerwise_prod`, `jacobian_determinant_full`, `jacobian_determinant_layerwise_prod`), `source_layer_idx`/`target_layer_idx`, `n_points`, `radius`, `reference_point`, `n_pca_components`, `log_scale`.
- **`vis_plateaus/contour_config.yaml`**: Same spatial settings plus `n_contour_levels`, `contour_level_spacing` (`even`/`log`), and `sub_reference_points` for marking additional points on the plot.

## Key Flags

| Script | Flag | Description |
|--------|------|-------------|
| `train.py` | `--multi_seed` | Train `n_runs` seeds under one directory |
| `train.py` | `--dry_run` | Single epoch for testing |
| `interpolate_and_record_activations.py` | `--interpolate_only_first_layer` | Only interpolate at the first layer (faster) |
| `interpolate_and_record_activations.py` | `--freeze_attention` / `--freeze_mlp` | Freeze attention or MLP outputs during interpolation |
| All `*_plots.py` / `jacobians_*.py` | `--multi_seed` | Aggregate across seed checkpoints with CIs |
| `relative_distances_layerwise_plots.py` | `--single_layer N` | Plot only for target layer N |
| `visualize_plateaus.py` / `visualize_contours.py` | `--model_path` | Override checkpoint path from config |
| `visualize_plateaus.py` / `visualize_contours.py` | `--multi_seed` | Average metrics across seed checkpoints |

## Model Architecture

- **`ResNetMLP`**: Projects 2D input to `hidden_dim` via a linear layer, then passes through `num_blocks` residual blocks (each: LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU -> Linear + skip connection), followed by a final LayerNorm -> ReLU -> output linear.
- **`ResNetMLPSkeleton`**: Same residual blocks but no input projection — operates directly in `input_dim` space (R^n -> R^n through blocks).
- Each `ResidualBlock` has `nn.Identity()` hook points (`hook_resid_pre`, `hook_mlp_out`, `hook_resid_post`) for activation recording. Top-level models add `hook_input` and `hook_embed`.

## Acknowledgements

This project builds upon concepts from [MShinkle/activation_plateau_mechanisms](https://github.com/MShinkle/activation_plateau_mechanisms).
