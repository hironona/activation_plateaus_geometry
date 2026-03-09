# Activation Plateau Geometry

> [!IMPORTANT]
> **Work in Progress**: This project is actively under development. Features and APIs may change.

This repository focuses on analyzing the geometry of activation plateaus and residual stream dynamics in deep neural networks. It currently supports experimentation with ResNet-like MLPs on toy tasks (classification and regression) and includes a suite of tools for geometric analysis of latent spaces.

## Features

- **Modular Training Pipeline**: A clean, configurable PyTorch-based training framework located in `train/` for:
  - **ResNet MLPs**: Custom MLP architectures with residual connections.
  - **Toy Tasks**: Support for synthetic tasks like spiral classification (`class_spiral`) and sine wave regression (`reg_sine_wave`).
  - **Checkpointing**: Automatic saving of model states and configurations.
  - **HuggingFace Upload**: `train/upload_hf.py` to push checkpoints to the Hub.

- **Interpolation & Metric Visualization** (`vis_plots/`): Scripts to investigate trained models via activation interpolation:
  - **Activation Interpolation**: `interpolate_and_record_activations.py` captures activations while processing interpolated inputs.
  - **Metric Visualization**: Tools to plot step sizes, relative distances, Hamming distances, and spline approximations of activation paths.
  - **Jacobian Analysis**: Scripts to compute and analyze layer-wise, attention, MLP, and full-residual Jacobians.

- **Plateau Geometry Visualization** (`vis_plateaus/`): Directly visualize metric landscapes over input or activation space for toy models:
  - Colors sampled points by L2 norm, Jacobian norm, or Jacobian determinant (full or layerwise product).
  - Supports PCA projection for high-dimensional activation spaces.

## Project Structure

```
.
├── train/                  # Training pipeline
│   ├── train.py           # Main training entry point
│   ├── model.py           # ResNetMLP and ResNetMLPSkeleton architectures
│   ├── data.py            # Toy dataset generators
│   ├── upload_hf.py       # Upload checkpoints to HuggingFace Hub
│   └── config.yaml        # Training configuration (hyperparams, task settings)
├── vis_plots/              # Interpolation recording and metric plots
│   ├── interpolate_and_record_activations.py  # Core script for generating activation data
│   ├── *_plots.py         # Plotting scripts for various metrics
│   ├── jacobians_*.py     # Jacobian analysis scripts (layerwise, attention, mlp, full_residual)
│   ├── utils.py           # Shared utilities (model loaders, interpolation, metrics)
│   └── config.yaml        # Analysis configuration (models, interpolation pairs, image paths)
├── vis_plateaus/           # Plateau geometry visualization (toy ResNet only)
│   ├── visualize_plateaus.py  # Main script for plateau geometry plots
│   ├── compute_metrics.py     # Jacobian norm/determinant metric computation
│   ├── utils.py               # Shared utilities for this pipeline
│   └── config.yaml            # Plateau vis settings (metric, layers, resolution)
├── run/                    # Shell scripts for running experiments
│   ├── full_experiment.sh     # Full GPT-2 experiment
│   ├── toy_resnet_vis.sh      # Toy ResNet interpolation pipeline
│   ├── vis_plateaus.sh        # Plateau geometry visualization (loops over noise levels)
│   ├── resnet_vis.sh          # ResNet-101 plots
│   └── vit_vis.sh             # ViT (DINOv2) plots
├── images/                 # Local image inputs for ViT/ResNet interpolation
├── activations/            # Output directory for recorded activations
├── checkpoints/            # Output directory for training checkpoints
└── plots/                  # Output directory for generated figures
```

## Quick Start

### 1. Installation

Install the required dependencies using `uv`:

```bash
uv sync
```

### 2. Training a Model

To train a ResNet MLP on the default toy task (e.g., spiral classification):

```bash
python train/train.py --config train/config.yaml
```

This will create a timestamped directory in `checkpoints/` containing the model weights (`checkpoint_epoch_X.pt`).

### 3. Running Analysis

After training, you can analyze the model's activation geometry. Ensure `vis_plots/config.yaml` points to your trained model checkpoint (under `model_names: toy_resnet`).

**Option A — Full interpolation + metric plots pipeline:**

```bash
bash run/toy_resnet_vis.sh
```

This script will:
1. Run `vis_plots/interpolate_and_record_activations.py` to generate data in `activations/`.
2. Run various plotting scripts to generate figures in `plots/`.

**Option B — Plateau geometry visualization:**

```bash
bash run/vis_plateaus.sh
```

Or directly:

```bash
uv run vis_plateaus/visualize_plateaus.py --model_type toy_resnet --data_type class_spiral \
    --model_path checkpoints/class_spiral/ResNetMLP/<config_name> --multi_seed
```

## Configuration

- **Training** (`train/config.yaml`): model dimensions (`hidden_dim`, `num_blocks`), training settings (`lr`, `batch_size`), task (`class_spiral` vs `reg_sine_wave`), and multi-seed `n_runs`.
- **Interpolation & plots** (`vis_plots/config.yaml`): interpolation steps (`n_steps`), model paths, input pairs (2D spiral points, local image paths from `images/`, text token pairs).
- **Plateau visualization** (`vis_plateaus/config.yaml`): metric (`l2_norm`, `jacobian_norm_full`, etc.), source/target layer indices, grid resolution (`n_points`), PCA components, log scale, model path.

## Acknowledgements

This project builds upon usage and concepts from the original repository by [MShinkle](https://github.com/MShinkle/activation_plateau_mechanisms).
