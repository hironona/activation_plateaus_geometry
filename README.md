# Activation Plateau Geometry

> [!IMPORTANT]
> **Work in Progress**: This project is actively under development. Features and APIs may change.

This repository focuses on analyzing the geometry of activation plateaus and residual stream dynamics in deep neural networks. It currently supports experimentation with ResNet-like MLPs on toy tasks (classification and regression) and includes a suite of tools for geometric analysis of latent spaces.

## Features

- **Modular Training Pipeline**: A clean, configurable PyTorch-based training framework located in `train/` for:
  - **ResNet MLPs**: Custom MLP architectures with residual connections.
  - **Toy Tasks**: Support for synthetic tasks like spiral classification (`class_spiral`) and sine wave regression (`reg_sine_wave`).
  - **Checkpointing**: Automatic saving of model states and configurations.

- **Geometric Analysis**: A set of scripts in `scripts/` to investigate the trained models:
  - **Activation Interpolation**: `interpolate_and_record_activations.py` captures activations while processing interpolated inputs.
  - **Metric Visualization**: Tools to plot step sizes, relative distances, Hamming distances, and spline approximations of activation paths.
  - **Jacobian Analysis**: Scripts to compute and analyze layer-wise and full-residual Jacobians.

## Project Structure

```
.
├── train/                  # Training pipeline
│   ├── train.py           # Main training entry point
│   ├── model.py           # ResNetMLP and ResNetMLPSkeleton architectures
│   ├── data.py            # Toy dataset generators
│   └── config.yaml        # Training configuration (hyperparams, task settings)
├── scripts/                # Analysis and visualization tools
│   ├── interpolate_....py # Core script for generating activation data
│   ├── *_plots.py         # Plotting scripts for various metrics
│   ├── jacobians_*.py     # Jacobian analysis scripts
│   └── config.yaml        # Analysis configuration (models, interpolation pairs)
├── run/                    # Shell scripts for running experiments
├── activations/            # Output directory for recorded activations
├── checkpoints/            # Output directory for training checkpoints
└── plots/                  # Output directory for generated figures
```

## Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Training a Model

To train a ResNet MLP on the default toy task (e.g., spiral classification):

```bash
python train/train.py --config train/config.yaml
```

This will create a timestamped directory in `checkpoints/` containing the model weights (`checkpoint_epoch_X.pt`).

### 3. Running Analysis

After training, you can analyze the model's activation geometry.

First, ensure `scripts/config.yaml` points to your trained model checkpoint (under `model_names: toy_resnet`) or use the command line arguments to specify the model type.

To run the full visualization pipeline for a toy ResNet model:

```bash
bash run/toy_resnet_vis.sh
```

This script will:
1.  Run `interpolate_and_record_activations.py` to generate data in `activations/`.
2.  Run various plotting scripts to generate figures in `plots/`.

## Configuration

- **Training**: valid parameters for `train/config.yaml` include model dimensions (`hidden_dim`, `num_blocks`), training settings (`lr`, `batch_size`), and task specification (`class_spiral` vs `reg_sine_wave`).
- **Analysis**: `scripts/config.yaml` controls the interpolation steps (`n_steps`), specific model paths, and the input pairs used for interpolation (e.g., specific points in the spiral 2D plane).

## Acknowledgements

This project builds upon usage and concepts from the original repository by [MShinkle](https://github.com/MShinkle/activation_plateau_mechanisms).