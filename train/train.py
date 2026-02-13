import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import numpy as np

from model import ResNetMLP, ResNetMLPSkeleton
from data import ToyDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_single_model(config: dict):
    # Set Seed
    seed = config['training']['seed']
    set_seed(seed)

    model_type = config['model_type']

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print(f"Loading data for task: {config['task']['name']}")
    task_name = config['task']['name']
    num_samples = config['task']['num_samples']
    noise_std = config['task']['noise_std']
    num_classes = config['task']['num_classes']
    distribution = config['task']['distribution']
    shuffle_per_epoch = config['task']['shuffle_per_epoch']
    dataset = ToyDataset(task_name, num_samples, noise_std, num_classes, distribution, seed=seed)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    classification = task_name.startswith("class")

    # Model
    print("Initializing model...")
    if model_type == "ResNetMLP":
        model = ResNetMLP(
            input_dim=config['model']['input_dim'],
            output_dim=config['model']['output_dim'],
            hidden_dim=config['model']['hidden_dim'],
            resblock_width=config['model']['resblock_width'],
            num_blocks=config['model']['num_blocks'],
            dropout=config['model']['dropout'],
        ).to(device)
    elif model_type == "ResNetMLPSkeleton":
        model = ResNetMLPSkeleton(
            input_dim=config['model']['input_dim'],
            output_dim=config['model']['output_dim'],
            hidden_dim=config['model']['hidden_dim'],
            resblock_width=config['model']['resblock_width'],
            num_blocks=config['model']['num_blocks'],
            dropout=config['model']['dropout'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    if classification and num_classes > 2:
        criterion = nn.CrossEntropyLoss()
    elif classification and num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    # Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = config['training']['checkpoint_name']
    checkpoint_dir = f"{config['training']['checkpoint_dir']}/{task_name}/{model_type}/{checkpoint_name}/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training Loop
    epochs = 1 if config.get('dry_run', False) else config['training']['epochs']
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        if shuffle_per_epoch:
            dataset.shuffle()
            dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

        # Save Checkpoint
        if epoch % config['training']['save_every'] == 0 or epoch == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save config
    with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    with open(os.path.join(checkpoint_dir, "description.txt"), "w") as f:
        f.write(config['training']['checkpoint_description'])

    print("Training finished.")

def main():
    parser = argparse.ArgumentParser(description="Train ResNet MLP on Toy Tasks")
    parser.add_argument("--config", type=str, default="train/config.yaml", help="Path to config file")
    parser.add_argument("--dry_run", action="store_true", help="Run a single epoch for testing")
    parser.add_argument("--multi_seed", action="store_true", help="Aggregate across multiple seeds (toy_resnet only)")
    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)
    
    seed = config['training']['seed']
    n_runs = config['training']['n_runs']

    if args.multi_seed:
        config['training']['checkpoint_name'] += f"-multi_seed_{n_runs}_runs"
        print(f"Running for {n_runs} runs...")
        for seed in range(seed, seed + n_runs):
            print(f"\n=== Training with seed {seed} ===")
            config['training']['seed'] = seed
            train_single_model(config)
    else:
        print(f"Running for a single run...")
        train_single_model(config)

if __name__ == "__main__":
    main()
