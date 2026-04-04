import argparse
import json
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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

def compute_accuracy(outputs, targets, num_classes):
    """Compute accuracy for classification tasks."""
    if num_classes > 2:
        preds = outputs.argmax(dim=1)
        labels = targets.argmax(dim=1)
    else:
        preds = (outputs.squeeze() > 0).long()
        labels = targets.squeeze().long()
    return (preds == labels).float().mean().item()

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, classification, num_classes):
    """Run evaluation on a dataloader, return avg loss and accuracy (if classification)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        if classification:
            total_correct += compute_accuracy(outputs, targets, num_classes) * inputs.size(0)
        total_samples += inputs.size(0)

        # Clear cache during evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del inputs, targets, outputs, loss

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples if classification else None
    return avg_loss, avg_acc

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

    # Train/val split
    val_split = config['training'].get('val_split', 0.2)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']), weight_decay=float(config['training']['weight_decay']))
    if classification and num_classes > 2:
        criterion = nn.CrossEntropyLoss()
    elif classification and num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    # LR Scheduler
    epochs = 1 if config.get('dry_run', False) else config['training']['epochs']
    lr_scheduler_type = config['training'].get('lr_scheduler', 'none')
    scheduler = None
    if lr_scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = config['training']['checkpoint_name']
    checkpoint_dir = f"{config['training']['checkpoint_dir']}/{task_name}/{model_type}/{checkpoint_name}/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Metrics history
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
    }
    if classification:
        metrics_history['train_acc'] = []
        metrics_history['val_acc'] = []

    best_val_loss = float('inf')

    # Training Loop
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        if shuffle_per_epoch:
            dataset.shuffle()
            # Recreate loaders after shuffle (underlying data changed)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size_actual = inputs.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual
            if classification:
                total_correct += compute_accuracy(outputs, targets, num_classes) * batch_size_actual

            # Update progress bar with metrics
            postfix = {"loss": f"{loss.item():.4f}"}
            if classification:
                batch_acc = compute_accuracy(outputs, targets, num_classes)
                postfix["acc"] = f"{batch_acc:.4f}"
            progress_bar.set_postfix(postfix)

            # Clear cache to prevent OOM
            del inputs, targets, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples if classification else None

        # Validation
        avg_val_loss, val_acc = evaluate(model, val_loader, criterion, device, classification, num_classes)

        # Clear cache after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # LR scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # Log metrics
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['lr'].append(current_lr)
        if classification:
            metrics_history['train_acc'].append(train_acc)
            metrics_history['val_acc'].append(val_acc)

        # Print epoch summary
        summary = f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
        if classification:
            summary += f", train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        summary += f", lr={current_lr:.6f}"
        print(summary)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, best_path)

        # Save periodic checkpoint
        if epoch % config['training']['save_every'] == 0 or epoch == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save metrics history
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Save config
    with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    with open(os.path.join(checkpoint_dir, "description.txt"), "w") as f:
        f.write(config['training']['checkpoint_description'])

    # Clean up
    del model, optimizer, criterion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Training finished. Best val_loss: {best_val_loss:.4f}")

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

    if args.dry_run:
        config['dry_run'] = True

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
