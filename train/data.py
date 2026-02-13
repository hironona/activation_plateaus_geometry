import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class ToyDataset(Dataset):
    """
    Synthetic dataset for toy regression tasks.
    Generates data on-the-fly (or pre-generated in memory) for consistency.
    """
    def __init__(self, task_name: str, num_samples: int = 10000, noise_std: float = 0.05, num_classes: int = 3, distribution: str = 'uniform', seed=0):
        self.task_name = task_name
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.num_classes = num_classes
        self.distribution = distribution
        self.seed = seed

        self.shuffle_count = 0

        self.data, self.targets = self._generate_data()
    
    def shuffle(self):
        """Shuffle the dataset by incrementing the shuffle count."""
        self.shuffle_count += 1
        self.data, self.targets = self._generate_data()
    
    def _sample(self):
        # reset seed for each shuffle
        rng = np.random.default_rng(self.seed + self.shuffle_count * 1000)

        # Generate 2D inputs in range [-1, 1]
        if self.distribution == 'uniform':
            X = rng.uniform(-1, 1, size=(self.num_samples, 2)).astype(np.float32)
        elif self.distribution == 'normal':
            X = rng.normal(0, 1, size=(self.num_samples, 2)).astype(np.float32)
        return X

    def _get_y(self, X):
        if self.task_name == "reg_sine_wave":
            # Task: f(x1, x2) = {sin(x1) + cos(x2)} * INDICATOR(x1 < 0)
            # Hamayun et al. (2024): https://arxiv.org/pdf/2402.15555
            x1 = X[:, 0]
            x2 = X[:, 1]
            indicator = np.where(x1 < 0, 1, 0)
            y = (np.sin(x1) + np.cos(x2)) * indicator

        elif self.task_name == "class_spiral":
            curvature = 4.0
            
            # Convert to Polar Coordinates relative to center (0,0)
            r = np.linalg.norm(X, axis=1)
            theta = np.arctan2(X[:, 1], X[:, 0])
            
            # Points further out are rotated more, creating the spiral effect.
            theta_twisted = theta + (curvature * r)
            
            # Rescale angles to [0, 2pi]
            theta_twisted = theta_twisted % (2 * np.pi)
            
            # Assign Classes based on Angle Sectors
            y = np.floor((theta_twisted / (2 * np.pi)) * self.num_classes).reshape(-1, 1).astype(int)

            # if num_classes == 2, we don't need to one-hot encode the targets
            if self.num_classes > 2:
                # One-hot encode the targets
                y = np.eye(self.num_classes)[y.squeeze(1)]

        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        return y.astype(np.float32)

    def _generate_data(self):
        X = self._sample()
        y = self._get_y(X)
        
        # Add some small Gaussian noise to the inputs
        X += np.random.normal(0, self.noise_std, size=X.shape)

        return torch.from_numpy(X), torch.from_numpy(y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def visualize_dataset():
    print("Generating Spiral Dataset...")
    num_classes = 2
    dataset = ToyDataset(task_name="class_spiral", num_samples=5000, num_classes=num_classes, noise_std=0.05)
    X = dataset.data.numpy()
    y = dataset.targets.numpy()

    # Create the plot
    plt.figure(figsize=(8, 8))
    
    # Scatter plot: Color points by class label
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', s=5, alpha=0.6)
    
    # Aesthetics
    plt.colorbar(scatter, label="Class Label")
    plt.title(f"Twisted Spiral Task ({num_classes} Classes)\nManifold for ResNet Activation Analysis")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Show plot
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    visualize_dataset()