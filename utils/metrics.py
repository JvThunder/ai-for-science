"""
Metrics logging and visualization utilities.

Provides MetricsLogger class for tracking and visualizing training metrics:
- Loss curves (training vs validation)
- Learning rate schedule
- Distance and angle accuracy metrics
- Customizable plots
- Weights & Biases (wandb) integration

Also includes utilities for generating training reports.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import json
import os
from pathlib import Path

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class MetricsLogger:
    """
    Logger for training metrics with plotting capabilities.

    Tracks:
    - Training and validation losses (total, distance, geometry)
    - Learning rates
    - Angle errors (MAE, RMSE)
    - Custom metrics

    Methods:
    - log(): Record metrics for an epoch
    - save(): Save metrics to JSON
    - plot(): Generate visualization plots
    - summary(): Print training summary
    """

    def __init__(self, log_dir: str = "logs", use_wandb: bool = False):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to save logs and plots
            use_wandb: Enable Weights & Biases logging
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Wandb setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Install with: pip install wandb")
            print("Continuing without wandb logging...")

        # Metrics storage
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Component losses
        self.train_distance_losses = []
        self.train_geometry_losses = []
        self.val_distance_losses = []
        self.val_geometry_losses = []

        # Angle metrics
        self.distance_maes = []
        self.distance_rmses = []
        self.theta_maes = []
        self.phi_maes = []

        # Custom metrics
        self.custom_metrics = {}

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict] = None,
        lr: Optional[float] = None
    ):
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            metrics: Dictionary of additional metrics
            lr: Current learning rate
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)

        if val_loss is not None:
            self.val_losses.append(val_loss)

        if lr is not None:
            self.learning_rates.append(lr)

        # Store component losses
        if metrics:
            if 'train_distance_loss' in metrics:
                self.train_distance_losses.append(metrics['train_distance_loss'])
            if 'train_geometry_loss' in metrics:
                self.train_geometry_losses.append(metrics['train_geometry_loss'])
            if 'val_distance_loss' in metrics:
                self.val_distance_losses.append(metrics['val_distance_loss'])
            if 'val_geometry_loss' in metrics:
                self.val_geometry_losses.append(metrics['val_geometry_loss'])

            # Store angle metrics
            if 'distance_mae' in metrics:
                self.distance_maes.append(metrics['distance_mae'])
            if 'distance_rmse' in metrics:
                self.distance_rmses.append(metrics['distance_rmse'])
            if 'theta_mae' in metrics:
                self.theta_maes.append(metrics['theta_mae'])
            if 'phi_mae' in metrics:
                self.phi_maes.append(metrics['phi_mae'])

            # Store custom metrics
            for key, value in metrics.items():
                if key not in self.custom_metrics:
                    self.custom_metrics[key] = []
                self.custom_metrics[key].append(value)

        # Log to wandb if enabled
        if self.use_wandb:
            wandb_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/learning_rate': lr
            }

            if val_loss is not None:
                wandb_dict['val/loss'] = val_loss

            if metrics:
                # Component losses
                if 'train_distance_loss' in metrics:
                    wandb_dict['train/distance_loss'] = metrics['train_distance_loss']
                if 'train_geometry_loss' in metrics:
                    wandb_dict['train/geometry_loss'] = metrics['train_geometry_loss']
                if 'val_distance_loss' in metrics and metrics['val_distance_loss'] is not None:
                    wandb_dict['val/distance_loss'] = metrics['val_distance_loss']
                if 'val_geometry_loss' in metrics and metrics['val_geometry_loss'] is not None:
                    wandb_dict['val/geometry_loss'] = metrics['val_geometry_loss']

                # Angle metrics
                if 'distance_mae' in metrics:
                    wandb_dict['metrics/distance_mae'] = metrics['distance_mae']
                if 'distance_rmse' in metrics:
                    wandb_dict['metrics/distance_rmse'] = metrics['distance_rmse']
                if 'theta_mae' in metrics:
                    wandb_dict['metrics/theta_mae'] = metrics['theta_mae']
                if 'phi_mae' in metrics:
                    wandb_dict['metrics/phi_mae'] = metrics['phi_mae']

            wandb.log(wandb_dict, step=epoch)

    def save(self, filename: str = "metrics.json"):
        """
        Save metrics to JSON file.

        Args:
            filename: Filename for JSON (saved in log_dir)
        """
        filepath = os.path.join(self.log_dir, filename)

        metrics_dict = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'train_distance_losses': self.train_distance_losses,
            'train_geometry_losses': self.train_geometry_losses,
            'val_distance_losses': self.val_distance_losses,
            'val_geometry_losses': self.val_geometry_losses,
            'distance_maes': self.distance_maes,
            'distance_rmses': self.distance_rmses,
            'theta_maes': self.theta_maes,
            'phi_maes': self.phi_maes,
            'custom_metrics': self.custom_metrics
        }

        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"Metrics saved to: {filepath}")

    def load(self, filename: str = "metrics.json"):
        """
        Load metrics from JSON file.

        Args:
            filename: Filename to load from
        """
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'r') as f:
            metrics_dict = json.load(f)

        self.epochs = metrics_dict.get('epochs', [])
        self.train_losses = metrics_dict.get('train_losses', [])
        self.val_losses = metrics_dict.get('val_losses', [])
        self.learning_rates = metrics_dict.get('learning_rates', [])
        self.train_distance_losses = metrics_dict.get('train_distance_losses', [])
        self.train_geometry_losses = metrics_dict.get('train_geometry_losses', [])
        self.val_distance_losses = metrics_dict.get('val_distance_losses', [])
        self.val_geometry_losses = metrics_dict.get('val_geometry_losses', [])
        self.distance_maes = metrics_dict.get('distance_maes', [])
        self.distance_rmses = metrics_dict.get('distance_rmses', [])
        self.theta_maes = metrics_dict.get('theta_maes', [])
        self.phi_maes = metrics_dict.get('phi_maes', [])
        self.custom_metrics = metrics_dict.get('custom_metrics', {})

        print(f"Metrics loaded from: {filepath}")

    def plot(self, save: bool = True, show: bool = False):
        """
        Generate training visualization plots.

        Creates a multi-panel figure with:
        - Training vs validation loss curves
        - Component losses (distance, geometry)
        - Learning rate schedule
        - Angle error metrics

        Args:
            save: Save plot to log_dir
            show: Display plot interactively
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

        # Plot 1: Total loss curves
        ax = axes[0, 0]
        if len(self.train_losses) > 0:
            ax.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        if len(self.val_losses) > 0:
            val_epochs = self.epochs[:len(self.val_losses)]
            ax.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Component losses
        ax = axes[0, 1]
        if len(self.train_distance_losses) > 0:
            ax.plot(self.epochs[:len(self.train_distance_losses)],
                   self.train_distance_losses, 'b--', label='Train Distance', alpha=0.7)
        if len(self.train_geometry_losses) > 0:
            ax.plot(self.epochs[:len(self.train_geometry_losses)],
                   self.train_geometry_losses, 'b:', label='Train Geometry', alpha=0.7)
        if len(self.val_distance_losses) > 0:
            val_epochs = self.epochs[:len(self.val_distance_losses)]
            ax.plot(val_epochs, self.val_distance_losses, 'r--', label='Val Distance', alpha=0.7)
        if len(self.val_geometry_losses) > 0:
            val_epochs = self.epochs[:len(self.val_geometry_losses)]
            ax.plot(val_epochs, self.val_geometry_losses, 'r:', label='Val Geometry', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Component Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Learning rate
        ax = axes[1, 0]
        if len(self.learning_rates) > 0:
            ax.plot(self.epochs[:len(self.learning_rates)],
                   self.learning_rates, 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        # Plot 4: Angle errors
        ax = axes[1, 1]
        if len(self.distance_maes) > 0:
            ax.plot(self.epochs[:len(self.distance_maes)],
                   self.distance_maes, 'b-', label='Distance MAE (Å)', linewidth=2)
        if len(self.theta_maes) > 0:
            theta_epochs = self.epochs[:len(self.theta_maes)]
            ax.plot(theta_epochs, self.theta_maes, 'r-', label='Theta MAE (°)', linewidth=2)
        if len(self.phi_maes) > 0:
            phi_epochs = self.epochs[:len(self.phi_maes)]
            ax.plot(phi_epochs, self.phi_maes, 'g-', label='Phi MAE (°)', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')
        ax.set_title('Angle Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save:
            plot_path = os.path.join(self.log_dir, 'training_metrics.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")

        # Show plot
        if show:
            plt.show()
        else:
            plt.close()

    def summary(self):
        """
        Print training summary statistics.
        """
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)

        if len(self.train_losses) > 0:
            print(f"\nTotal Epochs: {len(self.epochs)}")
            print(f"Final Training Loss: {self.train_losses[-1]:.6f}")

            if len(self.val_losses) > 0:
                print(f"Final Validation Loss: {self.val_losses[-1]:.6f}")
                print(f"Best Validation Loss: {min(self.val_losses):.6f} (Epoch {self.val_losses.index(min(self.val_losses)) + 1})")

            if len(self.distance_maes) > 0:
                print(f"\nFinal Distance MAE: {self.distance_maes[-1]:.4f} Å")

            if len(self.theta_maes) > 0:
                print(f"Final Theta MAE: {self.theta_maes[-1]:.2f}°")

            if len(self.phi_maes) > 0:
                print(f"Final Phi MAE: {self.phi_maes[-1]:.2f}°")

        print("="*60 + "\n")

    def finish(self):
        """
        Finish logging session (closes wandb if enabled).
        """
        if self.use_wandb:
            wandb.finish()
            print("Wandb run finished")


def plot_prediction_comparison(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Predictions vs Targets",
    xlabel: str = "Target",
    ylabel: str = "Prediction",
    save_path: Optional[str] = None
):
    """
    Create scatter plot comparing predictions to targets.

    Args:
        predictions: Array of predicted values
        targets: Array of target values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(8, 8))

    plt.scatter(targets, predictions, alpha=0.5, s=1)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.close()
