"""
Loss curve visualization script
Plots loss curves during training (can be used with wandb or local logging)
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(log_file=None, wandb_run_id=None, output_dir='./results/'):
    """
    Plot loss curves from training logs
    
    Args:
        log_file: path to local training log file (JSON format)
        wandb_run_id: W&B run ID for online logging
        output_dir: directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if wandb_run_id:
        try:
            import wandb
            api = wandb.Api()
            run = api.run(f"wandb_entity/project_name/{wandb_run_id}")
            
            # Extract metrics from W&B
            history = run.history()
            
            epochs = history['epoch'].values if 'epoch' in history.columns else np.arange(len(history))
            train_loss = history['train loss'].values if 'train loss' in history.columns else None
            val_loss = history['validation loss'].values if 'validation loss' in history.columns else None
            train_psnr = history['train PSNR'].values if 'train PSNR' in history.columns else None
            val_psnr = history['validation PSNR'].values if 'validation PSNR' in history.columns else None
            
            print(f"Loaded training history from W&B run {wandb_run_id}")
        except Exception as e:
            print(f"Error loading from W&B: {e}")
            return
    
    elif log_file and Path(log_file).exists():
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            epochs = data.get('epochs', [])
            train_loss = data.get('train_loss', [])
            val_loss = data.get('val_loss', [])
            train_psnr = data.get('train_psnr', [])
            val_psnr = data.get('val_psnr', [])
            
            print(f"Loaded training history from {log_file}")
        except Exception as e:
            print(f"Error loading log file: {e}")
            return
    else:
        print("Please provide either --log-file or --wandb-run-id")
        return
    
    # Plot 1: Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if train_loss is not None:
        axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    if val_loss is not None:
        axes[0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Reconstruction Loss Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: PSNR curves
    if train_psnr is not None:
        axes[1].plot(epochs, train_psnr, 'b-', label='Train PSNR', linewidth=2)
    if val_psnr is not None:
        axes[1].plot(epochs, val_psnr, 'r-', label='Validation PSNR', linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_title('PSNR Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'loss_curves.png', dpi=150, bbox_inches='tight')
    print(f"Loss curves saved to {Path(output_dir) / 'loss_curves.png'}")
    
    # Print summary statistics
    if train_loss:
        print(f"\nTraining Summary:")
        print(f"  Initial Train Loss: {train_loss[0]:.6f}")
        print(f"  Final Train Loss:   {train_loss[-1]:.6f}")
        print(f"  Loss Reduction:     {(train_loss[0] - train_loss[-1])/train_loss[0]*100:.1f}%")
    
    if val_loss:
        print(f"\nValidation Summary:")
        print(f"  Best Val Loss: {min(val_loss):.6f}")
        print(f"  Final Val Loss: {val_loss[-1]:.6f}")
    
    if train_psnr and val_psnr:
        print(f"\nPSNR Improvement:")
        print(f"  Initial PSNR: {train_psnr[0]:.2f} dB (train), {val_psnr[0]:.2f} dB (val)")
        print(f"  Final PSNR:   {train_psnr[-1]:.2f} dB (train), {val_psnr[-1]:.2f} dB (val)")


def get_args():
    parser = argparse.ArgumentParser(description='Plot training loss curves')
    parser.add_argument('--log-file', type=str, metavar='FILE',
                        help='Path to training log file (JSON format)')
    parser.add_argument('--wandb-run-id', type=str, metavar='ID',
                        help='W&B run ID for online logging')
    parser.add_argument('--output-dir', '-o', type=str, default='./results/',
                        help='Output directory for plots')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    plot_loss_curves(args.log_file, args.wandb_run_id, args.output_dir)
