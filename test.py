"""
Testing script for MRI reconstruction network
Evaluates the model on test set and generates quantitative metrics and visual comparisons
"""
import argparse
import logging
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset
from utils.utils import calculate_psnr, calculate_ssim

dir_img_und = Path('./data/img-und/')
dir_img_full = Path('./data/img-full/')
dir_results = Path('./results/')


def test_model(model, test_loader, device, save_visuals=True):
    """
    Test the model on test set and calculate metrics
    
    Args:
        model: trained model
        test_loader: test dataloader
        device: device to run on
        save_visuals: whether to save visual comparisons
    
    Returns:
        dict containing average metrics
    """
    model.eval()
    
    all_psnr_before = []  # PSNR between undersampled and ground truth
    all_psnr_after = []   # PSNR between reconstructed and ground truth
    all_ssim_before = []  # SSIM before reconstruction
    all_ssim_after = []   # SSIM after reconstruction
    all_loss = []
    
    criterion = torch.nn.MSELoss()
    
    if save_visuals:
        Path(dir_results).mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing', unit='batch')):
            images_und = batch['img_und'].to(device, dtype=torch.float32)
            true_image_full = batch['img_full'].to(device, dtype=torch.float32)
            
            # Forward pass
            img_full_pred = model(images_und)
            
            # Calculate loss
            loss = criterion(img_full_pred, true_image_full)
            all_loss.append(loss.item())
            
            # Calculate metrics for each image in batch
            batch_size = images_und.shape[0]
            for i in range(batch_size):
                und_img = images_und[i].detach().cpu().numpy()
                pred_img = img_full_pred[i].detach().cpu().numpy()
                true_img = true_image_full[i].detach().cpu().numpy()
                
                # PSNR/SSIM before reconstruction (undersampled vs ground truth)
                psnr_before = calculate_psnr(und_img, true_img, max_val=1.0)
                ssim_before = calculate_ssim(und_img, true_img, max_val=1.0)
                all_psnr_before.append(psnr_before)
                all_ssim_before.append(ssim_before)
                
                # PSNR/SSIM after reconstruction (predicted vs ground truth)
                psnr_after = calculate_psnr(pred_img, true_img, max_val=1.0)
                ssim_after = calculate_ssim(pred_img, true_img, max_val=1.0)
                all_psnr_after.append(psnr_after)
                all_ssim_after.append(ssim_after)
                
                # Save visual comparison
                if save_visuals and batch_idx < 5:  # Save first 5 batches
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Handle channel dimension
                    und_display = und_img[0] if und_img.ndim == 3 else und_img
                    pred_display = pred_img[0] if pred_img.ndim == 3 else pred_img
                    true_display = true_img[0] if true_img.ndim == 3 else true_img
                    
                    axes[0].imshow(und_display, cmap='gray')
                    axes[0].set_title(f'Undersampled\nPSNR: {psnr_before:.2f} dB\nSSIM: {ssim_before:.4f}')
                    axes[0].axis('off')
                    
                    axes[1].imshow(pred_display, cmap='gray')
                    axes[1].set_title(f'Reconstructed\nPSNR: {psnr_after:.2f} dB\nSSIM: {ssim_after:.4f}')
                    axes[1].axis('off')
                    
                    axes[2].imshow(true_display, cmap='gray')
                    axes[2].set_title('Ground Truth')
                    axes[2].axis('off')
                    
                    plt.suptitle(f'Test Sample {batch_idx * batch_size + i}')
                    plt.tight_layout()
                    plt.savefig(str(dir_results / f'comparison_batch{batch_idx}_sample{i}.png'), dpi=100, bbox_inches='tight')
                    plt.close()
    
    # Calculate average metrics
    metrics = {
        'avg_loss': np.mean(all_loss),
        'avg_psnr_before': np.mean(all_psnr_before),
        'avg_psnr_after': np.mean(all_psnr_after),
        'std_psnr_before': np.std(all_psnr_before),
        'std_psnr_after': np.std(all_psnr_after),
        'avg_ssim_before': np.mean(all_ssim_before),
        'avg_ssim_after': np.mean(all_ssim_after),
        'std_ssim_before': np.std(all_ssim_before),
        'std_ssim_after': np.std(all_ssim_after),
        'psnr_improvement': np.mean(all_psnr_after) - np.mean(all_psnr_before),
        'ssim_improvement': np.mean(all_ssim_after) - np.mean(all_ssim_before),
    }
    
    return metrics, all_psnr_before, all_psnr_after, all_ssim_before, all_ssim_after


def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet reconstruction model')
    parser.add_argument('--model', '-m', type=str, required=True, metavar='FILE',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, 
                        help='Batch size for testing')
    parser.add_argument('--scale', '-s', type=float, default=1.0, 
                        help='Downscaling factor of the images')
    parser.add_argument('--bilinear', action='store_true', default=False, 
                        help='Use bilinear upsampling')
    parser.add_argument('--no-save-visuals', action='store_true', default=False,
                        help='Do not save visual comparisons')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Create model
    model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
    model = model.to(device=device)
    
    # Load checkpoint
    logging.info(f'Loading model from {args.model}')
    state_dict = torch.load(args.model, map_location=device)
    state_dict.pop('mask_values', None)
    model.load_state_dict(state_dict)
    logging.info('Model loaded successfully!')
    
    # Create dataset
    dataset = BasicDataset(dir_img_und, dir_img_full, args.scale)
    
    # Use last 20% as test set (following common split patterns)
    n_total = len(dataset)
    n_test = int(0.2 * n_total)
    n_train_val = n_total - n_test
    _, test_set = random_split(dataset, [n_train_val, n_test], 
                               generator=torch.Generator().manual_seed(0))
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, 
                                              shuffle=False, num_workers=0, pin_memory=True)
    
    # Run testing
    logging.info(f'Starting evaluation on {len(test_set)} test samples...')
    metrics, psnr_before, psnr_after, ssim_before, ssim_after = test_model(
        model, test_loader, device, save_visuals=not args.no_save_visuals
    )
    
    # Print results
    logging.info('\n' + '='*60)
    logging.info('TEST RESULTS')
    logging.info('='*60)
    logging.info(f'Test Loss:                 {metrics["avg_loss"]:.6f}')
    logging.info(f'\nBefore Reconstruction:')
    logging.info(f'  PSNR: {metrics["avg_psnr_before"]:.2f} ± {metrics["std_psnr_before"]:.2f} dB')
    logging.info(f'  SSIM: {metrics["avg_ssim_before"]:.4f} ± {metrics["std_ssim_before"]:.4f}')
    logging.info(f'\nAfter Reconstruction:')
    logging.info(f'  PSNR: {metrics["avg_psnr_after"]:.2f} ± {metrics["std_psnr_after"]:.2f} dB')
    logging.info(f'  SSIM: {metrics["avg_ssim_after"]:.4f} ± {metrics["std_ssim_after"]:.4f}')
    logging.info(f'\nImprovement:')
    logging.info(f'  PSNR: +{metrics["psnr_improvement"]:.2f} dB ({metrics["psnr_improvement"]/metrics["avg_psnr_before"]*100:.1f}%)')
    logging.info(f'  SSIM: +{metrics["ssim_improvement"]:.4f} ({metrics["ssim_improvement"]/metrics["avg_ssim_before"]*100:.1f}%)')
    logging.info('='*60 + '\n')
    
    if not args.no_save_visuals:
        logging.info(f'Visual comparisons saved to {dir_results}')
    
    # Save metrics to file
    metrics_file = dir_results / 'test_metrics.txt'
    Path(dir_results).mkdir(parents=True, exist_ok=True)
    with open(metrics_file, 'w') as f:
        f.write('TEST METRICS\n')
        f.write('='*60 + '\n')
        f.write(f'Test Loss:                 {metrics["avg_loss"]:.6f}\n')
        f.write(f'\nBefore Reconstruction:\n')
        f.write(f'  PSNR: {metrics["avg_psnr_before"]:.2f} ± {metrics["std_psnr_before"]:.2f} dB\n')
        f.write(f'  SSIM: {metrics["avg_ssim_before"]:.4f} ± {metrics["std_ssim_before"]:.4f}\n')
        f.write(f'\nAfter Reconstruction:\n')
        f.write(f'  PSNR: {metrics["avg_psnr_after"]:.2f} ± {metrics["std_psnr_after"]:.2f} dB\n')
        f.write(f'  SSIM: {metrics["avg_ssim_after"]:.4f} ± {metrics["std_ssim_after"]:.4f}\n')
        f.write(f'\nImprovement:\n')
        f.write(f'  PSNR: +{metrics["psnr_improvement"]:.2f} dB ({metrics["psnr_improvement"]/metrics["avg_psnr_before"]*100:.1f}%)\n')
        f.write(f'  SSIM: +{metrics["ssim_improvement"]:.4f} ({metrics["ssim_improvement"]/metrics["avg_ssim_before"]*100:.1f}%)\n')
    
    logging.info(f'Metrics saved to {metrics_file}')
