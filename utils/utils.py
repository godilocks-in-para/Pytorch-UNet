import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, List, Dict
import warnings

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def calculate_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    计算两张图像之间的PSNR
    
    Args:
        pred: 预测图像，形状 [H, W] 或 [H, W, C]
        target: 目标图像，相同形状
        max_val: 图像最大值（[0,1]范围则为1.0，[0,255]则为255）
    
    Returns:
        psnr值（dB）
    """
    # 确保是numpy数组
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # 处理可能的批次维度
    if pred.ndim == 4:  # [B, C, H, W]
        pred = pred[0]
    if target.ndim == 4:
        target = target[0]
    
    # 转置为skimage需要的格式 [H, W, C]
    if pred.ndim == 3 and pred.shape[0] in [1, 3]:  # [C, H, W]
        pred = np.transpose(pred, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))
    
    # 处理单通道图像 [H, W] -> 无需转换
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        psnr_value = psnr(target, pred, data_range=max_val)
    
    return psnr_value

def calculate_ssim(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    计算两张图像之间的SSIM
    
    Args:
        pred: 预测图像，形状 [H, W] 或 [H, W, C] 或 [C, H, W]
        target: 目标图像，相同形状
        max_val: 图像最大值
    
    Returns:
        ssim值（0-1之间）
    """
    # 确保是numpy数组
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # 处理批次维度
    if pred.ndim == 4:  # [B, C, H, W]
        pred = pred[0]
    if target.ndim == 4:
        target = target[0]
    
    # 转置为skimage需要的格式 [H, W, C]
    if pred.ndim == 3 and pred.shape[0] in [1, 3]:  # [C, H, W]
        pred = np.transpose(pred, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))
    
    # 对于单通道图像，需要确保形状是 [H, W]
    if pred.ndim == 3 and pred.shape[2] == 1:
        pred = pred.squeeze(axis=2)
        target = target.squeeze(axis=2)
    
    # SSIM参数设置
    multichannel = (pred.ndim == 3 and pred.shape[2] > 1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssim_value = ssim(target, pred, 
                         data_range=max_val,
                         multichannel=multichannel,
                         win_size=None)  # 自动选择窗口大小
    
    return ssim_value

def calculate_metrics_for_slice(pred_slice: np.ndarray, 
                                target_slice: np.ndarray,
                                max_val: float = 1.0) -> Dict[str, float]:
    """
    计算单个切片的重建指标
    
    Args:
        pred_slice: 重建的切片
        target_slice: ground truth切片
        max_val: 图像最大值
    
    Returns:
        包含PSNR和SSIM的字典
    """
    psnr_value = calculate_psnr(pred_slice, target_slice, max_val)
    ssim_value = calculate_ssim(pred_slice, target_slice, max_val)
    
    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value
    }

def calculate_metrics_3d_volume(pred_volume: np.ndarray,
                                target_volume: np.ndarray,
                                slice_axis: int = 2,
                                max_val: float = 1.0) -> Dict[str, List[float]]:
    """
    计算3D体积中每个切片的指标
    
    Args:
        pred_volume: 预测的3D体积 [D, H, W] 或 [C, D, H, W]
        target_volume: 目标3D体积
        slice_axis: 切片轴（0=深度，1=高度，2=宽度）
        max_val: 图像最大值
    
    Returns:
        包含每个切片PSNR和SSIM的字典
    """
    # 处理输入格式
    if pred_volume.ndim == 4:  # [C, D, H, W]
        pred_volume = pred_volume[0]  # 假设单通道
        target_volume = target_volume[0]
    
    # 确保切片轴是第一个维度
    if slice_axis != 0:
        pred_volume = np.moveaxis(pred_volume, slice_axis, 0)
        target_volume = np.moveaxis(target_volume, slice_axis, 0)
    
    n_slices = pred_volume.shape[0]
    psnr_list = []
    ssim_list = []
    
    for i in range(n_slices):
        pred_slice = pred_volume[i]
        target_slice = target_volume[i]
        
        # 确保是2D图像
        if pred_slice.ndim == 3 and pred_slice.shape[0] == 1:
            pred_slice = pred_slice.squeeze(0)
            target_slice = target_slice.squeeze(0)
        
        metrics = calculate_metrics_for_slice(pred_slice, target_slice, max_val)
        psnr_list.append(metrics['PSNR'])
        ssim_list.append(metrics['SSIM'])
    
    return {
        'PSNR_per_slice': psnr_list,
        'SSIM_per_slice': ssim_list,
        'PSNR_mean': np.mean(psnr_list),
        'PSNR_std': np.std(psnr_list),
        'SSIM_mean': np.mean(ssim_list),
        'SSIM_std': np.std(ssim_list)
    }

def print_metrics_summary(before_metrics: Dict, after_metrics: Dict):
    """
    打印重建前后的指标对比
    
    Args:
        before_metrics: 重建前的指标（欠采样图像）
        after_metrics: 重建后的指标
    """
    print("\n" + "="*60)
    print("Reconstruction Quality Assessment")
    print("="*60)
    
    print(f"\n{'Metric':<15} {'Before':<15} {'After':<15} {'Improvement':<15}")
    print("-"*60)
    
    for metric in ['PSNR', 'SSIM']:
        before_mean = before_metrics[f'{metric}_mean']
        after_mean = after_metrics[f'{metric}_mean']
        improvement = after_mean - before_mean
        
        print(f"{metric:<15} {before_mean:<15.2f} {after_mean:<15.2f} {improvement:+.2f}")
    
    print(f"\n{'Before':<15} Mean ± Std (PSNR/SSIM): {before_metrics['PSNR_mean']:.2f}±{before_metrics['PSNR_std']:.2f} / {before_metrics['SSIM_mean']:.4f}±{before_metrics['SSIM_std']:.4f}")
    print(f"{'After':<15} Mean ± Std (PSNR/SSIM): {after_metrics['PSNR_mean']:.2f}±{after_metrics['PSNR_std']:.2f} / {after_metrics['SSIM_mean']:.4f}±{after_metrics['SSIM_std']:.4f}")
    print("="*60 + "\n")