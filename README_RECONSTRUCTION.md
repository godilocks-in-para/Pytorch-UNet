# MRI Reconstruction Network - Deep Learning Project

This project implements a U-Net based deep learning network for MRI reconstruction from undersampled k-space data.

## Project Overview

**Objective:** Remove aliasing artifacts from undersampled MRI images using a deep learning reconstruction network.

### Key Features

- **Architecture:** U-Net (PyTorch implementation)
- **Loss Function:** L2 Loss (Mean Squared Error)
- **Evaluation Metrics:** PSNR and SSIM
- **Learning Rate Strategy:** ReduceLROnPlateau
- **Dataset Splitting:** Train/Validation/Test with configurable ratios

## 关于 Weights & Biases (W&B)

**Weights & Biases** 是一个强大的机器学习实验跟踪平台，提供：

1. **实时监控** - 在云端实时显示训练进度
2. **超参数记录** - 自动保存所有配置和参数
3. **指标可视化** - 自动绘制损失曲线、精度曲线等
4. **媒体日志** - 可视化图像、模型架构、混淆矩阵等
5. **协作功能** - 方便与他人分享实验结果

### 使用方法

```bash
# 注册W&B账户（首次使用）
pip install wandb
wandb login

# 在代码中自动使用W&B
python train.py --epochs 20 --batch-size 4

# 访问https://wandb.ai查看实时训练进度
```

## 功能实现清单

✅ **Architecture Implementation**
- U-Net with encoder-decoder structure
- Skip connections for feature preservation
- Configurable depth and upsampling methods

✅ **Loss Function Design**
- L2 Loss (MSELoss) for pixel-wise reconstruction
- Implemented in `train.py`

✅ **Model Training**
- Training/Validation/Test split (80/10/10)
- Configurable batch size
- ReduceLROnPlateau learning rate decay strategy
- Mixed precision training (AMP) support
- Gradient clipping and checkpointing

✅ **Quantitative Evaluation**
- PSNR (Peak Signal-to-Noise Ratio) calculation
- SSIM (Structural Similarity Index) calculation
- Metrics saved in test results

✅ **Real-time Visualization in W&B**
- Undersampled images logged every 20 steps
- Reconstructed images logged every 20 steps
- Ground truth images for comparison
- Training and validation metrics tracked
- Parameter histograms monitored

✅ **Visual Comparisons**
- Side-by-side comparison plots (Undersampled vs Reconstructed vs Ground Truth)
- Per-slice PSNR and SSIM values displayed
- Automatic saving to `./results/` directory

## Directory Structure

```
Pytorch-UNet/
├── train.py              # Main training script
├── test.py              # Testing and evaluation script
├── predict.py           # Inference/prediction script
├── plot_loss.py         # Loss curve visualization
├── unet/
│   ├── unet_model.py   # U-Net architecture
│   └── unet_parts.py   # Building blocks (DoubleConv, Down, Up, etc.)
├── utils/
│   ├── data_loading.py # Dataset class and preprocessing
│   └── utils.py        # PSNR, SSIM calculation functions
├── data/
│   ├── img-und/        # Undersampled MRI images
│   └── img-full/       # Fully sampled ground truth images
├── checkpoints/        # Saved model weights
└── results/            # Test results and visualizations
```

## Usage

### Training

```bash
# Basic training
python train.py --epochs 50 --batch-size 4 --learning-rate 1e-4

# With mixed precision (faster)
python train.py --epochs 50 --batch-size 8 --amp

# Resume from checkpoint
python train.py --epochs 50 --batch-size 4 --load checkpoints/checkpoint_epoch10.pth
```

### Testing and Evaluation

```bash
# Run comprehensive test evaluation
python test.py --model checkpoints/checkpoint_epoch50.pth --batch-size 4

# Save test results with visual comparisons
python test.py --model checkpoints/checkpoint_epoch50.pth --batch-size 1
```

### Prediction on New Images

```bash
# Single image prediction
python predict.py --model checkpoints/checkpoint_epoch50.pth --input undersampled.png --output reconstructed.png

# Batch prediction with visualization
python predict.py --model checkpoints/checkpoint_epoch50.pth --input image1.png image2.png --viz
```

### Visualize Training Progress

```bash
# Plot loss curves from W&B
python plot_loss.py --wandb-run-id <your_run_id>
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch-size` | 1 | Batch size for training |
| `--learning-rate` | 1e-5 | Initial learning rate |
| `--validation` | 10.0 | Validation set percentage (0-100) |
| `--scale` | 1.0 | Image downscaling factor |
| `--amp` | False | Enable mixed precision training |
| `--bilinear` | False | Use bilinear upsampling (default: transposed conv) |

## Key Implementation Details

### Data Split
- **Training:** 80% of dataset
- **Validation:** 10% of dataset  
- **Testing:** 10% of dataset

### Optimizer & Scheduler
- Optimizer: RMSprop with momentum=0.999
- Scheduler: ReduceLROnPlateau
  - Mode: minimize (validation loss)
  - Patience: 5 epochs
  - Factor: 0.1 (reduces LR by 10x when plateau)

### Loss Function
- L2 Loss (Mean Squared Error): Pixel-wise reconstruction error

### Metrics
- **PSNR:** Peak Signal-to-Noise Ratio (higher is better)
- **SSIM:** Structural Similarity Index (closer to 1 is better)

## Expected Results

After training on typical MRI data:
- **PSNR improvement:** 3-8 dB (from undersampled baseline)
- **SSIM improvement:** 0.05-0.15
- **Typical PSNR:** 25-35 dB after reconstruction

## W&B Dashboard Features

When logging with W&B, you can view:

1. **Charts** - Real-time loss and PSNR curves
2. **Images** - Training samples showing:
   - Undersampled input (with aliasing)
   - Reconstructed output (network result)
   - Ground truth (fully sampled reference)
3. **Histograms** - Weight and gradient distributions for debugging
4. **Config** - All hyperparameters recorded
5. **System** - GPU usage, training time, etc.

## Dependencies

```
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.23.0
Pillow>=9.3.0
tqdm>=4.64.0
wandb>=0.13.0
scikit-image>=0.19.0
matplotlib>=3.6.0
```

## Example Output

```
Starting training:
    Epochs:          50
    Batch size:      4
    Learning rate:   1e-04
    Training size:   720
    Validation size: 90
    Test size:       90
    Device:          cuda

Epoch 1/50: 100%|█████| 180/180 [00:45<00:00, 3.98 img/s]
Validation Loss: 0.015234, Validation PSNR: 28.41 dB

...

TEST RESULTS
============================================================
Test Loss:                 0.014521

Before Reconstruction:
  PSNR: 24.32 ± 1.45 dB
  SSIM: 0.6821 ± 0.0523

After Reconstruction:
  PSNR: 30.18 ± 1.23 dB
  SSIM: 0.8234 ± 0.0412

Improvement:
  PSNR: +5.86 dB (24.1%)
  SSIM: +0.1413 (20.7%)
============================================================
```

## Deliverables

✅ **Model Definition Code**
- `unet/unet_model.py` - Complete U-Net architecture
- `unet/unet_parts.py` - Modular building blocks

✅ **Training Logs**
- Real-time monitoring via Weights & Biases
- Loss curves visualization in `results/loss_curves.png`

✅ **Visual Comparisons**
- Side-by-side reconstruction results
- PSNR/SSIM values displayed
- Saved in `results/comparison_*.png`

✅ **Quantitative Metrics**
- Test metrics summary in `results/test_metrics.txt`
- Per-slice PSNR and SSIM values

## Troubleshooting

**CUDA Out of Memory:**
```bash
python train.py --batch-size 1 --amp
```

**Slow Training:**
```bash
python train.py --batch-size 8 --amp --bilinear
```

**W&B Connection Issues:**
```bash
wandb offline  # Run in offline mode
```

## References

- U-Net: [Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Weights & Biases: [wandb.ai](https://wandb.ai)

## License

MIT License
