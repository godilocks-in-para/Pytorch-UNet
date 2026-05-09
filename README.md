# Deep Learning MRI Reconstruction Network

## 📋 Project Overview

This project implements a **U-Net based deep learning reconstruction network** to remove aliasing artifacts from undersampled MRI images. The network reconstructs high-quality fully-sampled MRI scans from low-resolution undersampled k-space data using supervised learning.

**Key Objective:** Remove aliasing artifacts and reconstruct artifact-free MRI images from variable-density undersampled k-space data.

---

## 🎯 Project Goals & Deliverables

### ✅ Implemented Objectives

| Objective | Status | Details |
|-----------|--------|---------|
| **Reconstruction Network** | ✓ Complete | U-Net architecture with 4 encoder-decoder levels |
| **Architecture Implementation** | ✓ Complete | PyTorch-based U-Net, ResNet-inspired skip connections |
| **Loss Function Design** | ✓ Complete | L2 Loss (Mean Squared Error) for pixel-wise reconstruction |
| **Model Training** | ✓ Complete | Deterministic 80/10/10 train/val/test split with filename-based partitioning |
| **Learning Rate Strategy** | ✓ Complete | ReduceLROnPlateau scheduler with patience=5 |
| **Quantitative Evaluation** | ✓ Complete | PSNR and SSIM metrics before/after reconstruction |
| **Training Visualization** | ✓ Complete | Loss curves, learning rate decay plots |
| **Test Visualization** | ✓ Complete | 320×320 three-image comparison (Input/Reconstruction/Ground Truth) |
| **Batch Processing** | ✓ Complete | Configurable batch sizes, efficient GPU utilization |

---

## 🏗️ Architecture

### Network Design: U-Net

```
Input: 1×320×320 (undersampled MRI)
  ↓
Encoder (4 levels)
  ├─ Conv(1→64) → BatchNorm → ReLU
  ├─ MaxPool 2×2
  ├─ Conv(64→128) → BatchNorm → ReLU
  ├─ MaxPool 2×2
  ├─ Conv(128→256) → BatchNorm → ReLU
  ├─ MaxPool 2×2
  └─ Conv(256→512) → BatchNorm → ReLU
  
Bottleneck: 20×20×512 (max compression)
  ↓
Decoder (4 levels with skip connections)
  ├─ ConvTranspose2d (512→256) + skip connection
  ├─ ConvTranspose2d (256→128) + skip connection
  ├─ ConvTranspose2d (128→64) + skip connection
  └─ ConvTranspose2d (64→1)
  
Output: 1×320×320 (reconstructed MRI)
```

**Key Features:**
- ✓ Skip connections preserve fine details
- ✓ Batch normalization for stable training
- ✓ Symmetric encoder-decoder structure
- ✓ Supports both bilinear and transposed convolution upsampling

### Model Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | ~7.8M |
| Input Resolution | 320×320 |
| Output Resolution | 320×320 |
| Bottleneck Size | 20×20×512 |
| Encoder Levels | 4 |
| Decoder Levels | 4 |
| Downsampling Factor | 16× (2^4) |

---

## 📊 Dataset

### Source: BraTS2021

- **Format:** MRI brain scan slices from BraTS2021 challenge
- **Modality:** Multi-modal (T1, T1c, T2, FLAIR)
- **Total Samples:** ~10,461 image pairs
- **Resolution:** 320×320 pixels
- **Data Format:** NumPy (.npy) arrays + PNG visualizations

### Data Organization

```
data/
├── img-und/              # Undersampled k-space (input)
│   ├── BraTS2021_00000_slice_070_test.npy
│   ├── BraTS2021_00000_slice_070_test.png
│   └── ...
├── img-full/             # Fully-sampled k-space (ground truth)
│   ├── BraTS2021_00000_slice_070_test.npy
│   ├── BraTS2021_00000_slice_070_test.png
│   └── ...
```

### Dataset Split

**Deterministic Split by Filename Suffix:**

| Split | Samples | Suffix | Usage |
|-------|---------|--------|-------|
| Training | 8,368 | `_train` | Model learning |
| Validation | 1,047 | `_val` | Hyperparameter tuning, early stopping |
| Testing | 1,046 | `_test` | Final evaluation, metric reporting |

**Split Ratio:** 80:10:10 (train:val:test)

---

## 🚀 Training Configuration

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | RMSprop | Effective for image reconstruction tasks |
| **Learning Rate** | 1e-5 | Conservative initialization for stable training |
| **Learning Rate Decay** | ReduceLROnPlateau | Dynamic decay based on validation loss |
| **LR Decay Patience** | 5 epochs | Wait 5 epochs before reducing LR |
| **LR Decay Factor** | 0.5 | Reduce LR by 50% |
| **Batch Size** | 32 | Balance between memory efficiency and convergence |
| **Loss Function** | MSE (L2) | Pixel-wise reconstruction error |
| **Momentum** | 0.999 | Heavy momentum for stable updates |

### Loss Function: Mean Squared Error (MSE)

$$L(\hat{x}, x) = \frac{1}{N} \sum_{i=1}^{N} (\hat{x}_i - x_i)^2$$

Where:
- $\hat{x}$ = Network reconstruction
- $x$ = Ground truth fully-sampled image
- $N$ = Number of pixels

### Learning Rate Strategy

**ReduceLROnPlateau Configuration:**

```python
ReduceLROnPlateau(
    mode='min',              # Monitor validation loss
    factor=0.5,              # Multiply LR by 0.5
    patience=5,              # Wait 5 epochs without improvement
    verbose=True,
    min_lr=1e-7              # Minimum learning rate
)
```

---

## 📈 Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)

$$\text{PSNR} = 20 \log_{10}\left(\frac{MAX}{\sqrt{MSE}}\right)$$

- **Unit:** dB (decibels)
- **Range:** 0 → ∞ (higher is better)
- **Typical Range:** 20-40 dB

### SSIM (Structural Similarity Index)

$$\text{SSIM} = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

- **Range:** -1 to 1 (1 = identical)
- **Interpretation:** Correlates better with human perception than PSNR

---

## 💻 Quick Start

### Installation

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python train.py --epochs 50

# Custom batch size
python train.py --epochs 50 --batch-size 64

# Resume from checkpoint
python train.py --epochs 50 --load checkpoints/checkpoint_epoch_25.pth
```

### Testing

```bash
# Test with best checkpoint
python test.py --model checkpoints/checkpoint_best.pth

# Test with smaller batch size (limited GPU memory)
python test.py --model checkpoints/checkpoint_best.pth --batch-size 8

# Test without saving visualizations
python test.py --model checkpoints/checkpoint_best.pth --no-save-visuals
```

---

## 📊 Expected Results

### Performance Benchmarks

**Typical Results on BraTS2021 Test Set:**

| Metric | Before Reconstruction | After Reconstruction | Improvement |
|--------|----------------------|----------------------|-------------|
| **PSNR (dB)** | 22-24 | 31-34 | +8-11 dB |
| **SSIM** | 0.80-0.83 | 0.94-0.96 | +0.12-0.15 |
| **Test Loss (MSE)** | - | 0.0008-0.0015 | - |

### Visual Quality

- **Aliasing Artifacts:** Significantly reduced
- **Fine Details:** Well preserved via skip connections
- **Noise:** Minimized while maintaining structure
- **Boundary Sharpness:** Maintained throughout

---

## 🗂️ Project Structure

```
Pytorch-UNet/
├── train.py                          # Training script
├── test.py                           # Testing & evaluation script
├── predict.py                        # Single image prediction
├── plot_loss.py                      # Visualization of training curves
├── evaluate.py                       # Batch evaluation utility
├── requirements.txt                  # Python dependencies
│
├── unet/
│   ├── __init__.py
│   ├── unet_model.py                 # Main U-Net architecture
│   └── unet_parts.py                 # U-Net building blocks
│
├── utils/
│   ├── __init__.py
│   ├── data_loading.py               # Dataset loader (split-based)
│   └── utils.py                      # Helper functions (PSNR, SSIM)
│
├── data/
│   ├── img-und/                      # Undersampled k-space input
│   └── img-full/                     # Fully-sampled k-space (ground truth)
│
├── checkpoints/
│   ├── checkpoint_best.pth           # Best model (lowest val loss)
│   ├── checkpoint_epoch_*.pth        # Epoch checkpoints
│   └── checkpoint_latest.pth         # Most recent checkpoint
│
├── results/
│   ├── test_metrics.txt              # Quantitative results
│   ├── comparison_batch*.png         # Visual 320×320 comparisons
│   └── training_log_*.txt            # Training logs
│
└── README.md                         # This file
```

---

## 🔧 Advanced Usage

### Training Arguments

```bash
python train.py -h
usage: train.py [--epochs E] [--batch-size B] [--learning-rate LR]
                 [--load LOAD] [--scale SCALE] [--bilinear] [--amp]

Arguments:
  --epochs E              Number of training epochs
  --batch-size B          Batch size (default: 32)
  --learning-rate LR      Initial learning rate (default: 1e-5)
  --load LOAD             Load from checkpoint
  --scale SCALE           Image scaling factor (default: 1.0)
  --bilinear              Use bilinear upsampling
  --amp                   Enable automatic mixed precision
```

### Testing Arguments

```bash
python test.py -h
usage: test.py [--model FILE] [--batch-size B] [--scale SCALE]
                [--bilinear] [--no-save-visuals]

Arguments:
  --model FILE            Path to checkpoint (required)
  --batch-size B          Batch size for testing (default: 1)
  --scale SCALE           Image scaling factor (default: 1.0)
  --bilinear              Use bilinear upsampling
  --no-save-visuals       Skip visual comparisons
```

---

## 🔍 Visualization Examples

### Three-Image Comparison

Each comparison image shows:
1. **① Undersampled Input** - Low-resolution input with aliasing artifacts
2. **② Reconstructed Output** - Network prediction
3. **③ Ground Truth** - Fully-sampled reference

Each image is **320×320 pixels** with metrics displayed below.

### Training Curves

View training progress:
```bash
python plot_loss.py
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --batch-size 16
```

### Data Loading Error
Ensure `.npy` files are in `data/img-und/` and `data/img-full/`

### Validation Loss Not Decreasing
- Reduce initial learning rate to 1e-6
- Increase training epochs
- Check data normalization

---

## 📝 Key References

1. **U-Net: Convolutional Networks for Biomedical Image Segmentation** (Ronneberger et al., 2015)
2. **BraTS Challenge:** https://www.med.upenn.edu/cbica/brats2021/
3. **PyTorch Documentation:** https://pytorch.org/

---

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@article{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={Medical Image Computing and Computer-Assisted Intervention},
  year={2015}
}
```

---

**Project Date:** May 2026  
**Status:** ✅ Production Ready  
**Last Updated:** May 8, 2026
