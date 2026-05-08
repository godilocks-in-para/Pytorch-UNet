# 网络测试完整指南 (Testing Guide)

## 1. 快速开始 (Quick Start)

### 使用最佳 checkpoint 进行测试

```bash
# 假设最佳模型保存在 checkpoints/checkpoint_epoch_25.pth
python test.py --model checkpoints/checkpoint_epoch_25.pth --batch-size 32
```

输出示例：
```
============================================================
TEST RESULTS
============================================================
Test Loss:                 0.001234
Before Reconstruction:
  PSNR: 22.34 ± 1.23 dB
  SSIM: 0.8234 ± 0.0145
After Reconstruction:
  PSNR: 31.45 ± 0.95 dB
  SSIM: 0.9456 ± 0.0089
Improvement:
  PSNR: +9.11 dB (40.8%)
  SSIM: +0.1222 (14.8%)
```

---

## 2. 寻找最佳 Checkpoint

### 方法 A: 使用训练日志

每次训练会自动保存 checkpoints 到 `./checkpoints/` 目录。

```bash
# 列出所有 checkpoint
ls -la ./checkpoints/
```

查看最新或评分最高的模型：
```
checkpoint_epoch_1.pth    (earliest)
checkpoint_epoch_10.pth   (early stopping checkpoint)
checkpoint_best.pth       (BEST - recommended!)
checkpoint_latest.pth     (most recent)
```

**推荐**: 使用 `checkpoint_best.pth`（基于验证集最低损失选择）

### 方法 B: 从训练输出查找

训练时会打印验证指标：

```
Epoch 10/50
Training Loss: 0.002341
Validation Loss: 0.001892 ← Watch this value
```

保存模型时输出：
```
Checkpoint saved: epoch_10, val_loss=0.001892
```

---

## 3. 测试命令详解

### 基础命令

```bash
python test.py --model <checkpoint_path> [options]
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必需 | checkpoint 文件路径 |
| `--batch-size` | 16 | 测试 batch 大小 (可根据显存调整) |
| `--scale` | 1.0 | 图像缩放因子 (保持默认) |
| `--bilinear` | 否 | 是否使用双线性插值上采样 |
| `--no-save-visuals` | 否 | 不保存可视化结果 |

### 常见用法

#### 场景 1: 快速测试 (小 batch size)
```bash
python test.py --model checkpoints/checkpoint_best.pth --batch-size 8
```

#### 场景 2: 完整测试 + 保存可视化 (默认)
```bash
python test.py --model checkpoints/checkpoint_best.pth --batch-size 32
```

#### 场景 3: 仅测试指标，不保存图像 (节省空间)
```bash
python test.py --model checkpoints/checkpoint_best.pth --no-save-visuals
```

#### 场景 4: 测试多个 checkpoint (对比)
```bash
python test.py --model checkpoints/checkpoint_epoch_10.pth --batch-size 32
python test.py --model checkpoints/checkpoint_epoch_20.pth --batch-size 32
python test.py --model checkpoints/checkpoint_epoch_30.pth --batch-size 32
```

---

## 4. 理解测试输出

### 输出指标解释

**PSNR (峰值信噪比 - Peak Signal-to-Noise Ratio)**
- 单位: dB (分贝)
- 范围: 0~∞ (越高越好)
- 典型值: 20-40 dB
- 改进 +9 dB = 显著改进 ✓

**SSIM (结构相似性 - Structural Similarity)**
- 范围: -1 ~ 1 (1 = 完全相同)
- 人眼感知对标
- 改进 +0.12 = 较好改进 ✓

**Test Loss**
- MSE 损失值
- 越低越好
- 用于模型对标

### 结果文件

测试会生成以下文件：

```
./results/
├── test_metrics.txt          # 数值结果汇总
├── psnr_comparison.csv       # 样本级 PSNR 对比
├── ssim_comparison.csv       # 样本级 SSIM 对比
├── before_reconstruction/    # 重建前图像
│   ├── sample_0.png
│   ├── sample_1.png
│   └── ...
└── after_reconstruction/     # 重建后图像
    ├── sample_0.png
    ├── sample_1.png
    └── ...
```

---

## 5. 工作流程详解

### 数据分割机制 (现已修复)

**之前** (随机分割 - 不确定性):
```python
_, test_set = random_split(dataset, [80%, 20%])  # ❌ 每次运行可能不同
```

**现在** (基于文件名后缀 - 确定性):
```python
test_set = BasicDataset(..., split='test')  # ✓ 总是加载 *_test.npy 文件
```

### Checkpoint 加载流程

```python
# 1. 加载模型架构
model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)

# 2. 加载预训练权重
checkpoint = torch.load(args.model, map_location=device)
model.load_state_dict(checkpoint)  # 或 state_dict 的子集

# 3. 设置评估模式
model.eval()

# 4. 禁用梯度计算
with torch.no_grad():
    # 运行推理
```

### 测试评估流程

```python
for batch in test_loader:
    # 1. 前向传播
    reconstruction = model(undersampled_image)
    
    # 2. 计算损失
    loss = MSE(reconstruction, fully_sampled_reference)
    
    # 3. 计算指标
    psnr_before = calculate_psnr(undersampled, reference)
    psnr_after = calculate_psnr(reconstruction, reference)
    
    # 4. 可视化 (可选)
    save_comparison_image(before, after, reference)
```

---

## 6. 故障排除

### 错误 1: 文件不存在
```
RuntimeError: No such file or directory: 'checkpoints/checkpoint_best.pth'
```

**解决方案**: 
```bash
# 检查 checkpoint 是否存在
ls checkpoints/

# 确保路径正确
python test.py --model ./checkpoints/checkpoint_best.pth
```

### 错误 2: CUDA 内存不足
```
RuntimeError: CUDA out of memory
```

**解决方案**: 减小 batch size
```bash
python test.py --model checkpoints/checkpoint_best.pth --batch-size 4
```

### 错误 3: 数据分割问题
```
AssertionError: Either no image or multiple images found
```

**原因**: 旧版本 glob 问题（已修复）
**检查**: 确保 `data/img-und/` 中文件格式为 `*_test.npy`

```bash
# 验证文件存在
ls data/img-und/*_test.npy
```

---

## 7. 性能对标

### 预期性能指标 (BraTS2021 MRI 重建)

| 度量 | 典型值 | 优秀值 |
|------|--------|--------|
| PSNR before | 20-25 dB | - |
| PSNR after | 30-35 dB | >35 dB |
| PSNR 改进 | +8-12 dB | >10 dB |
| SSIM before | 0.70-0.80 | - |
| SSIM after | 0.92-0.96 | >0.95 |
| Test Loss | 0.0005-0.001 | <0.0005 |

### 我的模型结果示例
```
PSNR Improvement: +9.11 dB (40.8% relative)
SSIM Improvement: +0.1222 (14.8% relative)
Test Loss: 0.001234
↑ 这是很好的性能！
```

---

## 8. 高级用法

### 在 Python 脚本中集成测试

```python
import torch
from unet import UNet
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader

# 加载模型
model = UNet(n_channels=1, n_classes=1)
checkpoint = torch.load('checkpoints/checkpoint_best.pth', map_location='cuda:0')
model.load_state_dict(checkpoint)
model.eval()

# 加载测试数据
test_set = BasicDataset('./data/img-und', './data/img-full', scale=1.0, split='test')
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# 运行推理
with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_loader):
        predictions = model(images.cuda())
        # 处理结果...
```

### 比较多个模型

```bash
# 创建测试脚本
for epoch in 5 10 15 20 25 30; do
    echo "Testing epoch_$epoch..."
    python test.py --model checkpoints/checkpoint_epoch_${epoch}.pth
done

# 对比结果
cat results/test_metrics.txt
```

---

## 9. 常见问题 (FAQ)

**Q: test.py 和 train.py 的区别是什么?**
- `train.py`: 在训练集上学习参数，在验证集上调整学习率
- `test.py`: 在测试集上评估最终模型性能

**Q: 为什么要用最佳 checkpoint 而不是最新的?**
- 最新 checkpoint 可能过拟合；最佳基于验证集性能选择

**Q: 测试需要多长时间?**
- 1,046 张测试图像，batch_size=32: ~2-3 分钟 (GPU)

**Q: 可以修改测试集分割比例吗?**
- 目前固定为文件名后缀 (_test)；可修改 `data_loading.py` 的 split 逻辑

**Q: 可视化图像太多了怎么办?**
```bash
# 不保存可视化，节省空间
python test.py --model checkpoints/checkpoint_best.pth --no-save-visuals
```

---

## 10. 完整工作流示例

```bash
# 步骤 1: 确认 checkpoint 存在
ls -lh checkpoints/ | tail -5

# 步骤 2: 使用最佳模型测试
python test.py --model checkpoints/checkpoint_best.pth --batch-size 32

# 步骤 3: 检查结果
cat results/test_metrics.txt

# 步骤 4: 查看可视化
# 在 results/before_reconstruction/ 和 after_reconstruction/ 中查看图像

# 步骤 5 (可选): 对比多个 epoch
python test.py --model checkpoints/checkpoint_epoch_20.pth --batch-size 32
python test.py --model checkpoints/checkpoint_epoch_25.pth --batch-size 32
# 比较 results/test_metrics.txt 的输出
```

---

## 总结

| 操作 | 命令 |
|------|------|
| 快速测试 | `python test.py --model checkpoints/checkpoint_epoch50.pth` |
| 列出检查点 | `ls -la checkpoints/` |
| 仅输出指标 | `python test.py --model checkpoints/checkpoint_best.pth --no-save-visuals` |
| 小显存模式 | `python test.py --model checkpoints/checkpoint_best.pth --batch-size 4` |
| 查看结果 | `cat results/test_metrics.txt` |

祝测试顺利！ 🎉
