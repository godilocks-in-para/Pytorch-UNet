import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

# def predict_img(net,
#                 full_img,
#                 device,
#                 scale_factor=1,
#                 out_threshold=0.5):
#     net.eval()
#     img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)

#     with torch.no_grad():
#         output = net(img).cpu()
#         output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
#         if net.n_classes > 1:
#             mask = output.argmax(dim=1)
#         else:
#             mask = torch.sigmoid(output) > out_threshold

#     return mask[0].long().squeeze().numpy()
def predict_img(net, img_in, device, scale_factor=1):
    """
    预测重建图像
    
    Args:
        net: 重建模型
        img_in: 输入图像（欠采样图像）
        device: 计算设备
        scale_factor: 缩放因子
    
    Returns:
        重建后的图像，numpy array，形状 [H, W]
    """
    net.eval()
    
    # 预处理
    img = torch.from_numpy(BasicDataset.preprocess(None, img_in, scale_factor, is_mask=False))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (img_in.size[1], img_in.size[0]), mode='bilinear')
        output = torch.clamp(output, 0, 1)  # 确保在[0,1]范围内
    
    # 返回单通道图像 [H, W]
    return output[0, 0].numpy()  # 假设单通道输出

def get_args():
    parser = argparse.ArgumentParser(description='Predict reconstructed images from undersampled input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output images')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of output channels')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


# def mask_to_image(mask: np.ndarray, mask_values):
#     if isinstance(mask_values[0], list):
#         out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
#     elif mask_values == [0, 1]:
#         out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
#     else:
#         out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

#     if mask.ndim == 3:
#         mask = np.argmax(mask, axis=0)

#     for i, v in enumerate(mask_values):
#         out[mask == i] = v

#     return Image.fromarray(out)
def array_to_image(array: np.ndarray):
    """
    将重建图像数组转换为PIL Image（简化版）
    假设输入范围[0,1]，形状[H,W]或[H,W,C]
    
    Args:
        array: 重建图像数组，值域[0,1]
    
    Returns:
        PIL Image对象
    """
    # 处理维度 [C, H, W] -> [H, W, C]
    if array.ndim == 3 and array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))
    
    # 处理单通道 [H, W, 1] -> [H, W]
    if array.ndim == 3 and array.shape[2] == 1:
        array = array.squeeze(axis=2)
    
    # 归一化到[0, 255]
    array = np.clip(array, 0, 1) * 255
    array = array.astype(np.uint8)
    
    # 转换为PIL Image
    if array.ndim == 2:
        return Image.fromarray(array, mode='L')
    elif array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array, mode='RGB')
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    # 移除旧的mask_values键（如果存在）以支持向后兼容
    state_dict.pop('mask_values', None)
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img_und_in = Image.open(filename)

        img_full_pred = predict_img(net=net,
                           img_in=img_und_in,
                           device=device,
                           scale_factor=args.scale)

        if not args.no_save:
            out_filename = out_files[i]
            result = array_to_image(img_full_pred)
            result.save(out_filename)
            logging.info(f'Reconstruction saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            # 可视化：显示输入和预测的重建图像
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(np.array(img_und_in), cmap='gray')
            axes[0].set_title('Undersampled Input')
            axes[0].axis('off')
            axes[1].imshow(img_full_pred, cmap='gray')
            axes[1].set_title('Reconstructed Image')
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
