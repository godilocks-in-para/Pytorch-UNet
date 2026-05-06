import torch
import torch.nn.functional as F
from tqdm import tqdm

# from utils.dice_score import multiclass_dice_coeff, dice_coeff

from utils.utils import calculate_psnr, calculate_ssim

@torch.inference_mode()
# def evaluate(net, dataloader, device, amp):
#     net.eval()
#     num_val_batches = len(dataloader)
#     dice_score = 0

#     # iterate over the validation set
#     with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
#         for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#             image, mask_true = batch['image'], batch['mask']

#             # move images and labels to correct device and type
#             image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
#             mask_true = mask_true.to(device=device, dtype=torch.long)

#             # predict the mask
#             mask_pred = net(image)

#             if net.n_classes == 1:
#                 assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
#                 mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#                 # compute the Dice score
#                 dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
#             else:
#                 assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
#                 # convert to one-hot format
#                 mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
#                 mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#                 # compute the Dice score, ignoring background
#                 dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

#     net.train()
#     return dice_score / max(num_val_batches, 1)

# @torch.inference_mode()
# def evaluate(model, dataloader, device, amp):
#     """
#     评估重建模型的PSNR分数（用于学习率调度器）
    
#     Args:
#         model: 重建模型
#         dataloader: 验证数据加载器
#         device: 计算设备
#         amp: 是否使用混合精度
    
#     Returns:
#         平均PSNR值
#     """
#     model.eval()
#     num_val_batches = len(dataloader)
#     total_psnr = 0.0
#     n_samples = 0
    
#     # 遍历验证集
#     with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
#         for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', 
#                          unit='batch', leave=False):
#             # 获取输入和目标
#             images_und = batch['img_und']
#             true_image_full = batch['img_full']
            
#             # 移动图像到正确的设备和类型
#             images_und = images_und.to(device=device, dtype=torch.float32, 
#                                       memory_format=torch.channels_last)
#             true_image_full = true_image_full.to(device=device, dtype=torch.float32)
            
#             # 预测重建图像
#             img_full_pred = model(images_und)
            
#             # 计算每个样本的PSNR
#             batch_size = images_und.shape[0]
#             for i in range(batch_size):
#                 # 计算MSE
#                 mse = torch.mean((img_full_pred[i] - true_image_full[i]) ** 2)
#                 if mse > 0:
#                     psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
#                     total_psnr += psnr.item()
#                     n_samples += 1
    
#     model.train()
#     avg_psnr = total_psnr / max(n_samples, 1)
#     return avg_psnr

@torch.inference_mode()
def evaluate(model, val_loader, device, amp):
    model.eval()
    total_loss = 0
    total_psnr = 0
    criterion = F.MSELoss()
    
    with torch.no_grad():
        for batch in val_loader:
            images_und = batch['img_und'].to(device, dtype=torch.float32)
            true_image_full = batch['img_full'].to(device, dtype=torch.float32)
            
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                img_full_pred = model(images_und)
                loss = criterion(img_full_pred, true_image_full)
                
                # 使用PSNR函数
                psnr_value = calculate_psnr(img_full_pred, true_image_full, max_val=1.0)
                total_psnr += psnr_value
            
            total_loss += loss.item()
    model.train()
    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    return avg_loss, avg_psnr

