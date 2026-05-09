import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
# from utils.dice_score import dice_loss
from utils.utils import calculate_psnr, calculate_ssim

dir_img_und = Path('./data/img-und/')
dir_img_full = Path('./data/img-full/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        test_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create datasets based on split (from filenames: _train, _val, _test)
    train_set = BasicDataset(dir_img_und, dir_img_full, img_scale, split='train')
    val_set = BasicDataset(dir_img_und, dir_img_full, img_scale, split='val')
    test_set = BasicDataset(dir_img_und, dir_img_full, img_scale, split='test')
    
    n_train = len(train_set)
    n_val = len(val_set)
    n_test = len(test_set)
    
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, test_percent=test_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Test size:       {n_test}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                          lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)

    # 修改调度器：监控验证损失（'min' 模式）而不是 Dice
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: minimize reconstruction loss

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 修改损失函数：使用 MSE Loss (L2 Loss)
    criterion = nn.MSELoss()

    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images_und, true_image_full = batch['img_und'], batch['img_full']

                assert images_und.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images_und have {images_und.shape[1]} channels. Please check that ' \
                    'the images_und are loaded correctly.'

                images_und = images_und.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_image_full = true_image_full.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    img_full_pred = model(images_und)
                    loss = criterion(img_full_pred, true_image_full)
                        

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images_und.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                if global_step % 100 == 0:
                    experiment.log({
                    'train PSNR': calculate_psnr(img_full_pred, true_image_full),
                    'train SSIM': calculate_ssim(img_full_pred, true_image_full),
                    'step': global_step,
                    'epoch': epoch})
                    
                # if global_step % 400 == 0:
                #     # 记录第一张图像的欠采样、重建和真实图像
                #     try:
                #         und_img = images_und[0, 0].detach().cpu().numpy()
                #         pred_img = img_full_pred[0, 0].detach().cpu().numpy()
                #         true_img = true_image_full[0, 0].detach().cpu().numpy()
                #         log_dict_400={}
                #         log_dict_400['train/undersampled'] = wandb.Image(und_img, caption='Undersampled Input')
                #         log_dict_400['train/reconstructed'] = wandb.Image(pred_img, caption='Reconstructed')
                #         log_dict_400['train/ground_truth'] = wandb.Image(true_img, caption='Ground Truth')
                #     except:
                #         pass

                if global_step % 10 == 0:
                    log_dict_10 = {
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                    }
                    experiment.log(log_dict_10)
                    
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
                    
            # histograms = {}
            # for tag, value in model.named_parameters():
            #     tag = tag.replace('/', '.')
            #     if not (torch.isinf(value) | torch.isnan(value)).any():
            #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
            #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_loss, val_psnr, val_ssim = evaluate(model, val_loader, device, amp)  # 新写的评估函数
            scheduler.step(val_loss)  # 监控验证损失（'min'模式）
                        
            logging.info(f'Validation Loss: {val_loss:.6f}, Validation PSNR: {val_psnr:.2f} dB, Validation SSIM: {val_ssim:.4f}')
                            
            val_log_dict = {
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation loss': val_loss,
                'validation PSNR': val_psnr,
                'validation SSIM': val_ssim,
                'epoch': epoch,
                'step': global_step,
                # **histograms
            }


            
            # =========================================
            # Validation visualization
            # =========================================

            # try:
            #     model.eval()

            #     with torch.no_grad():

            #         for idx, image in enumerate(val_loader):

            #             # 只保存前5张
            #             if idx >= 5:
            #                 break

            #             images_und_val = image['img_und'].to(
            #                 device=device,
            #                 dtype=torch.float32,
            #                 memory_format=torch.channels_last
            #             )

            #             true_image_full_val = image['img_full'].to(
            #                 device=device,
            #                 dtype=torch.float32
            #             )

            #             with torch.autocast(
            #                 device.type if device.type != 'mps' else 'cpu',
            #                 enabled=amp
            #             ):
            #                 img_full_pred_val = model(images_und_val)

            #             # 只取batch中的第一张
            #             und_img_val = images_und_val[0, 0].detach().cpu().numpy()
            #             pred_img_val = img_full_pred_val[0, 0].detach().cpu().numpy()
            #             true_img_val = true_image_full_val[0, 0].detach().cpu().numpy()

            #             val_log_dict[f'val/undersampled_{idx}'] = wandb.Image(
            #                 und_img_val,
            #                 caption=f'Val Undersampled {idx}'
            #             )

            #             val_log_dict[f'val/reconstructed_{idx}'] = wandb.Image(
            #                 pred_img_val,
            #                 caption=f'Val Reconstructed {idx}'
            #             )

            #             val_log_dict[f'val/ground_truth_{idx}'] = wandb.Image(
            #                 true_img_val,
            #                 caption=f'Val Ground Truth {idx}'
            #             )

            # except Exception as e:
            #     print(e)
            
            try:
                experiment.log(val_log_dict)
            except:
                pass
            
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images_und and target images_full')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images_und')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images_und
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        # 移除旧的mask_values键（如果存在）以支持向后兼容
        state_dict.pop('mask_values', None)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
