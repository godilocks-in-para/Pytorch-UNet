import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', split: str = None):
        """
        Args:
            images_dir: Path to undersampled images directory
            mask_dir: Path to fully-sampled images directory
            scale: Image scaling factor
            mask_suffix: Suffix for mask files
            split: Filter by split ('train', 'val', 'test', or None for all)
        """
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.split = split

        # Get all files and filter by split if specified
        all_ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        
        # Filter by split: files end with _train, _val, or _test
        if split is not None:
            assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got {split}"
            self.ids = [id for id in all_ids if id.endswith(f'_{split}')]
        else:
            self.ids = all_ids
        
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir} (split={split}), make sure you put your images there')

        logging.info(f'Creating {split or "combined"} dataset with {len(self.ids)} examples')
        # 对于重建任务，不需要扫描mask文件获取unique values
        self.mask_values = None  # 重建任务中不需要mask_values映射

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        # 对于重建任务，两张图像（欠采样和全采样）都应该被当作图像（非mask）处理
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # 两张图像都使用相同的预处理（是_mask=False）
        img = self.preprocess(None, img, self.scale, is_mask=False)
        mask = self.preprocess(None, mask, self.scale, is_mask=False)

        return {
            'img_und': torch.as_tensor(img.copy()).float().contiguous(),
            'img_full': torch.as_tensor(mask.copy()).float().contiguous()
        }

