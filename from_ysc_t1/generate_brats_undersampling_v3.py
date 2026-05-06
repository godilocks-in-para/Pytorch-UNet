#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
generate_brats_undersampling.py

Task 1 for "Multi-contrast MRI Reconstruction from Undersampled Data":
Batch-generate undersampled T2 MRI examples from BraTS .nii.gz volumes.

For each selected 2D T2 slice, this script generates:
  1. full image PNG
  2. sampling mask PNG
  3. aliased image PNG
  4. comparison PNG: mask / full / aliased
  5. optional separate NPY files
  6. NPZ sample file for later model training:
        k_und, mask, im_gt, im_und, k_full, metadata

Default undersampling:
  - 2D random variable-density k-space mask
  - acceleration factor R = 5
  - slice selection mode: uniform or largest

Example:
  python generate_brats_undersampling.py ^
    --data_root "D:/BraTS/BraTS2021_Training_Data" ^
    --output_root "D:/BraTS/undersampling_output" ^
    --acceleration 5 ^
    --num_slices_per_case 5 ^
    --save_npy ^
    --save_npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ============================================================
# FFT / IFFT
# ============================================================

def fft2c(img: np.ndarray) -> np.ndarray:
    """
    Centered 2D FFT.

    Input:
        img: 2D real-valued image, shape [H, W]

    Output:
        centered complex k-space, shape [H, W]
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """
    Centered 2D inverse FFT.

    Input:
        kspace: centered complex k-space, shape [H, W]

    Output:
        complex image, shape [H, W]
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))


# ============================================================
# Image normalization / saving
# ============================================================

def normalize_image(img: np.ndarray, percentile_clip: bool = True) -> np.ndarray:
    """
    Normalize image to [0, 1] for visualization and training.

    percentile_clip=True:
        Uses 1st and 99th percentile clipping to improve display contrast.
    """
    img = img.astype(np.float32)

    if np.allclose(img.max(), img.min()):
        return np.zeros_like(img, dtype=np.float32)

    if percentile_clip:
        p1, p99 = np.percentile(img, [1, 99])
        if p99 > p1:
            img = np.clip(img, p1, p99)

    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img.astype(np.float32)


def save_gray_image(img: np.ndarray, save_path: Path, title: str | None = None) -> None:
    """Save one grayscale PNG image."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def save_comparison_figure(
    mask: np.ndarray,
    full_img: np.ndarray,
    aliased_img: np.ndarray,
    save_path: Path,
) -> None:
    """Save a side-by-side figure: mask / full / aliased."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("Sampling Mask")
    axes[0].axis("off")

    axes[1].imshow(full_img, cmap="gray")
    axes[1].set_title("Fully Sampled T2")
    axes[1].axis("off")

    axes[2].imshow(aliased_img, cmap="gray")
    axes[2].set_title("Aliased Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()


# ============================================================
# Mask generation
# ============================================================

def generate_variable_density_mask(
    shape: Tuple[int, int],
    acceleration: int = 5,
    center_fraction: float = 0.08,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a 2D random variable-density undersampling mask.

    Requirement:
        Random variable-density mask with acceleration factor = 5.

    Meaning:
        acceleration = 5 means roughly 1/5 k-space points are sampled.
        center_fraction controls a fully-sampled low-frequency center region.

    Returns:
        mask: float32 array, shape [H, W], values in {0, 1}
    """
    if acceleration <= 0:
        raise ValueError(f"acceleration must be positive, got {acceleration}")

    if not (0.0 <= center_fraction <= 1.0):
        raise ValueError(f"center_fraction must be in [0, 1], got {center_fraction}")

    rng = np.random.default_rng(seed)
    H, W = shape

    total_points = H * W
    target_samples = int(round(total_points / acceleration))

    # Normalized distance from k-space center
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy = (yy - cy) / (H / 2.0)
    xx = (xx - cx) / (W / 2.0)
    rr = np.sqrt(xx**2 + yy**2)

    mask = np.zeros((H, W), dtype=np.float32)

    # Fully sampled low-frequency center
    ch = max(1, int(round(H * center_fraction)))
    cw = max(1, int(round(W * center_fraction)))

    y0 = max(0, H // 2 - ch // 2)
    y1 = min(H, y0 + ch)
    x0 = max(0, W // 2 - cw // 2)
    x1 = min(W, x0 + cw)

    mask[y0:y1, x0:x1] = 1.0

    center_count = int(mask.sum())
    remaining = max(0, target_samples - center_count)

    # Outside center: random sampling with higher probability near center
    outside = mask == 0

    # Probability decays from center to high frequency region
    prob = 1.0 / (1.0 + (rr / 0.45) ** 4)
    prob = prob.astype(np.float64)
    prob[~outside] = 0.0

    outside_indices = np.flatnonzero(outside.ravel())
    outside_prob = prob.ravel()[outside_indices]

    if outside_prob.sum() > 0 and remaining > 0:
        outside_prob = outside_prob / outside_prob.sum()
        remaining = min(remaining, len(outside_indices))
        chosen = rng.choice(
            outside_indices,
            size=remaining,
            replace=False,
            p=outside_prob,
        )
        mask.ravel()[chosen] = 1.0

    return mask.astype(np.float32)


# ============================================================
# Slice selection
# ============================================================

def select_valid_slices(
    volume: np.ndarray,
    min_nonzero_ratio: float = 0.02,
    num_slices_per_case: int = 5,
    slice_select_mode: str = "uniform",
) -> List[int]:
    """
    Select useful 2D slices from a 3D MRI volume.

    BraTS volumes contain many empty or nearly empty slices.
    This function first skips slices with too few nonzero pixels.

    slice_select_mode:
        uniform:
            Uniformly select slices from all valid slices.
            This keeps coverage from top to bottom of the brain.

        largest:
            Select slices around the slice with the largest nonzero area.
            This usually keeps slices where the brain occupies a larger
            image region, which is often more useful for later training.

    Args:
        volume:
            3D T2 volume, expected shape [H, W, D].
        min_nonzero_ratio:
            Skip slice if nonzero pixels / total pixels is smaller than this.
        num_slices_per_case:
            Number of slices to select for each case.
            If <= 0, all valid slices are selected.
        slice_select_mode:
            "uniform" or "largest".

    Returns:
        List of selected slice indices.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    if slice_select_mode not in {"uniform", "largest"}:
        raise ValueError(
            f"slice_select_mode must be 'uniform' or 'largest', got {slice_select_mode}"
        )

    D = volume.shape[2]
    valid_indices: List[int] = []
    valid_ratios: List[float] = []

    for i in range(D):
        sl = volume[:, :, i]
        nonzero_ratio = np.count_nonzero(sl) / sl.size
        if nonzero_ratio >= min_nonzero_ratio:
            valid_indices.append(i)
            valid_ratios.append(float(nonzero_ratio))

    if len(valid_indices) == 0:
        return []

    if num_slices_per_case <= 0 or len(valid_indices) <= num_slices_per_case:
        return valid_indices

    if slice_select_mode == "uniform":
        positions = np.linspace(0, len(valid_indices) - 1, num_slices_per_case)
        selected = [valid_indices[int(round(p))] for p in positions]
        selected = sorted(list(set(selected)))
        return selected

    # slice_select_mode == "largest"
    # Pick a contiguous window around the slice with the largest nonzero area.
    max_pos = int(np.argmax(np.array(valid_ratios)))
    half = num_slices_per_case // 2

    start = max_pos - half
    end = start + num_slices_per_case

    if start < 0:
        start = 0
        end = num_slices_per_case

    if end > len(valid_indices):
        end = len(valid_indices)
        start = max(0, end - num_slices_per_case)

    selected = valid_indices[start:end]
    selected = sorted(list(set(selected)))
    return selected


# ============================================================
# One slice processing
# ============================================================

def process_one_slice(
    slice_img: np.ndarray,
    out_dir: Path,
    case_id: str,
    slice_idx: int,
    acceleration: int = 5,
    center_fraction: float = 0.08,
    seed: int | None = None,
    save_npy: bool = True,
    save_npz: bool = True,
) -> None:
    """
    Process one 2D T2 slice.

    Generates:
        PNG:
            *_full.png
            *_mask.png
            *_aliased.png
            *_comparison.png

        Optional NPY:
            *_full.npy
            *_mask.npy
            *_kspace_full.npy
            *_kspace_und.npy
            *_aliased.npy

        Optional NPZ:
            *_sample.npz
              - k_und: undersampled k-space, complex64, [H, W]
              - mask: undersampling mask, float32, [H, W]
              - im_gt: fully sampled image, float32, [H, W]
              - im_und: aliased image, float32, [H, W]
              - k_full: fully sampled k-space, complex64, [H, W]
              - case_id
              - slice_idx
              - acceleration
              - center_fraction
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ground truth image
    full_img = normalize_image(slice_img)

    # Fully sampled k-space
    kspace_full = fft2c(full_img)

    # Undersampling mask
    mask = generate_variable_density_mask(
        full_img.shape,
        acceleration=acceleration,
        center_fraction=center_fraction,
        seed=seed,
    )

    # Undersampled k-space
    kspace_und = kspace_full * mask

    # IFFT to aliased image
    aliased_complex = ifft2c(kspace_und)
    aliased_img = np.abs(aliased_complex)
    aliased_img = normalize_image(aliased_img)

    prefix = f"{case_id}_slice_{slice_idx:03d}"

    # 1. PNG outputs for visualization/report
    save_gray_image(
        full_img,
        out_dir / f"{prefix}_full.png",
        title="Fully Sampled T2",
    )
    save_gray_image(
        mask,
        out_dir / f"{prefix}_mask.png",
        title="Sampling Mask",
    )
    save_gray_image(
        aliased_img,
        out_dir / f"{prefix}_aliased.png",
        title="Aliased Image",
    )
    save_comparison_figure(
        mask,
        full_img,
        aliased_img,
        out_dir / f"{prefix}_comparison.png",
    )

    # 2. Separate NPY outputs for debugging
    if save_npy:
        np.save(out_dir / f"{prefix}_full.npy", full_img.astype(np.float32))
        np.save(out_dir / f"{prefix}_mask.npy", mask.astype(np.float32))
        np.save(out_dir / f"{prefix}_kspace_full.npy", kspace_full.astype(np.complex64))
        np.save(out_dir / f"{prefix}_kspace_und.npy", kspace_und.astype(np.complex64))
        np.save(out_dir / f"{prefix}_aliased.npy", aliased_img.astype(np.float32))

    # 3. Packed NPZ output for later PyTorch Dataset
    if save_npz:
        np.savez_compressed(
            out_dir / f"{prefix}_sample.npz",
            # Lab-style core fields
            k_und=kspace_und.astype(np.complex64),
            mask=mask.astype(np.float32),
            im_gt=full_img.astype(np.float32),
            im_und=aliased_img.astype(np.float32),
            # Extra field for debugging / data consistency experiments
            k_full=kspace_full.astype(np.complex64),
            # Metadata
            case_id=np.array(case_id),
            slice_idx=np.array(slice_idx, dtype=np.int32),
            acceleration=np.array(acceleration, dtype=np.int32),
            center_fraction=np.array(center_fraction, dtype=np.float32),
        )


# ============================================================
# One case processing
# ============================================================

def process_case(
    case_dir: Path,
    output_root: Path,
    acceleration: int = 5,
    center_fraction: float = 0.08,
    num_slices_per_case: int = 5,
    min_nonzero_ratio: float = 0.02,
    slice_select_mode: str = "uniform",
    save_npy: bool = True,
    save_npz: bool = True,
    seed: int = 42,
) -> None:
    """Process one BraTS case folder containing *_t2.nii.gz."""
    case_id = case_dir.name

    t2_files = sorted(list(case_dir.glob("*_t2.nii.gz")))
    if len(t2_files) == 0:
        print(f"[Skip] {case_id}: no *_t2.nii.gz found.")
        return

    if len(t2_files) > 1:
        print(f"[Warn] {case_id}: found multiple T2 files, use first one: {t2_files[0].name}")

    t2_path = t2_files[0]
    print(f"[Processing] {case_id} -> {t2_path.name}")

    volume = nib.load(str(t2_path)).get_fdata().astype(np.float32)

    if volume.ndim != 3:
        print(f"[Skip] {case_id}: T2 volume is not 3D, shape={volume.shape}")
        return

    slice_indices = select_valid_slices(
        volume,
        min_nonzero_ratio=min_nonzero_ratio,
        num_slices_per_case=num_slices_per_case,
        slice_select_mode=slice_select_mode,
    )

    if len(slice_indices) == 0:
        print(f"[Skip] {case_id}: no valid slices.")
        return

    case_out_dir = output_root / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)

    for local_i, sidx in enumerate(slice_indices):
        sl = volume[:, :, sidx]

        process_one_slice(
            slice_img=sl,
            out_dir=case_out_dir,
            case_id=case_id,
            slice_idx=sidx,
            acceleration=acceleration,
            center_fraction=center_fraction,
            seed=seed + local_i,
            save_npy=save_npy,
            save_npz=save_npz,
        )

    print(f"[Done] {case_id}: saved {len(slice_indices)} slice group(s).")


# ============================================================
# Find BraTS cases
# ============================================================

def find_case_dirs(data_root: Path) -> List[Path]:
    """
    Recursively find case directories containing *_t2.nii.gz.

    Supports structures like:
        data_root/
          BraTS2021_00000/
            BraTS2021_00000_t2.nii.gz

    Also supports:
        data_root/
          BraTS2021_Training_Data/
            BraTS2021_00000/
              BraTS2021_00000_t2.nii.gz
    """
    case_dirs = []
    for t2_path in data_root.rglob("*_t2.nii.gz"):
        case_dirs.append(t2_path.parent)

    case_dirs = sorted(list(set(case_dirs)))
    return case_dirs


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate undersampled BraTS T2 MRI examples."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder of BraTS data. The script recursively searches for *_t2.nii.gz.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output folder for generated PNG/NPY/NPZ files.",
    )
    parser.add_argument(
        "--acceleration",
        type=int,
        default=5,
        help="Acceleration factor R. R=5 means about 20%% k-space samples are kept.",
    )
    parser.add_argument(
        "--center_fraction",
        type=float,
        default=0.08,
        help="Fully sampled center fraction of k-space. Default: 0.08.",
    )
    parser.add_argument(
        "--num_slices_per_case",
        type=int,
        default=5,
        help="Number of valid 2D slices selected per case. Use 0 or negative to save all valid slices.",
    )
    parser.add_argument(
        "--min_nonzero_ratio",
        type=float,
        default=0.02,
        help="Skip slices whose nonzero-pixel ratio is below this threshold.",
    )
    parser.add_argument(
        "--slice_select_mode",
        type=str,
        default="uniform",
        choices=["uniform", "largest"],
        help=(
            "Slice selection strategy. "
            "'uniform' uniformly samples valid slices; "
            "'largest' selects slices around the slice with the largest nonzero area."
        ),
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Save separate .npy files for full/mask/kspace/aliased.",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Save packed .npz sample files for later training.",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="Maximum number of cases to process. 0 means all cases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mask generation.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    case_dirs = find_case_dirs(data_root)

    if len(case_dirs) == 0:
        print(f"[Error] No case folders with *_t2.nii.gz found under: {data_root}")
        print("Please check whether the BraTS .tar files have been fully extracted.")
        return

    print(f"[Info] Found {len(case_dirs)} case folder(s).")

    if args.max_cases > 0:
        case_dirs = case_dirs[: args.max_cases]
        print(f"[Info] Only processing first {len(case_dirs)} case(s).")

    print("[Info] Configuration:")
    print(f"  data_root           = {data_root}")
    print(f"  output_root         = {output_root}")
    print(f"  acceleration        = {args.acceleration}")
    print(f"  center_fraction     = {args.center_fraction}")
    print(f"  num_slices_per_case = {args.num_slices_per_case}")
    print(f"  min_nonzero_ratio   = {args.min_nonzero_ratio}")
    print(f"  slice_select_mode   = {args.slice_select_mode}")
    print(f"  save_npy            = {args.save_npy}")
    print(f"  save_npz            = {args.save_npz}")
    print(f"  seed                = {args.seed}")

    for case_i, case_dir in enumerate(case_dirs):
        case_seed = args.seed + case_i * 1000
        process_case(
            case_dir=case_dir,
            output_root=output_root,
            acceleration=args.acceleration,
            center_fraction=args.center_fraction,
            num_slices_per_case=args.num_slices_per_case,
            min_nonzero_ratio=args.min_nonzero_ratio,
            slice_select_mode=args.slice_select_mode,
            save_npy=args.save_npy,
            save_npz=args.save_npz,
            seed=case_seed,
        )

    print("[All Done]")


if __name__ == "__main__":
    main()
