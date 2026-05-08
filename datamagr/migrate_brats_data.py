"""
BraTS Undersampled Data Migration Script
=========================================
Migrates data from variable_density_line.7z into the data directory structure.

Data Structure:
- Source: variable_density_line.7z/variable_density_line/BraTS2021_XXXXX/
  - *_aliased.npy: Undersampled (aliased) image slices
  - *_full.npy: Fully-sampled (ground truth) image slices
  - *_kspace_und.npy: Undersampled k-space data
  - *_kspace_full.npy: Full k-space data
  - *_mask.npy: Undersampling mask
  - *.png: Visualization files

- Destination: ./data/
  - img-und/: Undersampled (aliased) images for input
  - img-full/: Fully-sampled (ground truth) images for target

Usage:
    python migrate_brats_data.py --source <source_dir> --output <output_dir> --split-ratio 0.8 0.1 0.1
"""

import os
import sys
import numpy as np
from pathlib import Path
from shutil import copy2
import argparse
from tqdm import tqdm
import random


def create_directories(output_dir):
    """Create necessary output directories."""
    img_und_dir = os.path.join(output_dir, 'img-und')
    img_full_dir = os.path.join(output_dir, 'img-full')
    
    os.makedirs(img_und_dir, exist_ok=True)
    os.makedirs(img_full_dir, exist_ok=True)
    
    return img_und_dir, img_full_dir


def get_patient_folders(source_dir):
    """Get list of all patient folders (BraTS2021_XXXXX)."""
    patients = []
    variable_density_path = os.path.join(source_dir, 'variable_density_line')
    
    if not os.path.exists(variable_density_path):
        raise FileNotFoundError(f"Cannot find variable_density_line folder at {variable_density_path}")
    
    for folder in sorted(os.listdir(variable_density_path)):
        patient_path = os.path.join(variable_density_path, folder)
        if os.path.isdir(patient_path) and folder.startswith('BraTS2021_'):
            patients.append((folder, patient_path))
    
    return patients


def get_image_slices(patient_path):
    """Get all undersampled and full-sampled image slices from a patient folder."""
    aliased_files = []
    full_files = []
    
    for file in os.listdir(patient_path):
        if file.endswith('_aliased.npy'):
            aliased_files.append(file)
        elif file.endswith('_full.npy'):
            full_files.append(file)
    
    return sorted(aliased_files), sorted(full_files)


def verify_image_pair(patient_path, aliased_file, full_file):
    """Verify that aliased and full images have matching slice numbers."""
    try:
        # Extract slice number from filenames
        # Format: BraTS2021_00000_slice_070_aliased.npy
        aliased_parts = aliased_file.split('_slice_')
        full_parts = full_file.split('_slice_')
        
        if len(aliased_parts) >= 2 and len(full_parts) >= 2:
            aliased_slice = aliased_parts[1].split('_')[0]
            full_slice = full_parts[1].split('_')[0]
            
            if aliased_slice != full_slice:
                return False
            
            # Verify data shapes match
            aliased_data = np.load(os.path.join(patient_path, aliased_file))
            full_data = np.load(os.path.join(patient_path, full_file))
            
            if aliased_data.shape != full_data.shape:
                print(f"Shape mismatch: {aliased_file} {aliased_data.shape} vs {full_file} {full_data.shape}")
                return False
            
            return True
    except Exception as e:
        print(f"Error verifying pair: {e}")
        return False
    
    return True


def migrate_images(source_dir, output_dir, split_ratio=(0.8, 0.1, 0.1), seed=0):
    """
    Migrate BraTS data to img-und/ and img-full/ directories.
    
    Args:
        source_dir: Path to variable_density_line.7z directory
        output_dir: Path to output data directory
        split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
        seed: Random seed for reproducible splits
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    img_und_dir, img_full_dir = create_directories(output_dir)
    
    # Get all patient folders
    patients = get_patient_folders(source_dir)
    print(f"Found {len(patients)} patient folders")
    
    # Collect all image pairs
    all_pairs = []
    for patient_name, patient_path in tqdm(patients, desc="Scanning patient folders"):
        aliased_files, full_files = get_image_slices(patient_path)
        
        # Match aliased and full files by slice number
        for aliased_file in aliased_files:
            # Find corresponding full file
            aliased_slice = aliased_file.split('_slice_')[1].split('_')[0]
            full_file = None
            
            for ff in full_files:
                if f'_slice_{aliased_slice}_full.npy' in ff:
                    full_file = ff
                    break
            
            if full_file and verify_image_pair(patient_path, aliased_file, full_file):
                all_pairs.append({
                    'patient': patient_name,
                    'patient_path': patient_path,
                    'aliased_file': aliased_file,
                    'full_file': full_file,
                    'slice_num': aliased_slice
                })
    
    print(f"Found {len(all_pairs)} valid image pairs")
    
    # Split data
    random.shuffle(all_pairs)
    train_count = int(len(all_pairs) * split_ratio[0])
    val_count = int(len(all_pairs) * split_ratio[1])
    
    train_pairs = all_pairs[:train_count]
    val_pairs = all_pairs[train_count:train_count + val_count]
    test_pairs = all_pairs[train_count + val_count:]
    
    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
    
    # Migrate training data
    print("\nMigrating training images...")
    for pair in tqdm(train_pairs, desc="Train"):
        migrate_pair(pair, img_und_dir, img_full_dir, 'train')
    
    # Migrate validation data
    print("\nMigrating validation images...")
    for pair in tqdm(val_pairs, desc="Val"):
        migrate_pair(pair, img_und_dir, img_full_dir, 'val')
    
    # Migrate test data
    print("\nMigrating test images...")
    for pair in tqdm(test_pairs, desc="Test"):
        migrate_pair(pair, img_und_dir, img_full_dir, 'test')
    
    # Print summary statistics
    print_migration_summary(all_pairs)
    
    print(f"\n✓ Migration completed!")
    print(f"  Undersampled images: {img_und_dir}")
    print(f"  Full-sampled images: {img_full_dir}")


def migrate_pair(pair, img_und_dir, img_full_dir, split):
    """Migrate a single image pair."""
    patient = pair['patient']
    patient_path = pair['patient_path']
    slice_num = pair['slice_num']
    
    aliased_file = pair['aliased_file']
    full_file = pair['full_file']
    
    # Create filenames with split prefix
    base_name = f"{patient}_slice_{slice_num}_{split}"
    
    # Save undersampled image
    aliased_data = np.load(os.path.join(patient_path, aliased_file))
    und_path = os.path.join(img_und_dir, f"{base_name}.npy")
    np.save(und_path, aliased_data)
    
    # Also save as PNG for visualization
    from PIL import Image
    und_img = np.uint8((aliased_data - aliased_data.min()) / 
                       (aliased_data.max() - aliased_data.min() + 1e-8) * 255)
    Image.fromarray(und_img).save(und_path.replace('.npy', '.png'))
    
    # Save full-sampled image
    full_data = np.load(os.path.join(patient_path, full_file))
    full_path = os.path.join(img_full_dir, f"{base_name}.npy")
    np.save(full_path, full_data)
    
    # Also save as PNG for visualization
    full_img = np.uint8((full_data - full_data.min()) / 
                        (full_data.max() - full_data.min() + 1e-8) * 255)
    Image.fromarray(full_img).save(full_path.replace('.npy', '.png'))


def print_migration_summary(all_pairs):
    """Print summary statistics about migrated data."""
    print("\n" + "="*60)
    print("Migration Summary")
    print("="*60)
    
    # Statistics
    unique_patients = len(set([p['patient'] for p in all_pairs]))
    total_slices = len(all_pairs)
    
    print(f"Total patient studies: {unique_patients}")
    print(f"Total slices migrated: {total_slices}")
    
    # Image statistics
    if all_pairs:
        sample_pair = all_pairs[0]
        aliased_data = np.load(os.path.join(sample_pair['patient_path'], 
                                            sample_pair['aliased_file']))
        full_data = np.load(os.path.join(sample_pair['patient_path'], 
                                         sample_pair['full_file']))
        
        print(f"\nImage properties:")
        print(f"  Shape: {aliased_data.shape}")
        print(f"  Dtype: {aliased_data.dtype}")
        print(f"  Aliased range: [{aliased_data.min():.2f}, {aliased_data.max():.2f}]")
        print(f"  Full range: [{full_data.min():.2f}, {full_data.max():.2f}]")
    
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate BraTS undersampled data')
    parser.add_argument('--source', type=str, 
                        default='../variable_density_line.7z',
                        help='Path to variable_density_line.7z directory')
    parser.add_argument('--output', type=str, 
                        default='./data',
                        help='Output directory for migrated data')
    parser.add_argument('--split-ratio', type=float, nargs=3,
                        default=[0.8, 0.1, 0.1],
                        help='Train/Val/Test split ratios')
    parser.add_argument('--seed', type=int, 
                        default=0,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate split ratio
    if not np.isclose(sum(args.split_ratio), 1.0):
        print("Error: Split ratios must sum to 1.0")
        sys.exit(1)
    
    try:
        migrate_images(args.source, args.output, 
                      split_ratio=tuple(args.split_ratio),
                      seed=args.seed)
    except Exception as e:
        print(f"Error during migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
