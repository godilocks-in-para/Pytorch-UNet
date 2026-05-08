"""
Quick-start wrapper for data migration
Automatically detects paths and runs migration with sensible defaults
"""

import os
import sys
from pathlib import Path


def find_source_path():
    """Auto-detect the source data path."""
    cwd = Path.cwd()
    
    # Check parent directory
    parent_source = cwd.parent / 'variable_density_line.7z'
    if parent_source.exists():
        return str(parent_source)
    
    # Check current directory
    curr_source = cwd / 'variable_density_line.7z'
    if curr_source.exists():
        return str(curr_source)
    
    # Check sibling directory
    sibling_source = cwd.parent / 'variable_density_line.7z'
    if sibling_source.exists():
        return str(sibling_source)
    
    return None


if __name__ == '__main__':
    print("="*70)
    print("BraTS Data Migration - Quick Start")
    print("="*70)
    
    # Find source
    source = find_source_path()
    output = './data'
    
    if not source:
        print("\n❌ Error: Cannot find 'variable_density_line.7z' directory")
        print("\nPlease ensure:")
        print("  1. You're running from the project root (where train.py is located)")
        print("  2. OR specify: python migrate_brats_data.py --source <path>")
        sys.exit(1)
    
    print(f"\n✓ Source found: {source}")
    print(f"✓ Output destination: {output}")
    
    # Import and run migration
    try:
        from migrate_brats_data import migrate_images
        
        print("\nStarting migration...")
        print("(This may take 5-15 minutes depending on disk speed)")
        print()
        
        migrate_images(
            source_dir=source,
            output_dir=output,
            split_ratio=(0.8, 0.1, 0.1),
            seed=0
        )
        
        print("\n" + "="*70)
        print("✓ MIGRATION COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Verify data: ls ./data/img-und/ | wc -l")
        print("  2. Train model: python train.py")
        print("  3. Evaluate: python test.py")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
