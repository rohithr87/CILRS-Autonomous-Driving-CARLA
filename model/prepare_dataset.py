#!/usr/bin/env python3
"""
Prepare collected data for Kaggle upload.
Resizes images from 800x600 to 200x88 (model input size).
"""

import os
import sys
import cv2
import shutil
from glob import glob

TARGET_WIDTH = 200
TARGET_HEIGHT = 88

DATA_DIR = '/home/rohith/carla_simulator/collected_data'
OUTPUT_DIR = '/home/rohith/carla_simulator/training_data'

def process_session(session_path, output_session_path):
    """Process one session folder."""
    
    session_name = os.path.basename(session_path)
    print(f"\n📂 Processing: {session_name}")
    
    # Create output directories
    output_images = os.path.join(output_session_path, 'images')
    os.makedirs(output_images, exist_ok=True)
    
    # Copy measurements.csv
    csv_src = os.path.join(session_path, 'measurements.csv')
    csv_dst = os.path.join(output_session_path, 'measurements.csv')
    shutil.copy2(csv_src, csv_dst)
    print(f"   ✓ Copied measurements.csv")
    
    # Copy summary if exists
    summary_src = os.path.join(session_path, 'summary.txt')
    if os.path.exists(summary_src):
        shutil.copy2(summary_src, os.path.join(output_session_path, 'summary.txt'))
    
    # Resize all images
    input_images = os.path.join(session_path, 'images')
    image_files = sorted(glob(os.path.join(input_images, '*.jpg')))
    
    total = len(image_files)
    print(f"   📸 Resizing {total} images to {TARGET_WIDTH}x{TARGET_HEIGHT}...")
    
    for i, img_path in enumerate(image_files):
        # Read image
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"   ⚠ Skipped corrupt image: {os.path.basename(img_path)}")
            continue
        
        # Resize to model input size
        resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Save with same filename
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_images, filename)
        cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Progress
        if (i + 1) % 5000 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"   📊 {i+1}/{total} ({pct:.1f}%)")
    
    print(f"   ✅ Done: {total} images resized")
    return total


def main():
    print("=" * 60)
    print("🔧 PREPARING DATA FOR KAGGLE UPLOAD")
    print("=" * 60)
    
    # Find all session folders
    sessions = sorted(glob(os.path.join(DATA_DIR, 'session*')))
    
    if not sessions:
        print(f"\n❌ No session folders found in {DATA_DIR}")
        return
    
    print(f"\n📁 Found {len(sessions)} sessions")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_frames = 0
    
    for session_path in sessions:
        session_name = os.path.basename(session_path)
        output_session = os.path.join(OUTPUT_DIR, session_name)
        
        frames = process_session(session_path, output_session)
        total_frames += frames
    
    # Check output size
    print(f"\n" + "=" * 60)
    print(f"✅ ALL SESSIONS PROCESSED")
    print(f"   Total frames: {total_frames}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"=" * 60)
    
    # Show size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(OUTPUT_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    size_gb = total_size / (1024**3)
    print(f"\n   📦 Total size: {size_gb:.2f} GB")
    print(f"\n   Next step: Compress for upload")
    print(f"   Run: cd ~/carla_simulator && tar -czf training_data.tar.gz training_data/")


if __name__ == '__main__':
    main()
