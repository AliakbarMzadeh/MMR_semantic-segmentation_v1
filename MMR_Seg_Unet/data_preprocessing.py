
"""
Data Preprocessing Script for SAR-RARP50 Dataset
Converts video + segmentation folders to ARCSeg format
"""

import os
import cv2
import zipfile
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import random

def extract_frames_from_video(video_path, seg_folder, output_images_dir, output_gt_dir, video_name):
    """
    Extract frames from video that match segmentation mask frame numbers
    """
    # Read video
    cap = cv2.VideoCapture(video_path)

    # Get all segmentation mask files
    seg_files = [f for f in os.listdir(seg_folder) if f.endswith('.png')]
    frame_numbers = [int(f.split('.')[0]) for f in seg_files]

    print(f"Processing {len(frame_numbers)} frames for {video_name}")

    for frame_num in tqdm(frame_numbers, desc=f"Extracting frames from {video_name}"):
        # Set video position to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save image
            image_filename = f"{video_name}_frame_{frame_num:09d}.png"
            image_path = os.path.join(output_images_dir, image_filename)

            # Save using PIL to ensure RGB format
            pil_image = Image.fromarray(frame)
            pil_image.save(image_path)

            # Copy corresponding segmentation mask
            seg_filename = f"{frame_num:09d}.png"
            seg_path = os.path.join(seg_folder, seg_filename)

            if os.path.exists(seg_path):
                gt_filename = f"{video_name}_frame_{frame_num:09d}.png"
                gt_path = os.path.join(output_gt_dir, gt_filename)
                shutil.copy2(seg_path, gt_path)

    cap.release()

def process_dataset(drive_dataset_path, output_base_path):
    """
    Process the entire SAR-RARP50 dataset
    """
    # Create output directories
    train_output_path = os.path.join(output_base_path, "sarrarp50")

    # Create train, val, test directories
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'groundtruth']:
            os.makedirs(os.path.join(train_output_path, split, subdir), exist_ok=True)

    # Process Train folder
    train_path = os.path.join(drive_dataset_path, "Train")
    test_path = os.path.join(drive_dataset_path, "Test")

    all_videos = []

    # Collect all video files from Train folder
    if os.path.exists(train_path):
        train_videos = [f for f in os.listdir(train_path) if f.endswith('.zip')]
        all_videos.extend([(os.path.join(train_path, v), v, 'train_source') for v in train_videos])

    # Collect all video files from Test folder
    if os.path.exists(test_path):
        test_videos = [f for f in os.listdir(test_path) if f.endswith('.zip')]
        all_videos.extend([(os.path.join(test_path, v), v, 'test_source') for v in test_videos])

    print(f"Found {len(all_videos)} video files total")

    # Split videos: Train folder → train/val, Test folder → test
    train_source_videos = [v for v in all_videos if v[2] == 'train_source']
    test_source_videos = [v for v in all_videos if v[2] == 'test_source']

    # Shuffle train videos for random split
    random.shuffle(train_source_videos)

    # Split train source into train/val (80/20)
    split_idx = int(0.8 * len(train_source_videos))
    train_videos = train_source_videos[:split_idx]
    val_videos = train_source_videos[split_idx:]
    test_videos = test_source_videos

    print(f"Train: {len(train_videos)}, Val: {len(val_videos)}, Test: {len(test_videos)}")

    # Process each split
    video_splits = [
        (train_videos, 'train'),
        (val_videos, 'val'),
        (test_videos, 'test')
    ]

    for video_list, split_name in video_splits:
        print(f"\nProcessing {split_name} split...")

        images_dir = os.path.join(train_output_path, split_name, 'images')
        gt_dir = os.path.join(train_output_path, split_name, 'groundtruth')

        for zip_path, zip_name, _ in tqdm(video_list, desc=f"Processing {split_name}"):
            video_name = zip_name.replace('.zip', '')

            # Extract zip to temporary directory
            temp_dir = f"/tmp/{video_name}"
            os.makedirs(temp_dir, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Find video file and segmentation folder
                video_file = None
                seg_folder = None

                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    if item.endswith('.avi'):
                        video_file = item_path
                    elif item == 'segmentation' and os.path.isdir(item_path):
                        seg_folder = item_path

                if video_file and seg_folder:
                    extract_frames_from_video(video_file, seg_folder, images_dir, gt_dir, video_name)
                else:
                    print(f"Warning: Could not find video or segmentation folder in {zip_name}")

            except Exception as e:
                print(f"Error processing {zip_name}: {e}")
            finally:
                # Clean up temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

    print(f"\nDataset processing complete!")
    print(f"Output saved to: {train_output_path}")

    # Print statistics
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(train_output_path, split, 'images')
        if os.path.exists(images_dir):
            num_images = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
            print(f"{split.capitalize()}: {num_images} images")

def calculate_dataset_statistics(dataset_path):
    """
    Calculate mean and std for the dataset (needed for training)
    """
    print("Calculating dataset statistics...")

    train_images_dir = os.path.join(dataset_path, "sarrarp50", "train", "images")

    if not os.path.exists(train_images_dir):
        print("Train images directory not found. Run preprocessing first.")
        return

    image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.png')]

    if len(image_files) == 0:
        print("No images found in train directory.")
        return

    # Sample subset for statistics (to avoid memory issues)
    sample_size = min(100, len(image_files))
    sample_files = random.sample(image_files, sample_size)

    mean_rgb = np.array([0.0, 0.0, 0.0])
    std_rgb = np.array([0.0, 0.0, 0.0])
    pixel_count = 0

    print(f"Calculating statistics from {sample_size} sample images...")

    # Calculate mean
    for img_file in tqdm(sample_files, desc="Calculating mean"):
        img_path = os.path.join(train_images_dir, img_file)
        img = np.array(Image.open(img_path)) / 255.0  # Normalize to [0,1]

        mean_rgb += np.mean(img.reshape(-1, 3), axis=0)
        pixel_count += img.shape[0] * img.shape[1]

    mean_rgb /= len(sample_files)

    # Calculate std
    for img_file in tqdm(sample_files, desc="Calculating std"):
        img_path = os.path.join(train_images_dir, img_file)
        img = np.array(Image.open(img_path)) / 255.0

        std_rgb += np.mean((img.reshape(-1, 3) - mean_rgb) ** 2, axis=0)

    std_rgb = np.sqrt(std_rgb / len(sample_files))

    print(f"\nDataset Statistics (from {sample_size} samples):")
    print(f"Mean [R, G, B]: [{mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}]")
    print(f"Std [R, G, B]: [{std_rgb[0]:.3f}, {std_rgb[1]:.3f}, {std_rgb[2]:.3f}]")

    return mean_rgb.tolist(), std_rgb.tolist()

if __name__ == "__main__":
    # Set paths
    drive_dataset_path = "/content/drive/MyDrive/Seg_Dataset"  # Update this path
    output_path = "/content/ARCSeg/src/data/datasets"

    # Process dataset
    process_dataset(drive_dataset_path, output_path)

    # Calculate statistics
    calculate_dataset_statistics(output_path)
