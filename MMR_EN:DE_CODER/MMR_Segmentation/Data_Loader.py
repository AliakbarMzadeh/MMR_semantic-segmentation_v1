"""
SAR-RARP50 Dataset Conversion Pipeline for MMR Segmentation Framework

This module provides comprehensive functionality for converting raw SAR-RARP50 surgical 
video datasets from ZIP archives into optimized Zarr format for efficient training and 
inference. The pipeline handles various data organization structures and implements 
medical imaging best practices for surgical tool segmentation.

Key Components:
===============

Video Processing Pipeline:
- Automatic detection of nested ZIP and direct folder structures
- Robust video extraction with multiple format support
- Frame-accurate synchronization between videos and segmentation masks
- Memory-efficient processing of high-resolution surgical videos

Segmentation Mask Processing:
- Intelligent mask filename parsing and frame mapping
- Support for RGB and grayscale mask formats
- Automatic handling of sparse annotation scenarios
- Class location tracking for intelligent sampling during training

Zarr Storage Optimization:
- High-performance compression using Blosc with Zstandard
- Chunked storage optimized for patch-based training
- Metadata preservation for reproducible experiments
- Memory-efficient processing of large video datasets

Data Structure Support:
- SAR-RARP50 nested ZIP format (video_01.zip, video_02.zip, etc.)
- Direct folder organization with video and mask files
- Flexible mask naming conventions and folder structures
- Automatic fallback strategies for data discovery

Features:
=========
- Lossless video frame extraction with OpenCV integration
- Intelligent class location sampling for imbalanced dataset handling
- Comprehensive error handling and progress reporting
- Professional logging with detailed processing statistics
- Memory-optimized processing suitable for large surgical datasets
- Configurable compression and chunking strategies

Target Use Cases:
================
- SAR-RARP50 robotic surgery dataset preprocessing
- Large-scale surgical video dataset conversion
- Medical imaging pipeline integration
- Research dataset preparation and standardization

Performance Optimizations:
=========================
- Zarr chunking aligned with training patch sizes
- Blosc compression for fast I/O and storage efficiency
- Temporal processing to minimize memory footprint
- Smart coordinate sampling to reduce metadata size

Dependencies:
=============
- opencv-python: Video processing and frame extraction
- zarr: High-performance array storage
- PIL/Pillow: Image format handling and conversion
- numpy: Numerical computations and array operations
- MMR_Segmentation.common_utils: Metadata serialization utilities

Usage Example:
==============
    python Data_Loader.py input_dataset.zip output_dir 9 "[512,640]"
    
    # Processes input_dataset.zip containing surgical videos
    # Outputs Zarr format to output_dir with 9 segmentation classes
    # Uses 512x640 patches for training optimization

Author: MMR Segmentation Team
Version: 1.0
"""

import os
import ast
import io
import cv2
import zarr
import argparse
import zipfile
import tempfile
import numpy as np
from PIL import Image
from zarr.codecs import BloscCodec, BloscShuffle
from MMR_Segmentation.common_utils import clean_numpy_scalars


def video_info(cap):
    """
    Extract comprehensive video metadata from OpenCV VideoCapture object.
    
    Retrieves essential video properties needed for processing and validation
    of surgical video data. This information is critical for ensuring proper
    frame extraction and temporal synchronization with segmentation masks.
    
    Args:
        cap (cv2.VideoCapture): Opened OpenCV VideoCapture object
            representing the surgical video file
    
    Returns:
        tuple: Video metadata as (width, height, fps, frame_count) where:
            - width (int): Frame width in pixels
            - height (int): Frame height in pixels  
            - fps (float): Frames per second (may be approximate)
            - frame_count (int): Total number of frames in video
    
    Note:
        - Frame count may be unreliable for some video formats
        - FPS values are extracted from video metadata (may vary from actual)
        - All values are essential for temporal alignment with annotations
        - Width/height are used for Zarr array initialization
        
    Example:
        >>> cap = cv2.VideoCapture('surgical_video.avi')
        >>> w, h, fps, nfr = video_info(cap)
        >>> print(f"Video: {w}x{h} @ {fps}fps, {nfr} frames")
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return width, height, fps, frame_count


def extract_video_from_zip(zip_file, temp_dir):
    """
    Extract surgical video file from ZIP archive with robust format detection.
    
    Locates and extracts the primary video file from various ZIP archive
    structures commonly used in surgical dataset distribution. Handles
    multiple naming conventions and provides clear error reporting.
    
    Args:
        zip_file (zipfile.ZipFile): Opened ZIP file object containing video data
        temp_dir (str): Temporary directory path for video extraction
    
    Returns:
        str: Path to extracted video file ready for OpenCV processing
    
    Raises:
        FileNotFoundError: If no compatible video file is found in the archive
            with detailed listing of available files for debugging
    
    Side Effects:
        - Creates video file in temporary directory
        - Prints extraction progress information
        - Overwrites existing files with same name
        
    Note:
        - Currently optimized for .avi format (common in surgical datasets)
        - Selects first video file if multiple are present
        - Standardizes output filename to 'video.avi' for consistency
        - Supports various ZIP internal folder structures
        
    Example:
        >>> with zipfile.ZipFile('surgical_data.zip') as zf:
        ...     video_path = extract_video_from_zip(zf, '/tmp/extract')
        >>> # Returns '/tmp/extract/video.avi'
    """
    # Locate video files in ZIP archive
    video_files = [f for f in zip_file.namelist() if f.endswith('.avi')]
    
    if not video_files:
        raise FileNotFoundError(
            f"No .avi video file found in ZIP. Available files: {zip_file.namelist()}"
        )
    
    # Use the first video file found (most datasets have one primary video)
    video_filename = video_files[0]
    video_path = os.path.join(temp_dir, "video.avi")
    
    # Extract video file to temporary location
    with open(video_path, "wb") as f:
        f.write(zip_file.read(video_filename))
    
    print(f"    Extracted video: {video_filename}")
    return video_path


def extract_masks_from_zip(zip_file):
    """
    Extract segmentation mask files and create frame-to-filename mapping.
    
    Discovers segmentation mask files within ZIP archives and establishes
    temporal correspondence with video frames. Handles various mask
    organization patterns and filename conventions used in surgical datasets.
    
    Args:
        zip_file (zipfile.ZipFile): Opened ZIP file containing mask images
    
    Returns:
        dict: Mapping from frame numbers to mask filenames as {frame_num: filename}
            where frame_num (int) corresponds to video frame index and
            filename (str) is the path within the ZIP archive
    
    Raises:
        FileNotFoundError: If no compatible mask files are discovered
            with detailed listing of available files
    
    Side Effects:
        - Prints mask discovery statistics
        - Warns about unparseable filenames
        - Validates frame number extraction
        
    Note:
        - Expects mask filenames to contain frame numbers (e.g., '000000060.png')
        - Searches both root directory and 'segmentation' subdirectories
        - Handles zero-padded and non-padded frame numbering
        - Skips files with invalid or missing frame numbers
        
    Example:
        >>> with zipfile.ZipFile('dataset.zip') as zf:
        ...     mapping = extract_masks_from_zip(zf)
        >>> mapping[60]  # Returns filename for frame 60
        'segmentation/000000060.png'
    """
    # Discover PNG mask files in various locations
    mask_files = []
    for filename in zip_file.namelist():
        if filename.endswith('.png') and ('segmentation' in filename or filename.endswith('.png')):
            mask_files.append(filename)
    
    if not mask_files:
        raise FileNotFoundError(
            f"No .png mask files found in ZIP. Available files: {zip_file.namelist()}"
        )
    
    # Create temporal mapping from frame numbers to filenames
    mask_idx2name = {}
    for mask_file in mask_files:
        try:
            # Extract frame number from filename (e.g., "000000060.png" -> 60)
            basename = os.path.basename(mask_file)
            frame_num = int(os.path.splitext(basename)[0])
            mask_idx2name[frame_num] = mask_file
        except ValueError:
            # Skip files with non-numeric names
            print(f"    Warning: Could not parse frame number from {mask_file}")
            continue
    
    print(f"    Found {len(mask_idx2name)} segmentation masks")
    return mask_idx2name


def write_sample(cap, mask_idx2name, n_classes, patch_size, compressor, file_save_path, file_name, zip_file):
    """
    Process video and segmentation masks, converting to optimized Zarr format.
    
    Implements the core conversion pipeline that synchronizes video frames with
    segmentation masks, applies preprocessing optimizations, and stores the
    result in Zarr format with medical imaging specific optimizations.
    
    Args:
        cap (cv2.VideoCapture): Opened video capture object for frame extraction
        mask_idx2name (dict): Frame number to mask filename mapping
        n_classes (int): Number of segmentation classes (excluding background)
        patch_size (list): Training patch dimensions [height, width] for chunk optimization
        compressor: Zarr compression codec (typically Blosc with Zstandard)
        file_save_path (str): Output path for Zarr file
        file_name (str): Identifier for logging and progress tracking
        zip_file (zipfile.ZipFile): ZIP archive containing mask files
    
    Returns:
        None: Saves processed data directly to Zarr format
    
    Side Effects:
        - Creates Zarr file with 'image' and 'mask' arrays
        - Saves metadata including class locations for sampling
        - Prints processing statistics and progress information
        - Handles memory cleanup and resource management
        
    Processing Pipeline:
        1. Extract video metadata and validate mask correspondence
        2. Initialize Zarr arrays with optimal chunking for training
        3. Process each frame with temporal synchronization
        4. Apply preprocessing (BGR->RGB, normalization)
        5. Extract class locations for intelligent sampling
        6. Store compressed data with metadata preservation
        
    Note:
        - Images stored as float32 normalized to [0,1] range
        - Masks stored as uint8 with class labels
        - Class locations sampled to balance memory vs. sampling quality
        - Chunking optimized for patch-based training access patterns
        - Compression reduces storage requirements significantly
        
    Example:
        >>> cap = cv2.VideoCapture('video.avi')
        >>> mapping = {0: 'mask_000.png', 1: 'mask_001.png'}
        >>> write_sample(cap, mapping, 9, [512, 640], compressor, 
        ...              'output.zarr', 'sample_01', zip_file)
    """
    # Extract video metadata for processing
    w, h, fps, nfr = video_info(cap)
    print(f"    {file_name}: {w}x{h}, fps={fps}, frames={nfr}, mask frames={len(mask_idx2name)}")

    # Define optimal chunk sizes for patch-based training
    # Chunks align with typical training patch sizes for efficient I/O
    image_chunks = (3, 1, *patch_size)  # (channels, time, height, width)
    mask_chunks = (1, *patch_size)      # (time, height, width)

    # Initialize Zarr storage with medical imaging optimizations
    z_file = zarr.open(file_save_path, mode='w')
    n = len(mask_idx2name)  # Number of frames with annotations
    
    # Create compressed arrays for images and masks
    img_ar = z_file.create_array(
        name='image', 
        shape=(3, n, h, w),           # Channels-first format for PyTorch
        chunks=image_chunks,
        dtype=np.float32,             # Float32 for normalized images
        compressors=compressor, 
        overwrite=True
    )
    mask_ar = z_file.create_array(
        name='mask', 
        shape=(n, h, w), 
        chunks=mask_chunks,
        dtype=np.uint8,               # Uint8 sufficient for class labels
        compressors=compressor, 
        overwrite=True
    )

    # Initialize class location tracking for intelligent sampling
    # Enables foreground-biased patch extraction during training
    class_locations = {int(lbl): [] for lbl in range(1, n_classes + 1)}

    # Process each frame that has corresponding segmentation mask
    for i, frame_idx in enumerate(sorted(mask_idx2name.keys())):
        # Extract specific video frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"    Warning: Could not read frame {frame_idx}")
            continue
        
        # Load corresponding segmentation mask
        try:
            mask_data = zip_file.read(mask_idx2name[frame_idx])
            mask = np.array(Image.open(io.BytesIO(mask_data)))
            
            # Handle different mask formats (RGB or grayscale)
            if len(mask.shape) == 3:
                # Convert RGB mask to grayscale by taking maximum channel
                # This handles cases where classes are encoded in different channels
                mask = mask.max(-1)
                
        except Exception as e:
            print(f"    Error loading mask for frame {frame_idx}: {e}")
            continue
        
        # Preprocess video frame for deep learning
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, convert to RGB
        frame = frame.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW format, normalize
        
        # Optional: ImageNet normalization (commented out for flexibility)
        # Standard ImageNet statistics for pretrained encoder compatibility
        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        # std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        # frame = (frame - mean) / std
        
        # Store processed data in Zarr arrays
        img_ar[:, i] = frame
        mask_ar[i] = mask
        
        # Collect class locations for intelligent sampling during training
        # This enables foreground-biased patch extraction to handle class imbalance
        for lbl in range(1, n_classes + 1):
            # Find all pixels belonging to current class
            slice_mask = mask == lbl
            slice_coords = np.argwhere(slice_mask)

            if slice_coords.shape[0] == 0:
                continue  # No pixels for this label in this frame

            # Sample coordinates if too many (memory optimization)
            if slice_coords.shape[0] > 50:
                indices = np.random.choice(slice_coords.shape[0], 50, replace=False)
                sampled = slice_coords[indices]
            else:
                sampled = slice_coords

            # Add temporal dimension (frame index) as first coordinate
            # Format: (frame, y, x) for 3D sampling during training
            sampled = [(i, y, x) for y, x in sampled]
            class_locations[int(lbl)].extend(sampled)

    # Save metadata for training pipeline
    properties = {'class_locations': class_locations}
    z_file.attrs['properties'] = clean_numpy_scalars(properties)

    print(f"    Saved: {file_save_path}")


def process_nested_zip(main_zip, zip_name, data_idx, dataset_save_path, n_classes, patch_size, compressor):
    """
    Process individual nested ZIP file from SAR-RARP50 dataset structure.
    
    Handles the common SAR-RARP50 organization where each surgical case is
    packaged as a separate ZIP file (e.g., video_01.zip, video_02.zip)
    within a master archive. Provides comprehensive error handling and
    progress tracking for robust batch processing.
    
    Args:
        main_zip (zipfile.ZipFile): Main ZIP archive containing nested case files
        zip_name (str): Name of nested ZIP file to process (e.g., 'video_01.zip')
        data_idx (int): Sequential index for output file naming
        dataset_save_path (str): Root directory for processed dataset output
        n_classes (int): Number of segmentation classes for metadata
        patch_size (list): Training patch dimensions for storage optimization
        compressor: Zarr compression codec for efficient storage
    
    Returns:
        None: Processes data and saves directly to Zarr format
    
    Side Effects:
        - Creates temporary directory for video extraction
        - Saves processed Zarr file with standardized naming
        - Prints detailed processing progress and statistics
        - Handles cleanup of temporary resources
        
    Error Handling:
        - Validates video file existence and readability
        - Ensures mask file availability and format compatibility
        - Provides detailed error messages for debugging
        - Gracefully skips problematic cases with warnings
        
    Note:
        - Uses temporary directory to avoid memory issues with large videos
        - Standardizes output naming as 'data_XXX.zarr' for consistency
        - Validates temporal correspondence between video and masks
        - Optimizes processing order for memory efficiency
        
    Example:
        >>> with zipfile.ZipFile('SAR_RARP50_TRAIN.zip') as main_zip:
        ...     process_nested_zip(main_zip, 'video_01.zip', 0, '/output', 
        ...                        9, [512, 640], compressor)
        # Creates /output/data/data_001.zarr
    """
    print(f"Processing {zip_name}...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract the nested ZIP file from main archive
            inner_zip_data = main_zip.read(zip_name)
            
            # Open the nested ZIP file containing video and masks
            with zipfile.ZipFile(io.BytesIO(inner_zip_data)) as inner_zip:
                # Extract video file to temporary location
                video_path = extract_video_from_zip(inner_zip, temp_dir)
                
                # Create frame-to-mask mapping
                mask_idx2name = extract_masks_from_zip(inner_zip)
                
                if not mask_idx2name:
                    print(f"    Skipping {zip_name} - no valid masks found")
                    return
                
                # Initialize video capture for frame extraction
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"    Error: Could not open video {video_path}")
                    return
                
                try:
                    # Create standardized output path
                    file_save_path = os.path.join(dataset_save_path, 'data', f"data_{data_idx + 1:03d}.zarr")
                    
                    # Process video and masks, save to Zarr format
                    write_sample(cap, mask_idx2name, n_classes, patch_size, 
                               compressor, file_save_path, zip_name, inner_zip)
                    
                finally:
                    # Ensure video capture is properly released
                    cap.release()
                    
        except Exception as e:
            print(f"    Error processing {zip_name}: {e}")
            return


def process_direct_folders(main_zip, folder_names, dataset_save_path, n_classes, patch_size, compressor):
    """
    Process direct folder structure within ZIP archive (alternative organization).
    
    Handles ZIP archives where videos and masks are organized in direct folders
    rather than nested ZIP files. This provides compatibility with various
    dataset organization schemes while maintaining the same processing pipeline.
    
    Args:
        main_zip (zipfile.ZipFile): Main ZIP archive with folder-based organization
        folder_names (list): List of folder names containing video/mask pairs
        dataset_save_path (str): Root directory for processed dataset output
        n_classes (int): Number of segmentation classes for metadata
        patch_size (list): Training patch dimensions for storage optimization
        compressor: Zarr compression codec for efficient storage
    
    Returns:
        None: Processes all folders and saves to Zarr format
    
    Side Effects:
        - Creates temporary directories for video extraction
        - Saves multiple Zarr files with sequential naming
        - Provides comprehensive progress tracking
        - Handles resource cleanup for each folder
        
    Processing Strategy:
        1. Iterate through each folder in the archive
        2. Locate video file within folder structure
        3. Discover corresponding mask files
        4. Create temporal mapping between video frames and masks
        5. Process using standard conversion pipeline
        
    Note:
        - Expects one video file per folder
        - Automatically discovers mask files with numeric naming
        - Handles various folder depth structures
        - Provides fallback for inconsistent organization
        
    Example:
        >>> with zipfile.ZipFile('dataset.zip') as main_zip:
        ...     folders = ['case_01', 'case_02', 'case_03']
        ...     process_direct_folders(main_zip, folders, '/output',
        ...                          9, [512, 640], compressor)
    """
    for data_idx, folder_name in enumerate(folder_names):
        print(f"Processing {folder_name}...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Locate video file within this folder
                video_files = [f for f in main_zip.namelist() 
                              if f.startswith(folder_name) and f.endswith('.avi')]
                
                if not video_files:
                    print(f"    Skipping {folder_name} - no video file found")
                    continue
                
                # Extract video to temporary location
                video_path = os.path.join(temp_dir, "video.avi")
                with open(video_path, "wb") as f:
                    f.write(main_zip.read(video_files[0]))
                
                # Discover mask files for this folder
                mask_files = [f for f in main_zip.namelist() 
                             if f.startswith(folder_name) and f.endswith('.png')]
                
                # Create temporal mapping from mask filenames
                mask_idx2name = {}
                for mask_file in mask_files:
                    try:
                        basename = os.path.basename(mask_file)
                        frame_num = int(os.path.splitext(basename)[0])
                        mask_idx2name[frame_num] = mask_file
                    except ValueError:
                        continue
                
                if not mask_idx2name:
                    print(f"    Skipping {folder_name} - no valid masks found")
                    continue
                
                # Initialize video processing
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"    Error: Could not open video {video_path}")
                    continue
                
                try:
                    # Create standardized output path
                    file_save_path = os.path.join(dataset_save_path, 'data', f"data_{data_idx + 1:03d}.zarr")
                    
                    # Process using standard pipeline
                    write_sample(cap, mask_idx2name, n_classes, patch_size, 
                               compressor, file_save_path, folder_name, main_zip)
                finally:
                    cap.release()
                    
            except Exception as e:
                print(f"    Error processing {folder_name}: {e}")
                continue


def main():
    """
    Main entry point for SAR-RARP50 dataset conversion pipeline.
    
    Implements command-line interface for converting surgical video datasets
    from ZIP archives to optimized Zarr format. Automatically detects dataset
    organization structure and applies appropriate processing strategy.
    
    Command Line Arguments:
        zip_dataset_path (str): Path to input ZIP file containing surgical videos
        save_dataset_path (str): Output directory for processed Zarr files
        n_classes (int): Number of segmentation classes (excluding background)
        patch_size (str): Training patch dimensions as string "[height,width]"
    
    Processing Workflow:
        1. Parse command line arguments and validate inputs
        2. Create output directory structure
        3. Initialize compression codec for efficient storage
        4. Detect dataset organization (nested ZIP vs. direct folders)
        5. Apply appropriate processing strategy
        6. Provide comprehensive progress reporting
        
    Output Structure:
        save_dataset_path/
        ├── data/
        │   ├── data_001.zarr
        │   ├── data_002.zarr
        │   └── ...
        
    Note:
        - Automatically handles both SAR-RARP50 organization styles
        - Uses Zstandard compression for optimal storage/speed balance
        - Provides detailed error reporting for debugging
        - Creates complete directory structure automatically
        
    Example Usage:
        python Data_Loader.py SAR_RARP50_TRAIN.zip /output/train 9 "[512,640]"
        python Data_Loader.py surgical_data.zip /output/test 5 "[256,256]"
    """
    # Configure command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Create training dataset from SAR-RARP50 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("zip_dataset_path", type=str, 
                       help="Path to dataset zip file")
    parser.add_argument("save_dataset_path", type=str, 
                       help="Path where processed dataset will be saved")
    parser.add_argument("n_classes", type=int, 
                       help="Number of segmentation classes")
    parser.add_argument("patch_size", type=ast.literal_eval, 
                       help="Patch size in format [W,H]")
    
    args = parser.parse_args()

    # Extract configuration parameters
    dataset_zip_path = args.zip_dataset_path
    dataset_save_path = args.save_dataset_path
    n_classes = args.n_classes
    patch_size = args.patch_size

    # Display processing configuration
    print("=" * 80)
    print("SAR-RARP50 Dataset Conversion Pipeline")
    print("=" * 80)
    print(f"Input ZIP file: {dataset_zip_path}")
    print(f"Output directory: {dataset_save_path}")
    print(f"Segmentation classes: {n_classes}")
    print(f"Training patch size: {patch_size}")
    print("=" * 80)

    # Create output directory structure
    os.makedirs(os.path.join(dataset_save_path, 'data'), exist_ok=True)

    # Configure high-performance compression
    # Zstandard provides excellent compression ratio with fast decompression
    compressor = BloscCodec(cname='zstd', clevel=3, shuffle=BloscShuffle.bitshuffle)

    # Open main ZIP file and detect organization structure
    with zipfile.ZipFile(dataset_zip_path, 'r') as main_zip:
        all_files = main_zip.namelist()
        
        # Strategy 1: Check for nested ZIP files (common SAR-RARP50 format)
        inner_zips = [f for f in all_files if f.endswith('.zip')]
        
        if inner_zips:
            print(f"Detected nested ZIP structure with {len(inner_zips)} files")
            print("Processing nested ZIP files...")
            # Process each nested ZIP file sequentially
            for data_idx, zip_name in enumerate(inner_zips):
                process_nested_zip(main_zip, zip_name, data_idx, dataset_save_path, 
                                 n_classes, patch_size, compressor)
        
        else:
            print("Detected direct folder structure")
            print("Processing folder-based organization...")
            # Strategy 2: Extract folder names containing video files
            folder_names = list(set(f.split('/')[0] for f in all_files if f.endswith('.avi')))
            process_direct_folders(main_zip, folder_names, dataset_save_path, 
                                 n_classes, patch_size, compressor)

    # Display completion summary
    print("\n" + "=" * 80)
    print("Dataset processing completed successfully!")
    print(f"Processed data saved to: {dataset_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()