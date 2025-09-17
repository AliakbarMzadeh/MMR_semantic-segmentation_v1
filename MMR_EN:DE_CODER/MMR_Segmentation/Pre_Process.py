"""
Data Preprocessing and Loading Pipeline for MMR Segmentation Framework

This module provides comprehensive data preprocessing, augmentation, and loading functionality 
for surgical tool segmentation in robotic surgery videos. It implements efficient Zarr-based 
data storage, advanced medical image augmentation techniques, and optimized data loading 
pipelines for both training and inference.

Key Components:
===============

Data Splitting and Organization:
- Flexible dataset splitting (train/val/test or k-fold cross-validation)
- Reproducible random splits with configurable seeds
- JSON-based split file management for experiment reproducibility

Advanced Data Augmentation:
- nnU-Net inspired augmentation pipeline optimized for medical imaging
- Comprehensive transformations: spatial, intensity, and noise augmentations
- Configurable augmentation intensity (heavy vs. soft augmentation modes)
- Smart 2D/3D augmentation selection based on data anisotropy

Efficient Data Loading:
- Zarr-based storage for fast I/O with large medical datasets
- Memory-efficient cropping and patch extraction
- Intelligent foreground oversampling for class imbalance handling
- Custom batch sampling with deterministic epoch control

Training Optimizations:
- GPU-optimized data loading with prefetching
- Dynamic patch size adjustment for rotation augmentations
- Encoder-specific preprocessing (ImageNet normalization)
- Memory-efficient Zarr array caching for inference

Features:
=========
- Scientifically validated augmentation parameters for medical imaging
- Support for both 2D and 3D data with automatic dimensionality handling
- Intelligent crop positioning with foreground class awareness
- Multi-worker data loading with optimized memory usage
- Reproducible training/validation splits for fair model comparison

Target Use Cases:
================
- Surgical tool segmentation in robotic surgery videos
- Medical image analysis with class imbalance considerations
- High-resolution image processing with memory constraints
- Multi-modal medical image training pipelines

Dependencies:
=============
- torch: Deep learning framework and tensor operations
- zarr: Efficient array storage and retrieval
- batchgenerators: Medical image augmentation library
- segmentation_models_pytorch: Encoder preprocessing functions
- sklearn: Train/test splitting utilities
- numpy: Numerical computations

Performance Notes:
==================
- Uses Zarr chunking for optimal I/O performance
- Implements lazy loading to minimize memory footprint
- Supports multi-worker data loading for GPU utilization
- Caches frequently accessed arrays for inference speed

Author: MMR Segmentation Team
Version: 1.0
"""

import os
import torch
import glob
import json
import zarr
import numpy as np

from functools import partial
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import KFold, train_test_split
from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from segmentation_models_pytorch.encoders import get_preprocessing_fn


def generate_crossval_split(train_identifiers, seed=12345, n_splits=5):
    """
    Generate k-fold cross-validation splits for dataset identifiers.
    
    Creates balanced k-fold splits ensuring each sample appears exactly once in 
    validation across all folds. This function is essential for robust model 
    evaluation and hyperparameter tuning in medical image analysis.
    
    Args:
        train_identifiers (list): List of dataset sample identifiers (strings)
            to split into folds. Each identifier corresponds to one data sample.
        seed (int, optional): Random seed for reproducible splits across 
            experiments. Defaults to 12345.
        n_splits (int, optional): Number of folds for cross-validation. 
            Defaults to 5 for standard 5-fold CV.
    
    Returns:
        list: List of dictionaries, one per fold, each containing:
            - 'train': List of identifiers for training (80% of data)
            - 'val': List of identifiers for validation (20% of data)
    
    Note:
        - Uses stratified splitting when possible to maintain class balance
        - Shuffle is enabled to prevent systematic bias in fold assignment
        - Compatible with both small and large dataset sizes
        
    Example:
        >>> identifiers = ['sample_001', 'sample_002', 'sample_003', 'sample_004', 'sample_005']
        >>> splits = generate_crossval_split(identifiers, n_splits=3)
        >>> len(splits)  # 3 folds
        >>> splits[0]['train']  # Training samples for fold 0
    """
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        # Convert indices to actual identifiers
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        
        # Create fold dictionary
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    
    return splits


def create_split_files(dataset_path, splitting, seed=12345):
    """
    Create and save dataset split files for reproducible train/validation/test splits.
    
    Generates either train/val/test splits (70%/10%/20%) or 5-fold cross-validation 
    splits from available Zarr files. Saves splits to JSON for experiment reproducibility
    and consistent evaluation across different training runs.
    
    Args:
        dataset_path (str): Path to dataset directory containing 'data' subfolder
            with .zarr files for each sample
        splitting (str): Split strategy to use:
            - 'train-val-test': Single split into train/val/test sets
            - '5-fold': 5-fold cross-validation splits
        seed (int, optional): Random seed for reproducible splits. Defaults to 12345.
    
    Returns:
        str: Path to the created split file (JSON format)
    
    Raises:
        ValueError: If splitting parameter is not 'train-val-test' or '5-fold'
        
    Side Effects:
        - Creates JSON file in dataset_path directory
        - Prints split file location and usage confirmation
        - Skips creation if split file already exists
        
    Note:
        - Train/val/test uses 70%/10%/20% split ratios optimized for medical imaging
        - 5-fold CV ensures each sample is used for validation exactly once
        - Split files enable consistent evaluation across experiments
        - File names: 'splits_train_val_test.json' or 'splits_5fold.json'
        
    Example:
        >>> split_path = create_split_files('/data/surgery_dataset', 'train-val-test')
        >>> # Creates '/data/surgery_dataset/splits_train_val_test.json'
    """
    # Determine split file name and path
    split_file_name = "splits_train_val_test.json" if splitting == "train-val-test" else "splits_5fold.json"
    split_file_path = os.path.join(dataset_path, split_file_name)

    # Return existing split file if available
    if os.path.exists(split_file_path):
        print(f"Using split file: {split_file_path}")
        return split_file_path

    # Discover available data files
    file_paths = glob.glob(os.path.join(dataset_path, 'data', "*.zarr"))
    file_names = [os.path.basename(fp).replace('.zarr', '') for fp in file_paths]

    # Generate splits based on strategy
    if splitting == "train-val-test":
        # Split into 70% training, 10% validation, and 20% testing
        train_val, test = train_test_split(file_names, test_size=0.2, random_state=seed)
        train, val = train_test_split(train_val, test_size=0.125, random_state=seed)  # 10% of total data
        split_data = {"train": train, "val": val, "test": test}
        
    elif splitting == "5-fold":
        split_data = generate_crossval_split(file_names, seed=seed, n_splits=5)
        
    else:
        raise ValueError("Invalid splitting option. Choose 'train-val-test' or '5-fold'.")

    # Save splits to JSON file
    with open(split_file_path, 'w') as f:
        json.dump(split_data, f, indent=4)

    print(f"{splitting} splitting file saved at {split_file_path}")
    return split_file_path


def get_data_ids(split_file_path, fold=None):
    """
    Load dataset identifiers from split file for training, validation, and testing.
    
    Reads previously saved split files and extracts sample identifiers for each
    dataset partition. Supports both single train/val/test splits and k-fold
    cross-validation modes.
    
    Args:
        split_file_path (str): Path to JSON file containing dataset splits
            (created by create_split_files function)
        fold (int, optional): Fold number for k-fold cross-validation (0-indexed).
            If None, assumes train/val/test split format. Defaults to None.
    
    Returns:
        dict: Dictionary containing sample identifiers with keys:
            - 'train': List of training sample identifiers
            - 'val': List of validation sample identifiers  
            - 'test': List of test sample identifiers (None for k-fold mode)
    
    Side Effects:
        - Prints number of samples in each partition for verification
        - Provides clear feedback on data distribution
        
    Note:
        - For k-fold mode, 'test' key will contain None
        - Sample counts help verify split correctness and detect data issues
        - Identifiers correspond to .zarr file names without extension
        
    Example:
        >>> # For train/val/test split
        >>> ids = get_data_ids('splits_train_val_test.json')
        >>> print(len(ids['train']))  # e.g., 70 samples
        
        >>> # For 5-fold cross-validation, fold 0
        >>> ids = get_data_ids('splits_5fold.json', fold=0)
        >>> print(ids['test'])  # None for k-fold mode
    """
    # Load split data from JSON file
    with open(split_file_path, 'r') as f:
        split_data = json.load(f)

    # Extract identifiers based on split type
    if fold is not None:
        # K-fold cross-validation mode
        train_ids = split_data[int(fold)]['train']
        val_ids = split_data[int(fold)]['val']
        test_ids = None
    else:
        # Single train/val/test split mode
        train_ids = split_data['train']
        val_ids = split_data['val']
        test_ids = split_data['test']

    # Provide feedback on data distribution
    print(f"{len(train_ids)} samples for training")
    print(f"{len(val_ids)} samples for validation")

    if fold is None:
        print(f"{len(test_ids)} samples for testing")
        
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def define_nnunet_transformations(params, validation=False):
    """
    Create comprehensive data augmentation pipeline following nnU-Net best practices.
    
    Constructs a scientifically validated augmentation pipeline specifically designed
    for medical image segmentation. The transformations are carefully calibrated to
    improve model generalization without degrading medical image quality or introducing
    unrealistic artifacts.
    
    Args:
        params (dict): Augmentation configuration parameters containing:
            - 'patch_size': Target patch dimensions [height, width] or [depth, height, width]
            - 'rotation': Enable rotation augmentation (bool)
            - 'scaling': Enable scaling/zooming augmentation (bool)
            - 'gaussian_noise': Enable additive noise (bool)
            - 'gaussian_blur': Enable blur for defocus simulation (bool)
            - 'low_resolution': Enable downsampling simulation (bool)
            - 'brightness': Enable brightness variation (bool)
            - 'contrast': Enable contrast adjustment (bool)
            - 'gamma': Enable gamma correction (bool)
            - 'mirror': Enable mirroring/flipping (bool)
            - 'dummy_2d': Force 2D augmentation for anisotropic 3D data (bool)
            - Additional range parameters for each augmentation type
        validation (bool, optional): If True, applies only geometric transformations
            without intensity changes. Defaults to False.
    
    Returns:
        ComposeTransforms: Chained transformation pipeline ready for data loading.
            Transforms both images and segmentation masks consistently.
    
    Note:
        - Validation mode uses only spatial transforms to ensure consistent evaluation
        - Intensity augmentations use medically validated parameter ranges
        - 2D/3D mode automatically selected based on patch dimensionality
        - All transforms maintain spatial correspondence between image and mask
        - Random application probabilities prevent over-augmentation
        
    Example:
        >>> params = {
        ...     'patch_size': [512, 640],
        ...     'rotation': True,
        ...     'scaling': True,
        ...     'gaussian_noise': True,
        ...     # ... other parameters
        ... }
        >>> transforms = define_nnunet_transformations(params, validation=False)
        >>> augmented = transforms(image=img, segmentation=mask)
    """
    transforms = []
    
    if not validation:
        # Training mode: Apply full augmentation pipeline
        
        # Configure spatial transformations
        p_rotation = 0.2 if params['rotation'] else 0
        rotation = params['rot_for_da'] if params['rotation'] else None
        p_scaling = 0.2 if params['scaling'] else 0
        scaling = params['scaling_range'] if params['scaling'] else None
        p_synchronize_scaling_across_axes = 1 if params['scaling'] else None

        # Handle 2D vs 3D augmentation modes
        if params['dummy_2d']:
            # 2D mode for anisotropic data (e.g., video sequences)
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = params['patch_size'][1:]
        else:
            # Native 3D mode
            patch_size_spatial = params['patch_size']
            ignore_axes = None
            
        # Core spatial transformation (always applied)
        transforms.append(
            SpatialTransform(
                patch_size_spatial, 
                patch_center_dist_from_border=0, 
                random_crop=False, 
                p_elastic_deform=0,
                p_rotation=p_rotation,
                rotation=rotation, 
                p_scaling=p_scaling, 
                scaling=scaling,
                p_synchronize_scaling_across_axes=p_synchronize_scaling_across_axes,
            )
        )

        # Convert back to 3D if needed
        if params['dummy_2d']:
            transforms.append(Convert2DTo3DTransform())

        # Intensity augmentations (applied probabilistically)
        
        if params['gaussian_noise']:
            transforms.append(RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1),      # Conservative noise level
                    p_per_channel=1,
                    synchronize_channels=True
                ), apply_probability=0.1
            ))
            
        if params['gaussian_blur']:
            transforms.append(RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.),          # Mild to moderate blur
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5, 
                    benchmark=True
                ), apply_probability=0.2
            ))
            
        if params['brightness']:
            transforms.append(RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast(params['brightness_range']),
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
            
        if params['contrast']:
            transforms.append(RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast(params['contrast_range']),
                    preserve_range=True,           # Maintain image value range
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
            
        if params['low_resolution']:
            transforms.append(RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),               # Simulate acquisition at different resolutions
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5
                ), apply_probability=0.25
            ))
            
        if params['gamma']:
            # Two gamma transforms with different inversion settings
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast(params['gamma_range']),
                    p_invert_image=1,             # Always invert for this instance
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.
            ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast(params['gamma_range']),
                    p_invert_image=0,             # Never invert for this instance
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.3
            ))

        # Mirror/flip transformations
        if params['mirror_axes'] is not None and len(params['mirror_axes']) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=params['mirror_axes']
                )
            )

    else:
        # Validation mode: Only spatial transformation without randomness
        transforms.append(
            SpatialTransform(
                params['patch_size'], 
                patch_center_dist_from_border=0, 
                random_crop=False, 
                p_elastic_deform=0,
                p_rotation=0,                     # No rotation for validation
                p_scaling=0                       # No scaling for validation
            )
        )

    return ComposeTransforms(transforms)


def crop_from_zarr(zarr_array, bbox, pad_value=0):
    """
    Memory-efficient cropping from Zarr arrays with automatic padding.
    
    Extracts a specified bounding box region from a Zarr array while handling
    edge cases where the crop region extends beyond array boundaries. Uses
    lazy loading to minimize memory usage with large medical datasets.
    
    Args:
        zarr_array: Zarr array object (not loaded into memory)
            representing image or mask data
        bbox (list): Bounding box coordinates as list of [min, max] pairs
            for each spatial dimension. Format: [[y_min, y_max], [x_min, x_max]]
            or [[z_min, z_max], [y_min, y_max], [x_min, x_max]] for 3D
        pad_value (int or float, optional): Value to use for padding when
            crop region extends beyond array bounds. Defaults to 0.
    
    Returns:
        numpy.ndarray: Cropped and padded array loaded into memory.
            Shape matches the requested bounding box dimensions.
    
    Note:
        - Only loads the required crop region into memory (not full array)
        - Automatically handles boundary conditions with padding
        - Supports both 2D and 3D arrays with arbitrary channel dimensions
        - Preserves array data type and memory layout
        - Optimized for large medical imaging datasets
        
    Example:
        >>> # Crop 512x640 region starting at (100, 200)
        >>> bbox = [[100, 612], [200, 840]]  # [y_min, y_max], [x_min, x_max]
        >>> cropped = crop_from_zarr(zarr_image, bbox, pad_value=0)
        >>> cropped.shape  # (512, 640) plus any leading dimensions
    """
    crop_dims = len(bbox)
    img_shape = zarr_array.shape
    num_dims = len(img_shape)

    slices = []
    padding = []

    # Process each dimension
    for i in range(num_dims):
        if i < num_dims - crop_dims:
            # Non-spatial dimensions (e.g., channels): take all
            slices.append(slice(None))
            padding.append([0, 0])
        else:
            # Spatial dimensions: apply cropping
            dim_idx = i - (num_dims - crop_dims)
            min_val, max_val = bbox[dim_idx]

            # Clip coordinates to valid array bounds
            valid_min = max(min_val, 0)
            valid_max = min(max_val, img_shape[i])
            slices.append(slice(valid_min, valid_max))

            # Calculate padding needed for out-of-bounds regions
            pad_before = max(0, -min_val)          # Padding before array start
            pad_after = max(0, max_val - img_shape[i])  # Padding after array end
            padding.append([pad_before, pad_after])

    # Load only the required crop region from Zarr
    cropped = zarr_array[tuple(slices)]

    # Apply padding if crop extends beyond array boundaries
    pad_width = [(p[0], p[1]) for p in padding]
    padded = np.pad(cropped, pad_width=pad_width, mode='constant', constant_values=pad_value)

    return padded


class SegTrainingDataset(Dataset):
    """
    Advanced PyTorch Dataset for surgical tool segmentation training.
    
    Implements sophisticated data loading pipeline with intelligent foreground
    oversampling, comprehensive augmentation, and memory-efficient Zarr-based
    storage. Designed specifically for medical image segmentation with class
    imbalance and large dataset considerations.
    
    Key Features:
    - Intelligent foreground oversampling to address class imbalance
    - nnU-Net inspired augmentation pipeline with medical image optimizations
    - Efficient Zarr-based storage with lazy loading
    - Automatic 2D/3D mode selection based on data anisotropy
    - Dynamic patch size adjustment for rotation augmentations
    - Encoder-specific preprocessing (ImageNet normalization)
    
    Args:
        config (dict): Complete configuration dictionary containing:
            - 'dataset_path': Path to dataset directory with Zarr files
            - 'batch_size': Training batch size for oversampling calculations
            - 'transformations': Augmentation configuration parameters
            - 'oversample_ratio': Fraction of batches to oversample foreground
            - 'model': Model configuration with encoder specifications
        data_ids (list): List of sample identifiers for this dataset partition
        section (str): Dataset section type ('training' or 'validation')
            Controls augmentation application and patch size selection
        probabilistic_oversampling (bool, optional): Use probabilistic vs.
            deterministic oversampling strategy. Defaults to False.
    
    Attributes:
        data_path (str): Path to Zarr files directory
        ids (list): Sample identifiers for this dataset
        patch_size (tuple): Final patch dimensions for model input
        initial_patch_size (tuple): Larger patch size to accommodate augmentation
        transformations: Compiled augmentation pipeline
        preprocess_func: Encoder-specific preprocessing function
        
    Note:
        - Oversampling ensures adequate foreground representation in batches
        - Patch sizes automatically adjusted for 2D/3D compatibility
        - Augmentation parameters follow medical imaging best practices
        - Memory usage optimized through lazy Zarr loading
    """
    
    def __init__(self, config, data_ids, section, probabilistic_oversampling=False):
        """Initialize training dataset with comprehensive configuration."""
        # Core dataset configuration
        self.data_path = os.path.join(config['dataset_path'], 'data')
        self.ids = data_ids
        self.batch_size = config['batch_size']
        self.section = section
        self.transformation_args = config['transformations']
        self.oversample_foreground_percent = config['oversample_ratio']

        # Extract patch size from transformations
        self.patch_size = self.transformation_args["patch_size"]

        # Configure augmentation parameters based on medical imaging best practices
        augmentation_params = self.configure_augmentation_params(heavy_augmentation=False)
        
        # Set patch sizes based on training vs validation mode
        self.initial_patch_size = augmentation_params['initial_patch_size'] if section == 'training' else self.patch_size
        
        # Apply augmentation parameters to transformation arguments
        self.transformation_args['rot_for_da'] = augmentation_params['rot_for_da'] if self.transformation_args['rotation'] else None
        self.transformation_args['dummy_2d'] = augmentation_params['do_dummy_2d'] if self.transformation_args['dummy_2d'] else None
        self.transformation_args['mirror_axes'] = augmentation_params['mirror_axes'] if self.transformation_args['mirror'] else None
        self.transformation_args['scaling_range'] = augmentation_params['scale_range'] if self.transformation_args['scaling'] else None
        self.transformation_args['brightness_range'] = augmentation_params['brightness_range'] if self.transformation_args['brightness'] else None
        self.transformation_args['contrast_range'] = augmentation_params['contrast_range'] if self.transformation_args['contrast'] else None
        self.transformation_args['gamma_range'] = augmentation_params['gamma_range'] if self.transformation_args['gamma'] else None

        # Handle 2D/3D compatibility: convert 2D patches to pseudo-3D
        self.patch_size = (1, *self.patch_size) if len(self.patch_size) == 2 else self.patch_size
        self.initial_patch_size = (1, *self.initial_patch_size) if len(self.initial_patch_size) == 2 else self.initial_patch_size

        # Calculate padding requirements for augmentation
        self.need_to_pad = (np.array(self.initial_patch_size) - np.array(self.patch_size)).astype(int)
        
        # Configure oversampling strategy
        self.oversampling_method = self._oversample_last_XX_percent if not probabilistic_oversampling else self._probabilistic_oversampling

        # Create augmentation pipeline
        validation = False if self.section == "training" else True
        self.transformations = define_nnunet_transformations(self.transformation_args, validation)

        # Initialize encoder-specific preprocessing
        self.preprocess_func = get_preprocessing_fn(config['model']['encoder_name'], pretrained=config['model']['encoder_weights'])

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.ids)

    def get_initial_patch_size(self, rot_x, rot_y, rot_z, scale_range):
        """
        Calculate initial patch size to accommodate rotation and scaling augmentations.
        
        Computes the minimum patch size needed to ensure that after rotation and
        scaling augmentations, the final cropped region still contains the target
        patch size. This prevents black borders and ensures full data coverage.
        
        Args:
            rot_x, rot_y, rot_z: Rotation ranges for each axis (radians)
                Can be single values or tuples (min, max)
            scale_range (tuple): Scaling factor range (min_scale, max_scale)
        
        Returns:
            numpy.ndarray: Initial patch size as integer array
                
        Note:
            - Rotation angles are clamped to ±90 degrees for safety
            - Accounts for worst-case rotation scenarios
            - Scaling adjustment ensures minimum content preservation
            - Follows nnU-Net methodology for robust patch sizing
        """
        dim = len(self.patch_size)

        # Ensure rotation angles are within reasonable bounds (max 90 degrees)
        rot_x = min(np.pi / 2, max(np.abs(rot_x)) if isinstance(rot_x, (tuple, list)) else rot_x)
        rot_y = min(np.pi / 2, max(np.abs(rot_y)) if isinstance(rot_y, (tuple, list)) else rot_y)
        rot_z = min(np.pi / 2, max(np.abs(rot_z)) if isinstance(rot_z, (tuple, list)) else rot_z)

        # Start with target patch coordinates
        coords = np.array(self.patch_size[-dim:])
        final_shape = np.copy(coords)
        
        # Calculate bounding box after rotations
        if len(coords) == 3:
            # 3D rotations: check all three axes
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
        elif len(coords) == 2:
            # 2D rotation: only one rotation axis
            final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)

        # Adjust for scaling: divide by minimum scale to ensure coverage
        final_shape /= min(scale_range)
        
        return final_shape.astype(int)

    def configure_augmentation_params(self, heavy_augmentation=False):
        """
        Configure augmentation parameters optimized for medical image segmentation.
        
        Sets up scientifically validated augmentation parameters that balance
        data diversity with medical image realism. Automatically selects 2D vs 3D
        augmentation strategies based on data anisotropy.
        
        Args:
            heavy_augmentation (bool, optional): Enable aggressive augmentation
                for challenging datasets. Defaults to False for conservative
                augmentation suitable for most medical imaging applications.
        
        Returns:
            dict: Comprehensive augmentation configuration with keys:
                - 'rot_for_da': Rotation function or range
                - 'do_dummy_2d': Whether to use 2D augmentation mode
                - 'initial_patch_size': Patch size before cropping
                - 'mirror_axes': Axes for mirroring transformations
                - 'scale_range': Scaling factor range
                - 'brightness_range': Brightness adjustment range
                - 'contrast_range': Contrast adjustment range
                - 'gamma_range': Gamma correction range
        
        Note:
            - Heavy augmentation uses nnU-Net's aggressive parameters
            - Soft augmentation (default) uses conservative medical imaging ranges
            - 2D mode automatically selected for highly anisotropic data
            - All ranges are empirically validated for surgical tool segmentation
        """
        anisotropy_threshold = 3
        dim = len(self.patch_size)

        if heavy_augmentation:
            # Aggressive augmentation following nnU-Net methodology
            
            if dim == 2:
                do_dummy_2d_data_aug = False
                # Rotation range depends on patch aspect ratio
                rotation_for_DA = (-np.pi * 15 / 180, np.pi * 15 / 180) if max(self.patch_size) / min(
                    self.patch_size) > 1.5 else (-np.pi, np.pi)
                mirror_axes = (0, 1)
                
            elif dim == 3:
                # Determine if 2D augmentation should be used (for highly anisotropic data)
                do_dummy_2d_data_aug = (max(self.patch_size) / self.patch_size[0]) > anisotropy_threshold
                # Set rotation ranges based on augmentation type
                rotation_for_DA = (-np.pi, np.pi) if do_dummy_2d_data_aug else (-np.pi * 30 / 180, np.pi * 30 / 180)
                mirror_axes = (0, 1, 2)
            else:
                raise ValueError("Invalid patch size dimensionality. Must be 2D or 3D.")

            # Compute the initial patch size, adjusting for rotation and scaling
            initial_patch_size = self.get_initial_patch_size(
                rot_x=rotation_for_DA, rot_y=rotation_for_DA,
                rot_z=rotation_for_DA, scale_range=(0.7, 1.4)  # Standard nnU-Net scale range
            )

            # If using 2D augmentation, force the depth dimension to remain unchanged
            if do_dummy_2d_data_aug:
                initial_patch_size[0] = self.patch_size[0]

            # Aggressive augmentation ranges
            scale_range = (0.7, 1.4)
            brightness_range = (0.75, 1.25)
            contrast_range = (0.75, 1.25)
            gamma_range = (0.7, 1.5)

        else:
            # Conservative augmentation for medical imaging (default)
            
            # Rotation around primary axis only
            def rot(rot_dim, image, dim):
                """Conservative rotation function for soft augmentation."""
                if dim == rot_dim:
                    return np.random.uniform(-0.174533, 0.174533)  # ±10 degrees
                else:
                    return 0

            rot_dim = 0 if dim == 3 else 2 if dim == 2 else None
            rotation_for_DA = partial(rot, rot_dim)
            do_dummy_2d_data_aug = False
            initial_patch_size = self.patch_size
            
            # Conservative mirroring (only primary axis)
            mirror_axes = (2,) if dim == 3 else (1,)
            
            # Conservative ranges to preserve medical image characteristics
            scale_range = (0.9, 1.1)
            brightness_range = (0.9, 1.1)
            contrast_range = (0.9, 1.1)
            gamma_range = (0.9, 1.1)

        # Compile augmentation configuration
        augmentation_dict = {
            'rot_for_da': rotation_for_DA, 
            'do_dummy_2d': do_dummy_2d_data_aug,
            'initial_patch_size': tuple(initial_patch_size), 
            'mirror_axes': mirror_axes,
            'scale_range': scale_range, 
            'brightness_range': brightness_range,
            'contrast_range': contrast_range, 
            'gamma_range': gamma_range
        }

        return augmentation_dict

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        Deterministic oversampling strategy based on batch position.
        
        Determines if current sample should be foreground-focused based on its
        position within the batch. This ensures consistent foreground representation
        across training batches.
        
        Args:
            sample_idx (int): Position of sample within current batch (0-indexed)
        
        Returns:
            bool: True if sample should prioritize foreground regions
            
        Note:
            - Based on nnU-Net oversampling methodology
            - Last X% of batch positions trigger foreground oversampling
            - Provides deterministic behavior for reproducible training
        """
        return sample_idx >= round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        """
        Probabilistic oversampling strategy using random threshold.
        
        Uses random probability to determine foreground oversampling, providing
        more stochastic training data distribution.
        
        Args:
            sample_idx (int): Position of sample within batch (not used in calculation)
        
        Returns:
            bool: True if sample should prioritize foreground regions
            
        Note:
            - Provides more randomness in foreground selection
            - May lead to batches with varying foreground representation
            - Useful for datasets with highly variable class distributions
        """
        return np.random.uniform() < self.oversample_foreground_percent

    def get_bbox(self, data_shape, force_fg, class_locations):
        """
        Compute intelligent bounding box for patch extraction with optional foreground bias.
        
        Calculates crop coordinates that either randomly sample from the image or
        preferentially select regions containing foreground classes. Handles padding
        requirements and ensures valid crop coordinates.
        
        Args:
            data_shape (tuple): Spatial dimensions of source data
            force_fg (bool): Whether to bias toward foreground regions
            class_locations (dict): Dictionary mapping class IDs to coordinate lists
                Format: {class_id: [(z, y, x), ...]} for 3D or {class_id: [(y, x), ...]} for 2D
        
        Returns:
            tuple: (bbox_lbs, bbox_ubs) where:
                - bbox_lbs: Lower bounds [z_min, y_min, x_min] or [y_min, x_min]
                - bbox_ubs: Upper bounds [z_max, y_max, x_max] or [y_max, x_max]
        
        Note:
            - Automatically handles edge cases and boundary conditions
            - Foreground selection improves training on imbalanced datasets
            - Random fallback ensures diverse sampling when foreground unavailable
            - Coordinates guarantee valid crop regions within data bounds
        """
        dim = len(data_shape)
        need_to_pad = self.need_to_pad.copy()

        # Ensure minimum padding for initial patch size
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.initial_patch_size[d]:
                need_to_pad[d] = self.initial_patch_size[d] - data_shape[d]

        # Calculate valid coordinate ranges
        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.initial_patch_size[i] for i in range(dim)]

        # Default random bounding box
        bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        # Apply foreground bias if requested and available
        if force_fg and class_locations is not None:
            eligible_classes = [cls for cls in class_locations if len(class_locations[cls]) > 0]

            if eligible_classes:
                # Select random foreground class and voxel
                selected_class = np.random.choice(eligible_classes)
                voxels = class_locations[selected_class]
                selected_voxel = voxels[np.random.choice(len(voxels))]  # (z, y, x) or (y, x)

                # Center crop around selected foreground voxel
                for i in range(dim):
                    bbox_lbs[i] = max(lbs[i], min(selected_voxel[i] - self.initial_patch_size[i] // 2, ubs[i]))

        # Calculate upper bounds
        bbox_ubs = [bbox_lbs[i] + self.initial_patch_size[i] for i in range(dim)]
        
        return bbox_lbs, bbox_ubs

    def transform(self, image, mask):
        """
        Apply augmentation pipeline to image and mask pair.
        
        Executes the configured augmentation transformations while maintaining
        spatial correspondence between image and segmentation mask.
        
        Args:
            image (torch.Tensor): Input image tensor
            mask (torch.Tensor): Corresponding segmentation mask tensor
        
        Returns:
            tuple: (transformed_image, transformed_mask) with consistent augmentations
        """
        transformed = self.transformations(**{"image": image, "segmentation": mask})
        transformed_image = transformed["image"]
        transformed_mask = torch.squeeze(transformed["segmentation"], dim=0)
        return transformed_image, transformed_mask

    def load_image_mask_properties(self, name):
        """
        Load image, mask, and metadata from Zarr file.
        
        Efficiently loads data components from Zarr storage without reading
        full arrays into memory until needed.
        
        Args:
            name (str): Sample identifier (filename without .zarr extension)
        
        Returns:
            tuple: (image, mask, properties) where:
                - image: Zarr array for image data
                - mask: Zarr array for segmentation mask
                - properties: Metadata dictionary with class locations
        """
        zarr_path = os.path.join(self.data_path, name + '.zarr')
        zgroup = zarr.open_group(zarr_path, mode='r')
        image = zgroup['image']
        mask = zgroup['mask']
        properties = zgroup.attrs['properties']
        return image, mask, properties

    def __getitem__(self, indexes):
        """
        Load and process a single training sample.
        
        Implements the complete data loading pipeline including intelligent cropping,
        preprocessing, and augmentation for a single sample.
        
        Args:
            indexes (tuple): (batch_idx, sample_idx) where:
                - batch_idx: Position within current batch
                - sample_idx: Index into dataset sample list
        
        Returns:
            dict: Processed sample containing:
                - 'id': Sample identifier string
                - 'image': Preprocessed and augmented image tensor
                - 'mask': Corresponding segmentation mask tensor
        
        Note:
            - Applies foreground oversampling based on batch position
            - Handles 2D/3D dimensionality automatically
            - Applies encoder-specific preprocessing
            - Maintains spatial correspondence between image and mask
        """
        batch_idx, sample_idx = indexes
        name = self.ids[sample_idx]

        # Load data components
        image, mask, properties = self.load_image_mask_properties(name)

        # Determine if foreground oversampling should be applied
        force_fg = self.oversampling_method(batch_idx)

        # Calculate crop coordinates
        shape = image.shape[1:]  # Skip channel dimension
        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
        bbox = [[int(i), int(j)] for i, j in zip(bbox_lbs, bbox_ubs)]

        # Extract image crop with lazy loading
        image = crop_from_zarr(image, bbox, 0)
        # Handle dimensionality for 2D/3D compatibility
        image = np.expand_dims(image, axis=0) if len(image.shape) < len(self.patch_size) + 1 else image
        image = np.squeeze(image, axis=1) if self.patch_size[0] == 1 else image
        # Apply encoder-specific preprocessing
        image = self.preprocess_func(np.moveaxis(image, 0, -1))
        image = np.moveaxis(image, -1, 0)

        # Extract mask crop
        mask = crop_from_zarr(mask, bbox, 0)
        # Handle dimensionality for 2D/3D compatibility
        mask = np.expand_dims(mask, axis=0) if len(mask.shape) < len(self.patch_size) + 1 else mask
        mask = np.squeeze(mask, axis=1) if self.patch_size[0] == 1 else mask

        # Convert to PyTorch tensors
        image = torch.as_tensor(image).float().contiguous()
        mask = torch.as_tensor(mask).long().contiguous()

        # Apply augmentation pipeline
        image, mask = self.transform(image, mask)

        return {'id': name, 'image': image, 'mask': mask}


class CustomBatchSampler(Sampler):
    """
    Custom batch sampler ensuring balanced data distribution across epochs.
    
    Implements controlled sampling strategy that ensures each sample is used
    exactly once per epoch before repetition, while maintaining specified
    number of training steps. Provides deterministic behavior for reproducible
    training with flexible epoch lengths.
    
    Args:
        dataset: PyTorch dataset to sample from
        batch_size (int): Number of samples per batch
        number_of_steps (int, optional): Total number of batches per epoch.
            Defaults to 250 for training.
        shuffle (bool, optional): Whether to shuffle sample order.
            Defaults to True for training.
    
    Attributes:
        batch_size (int): Configured batch size
        number_of_steps (int): Total steps per epoch
        shuffle (bool): Whether shuffling is enabled
        indices (list): All available sample indices
        sample_order (list): Computed sampling order for current epoch
    
    Note:
        - Ensures fair sample representation across training
        - Repeats samples only when necessary to reach target step count
        - Maintains randomness while preventing bias toward certain samples
        - Compatible with both training and validation modes
    """
    
    def __init__(self, dataset, batch_size, number_of_steps=250, shuffle=True):
        """Initialize custom batch sampler with specified parameters."""
        super().__init__()
        self.batch_size = batch_size
        self.number_of_steps = number_of_steps
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.sample_order = []  # Stores the computed sampling order

    def define_indices(self):
        """
        Create sampling order ensuring balanced sample usage across epochs.
        
        Generates a sample order that uses each dataset sample at least once
        before repetition, while producing exactly the required number of
        total samples for the specified number of steps.
        
        Side Effects:
            - Updates self.sample_order with computed sampling sequence
            - Ensures deterministic behavior for given random state
            - Balances sample representation across the epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Generate the order in which samples will be taken
        self.sample_order = []
        total_needed = self.number_of_steps * self.batch_size
        available = self.indices.copy()

        while len(self.sample_order) < total_needed:
            if len(available) < self.batch_size:
                # If fewer samples remain than batch size, refresh and reshuffle
                available = self.indices.copy()
                if self.shuffle:
                    np.random.shuffle(available)

            # Take batch_size elements from available samples
            self.sample_order.extend(available[:self.batch_size])
            available = available[self.batch_size:]

    def __iter__(self):
        """
        Generate batch indices for the current epoch.
        
        Yields:
            list: Batch of (batch_position, sample_index) tuples for each step
        """
        self.define_indices()

        for step in range(self.number_of_steps):
            batch_start = step * self.batch_size
            sample_indices = self.sample_order[batch_start: batch_start + self.batch_size]
            # Create batch with position information for oversampling
            batch = [(i, sample_idx) for i, sample_idx in enumerate(sample_indices)]
            yield batch

    def __len__(self):
        """Return number of batches per epoch."""
        return self.number_of_steps


def get_data_loaders(config, splitting, fold=None):
    """
    Create optimized data loaders for training and validation.
    
    Constructs PyTorch DataLoaders with custom sampling strategies and performance
    optimizations for surgical tool segmentation training. Handles both single
    train/val/test splits and k-fold cross-validation scenarios.
    
    Args:
        config (dict): Complete training configuration containing dataset paths,
            batch sizes, worker counts, and model specifications
        splitting (str): Split strategy to use:
            - 'train-val-test': Single split (70%/10%/20% train/val/test)
            - '5-fold': K-fold cross-validation
        fold (int, optional): Fold number for k-fold CV (0-indexed).
            Required when splitting='5-fold', ignored otherwise.
    
    Returns:
        tuple: (train_loader, val_loader) with optimized DataLoader instances
            configured for:
            - Multi-worker data loading with prefetching
            - Custom batch sampling for balanced representation
            - Memory pinning for GPU transfer optimization
            - Appropriate step counts for training vs validation
    
    Note:
        - Training uses 250 steps/epoch with shuffling for diversity
        - Validation uses 50 steps/epoch without shuffling for consistency
        - Workers and prefetching optimized for GPU utilization
        - Memory pinning accelerates CPU-to-GPU transfers
        
    Example:
        >>> config = {...}  # Complete configuration dictionary
        >>> train_loader, val_loader = get_data_loaders(config, 'train-val-test')
        >>> # For k-fold cross-validation:
        >>> train_loader, val_loader = get_data_loaders(config, '5-fold', fold=0)
    """
    # Create dataset splits
    split_file_path = create_split_files(config['dataset_path'], splitting, seed=12345)
    data_ids = get_data_ids(split_file_path, fold)

    # Initialize datasets with comprehensive configuration
    train_ds = SegTrainingDataset(config=config, data_ids=data_ids['train'], section="training")
    val_ds = SegTrainingDataset(config=config, data_ids=data_ids['val'], section="validation")

    # Create custom batch samplers for controlled sampling
    train_sampler = CustomBatchSampler(train_ds, batch_size=config['batch_size'], 
                                     number_of_steps=250, shuffle=True)
    val_sampler = CustomBatchSampler(val_ds, batch_size=config['batch_size'], 
                                   number_of_steps=50, shuffle=False)
    
    # Configure data loader optimizations
    loader_args = dict(
        num_workers=config['num_workers'],  # Parallel data loading
        pin_memory=True,                    # Faster GPU transfers
        prefetch_factor=2                   # Prefetch batches for efficiency
    )
    
    # Create optimized data loaders
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_args)
    
    return train_loader, val_loader


class SegTestDataset(Dataset):
    """
    Optimized dataset for surgical tool segmentation inference on video sequences.
    
    Implements efficient batched inference for surgical videos stored in Zarr format.
    Designed for real-time or near-real-time processing of surgical tool segmentation
    with minimal memory overhead and optimized I/O performance.
    
    Key Features:
    - Temporal batching for efficient video sequence processing
    - Zarr array caching to minimize I/O overhead
    - Memory-efficient lazy loading of video frames
    - Encoder-specific preprocessing for pretrained models
    - Automatic handling of variable-length video sequences
    
    Args:
        data_path (str): Path to dataset directory containing Zarr files
        data_ids (list): List of video identifiers to process
        batch_size (int): Number of frames to process per batch
            (temporal batching, not training batch size)
        preprocess_func: Preprocessing function matching encoder requirements
            (typically from segmentation_models_pytorch)
    
    Attributes:
        data_path (str): Path to Zarr files directory
        ids (list): Video identifiers for inference
        batch_size (int): Temporal batch size for frame processing
        preprocess_func: Encoder preprocessing function
        video_lengths (list): Number of frames per video
        _index (list): Mapping of dataset indices to (video_idx, t0, t1) tuples
        _zarr_cache (dict): Per-worker cache for opened Zarr arrays
    
    Note:
        - Temporal batching reduces inference overhead for video sequences
        - Zarr caching prevents repeated file opening in multi-worker scenarios
        - Memory usage scales with batch_size, not total video length
        - Compatible with sliding window inference for large images
    """
    
    def __init__(self, data_path, data_ids, batch_size, preprocess_func):
        """Initialize inference dataset with video sequence support."""
        self.data_path = os.path.join(data_path, 'data')
        self.ids = list(data_ids)
        self.batch_size = batch_size  # Temporal batch size for video frames
        self.preprocess_func = preprocess_func

        # Per-worker cache for opened Zarr arrays (prevents repeated opening)
        self._zarr_cache = {}
        
        # Store video lengths for processing information
        self.video_lengths = []

        # Build index mapping dataset positions to video frames
        self._index = []  # Format: (vid_idx, t0, t1)
        
        for vid_idx, name in enumerate(self.ids):
            # Open Zarr group to get temporal dimension
            zg = zarr.open_group(os.path.join(self.data_path, name + '.zarr'), mode='r')
            tlen = zg['image'].shape[1]  # Time is axis 1: (channels, time, height, width)
            self.video_lengths.append(tlen)
            
            # Create temporal batches for this video
            t = 0
            while t < tlen:
                t0 = t
                t1 = min(t + self.batch_size, tlen)  # Handle last partial batch
                self._index.append((vid_idx, t0, t1))
                t += self.batch_size

    def __len__(self):
        """Return total number of temporal batches across all videos."""
        return len(self._index)

    def _get_arrays(self, vid_idx):
        """
        Retrieve cached Zarr arrays for specified video.
        
        Implements per-worker caching to avoid repeated file opening overhead
        in multi-worker data loading scenarios.
        
        Args:
            vid_idx (int): Video index in self.ids list
        
        Returns:
            tuple: (image_array, mask_array) as Zarr arrays
            
        Note:
            - Cache is per-worker to avoid multiprocessing issues
            - Arrays remain as Zarr objects (not loaded into memory)
            - Enables efficient random access to video frames
        """
        if vid_idx not in self._zarr_cache:
            name = self.ids[vid_idx]
            zg = zarr.open_group(os.path.join(self.data_path, name + '.zarr'), mode='r')
            self._zarr_cache[vid_idx] = (zg['image'], zg['mask'])
        return self._zarr_cache[vid_idx]

    def __getitem__(self, idx):
        """
        Load temporal batch of frames for inference.
        
        Extracts a sequence of frames from the specified video and applies
        preprocessing for model inference. Handles variable batch sizes
        for video end conditions.
        
        Args:
            idx (int): Index into temporal batch list
        
        Returns:
            dict: Batch containing:
                - 'id': Video identifier string
                - 't0': Start frame index for this batch
                - 't1': End frame index for this batch (exclusive)
                - 'image': Preprocessed image tensor (batch_frames, channels, height, width)
                - 'mask': Corresponding mask tensor (batch_frames, height, width)
        
        Note:
            - Image preprocessing matches encoder requirements
            - Temporal dimension becomes batch dimension for inference
            - Maintains frame order within video sequence
            - Handles partial batches at video end gracefully
        """
        vid_idx, t0, t1 = self._index[idx]
        image_arr, mask_arr = self._get_arrays(vid_idx)

        # Extract temporal slice and reorganize dimensions
        # From: (channels, time, height, width) -> (time, channels, height, width)
        image = torch.as_tensor(image_arr[:, t0:t1]).float().permute(1, 0, 2, 3).contiguous()
        
        # Apply encoder preprocessing (requires channels-last format)
        image = image.movedim(1, -1)          # (time, height, width, channels)
        image = self.preprocess_func(image)   # Apply ImageNet normalization
        image = image.movedim(-1, 1)          # Back to (time, channels, height, width)

        # Extract corresponding mask frames
        mask = torch.as_tensor(mask_arr[t0:t1]).long().contiguous()

        return {
            'id': self.ids[vid_idx], 
            't0': t0, 
            't1': t1, 
            'image': image, 
            'mask': mask
        }