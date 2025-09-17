"""
MMR Segmentation Model Training Script

This module provides the main training interface for the MMR (Multi-Modal Robotic) 
segmentation framework designed for surgical tool segmentation in robotic surgery videos.
It handles command-line argument parsing, dataset configuration, model initialization,
and orchestrates the complete training pipeline.

Key Features:
- Flexible dataset splitting (train-val-test or 5-fold cross-validation)
- Configurable training parameters via command line arguments
- Automatic checkpoint management and training continuation
- Integration with custom data loaders and model architecture
- Support for multi-class surgical tool segmentation
- Comprehensive error handling and validation

Target Applications:
- Robotic surgery tool tracking and identification
- Medical image analysis research
- Surgical workflow automation
- Clinical decision support systems

Author: MMR Segmentation Team
Version: 1.0.0
"""

import os
import ast
import argparse
import tempfile
from MMR_Segmentation.Pre_Process import get_data_loaders
from MMR_Segmentation.common_utils import get_config, check_and_convert_user_config_args
from MMR_Segmentation.Main_MMR_SegModel import SegModel


def parse_arguments():
    """
    Parse and validate command-line arguments for MMR model training.
    
    This function sets up the argument parser with all required and optional parameters
    for training the segmentation model. It handles both main training arguments and
    additional configuration arguments, performs validation checks, and ensures
    compatibility between different argument combinations.
    
    Returns:
        tuple: A tuple containing:
            - main_args (argparse.Namespace): Parsed main arguments including dataset_path,
              results_path, splitting strategy, n_classes, patch_size, fold, progress_bar,
              and continue_training flags
            - additional_args (dict): Additional configuration arguments converted to
              appropriate types for model configuration
    
    Raises:
        argparse.ArgumentError: If required arguments are missing or invalid combinations
                               are provided (e.g., missing fold for 5-fold validation)
        FileExistsError: If results_path already exists and continue_training is False
    
    Example:
        >>> main_args, config_args = parse_arguments()
        >>> print(f"Training on {main_args.n_classes} classes")
        >>> print(f"Using patch size: {main_args.patch_size}")
    
    Note:
        - When using '5-fold' splitting, the --fold argument (0-4) is mandatory
        - When using 'train-val-test' splitting, --fold should not be provided
        - Results path must not exist unless continuing training from checkpoint
    """
    # Initialize argument parser with descriptive help text
    parser = argparse.ArgumentParser(
        description="Train a model to segment surgical instruments."
    )
    
    # Required positional arguments for core training configuration
    parser.add_argument(
        "dataset_path", 
        type=str, 
        help="Path to the dataset."
    )
    parser.add_argument(
        "results_path", 
        type=str, 
        help="Path to save results."
    )
    parser.add_argument(
        "splitting", 
        choices=["train-val-test", "5-fold"],
        help="Choose either 'train-val-test' for a standard split or '5-fold' for cross-validation."
    )
    parser.add_argument(
        "n_classes", 
        type=int, 
        help="Number of classes."
    )
    parser.add_argument(
        "patch_size", 
        type=ast.literal_eval, 
        help="Patch size in format [W,H]"
    )
    
    # Optional arguments for advanced training configuration
    parser.add_argument(
        "-f", "--fold", 
        type=int, 
        choices=[0, 1, 2, 3, 4], 
        required=False, 
        default=None,
        help="Specify the fold index (0-4) when using 5-fold cross-validation."
    )
    parser.add_argument(
        "-p", "--progress_bar", 
        action="store_true", 
        help="Enable progress bar (default: False)"
    )
    parser.add_argument(
        "-c", "--continue_training", 
        action="store_true",
        help="Continue training from the last checkpoint (default: False)"
    )

    # Parse main arguments and capture additional configuration arguments
    main_args, additional_args = parser.parse_known_args()

    # Convert additional arguments to proper configuration format
    additional_args = check_and_convert_user_config_args(additional_args, mode='training')

    # Validation: Ensure --fold is provided only when --splitting is "5-fold"
    if main_args.splitting == "5-fold" and main_args.fold is None:
        parser.error("--fold is required when --splitting is set to '5-fold'")

    # Validation: Ensure --fold is None when --splitting is "train-val-test"
    if main_args.splitting == "train-val-test" and main_args.fold is not None:
        parser.error("--fold should not be provided when --splitting is set to 'train-val-test'")

    # Validation: Check if results path already exists (unless continuing training)
    if os.path.exists(main_args.results_path) and not main_args.continue_training:
        raise FileExistsError(f"Results path {main_args.results_path} already exists.")

    return main_args, additional_args


def main():
    """
    Main training execution function for MMR segmentation model.
    
    This function orchestrates the complete training pipeline including:
    - Temporary directory management for intermediate files
    - Argument parsing and validation
    - Configuration setup with user-provided parameters
    - Data loader initialization based on splitting strategy
    - Model instantiation and training execution
    
    The function uses a temporary directory context manager to ensure clean
    resource management and handles both standard train-val-test splits and
    k-fold cross-validation scenarios.
    
    Pipeline Steps:
    1. Create temporary directory for intermediate processing
    2. Parse and validate command-line arguments
    3. Extract training parameters from parsed arguments
    4. Configure training mode (new training vs. continuation)
    5. Initialize model configuration with all parameters
    6. Create data loaders based on splitting strategy
    7. Instantiate segmentation model with configuration
    8. Execute training with prepared data loaders
    
    Raises:
        FileExistsError: If results path exists and continue_training is False
        ValueError: If invalid splitting strategy or fold configuration is provided
        RuntimeError: If model initialization or training execution fails
    
    Example:
        >>> # Command line usage:
        >>> # python script.py /path/to/dataset /path/to/results train-val-test 9 "[512,640]" -p
        >>> main()
    
    Note:
        - Uses temporary directory for safe intermediate file handling
        - Automatically cleans up temporary files after execution
        - Supports both new training and checkpoint continuation
        - Configuration is fully customizable via command-line arguments
    """
    # Use temporary directory context manager for clean resource handling
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")

        # Parse command-line arguments and additional configuration
        args, config_args = parse_arguments()
        
        # Extract core training parameters from parsed arguments
        dataset_path = args.dataset_path
        results_path = args.results_path
        splitting = args.splitting
        n_classes = args.n_classes
        patch_size = args.patch_size
        fold = args.fold
        progress_bar = args.progress_bar
        continue_training = args.continue_training

        # Determine training mode based on continuation flag
        mode = 'continue_training' if continue_training else 'training'
        
        # Initialize comprehensive model configuration
        config = get_config(
            dataset_path, 
            results_path, 
            mode, 
            config_args=config_args, 
            n_classes=n_classes,
            patch_size=patch_size, 
            progress_bar=progress_bar
        )

        # Create data loaders based on splitting strategy and fold configuration
        train_loader, val_loader = get_data_loaders(config, splitting, fold)

        # Initialize segmentation model with prepared configuration
        model = SegModel(config=config)
        
        # Execute training pipeline with prepared data loaders
        model.train(train_loader=train_loader, val_loader=val_loader)


# Execute main function when script is run directly
if __name__ == "__main__":
    main()