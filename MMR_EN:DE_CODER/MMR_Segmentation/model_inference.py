"""
MMR Segmentation Model Inference Script

This module provides the inference interface for the MMR (Multi-Modal Robotic) 
segmentation framework, enabling prediction on new datasets using pre-trained models.
It supports both testing on completely new datasets and evaluation on the test 
partition of the original training dataset.

Key Features:
- Flexible inference on new datasets or training dataset test splits
- Automatic data discovery and preprocessing pipeline setup
- Efficient batch processing with configurable parameters
- Integration with pre-trained model checkpoints
- Support for sliding window inference on large images
- Comprehensive error handling and validation

Inference Modes:
1. New Dataset Testing: Infer on completely new data (different from training)
2. Training Dataset Testing: Evaluate on test partition of original training data

Target Applications:
- Clinical deployment for real-time surgical tool segmentation
- Research evaluation on new datasets
- Performance benchmarking across different surgical procedures
- Validation of model generalization capabilities

Author: MMR Segmentation Team
Version: 1.0.0
"""

import os
import glob
import tempfile
import argparse
from torch.utils.data import DataLoader
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from MMR_Segmentation.common_utils import get_config, check_and_convert_user_config_args
from MMR_Segmentation.Pre_Process import create_split_files, get_data_ids, SegTestDataset
from MMR_Segmentation.Main_MMR_SegModel import SegModel


def parse_arguments():
    """
    Parse and validate command-line arguments for MMR model inference.
    
    This function sets up the argument parser for inference-specific parameters
    and validates that the required pre-trained model exists at the results path.
    It handles both main inference arguments and additional configuration arguments
    for fine-tuning the inference process.
    
    Returns:
        tuple: A tuple containing:
            - main_args (argparse.Namespace): Parsed main arguments including 
              test_dataset_path and results_path
            - additional_args (dict): Additional configuration arguments converted 
              to appropriate types for inference configuration
    
    Raises:
        FileExistsError: If results_path does not exist (pre-trained model required)
        argparse.ArgumentError: If required arguments are missing or invalid
    
    Example:
        >>> main_args, config_args = parse_arguments()
        >>> print(f"Testing dataset: {main_args.test_dataset_path}")
        >>> print(f"Model location: {main_args.results_path}")
    
    Note:
        - The results_path must contain a pre-trained model from previous training
        - Additional arguments are automatically converted to proper configuration format
        - Inference mode is automatically detected based on dataset path comparison
    """
    # Initialize argument parser with descriptive help text for inference
    parser = argparse.ArgumentParser(
        description="Run inference on a data."
    )
    
    # Required positional arguments for inference configuration
    parser.add_argument(
        "test_dataset_path", 
        type=str, 
        help="Path to the dataset we want to infer on."
    )
    parser.add_argument(
        "results_path", 
        type=str, 
        help="Path to save results."
    )

    # Parse main arguments and capture additional configuration arguments
    main_args, additional_args = parser.parse_known_args()

    # Convert additional arguments to proper configuration format for testing mode
    additional_args = check_and_convert_user_config_args(additional_args, mode='testing')

    # Validation: Ensure results path exists (contains pre-trained model)
    if not os.path.exists(main_args.results_path):
        raise FileExistsError(
            f"Results path {main_args.results_path} should already exist. First, train a model."
        )

    return main_args, additional_args


def main():
    """
    Main inference execution function for MMR segmentation model.
    
    This function orchestrates the complete inference pipeline including:
    - Temporary directory management for intermediate processing
    - Argument parsing and validation
    - Configuration setup for inference mode
    - Automatic detection of inference scenario (new vs. training dataset)
    - Data loading and preprocessing pipeline setup
    - Model loading from pre-trained checkpoint
    - Inference execution on prepared test data
    
    Inference Pipeline Steps:
    1. Create temporary directory for intermediate processing
    2. Parse and validate command-line arguments
    3. Initialize inference configuration from pre-trained model
    4. Determine inference mode by comparing dataset paths
    5. Prepare test data IDs based on inference mode
    6. Set up preprocessing pipeline matching training configuration
    7. Create test dataset and data loader with optimal settings
    8. Load pre-trained model from checkpoint
    9. Execute inference on prepared test data
    
    Inference Modes:
    - **New Dataset Mode**: When test_dataset_path differs from training path,
      automatically discovers all .zarr files in the dataset for inference
    - **Training Dataset Mode**: When paths match, uses the test partition
      from the original train-val-test split for evaluation
    
    Raises:
        FileExistsError: If results path does not contain pre-trained model
        RuntimeError: If model loading or inference execution fails
        ValueError: If dataset configuration is incompatible with pre-trained model
    
    Example:
        >>> # Command line usage for new dataset:
        >>> # python inference_script.py /path/to/new/dataset /path/to/trained/model
        >>> main()
    
    Note:
        - Automatically handles different inference scenarios without user intervention
        - Uses optimal data loading settings for inference performance
        - Preserves all preprocessing configurations from training
        - Supports batch processing for efficient GPU utilization
    """
    # Use temporary directory context manager for clean resource handling
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")

        # Parse command-line arguments and additional configuration
        args, config_args = parse_arguments()
        
        # Extract core inference parameters from parsed arguments
        test_dataset_path = args.test_dataset_path
        results_path = args.results_path

        # Initialize inference configuration from pre-trained model
        config = get_config(
            test_dataset_path, 
            results_path, 
            mode='testing', 
            config_args=config_args
        )

        # Determine inference mode by comparing dataset paths
        train_path = config['dataset_path']
        
        if train_path != test_dataset_path:
            # NEW DATASET MODE: Testing on completely new data
            print("Training data path is different from dataset path. "
                  "Assuming we are in 'testing' mode and we are testing on a new dataset.")
            
            # Automatically discover all .zarr files in the new dataset
            file_paths = glob.glob(os.path.join(test_dataset_path, 'data', "*.zarr"))
            data_ids = [os.path.basename(fp).replace('.zarr', '') for fp in file_paths]
            print(f"{len(data_ids)} samples for testing")
            
        else:
            # TRAINING DATASET MODE: Testing on original dataset's test partition
            print("Testing on the test partition of the training dataset.")
            
            # Create split files using the same seed as training for consistency
            split_file_path = create_split_files(
                test_dataset_path, 
                'train-val-test', 
                seed=12345
            )
            
            # Extract test data IDs from the original split
            data_ids = get_data_ids(split_file_path)['test']

        # Set up preprocessing pipeline matching training configuration
        preprocess_func = get_preprocessing_fn(
            config['model']['encoder_name'], 
            pretrained=config['model']['encoder_weights']
        )
        
        # Create test dataset with optimized configuration for inference
        test_ds = SegTestDataset(
            data_path=test_dataset_path, 
            data_ids=data_ids, 
            batch_size=config['infer_batch_size'], 
            preprocess_func=preprocess_func
        )
        
        # Create data loader with optimal settings for inference performance
        test_loader = DataLoader(
            test_ds, 
            batch_size=None,  # Batch size handled by dataset
            shuffle=False,    # Maintain deterministic order
            num_workers=config['infer_num_workers'],
            pin_memory=True,         # Optimize GPU memory transfer
            persistent_workers=True, # Maintain worker processes
            prefetch_factor=1        # Optimize memory usage
        )

        # Initialize segmentation model from pre-trained checkpoint
        model = SegModel(config=config)
        
        # Execute inference pipeline on prepared test data
        model.run_inference(test_loader=test_loader)


# Execute main function when script is run directly
if __name__ == "__main__":
    main()