"""
Common Utilities for MMR Segmentation Framework

This module provides essential utility functions for the MMR (Multi-Modal Robotic) Segmentation
project, including configuration management, loss visualization, data serialization, and 
argument parsing for training and inference pipelines.

The utilities handle:
- Configuration file creation and management (YAML format)
- Training/validation loss plotting and visualization
- NumPy data type conversion for YAML serialization
- Command-line argument parsing and validation
- Default configuration generation for different operational modes

Key Features:
- Supports both training and inference configuration modes
- Provides flexible argument parsing with type safety
- Generates publication-ready loss plots with customizable scaling
- Ensures YAML compatibility through data type cleaning
- Validates user arguments against allowed parameter sets

Dependencies:
- matplotlib: For loss plot generation
- yaml: For configuration file handling
- numpy: For data type conversions
- ast: For safe string evaluation

Author: MMR Segmentation Team
Version: 1.0
"""

import os
import ast
import yaml
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def save_losses(loss_dict, save_path, log_scale=False):
    """
    Generate and save training/validation loss plots from loss dictionary.
    
    Creates a professional publication-ready plot showing training and validation
    loss curves over epochs. Supports both linear and logarithmic scaling for
    better visualization of loss trends.
    
    Args:
        loss_dict (dict): Dictionary containing loss values with keys:
            - 'train_loss': List of training loss values per epoch
            - 'val_loss': List of validation loss values per epoch
        save_path (str): Directory path where the loss plot will be saved
        log_scale (bool, optional): Whether to use logarithmic y-axis scaling.
            Defaults to False. Useful when loss values span multiple orders
            of magnitude.
    
    Returns:
        None: Saves plot as 'loss.png' in the specified directory
        
    Note:
        - Plot is saved at 300 DPI for publication quality
        - Creates save_path directory if it doesn't exist
        - Uses tight layout for optimal spacing
        - Automatically closes plot to free memory
    """
    # Ensure output directory exists
    os.makedirs(save_path, exist_ok=True)
    save_plot_path = os.path.join(save_path, "loss.png")

    # Generate epoch sequence for x-axis
    epochs = range(1, len(loss_dict['train_loss']) + 1)

    # Define human-readable labels for plot legend
    mapping_names_dict = {
        'train_loss': 'Train DiceCELoss', 
        'val_loss': 'Val DiceCELoss'
    }

    # Create high-quality figure
    plt.figure(figsize=(10, 8))

    # Plot available loss curves
    for key in mapping_names_dict:
        if key in loss_dict.keys():
            plt.plot(epochs, loss_dict[key], 
                    label=mapping_names_dict[key], 
                    linestyle='-')

    # Configure axes and scaling
    if log_scale:
        plt.yscale('log')
        plt.ylabel("log(loss)")
    else:
        plt.ylabel("loss")
    
    plt.xlabel("Epoch")
    plt.title("Losses per Epoch")
    plt.legend()
    plt.grid(True)

    # Save with publication quality settings
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory


def clean_numpy_scalars(obj):
    """
    Recursively convert NumPy data types to native Python types for YAML serialization.
    
    NumPy scalar types (np.float64, np.int64, etc.) are not directly serializable
    to YAML format. This function recursively traverses data structures and converts
    all NumPy scalars to their corresponding Python native types.
    
    Args:
        obj: Any Python object that may contain NumPy scalars. Supports:
            - NumPy scalars (np.float64, np.int64, etc.)
            - Tuples, lists, and dictionaries (recursively processed)
            - Native Python types (passed through unchanged)
    
    Returns:
        object: Same structure as input with all NumPy scalars converted to
                Python native types (int, float, bool, etc.)
    
    Example:
        >>> config = {'lr': np.float64(0.001), 'epochs': np.int64(100)}
        >>> clean_config = clean_numpy_scalars(config)
        >>> # Result: {'lr': 0.001, 'epochs': 100}
    """
    if isinstance(obj, np.generic):
        # Convert NumPy scalar to Python native type
        return obj.item()
    elif isinstance(obj, tuple):
        # Recursively process tuple elements
        return tuple(clean_numpy_scalars(x) for x in obj)
    elif isinstance(obj, list):
        # Recursively process list elements
        return [clean_numpy_scalars(x) for x in obj]
    elif isinstance(obj, dict):
        # Recursively process dictionary values
        return {k: clean_numpy_scalars(v) for k, v in obj.items()}
    else:
        # Return native Python types unchanged
        return obj


def get_default_config(n_classes, patch_size):
    """
    Generate comprehensive default configuration for MMR segmentation training.
    
    Creates a complete configuration dictionary with scientifically validated
    default parameters for medical image segmentation. The configuration includes
    data augmentation settings, training hyperparameters, model architecture
    specifications, and optimization parameters.
    
    Args:
        n_classes (int): Number of segmentation classes (excluding background).
            Background class is automatically added, so total classes = n_classes + 1
        patch_size (list): Spatial dimensions for training patches as [height, width].
            Example: [512, 640] for rectangular patches
    
    Returns:
        dict: Complete configuration dictionary with the following structure:
            - n_classes: Number of segmentation classes
            - transformations: Data augmentation parameters
            - training: Batch size, epochs, validation settings
            - optimization: Learning rate, scheduler, optimizer settings
            - model: Architecture and encoder specifications
            - inference: Sliding window and overlap parameters
    
    Note:
        - Uses UNet++ architecture with MobileNetV3 encoder by default
        - Includes comprehensive augmentation pipeline proven effective for medical imaging
        - Learning rate schedule uses polynomial decay for stable convergence
        - All parameters are empirically validated for surgical tool segmentation
    """
    # Training duration
    n_epochs = 200
    
    # Comprehensive data augmentation configuration
    # Each parameter enables/disables specific augmentation techniques
    transformations = {
        "patch_size": patch_size,           # Spatial dimensions for training patches
        "scaling": True,                    # Random scaling augmentation
        "rotation": True,                   # Random rotation augmentation
        "gaussian_noise": True,             # Additive Gaussian noise
        "gaussian_blur": True,              # Gaussian blur for defocus simulation
        "low_resolution": False,            # Low-resolution simulation (disabled by default)
        "brightness": True,                 # Brightness variation
        "contrast": True,                   # Contrast adjustment
        "gamma": True,                     # Gamma correction augmentation
        "mirror": True,                    # Mirror/flip augmentation
        "dummy_2d": False                  # 2D augmentation mode (disabled for 3D data)
    }
    
    # Complete configuration structure
    config = {
        # Core model parameters
        "n_classes": n_classes,
        "transformations": transformations,
        
        # Training data parameters
        "oversample_ratio": 0.33,          # Fraction of batches to oversample foreground
        "batch_size": 8,                   # Training batch size
        "num_workers": 8,                  # DataLoader worker processes
        
        # Inference parameters
        "infer_batch_size": 6,             # Inference batch size
        "infer_num_workers": 4,            # Inference worker processes
        "sw_batch_size": 24,               # Sliding window batch size
        "sw_overlap": 0.5,                 # Sliding window overlap ratio
        
        # Training schedule
        "n_epochs": n_epochs,
        "val_plot_interval": 10,           # Validation plotting frequency
        
        # Optimization parameters
        "grad_clip_max_norm": 12,          # Gradient clipping threshold
        "grad_accumulate_step": 1,         # Gradient accumulation steps
        
        # Learning rate scheduler (polynomial decay)
        "lr_scheduler": {
            "name": "PolynomialLR",
            "total_iters": n_epochs,
            "power": 0.9
        },
        
        # Optimizer configuration (AdamW for stable training)
        "optimizer": {
            "name": 'AdamW',
            "lr": 1e-4
        },
        
        # Model architecture specification
        "model": {
            "arch": 'UnetPlusPlus',                    # UNet++ architecture
            "encoder_name": 'tu-mobilenetv3_small_100', # MobileNetV3 encoder
            "encoder_weights": 'imagenet',              # ImageNet pretrained weights
            "in_channels": 3,                          # RGB input channels
            "classes": n_classes + 1                   # Total classes (including background)
        }
    }

    return config


def create_config(config, results_path):
    """
    Save configuration dictionary to YAML file with professional formatting.
    
    Serializes the configuration dictionary to a human-readable YAML file with
    optimized formatting for version control and manual editing. Removes YAML
    anchors and uses flow-style formatting for lists to improve readability.
    
    Args:
        config (dict): Configuration dictionary to save. Should be cleaned
            of NumPy types using clean_numpy_scalars() if necessary
        results_path (str): Directory path where config.yaml will be saved.
            Directory is created if it doesn't exist
    
    Returns:
        None: Saves config.yaml in the specified directory
        
    Side Effects:
        - Creates results_path directory if it doesn't exist
        - Prints confirmation message with full path to saved file
        - Overwrites existing config.yaml if present
        
    Note:
        - Uses custom YAML dumper to avoid anchors (&id001 references)
        - Forces lists to use flow style [item1, item2] for compactness
        - Preserves key order from original dictionary
        - Optimized for Git diff readability
    """
    config_save_path = os.path.join(results_path, 'config.yaml')
    os.makedirs(results_path, exist_ok=True)

    # Custom YAML dumper for clean, professional formatting
    class CustomDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            """Remove YAML anchors for cleaner output."""
            return True

    # Configure list representation for compact flow style
    def represent_list(dumper, data):
        """Force lists to use flow style: [item1, item2, item3]."""
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    CustomDumper.add_representer(list, represent_list)
    
    # Save configuration with optimized formatting
    with open(config_save_path, "w") as file:
        yaml.dump(config, file, sort_keys=False, Dumper=CustomDumper)
    
    print(f"Saved configuration at {config_save_path}")


def add_user_config_args(config, config_args):
    """
    Merge user-provided configuration arguments into base configuration.
    
    Intelligently merges user arguments into the base configuration dictionary,
    handling both flat parameters and nested dictionaries. Automatically updates
    dependent parameters (e.g., learning rate scheduler iterations when epochs change).
    
    Args:
        config (dict): Base configuration dictionary to modify in-place
        config_args (dict or None): User-provided configuration overrides.
            Can contain:
            - Flat parameters: {'batch_size': 16, 'n_epochs': 100}
            - Nested parameters: {'optimizer': {'lr': 0.001}}
            If None, returns config unchanged
    
    Returns:
        dict: Modified configuration with user arguments applied
        
    Side Effects:
        - Modifies the input config dictionary in-place
        - Automatically updates lr_scheduler['total_iters'] when n_epochs changes
        
    Example:
        >>> base_config = {'batch_size': 8, 'optimizer': {'lr': 1e-4}}
        >>> user_args = {'batch_size': 16, 'optimizer': {'weight_decay': 1e-5}}
        >>> result = add_user_config_args(base_config, user_args)
        >>> # Result: {'batch_size': 16, 'optimizer': {'lr': 1e-4, 'weight_decay': 1e-5}}
    """
    if config_args is not None:
        for key in config_args:
            if not isinstance(config_args[key], dict):
                # Handle flat parameters
                config[key] = config_args[key]
            else:
                # Handle nested dictionaries (merge, don't replace)
                for inner_key in config_args[key]:
                    config[key][inner_key] = config_args[key][inner_key]

            # Automatic dependency update: sync scheduler iterations with epochs
            if key == 'n_epochs' and 'total_iters' in config['lr_scheduler']:
                config['lr_scheduler']['total_iters'] = config_args[key]

    return config


def safe_eval(s: str):
    """
    Safely evaluate string representations of Python literals.
    
    Provides secure evaluation of string arguments from command line or configuration
    files. Handles common literals (booleans, None) and uses ast.literal_eval for
    safe evaluation of numbers, lists, dictionaries, and strings. Falls back to
    plain string if evaluation fails.
    
    Args:
        s (str): String representation to evaluate. Examples:
            - 'true', 'false' -> boolean values
            - 'none', 'null' -> None
            - '42', '3.14' -> numeric values
            - '[1, 2, 3]' -> list
            - '{"key": "value"}' -> dictionary
            - 'plain_text' -> string (fallback)
    
    Returns:
        object: Evaluated Python object of appropriate type, or original
                string if evaluation fails
    
    Security:
        - Uses ast.literal_eval for safe evaluation (no code execution)
        - No function calls or arbitrary code execution possible
        - Safe for processing untrusted input
        
    Example:
        >>> safe_eval('true')      # -> True
        >>> safe_eval('[1, 2, 3]') # -> [1, 2, 3]
        >>> safe_eval('hello')     # -> 'hello'
    """
    # Handle common string literals case-insensitively
    literal_mappings = {
        'true': True, 
        'false': False, 
        'none': None, 
        'null': None
    }
    
    if (normalized := s.lower()) in literal_mappings:
        return literal_mappings[normalized]
    
    try:
        # Safely evaluate Python literals (numbers, lists, dicts, strings)
        return ast.literal_eval(s)
    except Exception:
        # Fallback: return as plain string
        return s


def check_and_convert_user_config_args(args, mode):
    """
    Parse, validate, and convert command-line arguments for different operational modes.
    
    Processes raw command-line arguments into a structured configuration dictionary
    with proper validation against allowed parameters for training or inference modes.
    Handles argument prefixes for nested configurations and provides detailed error
    messages for invalid arguments.
    
    Args:
        args (list): Raw command-line arguments as strings. Should follow pattern:
            ['--arg1', 'value1', '--arg2', 'value2', ...]
            Leading dashes are automatically stripped
        mode (str): Operational mode for validation. Options:
            - 'training': Validates against training parameters
            - 'testing': Validates against inference parameters
    
    Returns:
        dict or None: Structured configuration dictionary with nested parameters
                     properly organized, or None if no arguments provided
        
    Raises:
        KeyError: If any argument is not allowed for the specified mode,
                 with detailed message showing valid arguments
        
    Example:
        >>> args = ['--batch_size', '16', '--optimizer_lr', '0.001']
        >>> result = check_and_convert_user_config_args(args, 'training')
        >>> # Result: {'batch_size': 16, 'optimizer': {'lr': 0.001}}
        
    Note:
        - Supports nested parameters using underscore notation: 'optimizer_lr'
        - Automatically converts string values using safe_eval()
        - Provides comprehensive validation with helpful error messages
    """
    # Clean argument names (remove leading dashes)
    args = [a.lstrip("-") for a in args]
    
    # Convert to key-value pairs and evaluate values safely
    args = {args[i]: safe_eval(args[i + 1]) for i in range(0, len(args), 2)}

    if args:
        # Define allowed arguments for each mode
        allowed_training_args = [
            "transformations_scaling", "transformations_rotation", "transformations_gaussian_noise",
            "transformations_gaussian_blur", "transformations_low_resolution", "transformations_brightness",
            "transformations_contrast", "transformations_gamma", "transformations_mirror", "transformations_dummy_2d",
            "oversample_ratio", "batch_size", "num_workers", "n_epochs", "val_plot_interval", 
            "grad_clip_max_norm", "grad_accumulate_step"
        ]
        
        allowed_testing_args = [
            "infer_batch_size", "infer_num_workers", "sw_batch_size", "sw_overlap"
        ]
        
        # Flexible arguments that allow nested parameters
        flexible_args = ["lr_scheduler", "optimizer", "model"]
        args_with_prefix = ["transformations", "lr_scheduler", "optimizer", "model"]

        args_dict = {}
        
        # Validate and process each argument
        for arg in args:
            # Check argument validity based on mode
            arg_wrong_for_training = (
                (arg not in allowed_training_args and mode == 'training') and 
                not any([item in arg for item in flexible_args])
            )
            arg_wrong_for_inference = arg not in allowed_testing_args and mode == 'testing'
            
            if arg_wrong_for_training or arg_wrong_for_inference:
                raise KeyError(
                    f"Argument '{arg}' is not an allowed argument. "
                    f"Valid training arguments: {allowed_training_args}. "
                    f"Valid inference arguments: {allowed_testing_args}. "
                    f"Valid flexible arguments: {[item + '_*' for item in flexible_args]}."
                )

            # Process flat vs. nested arguments
            if not any([item in arg for item in args_with_prefix]):
                # Flat argument
                args_dict[arg] = args[arg]
            else:
                # Nested argument with prefix (e.g., 'optimizer_lr')
                for item in args_with_prefix:
                    if item in arg:
                        if item not in args_dict.keys():
                            args_dict[item] = {}
                        
                        # Extract the nested key name
                        key_name = arg.replace(item + '_', '')
                        args_dict[item][key_name] = args[arg]

        return args_dict
    else:
        return None


def get_config(dataset_path, results_path, mode, config_args=None, n_classes=None, patch_size=None, progress_bar=False):
    """
    Generate or load configuration based on operational mode with user argument integration.
    
    Central configuration manager that handles different operational modes (training,
    continue_training, testing) with intelligent argument validation and model path
    resolution. Automatically manages configuration consistency across training sessions
    and provides flexible parameter overrides for inference.
    
    Args:
        dataset_path (str): Path to the dataset directory containing processed data
        results_path (str): Path to results directory for saving/loading configurations
        mode (str): Operational mode determining configuration behavior:
            - 'training': Create new configuration with defaults + user args
            - 'continue_training': Load existing config, validate consistency
            - 'testing': Load existing config, allow inference parameter overrides
        config_args (dict, optional): User-provided configuration overrides.
            Defaults to None
        n_classes (int, optional): Number of segmentation classes. Required for
            'training' mode, ignored otherwise
        patch_size (list, optional): Patch dimensions [height, width]. Required for
            'training' mode, ignored otherwise  
        progress_bar (bool, optional): Enable progress bar display. Defaults to False
    
    Returns:
        dict: Complete configuration dictionary with all parameters resolved,
              including automatically determined model paths and user overrides
    
    Raises:
        AssertionError: In 'continue_training' mode if user arguments differ from
                       saved configuration (prevents training inconsistencies)
        FileNotFoundError: If config.yaml not found in 'continue_training' or 'testing' modes
        
    Side Effects:
        - Creates and saves config.yaml in 'training' mode
        - Prints configuration file path in non-training modes
        - Prints changed arguments in 'testing' mode
        
    Note:
        - Automatically sets 'load_model_path' based on mode:
          * 'continue_training': uses last_model.pth
          * 'testing': uses best_model.pth
          * 'training': None (training from scratch)
        - Validates argument consistency in 'continue_training' to prevent errors
        - Allows flexible inference parameters in 'testing' mode
    """
    config_path = os.path.join(results_path, 'config.yaml')

    if mode == 'training':
        # Create new configuration for training from scratch
        config = get_default_config(n_classes, patch_size)
        config["dataset_path"] = dataset_path
        config = add_user_config_args(config, config_args)
        create_config(config, results_path)
        
    else:
        # Load existing configuration
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        print(f"Using configuration file: {config_path}")
        user_config = add_user_config_args(deepcopy(config), config_args)

        if mode == 'continue_training':
            # Ensure consistency when continuing training
            different_args = [
                (key, config[key], user_config[key]) 
                for key in user_config 
                if config[key] != user_config[key]
            ]
            
            if different_args:
                assertion_print = ", ".join([
                    f"arg '{item[0]}' - ({item[1]}, {item[2]})" 
                    for item in different_args
                ])
                raise AssertionError(
                    f"Different argument values were given between continue_training and training: "
                    f"{assertion_print}"
                )
                
        elif mode == 'testing':
            # Allow inference parameter overrides
            if config_args is not None:
                different_args = [
                    (key, config[key], user_config[key]) 
                    for key in user_config 
                    if config[key] != user_config[key]
                ]
                
                if different_args:
                    changed_args_print = "\n    ".join([
                        f"'{item[0]}': {item[1]} --> {item[2]}" 
                        for item in different_args
                    ])
                    print(f"Inference arguments changed:\n    {changed_args_print}")

        config = user_config

    # Add runtime configuration
    config['progress_bar'] = progress_bar
    config['results_path'] = results_path

    # Automatically determine model paths based on mode
    last_model_path = os.path.join(results_path, 'checkpoints', 'last_model.pth')
    best_model_path = os.path.join(results_path, 'checkpoints', 'best_model.pth')
    
    if mode == 'continue_training':
        config['load_model_path'] = last_model_path
    elif mode == 'testing':
        config['load_model_path'] = best_model_path
    else:
        config['load_model_path'] = None

    return config