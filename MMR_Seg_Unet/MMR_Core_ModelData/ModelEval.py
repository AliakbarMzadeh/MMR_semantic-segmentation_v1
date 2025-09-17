"""
ModelEval.py - Semantic Segmentation Model Evaluation

This module provides a standalone evaluation utility for testing trained semantic 
segmentation models on validation/test datasets. It loads pre-trained model checkpoints
and computes comprehensive evaluation metrics including IoU, Precision, Recall, and F1-score.

Key Features:
- Multi-architecture model loading support
- Comprehensive evaluation metrics (IoU, Precision, Recall, F1)  
- Flexible dataset configuration
- Sample prediction visualization and saving
- GPU acceleration support

Author: Medical Robotics Research Team
Compatible with: SAR-RARP50 and other surgical datasets
"""

# ================================ IMPORTS ================================

# System and utility imports
import argparse
import os

# Image processing
from PIL import Image

# PyTorch core imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

# Segmentation models pytorch for additional architectures
import segmentation_models_pytorch as smp

# Custom module imports
import utils

# Model architecture imports
from UArchModel.segnet import SegNet
from UArchModel.unet import UNet
from UArchModel.resnet_unet import ResNetUNet

# Data loading imports - Updated to use SAR-compatible loader
from SurgicalDataClass.dataloaders.SegNetDataLoaderV1_SAR import SegNetDataset

# ================================ ARGUMENT PARSER SETUP ================================

parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation Evaluation')

# Data loading parameters
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--batchSize', default=1, type=int,
                    help='Mini-batch size for evaluation (default: 1)')

# Model parameters  
parser.add_argument('--model_type', default='segnet', type=str,
                    help='model architecture (segnet, unet, resnet18, smp_UNet++, etc.)')
parser.add_argument('--bnMomentum', default=0.1, type=float,
                    help='Batch Norm Momentum (default: 0.1)')

# Image processing parameters
parser.add_argument('--imageSize', default=256, type=int,
                    help='height/width of the input image to the network')
parser.add_argument('--cropSize', default=-1, type=int,
                    help='crop size for evaluation (-1 for no cropping)')

# Dataset parameters
parser.add_argument('--data_dir', type=str, required=True,
                    help='data directory containing test images')
parser.add_argument('--json_path', type=str, required=True,
                    help='path to json file containing class information')
parser.add_argument('--dataset', type=str, default='sarrarp50',
                    help='dataset name (sarrarp50, synapse, cholec, etc.)')

# Model checkpoint parameters
parser.add_argument('--model', default='', type=str, metavar='PATH', required=True,
                    help='path to trained model checkpoint')

# Output parameters
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the evaluated images',
                    default='eval_results', type=str)
parser.add_argument('--saveTest', default='True', type=str,
                    help='Saves the validation/test images if True')

# ================================ HARDWARE SETUP ================================

use_gpu = torch.cuda.is_available()
print(f"GPU Available: {use_gpu}")
if use_gpu:
    print(f"GPU Device: {torch.cuda.get_device_name()}")

# ================================ MAIN FUNCTIONS ================================

def main():
    """
    Main evaluation function that orchestrates the model evaluation process.
    
    This function handles:
    - Argument parsing and validation
    - Data loading setup for test dataset
    - Model initialization and checkpoint loading
    - Evaluation execution with metrics computation
    - Results visualization and saving
    
    Returns:
        None: Prints evaluation results and optionally saves visualizations
    
    Raises:
        FileNotFoundError: If checkpoint or data files don't exist
        ValueError: If invalid model architecture is specified
    """
    # ===================== ARGUMENT PROCESSING =====================
    
    global args
    args = parser.parse_args()
    print("Evaluation Configuration:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Model Checkpoint: {args.model}")
    print(f"  Image Size: {args.imageSize}")
    print(f"  Batch Size: {args.batchSize}")
    print(f"  Save Results: {args.saveTest}")
    print()

    # Convert string boolean to actual boolean
    if args.saveTest == 'True':
        args.saveTest = True
    elif args.saveTest == 'False':
        args.saveTest = False

    # Create output directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created output directory: {args.save_dir}")

    # Enable cuDNN benchmark for optimal performance
    cudnn.benchmark = True

    # ===================== DATA LOADING SETUP =====================
    
    # Configure data transformations for evaluation
    data_transform = transforms.Compose([
        transforms.Resize((args.imageSize, args.imageSize), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    # Initialize test dataset
    try:
        image_dataset = SegNetDataset(
            root_dir=args.data_dir,
            crop_size=args.cropSize,
            json_path=args.json_path,
            sample='test',  # Evaluation mode
            dataset=args.dataset,
            image_size=[args.imageSize, args.imageSize],
            horizontal_flip=False,  # No augmentation for evaluation
            vertical_flip=False,
            rotate=False,
            brightness=False,
            contrast=False
        )
        print(f"Successfully loaded dataset with {len(image_dataset)} test images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=args.batchSize,
        shuffle=False,  # Keep order for evaluation
        num_workers=args.workers,
        pin_memory=True if use_gpu else False
    )

    # ===================== CLASS INFORMATION EXTRACTION =====================
    
    # Extract class information for model configuration
    classes = image_dataset.classes
    key = utils.disentangleKey(classes)
    num_classes = len(key)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(key.keys())}")
    print()

    # ===================== MODEL INITIALIZATION =====================
    
    # Initialize model based on specified architecture
    model = initialize_model(args.model_type, num_classes, args.bnMomentum)
    
    if model is None:
        print(f"Error: Unsupported model type '{args.model_type}'")
        return

    # ===================== CHECKPOINT LOADING =====================
    
    # Load trained model weights
    if os.path.isfile(args.model):
        print(f"=> Loading checkpoint '{args.model}'")
        try:
            checkpoint = torch.load(args.model, map_location='cpu' if not use_gpu else None)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                if 'epoch' in checkpoint:
                    print(f"=> Loaded checkpoint from epoch {checkpoint['epoch']}")
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
            
            print("=> Checkpoint loaded successfully")
        except Exception as e:
            print(f"=> Error loading checkpoint: {e}")
            return
    else:
        print(f"=> No checkpoint found at '{args.model}'")
        return

    print(f"Model Architecture: {args.model_type}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ===================== LOSS FUNCTION SETUP =====================
    
    # Define loss function for evaluation
    if args.dataset == "synapse":
        criterion = nn.CrossEntropyLoss(ignore_index=21)
    else:
        criterion = nn.CrossEntropyLoss()

    # Move model and loss to GPU if available
    if use_gpu:
        model.cuda()
        criterion.cuda()

    # ===================== EVALUATION EXECUTION =====================
    
    # Initialize evaluation metrics calculator
    evaluator = utils.Evaluate(key, use_gpu)

    # Run evaluation on test set
    print('=' * 60)
    print('RUNNING EVALUATION ON TEST SET')
    print('=' * 60)
    
    avg_loss = validate(dataloader, model, criterion, key, evaluator, args)
    
    print(f"\nAverage Test Loss: {avg_loss:.4f}")

    # ===================== METRICS COMPUTATION AND DISPLAY =====================
    
    print('=' * 60)
    print('EVALUATION METRICS')
    print('=' * 60)
    
    # Calculate IoU metrics
    IoU = evaluator.getIoU()
    mean_iou = torch.mean(IoU)
    print(f'Mean IoU: {mean_iou:.4f}')
    print('Class-wise IoU:')
    for i, iou in enumerate(IoU):
        print(f'  Class {i}: {iou:.4f}')
    
    # Calculate Precision, Recall, F1 metrics
    PRF1 = evaluator.getPRF1()
    precision, recall, F1 = PRF1[0], PRF1[1], PRF1[2]
    
    print(f'\nMean Precision: {torch.mean(precision):.4f}')
    print('Class-wise Precision:')
    for i, prec in enumerate(precision):
        print(f'  Class {i}: {prec:.4f}')
        
    print(f'\nMean Recall: {torch.mean(recall):.4f}')
    print('Class-wise Recall:')
    for i, rec in enumerate(recall):
        print(f'  Class {i}: {rec:.4f}')
        
    print(f'\nMean F1: {torch.mean(F1):.4f}')
    print('Class-wise F1:')
    for i, f1 in enumerate(F1):
        print(f'  Class {i}: {f1:.4f}')
    
    # ===================== RESULTS SUMMARY =====================
    
    print('\n' + '=' * 60)
    print('EVALUATION SUMMARY')
    print('=' * 60)
    print(f'Model: {args.model_type}')
    print(f'Dataset: {args.dataset}')
    print(f'Test Images: {len(image_dataset)}')
    print(f'Mean IoU: {mean_iou:.4f}')
    print(f'Mean Precision: {torch.mean(precision):.4f}')
    print(f'Mean Recall: {torch.mean(recall):.4f}')
    print(f'Mean F1-Score: {torch.mean(F1):.4f}')
    print(f'Average Loss: {avg_loss:.4f}')
    
    if args.saveTest:
        print(f'Results saved to: {args.save_dir}')


def initialize_model(model_type, num_classes, bn_momentum):
    """
    Initialize the specified model architecture.
    
    Args:
        model_type (str): Model architecture name
        num_classes (int): Number of segmentation classes
        bn_momentum (float): Batch normalization momentum
    
    Returns:
        torch.nn.Module: Initialized model or None if unsupported
    
    Example:
        >>> model = initialize_model('segnet', 5, 0.1)
    """
    if model_type == 'segnet':
        return SegNet(bn_momentum, num_classes)
    elif model_type == 'unet':
        return UNet(n_channels=3, n_classes=num_classes, bilinear=True)
    elif model_type == 'resnet18':
        return ResNetUNet(n_class=num_classes, resnet_model=18)
    elif model_type == 'smp_UNet++':
        return smp.UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights=None,  # No pretrained weights for evaluation
            in_channels=3,
            classes=num_classes
        )
    elif model_type == 'smp_unet18':
        return smp.Unet(
            encoder_name="resnet18", 
            encoder_weights=None,
            in_channels=3,
            classes=num_classes
        )
    elif model_type == 'smp_DeepLabV3+':
        return smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes
        )
    elif model_type == 'smp_MANet':
        return smp.MAnet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes
        )
    else:
        return None


def validate(val_loader, model, criterion, key, evaluator, args):
    """
    Run evaluation on the validation/test dataset.
    
    Args:
        val_loader (DataLoader): Data loader for evaluation dataset
        model (nn.Module): Neural network model to evaluate
        criterion (nn.Module): Loss function for evaluation
        key (dict): Class ID to RGB mapping dictionary
        evaluator (utils.Evaluate): Evaluation metrics calculator
        args (argparse.Namespace): Command line arguments
    
    Returns:
        float: Average validation loss across all batches
    
    Example:
        >>> avg_loss = validate(test_loader, model, criterion, key, evaluator, args)
    """
    # Switch model to evaluation mode
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    # Dataset-specific normalization parameters
    if args.dataset == "synapse":
        img_mean = [0.425, 0.304, 0.325]
        img_std = [0.239, 0.196, 0.202]
    elif args.dataset == "cholec":
        img_mean = [0.337, 0.212, 0.182]
        img_std = [0.278, 0.218, 0.185]
    elif args.dataset == "sarrarp50":
        img_mean = [0.485, 0.456, 0.406]  # ImageNet defaults
        img_std = [0.229, 0.224, 0.225]
    else:
        # Default normalization
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]

    print(f"Processing {len(val_loader)} batches...")
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for i, (img, gt, label) in enumerate(val_loader):
            
            # ===================== DATA PREPROCESSING =====================
            
            # Apply normalization
            img = utils.normalize(img, torch.Tensor(img_mean), torch.Tensor(img_std))
            
            # Move data to GPU if available
            if use_gpu:
                img = img.cuda()
                label = label.cuda()

            # ===================== FORWARD PASS =====================
            
            # Get model predictions
            seg = model(img)
            
            # Calculate loss
            loss = criterion(seg, label)
            total_loss += loss.item()
            num_batches += 1

            # ===================== METRICS COMPUTATION =====================
            
            # Convert predictions to class indices
            seg_pred = torch.argmax(seg, dim=1)
            
            # Convert ground truth to one-hot for evaluation
            from torch.nn.functional import one_hot
            oneHotGT = one_hot(label, len(key)).permute(0, 3, 1, 2).float()
            
            # Add batch results to evaluator
            evaluator.addBatch(seg, oneHotGT, args)

            # ===================== VISUALIZATION =====================
            
            # Display/save sample predictions
            if args.saveTest:
                utils.displaySamples(
                    img, seg_pred, gt, use_gpu, key, 
                    saveSegs=True, epoch=0, imageNum=i, 
                    save_dir=args.save_dir, total_epochs=1
                )

            # Progress update
            if (i + 1) % 10 == 0 or i == 0:
                print(f'  Processed batch {i+1}/{len(val_loader)} - Loss: {loss.item():.4f}')

    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    print(f"\nEvaluation completed successfully!")
    return avg_loss


# ================================ SCRIPT ENTRY POINT ================================

if __name__ == '__main__':
    main()