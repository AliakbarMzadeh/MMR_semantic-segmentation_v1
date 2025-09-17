"""
ModelTraining.py - Semantic Segmentation Training Framework

This module implements a comprehensive training pipeline for semantic segmentation models
on surgical datasets, with specialized support for SAR-RARP50. It supports multiple 
architectures (U-Net, SegNet, ResNet-UNet, UNet++) and provides flexible training 
configurations with mixed loss functions and comprehensive evaluation metrics.

Key Features:
- Multi-architecture support (SegNet, U-Net, ResNet-UNet, segmentation_models_pytorch)
- Mixed loss functions (CrossEntropy + Dice Loss)
- Comprehensive data augmentation pipeline
- Automated checkpointing and model saving
- Real-time training monitoring with visualization
- Support for multiple datasets with automatic train/val/test splits

Author: Medical Robotics Research Team
Dataset: Optimized for SAR-RARP50 surgical dataset
"""

# ================================ IMPORTS ================================

# PyTorch core imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
import segmentation_models_pytorch as smp

# General purpose imports
import argparse
import os

# Scientific computing and visualization
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot
from tqdm import tqdm
import random

# Custom module imports
import utils
from utils import dice
from dice_loss import DiceLoss
from skimage.metrics import hausdorff_distance

# Model architecture imports
from UArchModel.unet import UNet
from UArchModel.segnet import SegNet
from UArchModel.resnet_unet import ResNetUNet

# Data loading imports
from SurgicalDataClass.dataloaders.SegNetDataLoaderV1_SAR import SegNetDataset

# ================================ ARGUMENT PARSER SETUP ================================

parser = argparse.ArgumentParser(description='Semantic Segmentation Training Parameters')

# DATA PROCESSING ARGUMENTS
parser.add_argument('--workers', default=0, type=int, 
                    help='number of data loading workers (default: 0)')
parser.add_argument('--data_dir', type=str, 
                    help='data directory with train, test, and trainval image folders')
parser.add_argument('--json_path', type=str, 
                    help='path to json file containing class information for segmentation')
parser.add_argument('--dataset', type=str, 
                    help='dataset title (options: synapse / cholec / miccaiSegOrgans / miccaiSegRefined / sarrarp50)')

# MODEL PARAMETERS
parser.add_argument('--model', default='segnet', type=str, 
                    help='model architecture for segmentation (default: segnet)')
parser.add_argument('--batchnorm_momentum', default=0.1, type=float, 
                    help='batchnorm momentum for segnet')

# TRAINING PARAMETERS
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N', 
                    help='print frequency (default:1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=20, type=int, metavar='N', 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainBatchSize', default=32, type=int, 
                    help='Training Mini-batch size (default: 32)')
parser.add_argument('--valBatchSize', default=27, type=int, 
                    help='ValidationMini-batch size (default: 27)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', 
                    help='initial learning rate')
parser.add_argument('--optimizer', type=str, 
                    help='optimizer for the semantic segmentation network')
parser.add_argument('--wd', default=0.001, type=float, 
                    help='weight decay factor for optimizer')
parser.add_argument('--dice_loss_factor', default=0.5, type=float, 
                    help='loss weight factor for dice loss')
parser.add_argument('--lr_steps', default=2, type=int, 
                    help='number of steps to take with StepLR')
parser.add_argument('--step_gamma', default=0.1, type=float, 
                    help='gamma decay factor when stepping the Learning Rate')
parser.add_argument('--resnetModel', default=18, type=float, 
                    help='resnet model number')
parser.add_argument('--differential_lr', default=False, type=bool, 
                    help='use differential learning rate for pretrained encoder layers')

# IMAGE PROCESSING PARAMETERS
parser.add_argument('--resizedHeight', default=256, type=int, 
                    help='height of the input image to the network')
parser.add_argument('--resizedWidth', default=256, type=int, 
                    help='width of the resized image to the network')
parser.add_argument('--cropSize', default=256, type=int, 
                    help='height/width of the resized crop to the network')
parser.add_argument('--display_samples', default="False", type=str, 
                    help='Display samples during training / validation')
parser.add_argument('--save_samples', default="True", type=str, 
                    help='Save samples during final validation epoch')
parser.add_argument('--full_res_validation', default="False", type=str, 
                    help='Whether to validate your network on HD (1080x1920) Full-Resolution Images')

# EVALUATION PARAMETERS
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', 
                    help='evaluate model on validation set')

# OUTPUT AND SAVING PARAMETERS
parser.add_argument('--save_dir', dest='save_dir', 
                    help='The directory used to save the trained models', 
                    default='save_temp', type=str)
parser.add_argument('--seg_save_dir', dest='seg_save_dir', 
                    help='The directory used to save the segmentation results in the final epochs', 
                    type=str)
parser.add_argument('--saveSegs', default="True", type=str, 
                    help='Saves the validation/test images if True')

# ================================ HARDWARE SETUP ================================

# GPU detection and configuration
use_gpu = torch.cuda.is_available()
curr_device = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(curr_device)
device = torch.device('cuda' if use_gpu else 'cpu')

print("CUDA AVAILABLE:", use_gpu, flush=True)
print("CURRENT DEVICE:", curr_device, torch.cuda.device(curr_device), flush=True)
print("DEVICE NAME:", device_name, flush=True)

# ================================ REPRODUCIBILITY SETUP ================================

# Set seeds for reproducible results across multiple runs
seed = 6210  # Options: [6210, 2021, 3005]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================================ MAIN FUNCTIONS ================================

def main():
    """
    Main training function that orchestrates the entire training pipeline.
    
    This function handles:
    - Argument parsing and validation
    - Data loading and preprocessing setup
    - Model initialization and configuration
    - Loss function and optimizer setup
    - Training loop execution
    - Model checkpointing and result visualization
    
    Returns:
        None: Saves trained models and generates training curves
    
    Raises:
        ValueError: If invalid arguments are provided
        FileNotFoundError: If data directories or checkpoint files don't exist
    """
    # ===================== SETUP AND CONFIGURATION =====================
    
    global args
    args = parser.parse_args()
    print(f"args: {args}")

    # Logger setup for training monitoring
    log_path = os.path.join(args.save_dir, "train.log")
    logger = utils.get_logger("model", log_path)
    logger.info(f"args: {args}")

    image_size = [args.resizedHeight, args.resizedWidth]

    # ===================== DATA AUGMENTATION CONFIGURATION =====================
    
    # Configure data augmentation parameters
    rotate, horizontal_flip, vertical_flip = True, True, True
    logger.info(f"Data Augmentations: rotate={rotate}, horizontal_flip={horizontal_flip}, vertical_flip={vertical_flip}")

    # Dataset splits configuration (SAR-RARP50 has train/val/test, others have train/test)
    dataset_splits = ['train', 'val', 'test'] if args.dataset == 'sarrarp50' else ['train', 'test']
    
    # ===================== DATASET AND DATALOADER SETUP =====================
    
    # Initialize datasets for each split with augmentation parameters
    image_datasets = {x: SegNetDataset(os.path.join(args.data_dir, x), args.cropSize, args.json_path, x, args.dataset, 
                      image_size, rotate=rotate, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip, 
                      full_res_validation=args.full_res_validation) for x in dataset_splits}

    # Create dataloaders with appropriate batch sizes and settings
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.trainBatchSize if x == 'train' else args.valBatchSize,
                                                  shuffle=True if x == 'train' else False,
                                                  num_workers=args.workers,
                                                  pin_memory=True,
                                                  drop_last=True)
                   for x in dataset_splits}

    # Log dataset information
    dataset_sizes = {x: len(image_datasets[x]) for x in dataset_splits}
    logger.info(f"# of Training Images: {dataset_sizes['train']}")
    if 'val' in dataset_sizes:
        logger.info(f"# of Validation Images: {dataset_sizes['val']}")
    if 'test' in dataset_sizes:
        logger.info(f"# of Test Images: {dataset_sizes['test']}")

    # ===================== CLASS INFORMATION EXTRACTION =====================
    
    # Extract class information from dataset for model configuration
    classes = image_datasets['train'].classes
    print("\nCLASSES:", classes, "\n", flush=True)
    key = utils.disentangleKey(classes)
    print("KEY", key, "\n", flush=True)
    num_classes = len(key)
    print("NUM CLASSES:", num_classes, "\n", flush=True)
    
    # ===================== MODEL INITIALIZATION =====================
    
    # Initialize model based on specified architecture
    if args.model == 'segnet':
        model = SegNet(args.batchnorm_momentum, num_classes)
        model = model.to(device)
    elif args.model == 'unet':
        model = UNet(n_channels=3, n_classes=num_classes, bilinear=True)
        model = model.to(device)
    elif "resnet18" in args.model:
        model = ResNetUNet(n_class=num_classes, resnet_model=18)
        model = model.to(device)
    elif args.model == 'smp_UNet++':
        model = smp.UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        model = model.to(device)
    elif args.model == 'smp_unet18':
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        model = model.to(device)
    elif args.model == "smp_DeepLabV3+":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        model = model.to(device)
    elif args.model == "smp_MANet":
        model = smp.MAnet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        model = model.to(device)
    else:
        return "Model not available!"
    
    # Log model input dimensions
    if args.cropSize != -1:
        print(f"Model input size: (3, {args.cropSize}, {args.cropSize})")
    else:
        print(f"Model input size: (3, {args.resizedHeight}, {args.resizedWidth})")
        print("Model summary disabled due to compatibility issue")

    # ===================== DATASET-SPECIFIC NORMALIZATION PARAMETERS =====================
    
    # Set dataset-specific normalization parameters (computed from training data)
    if args.dataset == "synapse":
        image_mean = [0.425, 0.304, 0.325]  # mean [R, G, B]
        image_std = [0.239, 0.196, 0.202]   # standard deviation [R, G, B]
    elif args.dataset == "cholec":
        image_mean = [0.337, 0.212, 0.182]
        image_std = [0.278, 0.218, 0.185]
    elif args.dataset == "sarrarp50":
        # ImageNet defaults - update after computing dataset-specific statistics
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
    else:
        return "Dataset not available!"

    # ===================== CHECKPOINT RESUMING =====================
    
    # Resume training from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}

            new_weights = {}
            
            print("Pretrained Weight Dict:\n")

            for n, p in pretrained_dict.items():
                print(n, p.shape)

            # Load all parameters except conv_last (which is retrained from scratch)
            for n, p in pretrained_dict.items():
                if n.split(".")[0] != "conv_last":
                    new_weights[n] = p.data
            
            # Keep conv_last parameters from current model
            for n, p in model.named_parameters():
                if n.split(".")[0] == "conv_last":
                    new_weights[n] = p.data
                        
            model.state_dict().update(new_weights)
            model.load_state_dict(new_weights, strict=False)
            logger.info(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # ===================== LOSS FUNCTION SETUP =====================
    
    # Configure loss functions based on dice loss factor
    if args.dice_loss_factor == -1:
        logger.info("Training with CE Loss Only")
        dice_loss = None
    elif args.dice_loss_factor >= 0.0 and args.dice_loss_factor <= 1.0 and args.dataset == "synapse":
        logger.info(f"dice loss factor: {args.dice_loss_factor}")
        dice_loss = DiceLoss(ignore_index=21)  # Synapse dataset specific
    elif args.dice_loss_factor >= 0.0 and args.dice_loss_factor <= 1.0:
        logger.info(f"dice loss factor: {args.dice_loss_factor}")
        dice_loss = DiceLoss()  # Standard dice loss for other datasets
    else:
        raise ValueError("args.dice_loss_factor must be a float value from 0.0 to 1.0")
    
    # ===================== OPTIMIZATION SETUP =====================
    
    # Initialize CrossEntropy loss
    if args.dataset == "synapse":
        criterion = nn.CrossEntropyLoss(ignore_index=21)
    else:
        criterion = nn.CrossEntropyLoss()

    # Configure optimizer based on differential learning rate setting
    if args.differential_lr == False:
        # Standard optimization - same learning rate for all parameters
        if args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            logger.info(f"{args.optimizer} Optimizer LR = {args.lr} with WD = {args.wd}")
        elif args.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
            logger.info(f"{args.optimizer} Optimizer LR = {args.lr} with WD = {args.wd}")
        elif args.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            logger.info(f"{args.optimizer} Optimizer LR = {args.lr} with WD = {args.wd} and Momentum = 0.9")
    else:
        # Differential learning rate - different rates for encoder and decoder
        reduced_lr = args.lr * 0.1
        if args.optimizer == "Adam":
            optimizer = optim.Adam([
                        {'params': model.base_model.parameters(), 'lr': args.lr}], lr=reduced_lr, weight_decay=args.wd)
        elif args.optimizer == "SGD":
            optimizer = optim.SGD([{'params': model.base_model.parameters(), 'lr': args.lr}], lr=reduced_lr, weight_decay=args.wd, momentum=0.9)
        
        logger.info(f"{args.optimizer} Optimizer LR = {args.lr} for Base Model ({args.model}) Parameters and LR = {reduced_lr} for UNet Decoder Parameters with WD = {args.wd}")

    # Learning rate scheduler setup
    if args.lr_steps > 0:
        step_size = int(args.epochs // (args.lr_steps + 1))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args.step_gamma)
        logger.info(f"StepLR initialized with step size = {step_size} and gamma = 0.1")
    else:
        raise ValueError("args.lr_steps must be > 0")

    # Move loss function to GPU if available
    if use_gpu:
        criterion.cuda()

    # ===================== EVALUATION SETUP =====================
    
    # Initialize evaluation metrics tracker
    evaluator = utils.Evaluate(key, use_gpu)

    # Initialize metric tracking lists
    train_losses = []
    val_losses = []
    total_iou = []
    total_precision = []
    total_recall = []
    total_f1 = []

    # Training state tracking
    logger.info(f"Training Starting")
    best_f1 = 0
    best_epoch = 0

    # Determine validation split (use 'val' if available, otherwise 'test')
    val_split = 'val' if 'val' in dataloaders else 'test'

    # ===================== TRAINING LOOP =====================
    
    for epoch in range(args.epochs):
        
        # Enhanced metrics computation every 25 epochs (includes Dice & Hausdorff)
        if (epoch+1) == 1 or (epoch+1) % 25 == 0:
            train_loss, train_dice_coeff, train_haus_dist = train(
                dataloaders['train'], model, criterion, dice_loss, optimizer, 
                scheduler, epoch, key, train_losses, image_mean, image_std, logger, args)
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss}, Avg. Train DC={train_dice_coeff}, Avg. Train HD={train_haus_dist}, LR={optimizer.param_groups[0]['lr']}")

            val_loss, val_dice_coeff, val_haus_dist = validate(
                dataloaders[val_split], model, criterion, dice_loss, epoch, key, 
                evaluator, val_losses, image_mean, image_std, logger, args)
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Val Loss={val_loss}, Avg. Val DC={val_dice_coeff}, Avg. Val HD={val_haus_dist}, LR={optimizer.param_groups[0]['lr']}")
        else:
            # Standard training and validation (loss only)
            train_loss = train(
                dataloaders['train'], model, criterion, dice_loss, optimizer, 
                scheduler, epoch, key, train_losses, image_mean, image_std, logger, args)
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss}, LR={optimizer.param_groups[0]['lr']}")

            val_loss = validate(
                dataloaders[val_split], model, criterion, dice_loss, epoch, key, 
                evaluator, val_losses, image_mean, image_std, logger, args)
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Val Loss={val_loss}, LR={optimizer.param_groups[0]['lr']}")

        # Update learning rate
        scheduler.step()
        
        # ===================== METRICS CALCULATION AND LOGGING =====================
        
        print(f'\n>>>>>>>>>>>>>>>>>> Evaluation Metrics {epoch+1}/{args.epochs} <<<<<<<<<<<<<<<<<', flush=True)
        
        # Intersection over Union (IoU)
        IoU = evaluator.getIoU()
        print(f"Mean IoU = {torch.mean(IoU)}", flush=True)
        if (epoch + 1) % 25 == 0:
            print(f"Class-Wise IoU = {IoU}", flush=True)
        total_iou.append(torch.mean(IoU))

        # Precision, Recall, F1-Score
        PRF1 = evaluator.getPRF1()
        precision, recall, F1 = PRF1[0], PRF1[1], PRF1[2]

        print(f"Mean Precision = {torch.mean(precision)}", flush=True)
        total_precision.append(torch.mean(precision))

        print(f"Mean Recall = {torch.mean(recall)}", flush=True)
        total_recall.append(torch.mean(recall))

        print(f"Mean F1 = {torch.mean(F1)}", flush=True)
        if (epoch + 1) % 25 == 0:
            print(f"Class-Wise F1 = {F1}", flush=True)
        total_f1.append(torch.mean(F1))

        # ===================== MODEL CHECKPOINTING =====================
        
        # Save best model based on F1-score
        if torch.mean(F1) > best_f1:
            best_f1 = torch.mean(F1)
            best_epoch = epoch + 1

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(args.save_dir, f"{args.model}_{args.dataset}_bs{args.trainBatchSize}lr{args.lr}e{args.epochs}_checkpoint"))
            
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Mean IoU={torch.mean(IoU)}, Mean Precision={torch.mean(precision)}, Mean Recall={torch.mean(recall)}, Mean F1={torch.mean(F1)} (Best) (Saved)\n")
        else:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Mean IoU={torch.mean(IoU)}, Mean Precision={torch.mean(precision)}, Mean Recall={torch.mean(recall)}, Mean F1={torch.mean(F1)}\n")

        # Reset evaluator for next epoch
        evaluator.reset()

    # ===================== TRAINING COMPLETION SUMMARY =====================
    
    logger.info(f"(Training Complete): Best Mean F1={best_f1}, Best Epoch={best_epoch}")

    # ===================== RESULT VISUALIZATION =====================
    
    # Generate and save loss curves
    plt.plot(range(1, args.epochs+1), train_losses, color='blue')
    plt.plot(range(1, args.epochs+1), val_losses, color='black')
    plt.legend(["Train Loss", "Val Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves for {args.model} on {args.dataset} (bs{args.trainBatchSize}/lr{args.lr}/e{args.epochs})")
    figure_name = f"loss_{args.model}_{args.dataset}_bs{args.trainBatchSize}lr{args.lr}e{args.epochs}.png"
    plt.savefig(f"{args.save_dir}/{figure_name}")
    logger.info(f"Loss Curve saved to {args.save_dir}/{figure_name}")

    # Generate and save accuracy curves
    plt.clf()
    plt.plot(range(1, args.epochs+1), total_iou, color='blue')
    plt.plot(range(1, args.epochs+1), total_precision, color='red')
    plt.plot(range(1, args.epochs+1), total_recall, color='magenta')
    plt.plot(range(1, args.epochs+1), total_f1, color='black')
    plt.legend(["Mean IoU", "Mean Precision", "Mean Recall", "Mean F1"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curves for {args.model} on {args.dataset} (bs{args.trainBatchSize}/lr{args.lr}/e{args.epochs})")
    figure_name = f"acc_{args.model}_{args.dataset}_bs{args.trainBatchSize}lr{args.lr}e{args.epochs}.png"
    plt.savefig(f"{args.save_dir}/{figure_name}")
    logger.info(f"Accuracy Curve saved to {args.save_dir}/{figure_name}")


def train(train_loader, model, criterion, dice_loss, optimizer, scheduler, epoch, key, losses, img_mean, img_std, logger, args):
    """
    Execute one training epoch with forward and backward passes.
    
    Args:
        train_loader (DataLoader): Training data loader
        model (nn.Module): Neural network model
        criterion (nn.Module): Cross-entropy loss function
        dice_loss (DiceLoss or None): Dice loss function
        optimizer (torch.optim.Optimizer): Model optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        epoch (int): Current epoch number
        key (dict): Class ID to RGB mapping dictionary
        losses (list): List to store training losses
        img_mean (list): Dataset mean values [R, G, B]
        img_std (list): Dataset standard deviation values [R, G, B]
        logger (logging.Logger): Training logger
        args (argparse.Namespace): Training arguments
    
    Returns:
        float or tuple: Training loss (and optionally dice coefficient, hausdorff distance)
    
    Example:
        >>> train_loss = train(train_loader, model, criterion, dice_loss, 
        ...                   optimizer, scheduler, 0, key, [], img_mean, img_std, logger, args)
    """
    # Switch model to training mode (enables dropout, batch normalization updates)
    model.train()

    # Initialize progress bar for training batches
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader))

    # Initialize metrics tracking variables
    total_train_loss = 0
    total_dice_coeff = 0
    total_haus_dist = 0
    avg_dice_coeff = 0
    avg_haus_dist = 0
    total_samples = args.trainBatchSize

    # ===================== BATCH TRAINING LOOP =====================
    
    for i, (img, gt, label) in train_loop:

        # Handle different input dimensions based on cropping configuration
        if args.cropSize != -1:
            img = img.view(-1, 3, args.cropSize, args.cropSize)
            gt = gt.view(-1, 3, args.cropSize, args.cropSize)
        else:
            img = img.view(-1, 3, args.resizedHeight, args.resizedWidth)
            gt = gt.view(-1, 3, args.resizedHeight, args.resizedWidth)
        
        # Apply normalization using dataset-specific statistics
        img = utils.normalize(img, torch.Tensor(img_mean), torch.Tensor(img_std))

        # Move data to GPU if available
        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        # ===================== FORWARD PASS =====================
        
        # Compute model predictions
        seg = model(img)

        # ===================== LOSS CALCULATION =====================
        
        # Calculate loss based on dataset and dice loss configuration
        if args.dataset == "synapse":
            if args.dice_loss_factor != -1 and dice_loss != None:
                loss = (args.dice_loss_factor * dice_loss(seg, label)) + ((1 - args.dice_loss_factor) * criterion(seg, label))
            else:
                loss = criterion(seg, label)
        else:
            if args.dice_loss_factor != -1 and dice_loss != None:
                loss = (args.dice_loss_factor * dice_loss(seg, label)) + ((1 - args.dice_loss_factor) * criterion(seg, label))
            else:
                loss = criterion(seg, label)

        total_train_loss += loss.mean().item()

        # ===================== BACKWARD PASS AND OPTIMIZATION =====================
        
        # Clear gradients (PyTorch recommended approach)
        for param in model.parameters():
            param.grad = None

        # Compute gradients
        loss.backward()
        
        # Update model parameters
        optimizer.step()

        # Convert model output to class predictions
        seg = torch.argmax(seg, dim=1)

        # ===================== DETAILED METRICS COMPUTATION (EVERY 25 EPOCHS) =====================
        
        # Compute Dice coefficient and Hausdorff distance for detailed analysis
        if (epoch+1) == 1 or (epoch+1) % 25 == 0:
            # Process each image in the batch
            for seg_im, label_im in zip(seg, label):
                # Convert to one-hot encoding for metric calculation
                seg_im, label_im = one_hot(seg_im, len(key)), one_hot(label_im, len(key))
                seg_im, label_im = seg_im.cpu(), label_im.cpu()
                seg_im, label_im = seg_im.permute(2, 0, 1), label_im.permute(2, 0, 1)
                
                # Calculate Dice coefficient
                total_dice_coeff += dice(seg_im.data, label_im.data)

                # Handle Synapse dataset specific requirements
                if args.dataset == "synapse":
                    seg_im, label_im = seg_im[:21], label_im[:21]

                # Calculate Hausdorff distance for each class
                for seg_slice, label_slice in zip(seg_im, label_im):
                    seg_slice, label_slice = seg_slice.numpy(), label_slice.numpy()
                    # Handle infinite distances by capping at 1000
                    total_haus_dist += 1000 if hausdorff_distance(seg_slice, label_slice) == np.inf else hausdorff_distance(seg_slice, label_slice)
            
            # Calculate running averages
            avg_dice_coeff = total_dice_coeff / total_samples
            avg_haus_dist = total_haus_dist / total_samples
            total_samples += args.trainBatchSize

            # Update progress bar with detailed metrics
            train_loop.set_postfix(avg_loss=total_train_loss / (i + 1), 
                                 avg_dice=avg_dice_coeff, 
                                 avg_haus_dist=avg_haus_dist)
        else:
            # Update progress bar with loss only
            train_loop.set_postfix(avg_loss=total_train_loss / (i + 1))

        # Update progress bar description
        train_loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")
        
        # Display sample predictions if requested
        if args.display_samples == "True":
            utils.displaySamples(img, seg, gt, use_gpu, key, False, epoch, i)
        
    # Store epoch loss
    losses.append(total_train_loss / len(train_loop))

    # Return appropriate values based on metrics computation level
    if (epoch+1) == 1 or (epoch+1) % 25 == 0:
        return total_train_loss/len(train_loop), avg_dice_coeff, avg_haus_dist
    else:
        return total_train_loss/len(train_loop)


@torch.no_grad()  # Disable gradient computation for efficiency
def validate(val_loader, model, criterion, dice_loss, epoch, key, evaluator, losses, img_mean, img_std, logger, args):
    """
    Execute validation/evaluation on the validation dataset.
    
    Args:
        val_loader (DataLoader): Validation data loader
        model (nn.Module): Neural network model to evaluate
        criterion (nn.Module): Cross-entropy loss function
        dice_loss (DiceLoss or None): Dice loss function
        epoch (int): Current epoch number
        key (dict): Class ID to RGB mapping dictionary
        evaluator (utils.Evaluate): Evaluation metrics calculator
        losses (list): List to store validation losses
        img_mean (list): Dataset mean values [R, G, B]
        img_std (list): Dataset standard deviation values [R, G, B]
        logger (logging.Logger): Training logger
        args (argparse.Namespace): Training arguments
    
    Returns:
        float or tuple: Validation loss (and optionally dice coefficient, hausdorff distance)
    
    Example:
        >>> val_loss = validate(val_loader, model, criterion, dice_loss, 0, key, 
        ...                    evaluator, [], img_mean, img_std, logger, args)
    """
    # Switch model to evaluation mode (disables dropout, batch norm updates)
    model.eval()

    # Initialize metrics tracking variables
    total_val_loss = 0
    total_dice_coeff = 0
    total_haus_dist = 0
    avg_dice_coeff = 0
    avg_haus_dist = 0
    total_samples = args.valBatchSize

    # Initialize progress bar for validation batches
    val_loop = tqdm(enumerate(val_loader), total=len(val_loader))

    # ===================== VALIDATION LOOP =====================
    
    for i, (img, gt, label) in val_loop:
        # ===================== DATA PREPROCESSING =====================
        
        # Apply normalization using dataset-specific statistics
        img = utils.normalize(img, torch.Tensor(img_mean), torch.Tensor(img_std))
        
        # Convert ground truth to one-hot encoding for evaluation
        oneHotGT = one_hot(label, len(key)).permute(0, 3, 1, 2)

        # Move data to GPU if available
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
            oneHotGT = oneHotGT.cuda()

        # ===================== FORWARD PASS =====================
        
        # Compute model predictions (no gradient computation)
        seg = model(img)

        # ===================== LOSS CALCULATION =====================
        
        # Calculate validation loss using same configuration as training
        if args.dataset == "synapse":
            if args.dice_loss_factor != -1 and dice_loss != None:
                loss = (args.dice_loss_factor * dice_loss(seg, label)) + ((1 - args.dice_loss_factor) * criterion(seg, label))
            else:
                loss = criterion(seg, label)
        else:
            if args.dice_loss_factor != -1 and dice_loss != None:
                loss = (args.dice_loss_factor * dice_loss(seg, label)) + ((1 - args.dice_loss_factor) * criterion(seg, label))
            else:
                loss = criterion(seg, label)

        total_val_loss += loss.mean().item()

        # ===================== METRICS ACCUMULATION =====================
        
        # Add batch results to evaluator for IoU, Precision, Recall, F1 calculation
        evaluator.addBatch(seg, oneHotGT, args)

        # Convert predictions to class indices
        seg = torch.argmax(seg, dim=1)

        # ===================== DETAILED METRICS COMPUTATION (EVERY 25 EPOCHS) =====================
        
        # Compute detailed metrics for comprehensive evaluation
        if (epoch+1) == 1 or (epoch+1) % 25 == 0:
            # Process each image in the batch
            for seg_im, label_im in zip(seg, label):
                # Convert to one-hot encoding for metric calculation
                seg_im, label_im = one_hot(seg_im, len(key)), one_hot(label_im, len(key))
                seg_im, label_im = seg_im.cpu(), label_im.cpu()
                seg_im, label_im = seg_im.permute(2, 0, 1), label_im.permute(2, 0, 1)
                
                # Calculate Dice coefficient
                total_dice_coeff += dice(seg_im.data, label_im.data)

                # Handle Synapse dataset specific requirements
                if args.dataset == "synapse":
                    seg_im, label_im = seg_im[:21], label_im[:21]

                # Calculate Hausdorff distance for each class
                for seg_slice, label_slice in zip(seg_im, label_im):
                    seg_slice, label_slice = seg_slice.numpy(), label_slice.numpy()
                    # Handle infinite distances by capping at 1000
                    total_haus_dist += 1000 if hausdorff_distance(seg_slice, label_slice) == np.inf else hausdorff_distance(seg_slice, label_slice)
            
            # Calculate running averages
            avg_dice_coeff = total_dice_coeff / total_samples
            avg_haus_dist = total_haus_dist / total_samples
            total_samples += args.trainBatchSize

            # Update progress bar with detailed metrics
            val_loop.set_postfix(avg_loss=total_val_loss / (i + 1), 
                                avg_dice=avg_dice_coeff, 
                                avg_haus_dist=avg_haus_dist)
        else:
            # Update progress bar with loss only
            val_loop.set_postfix(avg_loss=total_val_loss / (i + 1))
        
        # Update progress bar description
        val_loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")

        # ===================== SAMPLE VISUALIZATION =====================
        
        # Display or save sample predictions based on configuration
        if args.display_samples == "True":
            utils.displaySamples(img, seg, gt, use_gpu, key, saveSegs=args.saveSegs, 
                               epoch=epoch, imageNum=i, save_dir=args.seg_save_dir, 
                               total_epochs=args.epochs)
        elif args.display_samples == "False" and args.save_samples == "True" and (epoch+1) == args.epochs:
            # Save samples only in final epoch if display is disabled
            utils.displaySamples(img, seg, gt, use_gpu, key, saveSegs=args.saveSegs, 
                               epoch=epoch, imageNum=i, save_dir=args.seg_save_dir, 
                               total_epochs=args.epochs)

    # Store epoch validation loss
    losses.append(total_val_loss / len(val_loop))

    # Return appropriate values based on metrics computation level
    if (epoch+1) == 1 or (epoch+1) % 25 == 0:
        return total_val_loss/len(val_loop), avg_dice_coeff, avg_haus_dist
    else:
        return total_val_loss/len(val_loop)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save model checkpoint to disk for resuming training or inference.
    
    Args:
        state (dict): Dictionary containing model state, optimizer state, and epoch info
            Expected keys: 'epoch', 'state_dict', 'optimizer'
        filename (str): Path where checkpoint will be saved
    
    Returns:
        None: Saves checkpoint file to specified location
    
    Example:
        >>> checkpoint_state = {
        ...     'epoch': 50,
        ...     'state_dict': model.state_dict(),
        ...     'optimizer': optimizer.state_dict()
        ... }
        >>> save_checkpoint(checkpoint_state, 'model_checkpoint.tar')
    """
    torch.save(state, filename)


# ================================ SCRIPT ENTRY POINT ================================

if __name__ == '__main__':
    main()