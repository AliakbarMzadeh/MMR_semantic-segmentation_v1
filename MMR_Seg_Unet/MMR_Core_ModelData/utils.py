"""
utils.py - Utility Functions for Semantic Segmentation

This module provides essential utility functions for semantic segmentation tasks,
including evaluation metrics computation, data visualization, image processing,
and logging capabilities. It serves as the backbone for training and evaluation
pipelines in surgical image segmentation.

Key Components:
- Evaluate class: Comprehensive metrics computation (IoU, Precision, Recall, F1)
- Visualization utilities: Sample display and result saving
- Data processing: Normalization, encoding, and format conversion
- Logging utilities: Professional logging setup
- Mathematical functions: Dice coefficient and other metrics

Author: Medical Robotics Research Team
Compatible with: PyTorch-based segmentation models
"""

# ================================ IMPORTS ================================

import torch
import numpy as np
import cv2
import os
import logging
from torch.nn.functional import one_hot

# ================================ EVALUATION UTILITIES ================================

class Evaluate():
    """
    Comprehensive evaluation metrics calculator for semantic segmentation.
    
    This class computes standard segmentation metrics including IoU (Intersection over Union),
    Precision, Recall, and F1-score. It accumulates predictions and ground truth over
    multiple batches and provides class-wise and mean metrics.
    
    Code adapted from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/eval/iouEval.py
    
    Attributes:
        num_classes (int): Number of segmentation classes
        key (dict): Class ID to RGB color mapping
        use_gpu (bool): Whether GPU acceleration is available
        tp (torch.Tensor): True positive counts per class
        fp (torch.Tensor): False positive counts per class  
        fn (torch.Tensor): False negative counts per class
    """

    def __init__(self, key, use_gpu):
        """
        Initialize the evaluation metrics calculator.
        
        Args:
            key (dict): Dictionary mapping class IDs to RGB color arrays
            use_gpu (bool): Whether GPU acceleration is available
        
        Example:
            >>> key = {0: np.array([0,0,0]), 1: np.array([255,0,0])}
            >>> evaluator = Evaluate(key, use_gpu=True)
        """
        self.num_classes = len(key)
        self.key = key
        self.use_gpu = use_gpu
        self.reset()

    def reset(self):
        """
        Reset all accumulated metrics to zero.
        
        This should be called at the beginning of each epoch to start
        fresh metric computation.
        
        Returns:
            None
        """
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives  
        self.fn = 0  # False Negatives

    def addBatch(self, seg, gt, args):
        """
        Add a batch of predictions and ground truth for metrics computation.
        
        This method processes model predictions and ground truth masks to accumulate
        true positives, false positives, and false negatives for each class.
        
        Args:
            seg (torch.Tensor): Model predictions with shape [batch_size, num_classes, H, W]
                                Should be softmax output before argmax
            gt (torch.Tensor): Ground truth one-hot encoded with shape [batch_size, num_classes, H, W]
            args (argparse.Namespace): Training arguments containing dataset information
        
        Returns:
            None: Updates internal TP, FP, FN counters
        
        Example:
            >>> # seg: [4, 5, 256, 256] - batch of 4 images, 5 classes
            >>> # gt:  [4, 5, 256, 256] - corresponding ground truth
            >>> evaluator.addBatch(seg, gt, args)
        """
        # Handle Synapse dataset specific requirements (limit to 21 classes)
        if args.dataset == "synapse":
            seg = seg[:, 0:21, :, :]
            gt = gt[:, 0:21, :, :]

        # Convert model predictions to class indices, then to one-hot
        seg = torch.argmax(seg, dim=1)  # [batch_size, H, W]
        seg = one_hot(seg, self.num_classes).permute(0, 3, 1, 2)  # [batch_size, num_classes, H, W]
        
        # Ensure both tensors are float for computation
        seg = seg.float()
        gt = gt.float()

        # Move tensors to GPU if not using GPU (this logic seems inverted - keeping as is for compatibility)
        if not self.use_gpu:
            seg = seg.cuda()
            gt = gt.cuda()

        # ===================== COMPUTE CONFUSION MATRIX ELEMENTS =====================
        
        # True Positives: prediction and ground truth both say class is present
        tpmult = seg * gt
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        
        # False Positives: prediction says class is present, ground truth says it's not
        fpmult = seg * (1 - gt)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        
        # False Negatives: prediction says class is not present, ground truth says it is
        fnmult = (1 - seg) * gt
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        # Accumulate metrics on CPU for numerical stability
        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        """
        Compute Intersection over Union (IoU) for each class.
        
        IoU is calculated as: TP / (TP + FP + FN)
        
        Returns:
            torch.Tensor: IoU scores for each class [num_classes]
        
        Example:
            >>> iou_scores = evaluator.getIoU()
            >>> mean_iou = torch.mean(iou_scores)
            >>> print(f"Mean IoU: {mean_iou:.4f}")
        """
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15  # Add epsilon to avoid division by zero
        iou = num / den
        return iou

    def getPRF1(self):
        """
        Compute Precision, Recall, and F1-score for each class.
        
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)  
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Returns:
            tuple: (precision, recall, f1) tensors each with shape [num_classes]
        
        Example:
            >>> precision, recall, f1 = evaluator.getPRF1()
            >>> print(f"Mean F1: {torch.mean(f1):.4f}")
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-15
        
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = (2 * precision * recall) / (precision + recall + epsilon)

        return precision, recall, f1

# ================================ LOGGING UTILITIES ================================

def get_logger(name, log_path=None):
    """
    Create a configured logger for training and evaluation monitoring.
    
    Sets up a logger with appropriate formatting and file handling for
    tracking training progress, metrics, and debug information.
    
    Args:
        name (str): Logger name identifier
        log_path (str, optional): Path to log file. If None, only console output
    
    Returns:
        logging.Logger: Configured logger instance
    
    Example:
        >>> logger = get_logger("training", "/path/to/train.log")
        >>> logger.info("Training started with 50 epochs")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatter with timestamp and message
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    # Add file handler if log path is provided
    if log_path:
        handler = logging.FileHandler(log_path, 'w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# ================================ VISUALIZATION UTILITIES ================================

def displaySamples(img, generated, gt, use_gpu, key, saveSegs, epoch, imageNum, save_dir=None, total_epochs=None):
    """
    Display and save sample predictions alongside ground truth.
    
    Creates a concatenated visualization showing original image, model prediction,
    and ground truth segmentation. Saves results during final epoch if specified.
    
    Args:
        img (torch.Tensor): Input image tensor [batch_size, 3, H, W]
        generated (torch.Tensor): Model predictions [batch_size, H, W] (class indices)
        gt (torch.Tensor): Ground truth image [batch_size, 3, H, W]
        use_gpu (bool): Whether tensors are on GPU
        key (dict): Class ID to RGB color mapping
        saveSegs (str or bool): Whether to save segmentation results ("True"/"False")
        epoch (int): Current epoch number
        imageNum (int): Image number within epoch
        save_dir (str, optional): Directory to save results
        total_epochs (int, optional): Total number of training epochs
    
    Returns:
        None: Saves visualization files to disk if requested
    
    Example:
        >>> displaySamples(img, pred, gt, True, class_key, "True", 
        ...               49, 0, "results/", 50)
    """
    # Move tensors to CPU for processing
    if use_gpu:
        img = img.cpu()
        generated = generated.cpu()

    # ===================== PROCESS GROUND TRUTH =====================
    
    # Convert ground truth to displayable format
    gt = gt.numpy()
    gt = np.transpose(np.squeeze(gt[0, :, :, :]), (1, 2, 0))  # [H, W, 3]
    gt = gt.astype(np.uint8)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB) / 255.0

    # ===================== PROCESS PREDICTIONS =====================
    
    # Convert predictions from class indices to RGB colors
    generated = generated.data.numpy()
    generated = reverseOneHot(generated, key)
    generated = np.squeeze(generated[0]).astype(np.uint8)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB) / 255.0

    # ===================== PROCESS INPUT IMAGE =====================
    
    # Convert input image to displayable format
    img = img.data.numpy()
    img = np.transpose(np.squeeze(img[0]), (1, 2, 0))  # [H, W, 3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ===================== CREATE CONCATENATED VISUALIZATION =====================
    
    # Stack images horizontally: [Input | Prediction | Ground Truth]
    stacked = np.concatenate((img, generated, gt), axis=1)

    # ===================== SAVE RESULTS =====================
    
    # Save visualization during final epoch
    if saveSegs == "True" and (epoch + 1) == total_epochs:
        file_name = f'epoch_{epoch}_img_{imageNum}.png'
        save_path = os.path.join(save_dir, file_name)
        print(f"Saving visualization: {save_path}")
        
        # Create directory if it doesn't exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Save image (convert back to 0-255 range)
        cv2.imwrite(save_path, (stacked * 255).astype(np.uint8))

    # Note: GUI display skipped for headless environments (Colab)
    return

# ================================ DATA PROCESSING UTILITIES ================================

def disentangleKey(key):
    """
    Parse class information from JSON format to dictionary mapping.
    
    Converts JSON class definitions to a dictionary mapping class IDs to RGB color arrays.
    Handles string parsing of color values from JSON format.
    
    Args:
        key (list): List of dictionaries containing class information
                   Each dict should have 'id' and 'color' keys
                   Color format: "(R,G,B)" as string
    
    Returns:
        dict: Mapping of class_id (int) to RGB color array (np.ndarray)
    
    Example:
        >>> json_classes = [
        ...     {'id': '0', 'color': '(0,0,0)'},
        ...     {'id': '1', 'color': '(255,0,0)'}
        ... ]
        >>> color_map = disentangleKey(json_classes)
        >>> # Returns: {0: np.array([0,0,0]), 1: np.array([255,0,0])}
    """
    dKey = {}
    
    for i in range(len(key)):
        # Extract class ID
        class_id = int(key[i]['id'])
        
        # Parse color string: "(R,G,B)" -> [R, G, B]
        c = key[i]['color']
        c = c.split(',')
        c0 = int(c[0][1:])      # Remove leading '('
        c1 = int(c[1])          # Middle value
        c2 = int(c[2][:-1])     # Remove trailing ')'
        
        # Create color array
        color_array = np.asarray([c0, c1, c2])
        dKey[class_id] = color_array

    return dKey


def generateLabel4CE(gt, key):
    """
    Generate class labels for Cross Entropy Loss from RGB ground truth masks.
    
    Converts RGB ground truth images to integer class labels by matching
    pixel colors to class definitions. Used for CrossEntropy loss computation.
    
    Args:
        gt (torch.Tensor): Ground truth RGB images [batch_size, 3, H, W]
        key (dict): Mapping from class_id to RGB color arrays
    
    Returns:
        torch.LongTensor: Class labels [batch_size, H, W] with integer class IDs
    
    Example:
        >>> gt_rgb = torch.rand(2, 3, 256, 256)  # Batch of 2 images
        >>> labels = generateLabel4CE(gt_rgb, class_key)
        >>> # Returns: torch.LongTensor with shape [2, 256, 256]
    """
    batch = gt.numpy()
    label = []
    
    # Process each image in the batch
    for i in range(len(batch)):
        img = batch[i, :, :, :]
        img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
        catMask = np.zeros((img.shape[0], img.shape[1]))

        # Assign pixels to class IDs based on RGB color matching
        for k in range(len(key)):
            rgb = key[k]
            # Find pixels matching this class color
            mask = np.where(np.all(img == rgb, axis=2))
            catMask[mask] = k
        
        # Convert to tensor and add to batch
        catMaskTensor = torch.from_numpy(catMask).unsqueeze(0)
        label.append(catMaskTensor)

    # Concatenate all images in batch
    label = torch.cat(label, 0)
    return label.long()


def reverseOneHot(batch, key):
    """
    Convert class indices back to RGB images for visualization.
    
    Takes model predictions (class indices) and converts them to RGB images
    using the class color mapping. Used for visualization and result saving.
    
    Args:
        batch (np.ndarray): Batch of class index predictions [batch_size, H, W]
        key (dict): Mapping from class_id to RGB color arrays
    
    Returns:
        np.ndarray: RGB images [batch_size, H, W, 3] with colors for each class
    
    Example:
        >>> predictions = np.array([[[0, 1, 0], [1, 1, 0]]])  # 1x2x3 predictions
        >>> rgb_result = reverseOneHot(predictions, class_key)
        >>> # Returns RGB image with class colors
    """
    generated = []

    # Process each image in the batch
    for i in range(len(batch)):
        vec = batch[i]
        idxs = vec

        # Initialize RGB output image
        segSingle = np.zeros([idxs.shape[0], idxs.shape[1], 3])

        # Map each class index to its RGB color
        for k in range(len(key)):
            rgb = key[k]
            mask = idxs == k
            segSingle[mask] = rgb

        # Add batch dimension and append
        segMask = np.expand_dims(segSingle, axis=0)
        generated.append(segMask)
    
    # Concatenate all images
    generated = np.concatenate(generated)
    return generated


def generateOneHot(gt, key):
    """
    Generate one-hot encoded tensors from RGB ground truth images.
    
    Converts RGB ground truth masks to one-hot encoded format where each
    class gets its own channel. Used for loss computation and evaluation.
    
    Args:
        gt (torch.Tensor): Ground truth RGB images [batch_size, 3, H, W]
        key (dict): Mapping from class_id to RGB color arrays
    
    Returns:
        torch.Tensor: One-hot encoded tensor [batch_size, num_classes, H, W]
    
    Example:
        >>> gt_rgb = torch.rand(1, 3, 256, 256)
        >>> onehot = generateOneHot(gt_rgb, class_key)
        >>> # Returns: [1, num_classes, 256, 256]
    """
    batch = gt.numpy()
    oneHot = None
    
    # Process each image in the batch
    for i in range(len(batch)):
        img = batch[i, :, :, :]
        img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
        
        # Create one-hot channels for each class
        for k in range(len(key)):
            catMask = np.zeros((img.shape[0], img.shape[1]))
            rgb = key[k]
            
            # Find pixels matching this class
            mask = np.where(np.all(img == rgb, axis=-1))
            catMask[mask] = 1

            # Convert to tensor
            catMaskTensor = torch.from_numpy(catMask).unsqueeze(0)
            
            # Accumulate channels
            if oneHot is not None:
                oneHot = torch.cat((oneHot, catMaskTensor), 0)
            else:
                oneHot = catMaskTensor

    # Reshape to proper batch format
    label = oneHot.view(len(batch), len(key), img.shape[0], img.shape[1])
    return label


def normalize(batch, mean, std):
    """
    Normalize a batch of images using provided mean and standard deviation.
    
    Applies per-channel normalization: (image - mean) / std
    Used for preprocessing images before feeding to neural networks.
    
    Args:
        batch (torch.Tensor): Batch of images [batch_size, 3, H, W]
        mean (torch.Tensor): Per-channel mean values [3]
        std (torch.Tensor): Per-channel standard deviation values [3]
    
    Returns:
        torch.Tensor: Normalized image batch [batch_size, 3, H, W]
    
    Example:
        >>> images = torch.rand(4, 3, 256, 256)
        >>> mean = torch.tensor([0.485, 0.456, 0.406])
        >>> std = torch.tensor([0.229, 0.224, 0.225])
        >>> normalized = normalize(images, mean, std)
    """
    # Reshape mean and std for broadcasting
    mean.unsqueeze_(1).unsqueeze_(1)  # [3, 1, 1]
    std.unsqueeze_(1).unsqueeze_(1)   # [3, 1, 1]
    
    concat = None
    
    # Normalize each image in the batch
    for i in range(len(batch)):
        img = batch[i, :, :, :]
        # Apply normalization: (img - mean) / std
        img = img.sub(mean).div(std).unsqueeze(0)

        # Accumulate normalized images
        if concat is not None:
            concat = torch.cat((concat, img), 0)
        else:
            concat = img

    return concat

# ================================ MATHEMATICAL METRICS ================================

def dice(im1, im2, empty_score=1.0):
    """
    Compute the Dice coefficient between two binary images.
    
    The Dice coefficient is a measure of overlap between two sets, commonly used
    for evaluating segmentation quality. It ranges from 0 (no overlap) to 1 (perfect overlap).
    
    Formula: Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Adopted from: https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137
    
    Args:
        im1 (array-like): First binary image/mask
        im2 (array-like): Second binary image/mask of same size
        empty_score (float): Score to return when both images are empty (default: 1.0)
    
    Returns:
        float: Dice coefficient in range [0, 1]
               1.0 = perfect overlap
               0.0 = no overlap
               empty_score = both images empty
    
    Raises:
        ValueError: If input images have different shapes
    
    Example:
        >>> mask1 = np.array([[1, 1, 0], [0, 1, 0]])
        >>> mask2 = np.array([[1, 0, 0], [0, 1, 1]]) 
        >>> dice_score = dice(mask1, mask2)
        >>> print(f"Dice coefficient: {dice_score:.3f}")
    
    Notes:
        - Input order is irrelevant (dice(A, B) == dice(B, A))
        - Non-boolean inputs are automatically converted to boolean
        - Handles edge case where both images are empty
    """
    # Convert inputs to boolean arrays
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    # Validate input shapes
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Handle empty images case
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient: 2 * intersection / (sum of both sets)
    intersection = np.logical_and(im1, im2)
    dice_coefficient = 2.0 * intersection.sum() / im_sum
    
    return dice_coefficient