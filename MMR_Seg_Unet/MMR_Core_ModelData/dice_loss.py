"""
dice_loss.py - Dice Loss Implementation for Semantic Segmentation

This module implements the Sørensen-Dice coefficient loss function, commonly used
in semantic segmentation tasks, especially for medical imaging where class 
imbalance is prevalent. The Dice loss is particularly effective for segmentation
tasks as it directly optimizes the overlap between predicted and ground truth masks.

Key Features:
- Differentiable Dice coefficient computation
- Numerical stability with epsilon smoothing
- Support for class ignoring (useful for background/void classes)
- Both functional and class-based interfaces
- GPU-accelerated computation with automatic device handling

Mathematical Background:
The Dice coefficient measures overlap between two sets and is defined as:
Dice = 2 * |A ∩ B| / (|A| + |B|)

The loss is computed as: Loss = 1 - Dice

Author: Medical Robotics Research Team
Based on: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
Reference: https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
"""

# ================================ IMPORTS ================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.utils.one_hot import one_hot

# ================================ FUNCTIONAL INTERFACE ================================

def dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1.0, ignore_index=None) -> torch.Tensor:
    r"""
    Compute Sørensen-Dice Coefficient loss for semantic segmentation.

    This function calculates the Dice loss, which is commonly used in semantic 
    segmentation tasks, particularly in medical imaging. The Dice coefficient 
    measures the overlap between predicted and ground truth segmentation masks.

    Mathematical Formula:
        Dice(x, class) = 2 * |X ∩ Y| / (|X| + |Y|)
        Loss(x, class) = 1 - Dice(x, class)

    Where:
    - X represents the predicted probability scores for each class
    - Y represents the one-hot encoded ground truth labels
    - |X ∩ Y| is the intersection (element-wise multiplication after softmax)
    - |X| + |Y| is the sum of both sets (cardinality)

    Args:
        input (torch.Tensor): Model predictions with shape (N, C, H, W)
                             Raw logits before softmax, where:
                             - N = batch size
                             - C = number of classes  
                             - H, W = spatial dimensions
        target (torch.Tensor): Ground truth labels with shape (N, H, W)
                              Integer class indices where each value is 0 ≤ target[i] ≤ C-1
        eps (float, optional): Epsilon value for numerical stability (default: 1.0)
                              Prevents division by zero when both prediction and target are empty
        ignore_index (int, optional): Class index to ignore during loss computation
                                    Useful for background or void classes (default: None)

    Returns:
        torch.Tensor: Scalar loss value (averaged across batch and classes)

    Raises:
        TypeError: If input is not a torch.Tensor
        ValueError: If input shape is not 4D (BxCxHxW)
        ValueError: If spatial dimensions of input and target don't match  
        ValueError: If input and target are on different devices

    Example:
        >>> # Binary segmentation example
        >>> batch_size, num_classes, height, width = 4, 2, 256, 256
        >>> predictions = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        >>> targets = torch.randint(0, num_classes, (batch_size, height, width))
        >>> loss = dice_loss(predictions, targets)
        >>> loss.backward()
        >>> print(f"Dice loss: {loss.item():.4f}")

        >>> # Multi-class segmentation with class ignoring
        >>> predictions = torch.randn(2, 5, 128, 128, requires_grad=True)  
        >>> targets = torch.randint(0, 5, (2, 128, 128))
        >>> loss = dice_loss(predictions, targets, eps=1e-6, ignore_index=0)  # Ignore background

    Notes:
        - Input logits are automatically converted to probabilities using softmax
        - Higher eps values provide more numerical stability but may affect gradient flow
        - The function averages loss across all classes and batch samples
        - For binary segmentation, use num_classes=2 (background + foreground)
    """
    # ===================== INPUT VALIDATION =====================
    
    # Type checking
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    # Shape validation - expect 4D tensor (batch, classes, height, width)
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    # Spatial dimensions must match between prediction and target
    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

    # Device consistency check
    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
    
    # ===================== PROBABILITY COMPUTATION =====================
    
    # Convert raw logits to probability distribution using softmax
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # ===================== ONE-HOT ENCODING =====================
    
    # Convert integer class labels to one-hot encoded format
    # Shape: (N, H, W) -> (N, C, H, W) where C = num_classes
    target_one_hot: torch.Tensor = one_hot(
        target, 
        num_classes=input.shape[1], 
        device=input.device, 
        dtype=input.dtype
    )

    # ===================== CLASS IGNORING (OPTIONAL) =====================
    
    # Remove ignored classes from computation if specified
    if ignore_index is not None:
        input_soft = input_soft[:, :ignore_index]
        target_one_hot = target_one_hot[:, :ignore_index]

    # ===================== DICE COEFFICIENT COMPUTATION =====================
    
    # Define spatial dimensions for summation (height and width)
    dims = (2, 3)
    
    # Calculate intersection: sum of element-wise multiplication
    # This represents |A ∩ B| in the Dice formula
    intersection = torch.sum(input_soft * target_one_hot, dims)
    
    # Calculate cardinality: sum of both prediction and target
    # This represents |A| + |B| in the Dice formula  
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    # Compute Dice score with numerical stability
    # Formula: (2 * intersection + eps) / (cardinality + eps)
    dice_score = (2.0 * intersection + eps) / (cardinality + eps)

    # ===================== LOSS COMPUTATION =====================
    
    # Convert Dice score to loss: Loss = 1 - Dice
    # Take mean across batch and classes
    dice_loss_value = torch.mean(-dice_score + 1.0)
    
    return dice_loss_value

# ================================ CLASS-BASED INTERFACE ================================

class DiceLoss(nn.Module):
    r"""
    PyTorch Module wrapper for Dice Loss computation.

    This class provides a convenient nn.Module interface for the Dice loss function,
    making it easy to integrate into PyTorch training pipelines. It maintains the
    same mathematical foundation as the functional version but with stateful configuration.

    The Dice coefficient measures the similarity between predicted and ground truth
    segmentation masks, making it particularly suitable for medical image segmentation
    where precise boundary delineation is crucial.

    Mathematical Formula:
        Dice(x, class) = 2 * |X ∩ Y| / (|X| + |Y|)
        Loss(x, class) = 1 - Dice(x, class)

    Attributes:
        eps (float): Epsilon value for numerical stability
        ignore_index (int or None): Class index to ignore during computation

    Shape:
        - Input: (N, C, H, W) where N=batch size, C=classes, H=height, W=width
        - Target: (N, H, W) with integer class indices 0 ≤ target[i] ≤ C-1  
        - Output: Scalar tensor representing the average loss

    Example:
        >>> # Initialize loss function
        >>> criterion = DiceLoss(eps=1e-6, ignore_index=0)
        >>> 
        >>> # Use in training loop
        >>> predictions = model(images)  # Shape: (batch, classes, H, W)
        >>> targets = ground_truth       # Shape: (batch, H, W) 
        >>> loss = criterion(predictions, targets)
        >>> 
        >>> # Backward pass
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

        >>> # Multi-class segmentation example
        >>> num_classes = 5
        >>> criterion = DiceLoss(eps=1.0)
        >>> input_tensor = torch.randn(8, num_classes, 256, 256, requires_grad=True)
        >>> target_tensor = torch.randint(0, num_classes, (8, 256, 256))
        >>> loss_value = criterion(input_tensor, target_tensor)
        >>> print(f"Dice Loss: {loss_value.item():.4f}")

    Notes:
        - Automatically handles softmax conversion of input logits
        - Supports gradient computation for backpropagation
        - Can be combined with other loss functions (e.g., CrossEntropy + Dice)
        - Suitable for both binary and multi-class segmentation tasks
        - Particularly effective for imbalanced datasets
    """

    def __init__(self, eps: float = 1.0, ignore_index=None) -> None:
        """
        Initialize the Dice Loss module.

        Args:
            eps (float, optional): Epsilon value for numerical stability (default: 1.0)
                                  Higher values provide more stability but may affect gradients
            ignore_index (int, optional): Class index to ignore during loss computation
                                        Commonly used to ignore background class (default: None)

        Example:
            >>> # Standard Dice loss for medical segmentation
            >>> dice_loss = DiceLoss(eps=1e-6)
            >>> 
            >>> # Dice loss ignoring background class (index 0)
            >>> dice_loss_no_bg = DiceLoss(eps=1.0, ignore_index=0)
        """
        super().__init__()
        self.eps: float = eps
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss for given input and target tensors.

        Args:
            input (torch.Tensor): Model predictions (raw logits) with shape (N, C, H, W)
            target (torch.Tensor): Ground truth class indices with shape (N, H, W)

        Returns:
            torch.Tensor: Computed Dice loss (scalar value)

        Example:
            >>> criterion = DiceLoss()
            >>> predictions = torch.randn(4, 3, 128, 128, requires_grad=True)
            >>> targets = torch.randint(0, 3, (4, 128, 128))
            >>> loss = criterion(predictions, targets)
            >>> # loss is now ready for backpropagation
        """
        return dice_loss(input, target, self.eps, self.ignore_index)

    def __repr__(self) -> str:
        """
        String representation of the DiceLoss module.

        Returns:
            str: Human-readable description of the module configuration
        """
        return f"DiceLoss(eps={self.eps}, ignore_index={self.ignore_index})"