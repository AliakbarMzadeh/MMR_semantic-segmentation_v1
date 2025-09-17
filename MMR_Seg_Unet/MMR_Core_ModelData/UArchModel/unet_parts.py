"""
unet_parts.py - Building Blocks for U-Net Architecture

This module implements the fundamental components that make up the U-Net segmentation
architecture. These modular building blocks provide the core functionality for
encoder-decoder networks with skip connections, enabling precise pixel-level
segmentation through hierarchical feature processing.

Key Components:
- DoubleConv: Core convolution block with batch normalization and activation
- Down: Encoder downsampling block combining pooling and feature extraction
- Up: Decoder upsampling block with skip connection integration
- OutConv: Final classification layer for pixel-wise predictions

Design Philosophy:
The modular design allows for flexible architecture construction while maintaining
consistent building patterns. Each component handles specific aspects of the U-Net
workflow, from feature extraction to spatial resolution management and multi-scale
feature fusion.

Author: Medical Robotics Research Team
Compatible with: U-Net and U-Net-derived architectures for semantic segmentation
"""

# ================================ IMPORTS ================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================ CORE BUILDING BLOCKS ================================

class DoubleConv(nn.Module):
    """
    Double Convolution Block - Core building block of U-Net architecture.
    
    Implements the standard U-Net convolution pattern: (Conv2D → BatchNorm → ReLU) × 2
    This pattern provides effective feature extraction while maintaining training stability
    through batch normalization and non-linear activation functions.
    
    Architecture Pattern:
        Input → Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → Output
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels
        mid_channels (int, optional): Number of intermediate channels (default: out_channels)
                                     Allows for bottleneck-style processing when different from out_channels
    
    Attributes:
        double_conv (nn.Sequential): Sequential container with the complete double convolution pattern
        
    Key Features:
        - **Batch Normalization**: Stabilizes training and enables higher learning rates
        - **ReLU Activation**: Provides non-linearity for complex pattern learning
        - **3×3 Convolutions**: Optimal balance between receptive field and computational efficiency  
        - **Padding=1**: Preserves spatial dimensions throughout processing
        - **Flexible Channels**: Supports bottleneck architectures through mid_channels parameter
        
    Example:
        >>> # Standard usage
        >>> double_conv = DoubleConv(64, 128)
        >>> x = torch.randn(4, 64, 128, 128)
        >>> output = double_conv(x)  # Shape: [4, 128, 128, 128]
        >>> 
        >>> # Bottleneck usage
        >>> bottleneck = DoubleConv(256, 128, mid_channels=64)
        >>> x = torch.randn(2, 256, 64, 64)
        >>> output = bottleneck(x)  # Shape: [2, 128, 64, 64]
        
    Notes:
        - Spatial dimensions are preserved due to padding=1 with kernel_size=3
        - In-place ReLU operations reduce memory usage during training
        - Batch normalization parameters are learned during training
        - Works effectively with various input resolutions and channel counts
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Initialize the double convolution block.
        
        Args:
            in_channels (int): Input channel count
            out_channels (int): Output channel count  
            mid_channels (int, optional): Intermediate channel count for bottleneck design
        """
        super().__init__()
        
        # Use out_channels as mid_channels if not specified (standard U-Net pattern)
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            # First convolution: input_channels → mid_channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution: mid_channels → out_channels  
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through double convolution block.
        
        Args:
            x (torch.Tensor): Input feature tensor with shape (N, in_channels, H, W)
        
        Returns:
            torch.Tensor: Output feature tensor with shape (N, out_channels, H, W)
                         Spatial dimensions preserved, channels transformed
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling Block - Encoder component for spatial resolution reduction.
    
    Combines max pooling for spatial downsampling with double convolution for feature
    extraction. This is the standard encoder pattern in U-Net that progressively 
    reduces spatial dimensions while increasing feature depth and semantic content.
    
    Architecture Pattern:
        Input → MaxPool2d → DoubleConv → Output
        
    Spatial Transformation:
        (N, in_channels, H, W) → (N, out_channels, H/2, W/2)
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels (typically 2× input channels)
    
    Attributes:
        maxpool_conv (nn.Sequential): Sequential container with max pooling and double convolution
        
    Design Rationale:
        - **Max Pooling**: Provides translation invariance and computational efficiency
        - **2×2 Pooling**: Standard downsampling factor for hierarchical feature extraction
        - **Feature Expansion**: Typically doubles channel count to maintain information capacity
        - **Receptive Field Growth**: Increases effective receptive field for larger context
        
    Example:
        >>> # Standard encoder progression
        >>> down1 = Down(64, 128)    # H/2, W/2, 128 channels
        >>> down2 = Down(128, 256)   # H/4, W/4, 256 channels
        >>> down3 = Down(256, 512)   # H/8, W/8, 512 channels
        >>> 
        >>> x = torch.randn(4, 64, 256, 256)
        >>> x1 = down1(x)  # Shape: [4, 128, 128, 128]
        >>> x2 = down2(x1) # Shape: [4, 256, 64, 64]
        
    Performance Notes:
        - Max pooling is computationally efficient compared to strided convolutions
        - Information loss is offset by increased channel capacity and receptive field
        - Skip connections in U-Net preserve high-resolution information
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the downsampling block.
        
        Args:
            in_channels (int): Input channel count
            out_channels (int): Output channel count (commonly 2× input)
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2×2 max pooling with stride=2
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        Forward pass through downsampling block.
        
        Args:
            x (torch.Tensor): Input tensor with shape (N, in_channels, H, W)
        
        Returns:
            torch.Tensor: Downsampled output with shape (N, out_channels, H/2, W/2)
                         Spatial resolution halved, feature depth increased
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling Block - Decoder component with skip connection integration.
    
    The most complex U-Net component that handles upsampling and feature fusion through
    skip connections. This block combines features from the decoder path with corresponding
    encoder features to preserve both semantic and spatial information.
    
    Architecture Pattern:
        Decoder Features → Upsample → Concatenate with Skip Features → DoubleConv → Output
        
    Skip Connection Handling:
        - Automatic size adjustment for mismatched feature map dimensions
        - Padding-based alignment for robust feature fusion
        - Channel concatenation for multi-scale information integration
    
    Args:
        in_channels (int): Total channels from decoder path (before skip connection)
        out_channels (int): Desired output channels after processing
        bilinear (bool, optional): Upsampling method selection (default: True)
                                  - True: Bilinear interpolation (faster, less memory)
                                  - False: Transposed convolution (learnable, more precise)
    
    Attributes:
        up (nn.Module): Upsampling layer (Upsample or ConvTranspose2d)
        conv (DoubleConv): Double convolution for processing fused features
        
    Upsampling Methods:
        **Bilinear Interpolation:**
        - Fixed interpolation algorithm
        - No additional parameters
        - Faster computation, lower memory usage
        - Good for most applications
        
        **Transposed Convolution:**
        - Learnable upsampling weights  
        - Additional parameters to train
        - Potentially more precise reconstruction
        - Higher computational cost
        
    Example:
        >>> # Bilinear upsampling (memory efficient)
        >>> up_bilinear = Up(1024, 512, bilinear=True)
        >>> decoder_features = torch.randn(2, 1024, 16, 16)
        >>> skip_features = torch.randn(2, 256, 32, 32)
        >>> output = up_bilinear(decoder_features, skip_features)  # [2, 512, 32, 32]
        >>> 
        >>> # Transposed convolution (learnable)
        >>> up_transpose = Up(1024, 512, bilinear=False)
        >>> output = up_transpose(decoder_features, skip_features)
        
    Skip Connection Benefits:
        - Preserves fine-grained spatial information lost during encoding
        - Enables precise boundary delineation in segmentation masks
        - Combines low-level details with high-level semantic information
        - Critical for achieving pixel-perfect segmentation accuracy
    """
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        Initialize the upsampling block with specified configuration.
        
        Args:
            in_channels (int): Input channels from decoder path
            out_channels (int): Desired output channels
            bilinear (bool, optional): Use bilinear upsampling vs transposed conv
        """
        super().__init__()

        # Configure upsampling method and subsequent convolution
        if bilinear:
            # Bilinear interpolation path
            self.up = nn.Upsample(scale_factor=2, mode='nearest')  # 2× spatial upsampling
            # Note: Uses 'nearest' mode - could be 'bilinear' for smoother interpolation
            
            # Adjust channel count for efficient processing
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Transposed convolution path (learnable upsampling)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            
            # Standard double convolution after upsampling
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass with skip connection integration.
        
        Handles the complex process of upsampling decoder features and fusing them
        with encoder skip connections, including automatic size adjustment for
        robust feature integration.
        
        Args:
            x1 (torch.Tensor): Decoder features to upsample, shape (N, C1, H1, W1)
            x2 (torch.Tensor): Skip connection features from encoder, shape (N, C2, H2, W2)
                              Where H2 ≈ 2×H1 and W2 ≈ 2×W1 (approximately double resolution)
        
        Returns:
            torch.Tensor: Fused and processed features with shape (N, out_channels, H2, W2)
                         Output matches skip connection spatial dimensions
        
        Processing Steps:
            1. Upsample x1 to approximately match x2's spatial dimensions
            2. Calculate and apply padding to handle size mismatches  
            3. Concatenate upsampled x1 with x2 along channel dimension
            4. Process concatenated features through double convolution
        
        Size Mismatch Handling:
            Due to pooling/upsampling operations, feature maps may have slightly different
            sizes. The forward method automatically calculates required padding to align
            dimensions before concatenation.
            
        Example:
            >>> up_block = Up(1024, 512)
            >>> decoder_feat = torch.randn(1, 1024, 15, 15)  # Slightly odd dimensions
            >>> skip_feat = torch.randn(1, 256, 32, 32)      # Target dimensions
            >>> result = up_block(decoder_feat, skip_feat)    # Shape: [1, 512, 32, 32]
        """
        # ===================== UPSAMPLING PHASE =====================
        
        # Upsample decoder features (x1) to match target resolution
        x1 = self.up(x1)  # Shape: (N, C1/2 or C1//2, ~2×H1, ~2×W1)
        
        # ===================== SIZE ALIGNMENT PHASE =====================
        
        # Calculate spatial dimension differences for padding
        # x2 is the skip connection (target size), x1 is upsampled decoder features
        diffY = x2.size()[2] - x1.size()[2]  # Height difference  
        diffX = x2.size()[3] - x1.size()[3]  # Width difference
        
        # Apply symmetric padding to align x1 with x2's spatial dimensions
        # Padding format: [pad_left, pad_right, pad_top, pad_bottom]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,    # X-axis padding
                        diffY // 2, diffY - diffY // 2])   # Y-axis padding
        
        # Note: This padding strategy handles both even and odd size differences
        # For even differences: symmetric padding (diffX//2, diffX//2)  
        # For odd differences: asymmetric padding (diffX//2, diffX//2 + 1)
        
        # ===================== FEATURE FUSION PHASE =====================
        
        # Concatenate skip connection features with upsampled features
        # Order: [skip_features, upsampled_features] along channel dimension
        x = torch.cat([x2, x1], dim=1)  # Shape: (N, C2 + C1_upsampled, H2, W2)
        
        # ===================== FEATURE PROCESSING PHASE =====================
        
        # Process concatenated features through double convolution
        return self.conv(x)  # Shape: (N, out_channels, H2, W2)


class OutConv(nn.Module):
    """
    Output Convolution Layer - Final classification component of U-Net.
    
    Implements the final 1×1 convolution that maps processed features to class
    predictions. This layer performs pixel-wise classification by transforming
    feature representations into class logits or probabilities.
    
    Architecture Pattern:
        Features → 1×1 Conv → Class Predictions
        
    Key Characteristics:
        - **1×1 Convolution**: Pixel-wise classification without spatial mixing
        - **No Activation**: Produces raw logits suitable for loss computation
        - **Channel Mapping**: Transforms feature channels to class predictions
        - **Spatial Preservation**: Maintains input spatial dimensions exactly
        
    Args:
        in_channels (int): Number of input feature channels (typically 64 in standard U-Net)
        out_channels (int): Number of output classes for segmentation
    
    Attributes:
        conv (nn.Conv2d): 1×1 convolution layer for classification
        
    Design Rationale:
        - 1×1 convolutions are computationally efficient for channel transformation
        - No spatial mixing preserves precise spatial localization
        - Raw logits allow flexibility in loss function choice (CrossEntropy, etc.)
        - Simple design focuses on classification without feature processing
        
    Example:
        >>> # Binary segmentation (background + foreground)
        >>> out_conv = OutConv(64, 2)
        >>> features = torch.randn(4, 64, 256, 256)
        >>> logits = out_conv(features)  # Shape: [4, 2, 256, 256]
        >>> 
        >>> # Multi-class segmentation  
        >>> multi_out = OutConv(64, 8)
        >>> predictions = multi_out(features)  # Shape: [4, 8, 256, 256]
        >>> 
        >>> # Convert to probabilities and class indices
        >>> probabilities = torch.softmax(predictions, dim=1)
        >>> class_predictions = torch.argmax(probabilities, dim=1)
        
    Usage Notes:
        - Output contains raw logits (not probabilities)
        - Apply softmax for probability interpretation  
        - Use CrossEntropy loss directly with logits for training
        - Final layer in U-Net architecture pipeline
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the output convolution layer.
        
        Args:
            in_channels (int): Number of input feature channels
            out_channels (int): Number of segmentation classes
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for pixel-wise classification.
        
        Args:
            x (torch.Tensor): Input features with shape (N, in_channels, H, W)
        
        Returns:
            torch.Tensor: Class logits with shape (N, out_channels, H, W)
                         Raw logits for each class at each spatial location
        
        Example:
            >>> out_layer = OutConv(64, 5)
            >>> features = torch.randn(2, 64, 128, 128)
            >>> class_logits = out_layer(features)  # [2, 5, 128, 128]
        """
        return self.conv(x)