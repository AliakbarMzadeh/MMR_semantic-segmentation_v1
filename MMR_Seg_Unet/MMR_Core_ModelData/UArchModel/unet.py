"""
unet.py - U-Net Architecture Implementation for Semantic Segmentation

This module implements the U-Net convolutional neural network architecture, originally
designed for biomedical image segmentation. U-Net is characterized by its symmetric
encoder-decoder structure with skip connections that preserve spatial information
during upsampling, making it highly effective for pixel-level prediction tasks.

Architecture Overview:
- Encoder (Contracting Path): Captures context through downsampling
- Decoder (Expansive Path): Enables precise localization through upsampling  
- Skip Connections: Combine high-resolution features from encoder with upsampled features
- Final Layer: Produces class probability maps for each pixel

Key Features:
- Symmetric encoder-decoder architecture with skip connections
- Configurable upsampling method (bilinear vs transposed convolution)
- Flexible input channels and output classes
- Efficient feature reuse through skip connections
- Suitable for medical image segmentation and dense prediction tasks



Author: Medical Robotics Research Team
Compatible with: PyTorch-based semantic segmentation pipelines
"""

# ================================ IMPORTS ================================

import torch.nn as nn
from .unet_parts import *

# ================================ U-NET ARCHITECTURE ================================

class UNet(nn.Module):
    """
    U-Net Convolutional Neural Network for Semantic Segmentation.
    
    This implementation follows the original U-Net architecture with a symmetric
    encoder-decoder structure and skip connections. The network consists of:
    
    1. **Encoder (Contracting Path)**: 4 downsampling blocks that capture context
       - Each block: 2 convolutions + max pooling
       - Feature channels: 64 → 128 → 256 → 512 → 1024
       
    2. **Decoder (Expansive Path)**: 4 upsampling blocks for precise localization
       - Each block: upsampling + concatenation + 2 convolutions
       - Feature channels: 1024 → 512 → 256 → 128 → 64
       
    3. **Skip Connections**: Direct feature concatenation between encoder and decoder
       - Preserves fine-grained spatial information lost during downsampling
       - Enables precise boundary delineation in segmentation masks
    
    Architecture Diagram:
    ```
    Input(3) → [64] → [128] → [256] → [512] → [1024]
                ↓      ↓       ↓       ↓        ↓
               Skip   Skip    Skip    Skip      |
                ↓      ↓       ↓       ↓        ↓
    Output(C) ← [64] ← [128] ← [256] ← [512] ← [1024]
    ```
    
    Args:
        n_channels (int): Number of input channels (e.g., 3 for RGB images)
        n_classes (int): Number of output segmentation classes
        bilinear (bool, optional): Use bilinear upsampling instead of transposed 
                                  convolutions (default: False)
    
    Attributes:
        n_channels (int): Number of input channels
        n_classes (int): Number of output classes
        bilinear (bool): Whether bilinear upsampling is used
        inc (DoubleConv): Initial convolution block
        down1-4 (Down): Encoder downsampling blocks
        up1-4 (Up): Decoder upsampling blocks  
        outc (OutConv): Final output convolution
    
    Example:
        >>> # Binary segmentation (background + foreground)
        >>> model = UNet(n_channels=3, n_classes=2, bilinear=True)
        >>> input_tensor = torch.randn(4, 3, 256, 256)
        >>> output = model(input_tensor)
        >>> print(f"Output shape: {output.shape}")  # [4, 2, 256, 256]
        
        >>> # Multi-class medical segmentation  
        >>> model = UNet(n_channels=1, n_classes=5, bilinear=False)
        >>> grayscale_input = torch.randn(2, 1, 512, 512)
        >>> segmentation = model(grayscale_input)
        >>> print(f"Classes: {segmentation.shape[1]}")  # 5
        
        >>> # RGB surgical image segmentation
        >>> surgical_model = UNet(n_channels=3, n_classes=8, bilinear=True)
        >>> rgb_images = torch.randn(8, 3, 256, 256)
        >>> masks = surgical_model(rgb_images)
    
    Notes:
        - **Bilinear upsampling**: Faster, less memory, but potentially less precise
        - **Transposed convolution**: Learnable upsampling, more parameters, higher precision
        - **Skip connections**: Essential for maintaining spatial resolution in segmentation
        - **Receptive field**: Gradually increases in encoder, preserves fine details in decoder
        - **Memory usage**: Proportional to input resolution and number of feature channels
    """

    def __init__(self, n_channels, n_classes, bilinear=False):
        """
        Initialize the U-Net architecture.
        
        Sets up the complete encoder-decoder structure with configurable upsampling
        method and appropriate feature channel dimensions throughout the network.
        
        Args:
            n_channels (int): Number of input channels
                             - 1 for grayscale images (e.g., X-rays, MRI)
                             - 3 for RGB images (e.g., surgical videos, photographs)
                             - 4 for RGBA images with transparency
            n_classes (int): Number of output segmentation classes
                            - 2 for binary segmentation (background + object)
                            - N for multi-class segmentation (N different structures)
            bilinear (bool, optional): Upsampling method selection (default: False)
                                     - True: Use bilinear interpolation (faster, less memory)
                                     - False: Use transposed convolutions (learnable, more precise)
        
        Example:
            >>> # Standard configuration for surgical image segmentation
            >>> model = UNet(n_channels=3, n_classes=5, bilinear=True)
            >>> 
            >>> # Medical imaging configuration  
            >>> model = UNet(n_channels=1, n_classes=3, bilinear=False)
            >>> 
            >>> # High-resolution segmentation with memory optimization
            >>> model = UNet(n_channels=3, n_classes=10, bilinear=True)
        """
        super(UNet, self).__init__()
        
        # Store configuration parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # ===================== ENCODER (CONTRACTING PATH) =====================
        
        # Initial convolution block: input_channels → 64 features
        # Processes raw input without spatial reduction
        self.inc = DoubleConv(n_channels, 64)
        
        # Downsampling blocks with increasing feature channels
        # Each block halves spatial dimensions and doubles feature channels
        self.down1 = Down(64, 128)      # 1/2 resolution, 128 features
        self.down2 = Down(128, 256)     # 1/4 resolution, 256 features  
        self.down3 = Down(256, 512)     # 1/8 resolution, 512 features
        
        # Adjust final encoder channels based on upsampling method
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # 1/16 resolution, 512 or 1024 features

        # ===================== DECODER (EXPANSIVE PATH) =====================
        
        # Upsampling blocks with decreasing feature channels
        # Each block doubles spatial dimensions and (optionally) halves feature channels
        self.up1 = Up(1024, 512 // factor, bilinear)    # 1/8 resolution, 256 or 512 features
        self.up2 = Up(512, 256 // factor, bilinear)     # 1/4 resolution, 128 or 256 features
        self.up3 = Up(256, 128 // factor, bilinear)     # 1/2 resolution, 64 or 128 features
        self.up4 = Up(128, 64, bilinear)                # Full resolution, 64 features

        # ===================== OUTPUT LAYER =====================
        
        # Final 1x1 convolution: 64 features → n_classes probability maps
        # Produces per-pixel class predictions
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Forward pass through the U-Net architecture.
        
        Implements the complete forward propagation including:
        1. Encoder feature extraction with skip connection storage
        2. Bottleneck processing at lowest resolution
        3. Decoder upsampling with skip connection integration
        4. Final classification layer
        
        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W) where:
                             - N: batch size
                             - C: number of input channels (must match n_channels)
                             - H, W: spatial dimensions (height, width)
        
        Returns:
            torch.Tensor: Output segmentation logits with shape (N, n_classes, H, W)
                         Raw logits (before softmax) for each class at each pixel
                         Apply softmax or sigmoid for probability interpretation
        
        Example:
            >>> model = UNet(3, 5)
            >>> input_batch = torch.randn(4, 3, 256, 256)
            >>> 
            >>> # Forward pass
            >>> logits = model(input_batch)
            >>> print(f"Logits shape: {logits.shape}")  # [4, 5, 256, 256]
            >>> 
            >>> # Convert to probabilities
            >>> probabilities = torch.softmax(logits, dim=1)
            >>> predictions = torch.argmax(probabilities, dim=1)  # Class indices
            >>> 
            >>> # Multi-scale input handling
            >>> small_input = torch.randn(2, 3, 128, 128)
            >>> small_output = model(small_input)  # [2, 5, 128, 128]
        
        Architecture Flow:
            Input → [DoubleConv] → [Down×4] → [Up×4] → [OutConv] → Output
                      ↓            ↓         ↗ (skip connections)
                   Features    Bottleneck   Feature Fusion
        
        Notes:
            - **Skip connections**: Encoder features are stored (x1-x4) and reused in decoder
            - **Spatial preservation**: Output spatial dimensions match input dimensions  
            - **Feature progression**: Features become more semantic (less spatial) in encoder
            - **Memory efficiency**: Intermediate features are reused rather than recomputed
        """
        # ===================== ENCODER FORWARD PASS =====================
        
        # Initial feature extraction (no spatial reduction)
        x1 = self.inc(x)        # Shape: (N, 64, H, W)
        
        # Progressive downsampling with feature extraction
        # Store intermediate features for skip connections
        x2 = self.down1(x1)     # Shape: (N, 128, H/2, W/2)
        x3 = self.down2(x2)     # Shape: (N, 256, H/4, W/4)  
        x4 = self.down3(x3)     # Shape: (N, 512, H/8, W/8)
        x5 = self.down4(x4)     # Shape: (N, 512|1024, H/16, W/16) - bottleneck
        
        # ===================== DECODER FORWARD PASS =====================
        
        # Progressive upsampling with skip connection integration
        # Each step combines upsampled features with corresponding encoder features
        x = self.up1(x5, x4)    # Shape: (N, 256|512, H/8, W/8)
        x = self.up2(x, x3)     # Shape: (N, 128|256, H/4, W/4)
        x = self.up3(x, x2)     # Shape: (N, 64|128, H/2, W/2)
        x = self.up4(x, x1)     # Shape: (N, 64, H, W)
        
        # ===================== OUTPUT GENERATION =====================
        
        # Final classification layer: features → class logits
        logits = self.outc(x)   # Shape: (N, n_classes, H, W)
        
        return logits

    def __repr__(self):
        """
        String representation of the U-Net model.
        
        Returns:
            str: Human-readable description of the model configuration
        """
        return (f"UNet(n_channels={self.n_channels}, n_classes={self.n_classes}, "
                f"bilinear={self.bilinear})")
