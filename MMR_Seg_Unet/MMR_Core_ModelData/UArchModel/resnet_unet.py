"""
resnet_unet.py - ResNet-UNet Hybrid Architecture for Semantic Segmentation

This module implements a hybrid architecture combining ResNet's powerful feature extraction
capabilities with U-Net's precise localization through skip connections. The architecture
uses a pretrained ResNet backbone as the encoder and a custom U-Net style decoder with
an additional original-size processing path for enhanced detail preservation.

Key Architectural Features:
- **Pretrained ResNet Encoder**: Leverages ImageNet-pretrained ResNet18/34 for robust feature extraction
- **U-Net Skip Connections**: Preserves spatial information through encoder-decoder connections
- **Original Size Processing**: Additional pathway for fine-grained detail preservation
- **Progressive Feature Fusion**: Systematic combination of multi-scale features
- **Transfer Learning**: Benefits from pretrained weights for faster convergence

Architecture Components:
1. ResNet Encoder (5 layers): Progressive downsampling with pretrained features
2. U-Net Decoder (4 stages): Progressive upsampling with skip connections
3. Original Size Path: Parallel processing for input-resolution features
4. Feature Fusion: Concatenation-based multi-scale feature integration

Adapted from: https://github.com/usuyama/pytorch-unet

Author: Medical Robotics Research Team
Compatible with: High-resolution segmentation tasks requiring fine detail preservation
"""

# ================================ IMPORTS ================================

import torch
import torch.nn as nn
from torchvision import models

# ================================ HELPER FUNCTIONS ================================

def convrelu(in_channels, out_channels, kernel, padding):
    """
    Create a convolution-ReLU block for feature processing.
    
    This helper function creates a standard building block consisting of a 2D
    convolution followed by ReLU activation, commonly used throughout the decoder.
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels  
        kernel (int): Convolution kernel size
        padding (int): Padding size for spatial dimension preservation
    
    Returns:
        nn.Sequential: Sequential container with Conv2d and ReLU layers
        
    Example:
        >>> conv_block = convrelu(256, 128, 3, 1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = conv_block(x)  # Shape: [1, 128, 32, 32]
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

# ================================ RESNET-UNET ARCHITECTURE ================================

class ResNetUNet(nn.Module):
    """
    ResNet-UNet hybrid architecture for semantic segmentation.
    
    This architecture combines the strengths of ResNet (robust feature extraction with
    pretrained weights) and U-Net (precise localization with skip connections). The
    design includes an additional original-size processing path for enhanced detail
    preservation, making it particularly effective for high-resolution segmentation tasks.
    
    Architecture Overview:
    ```
    Input ──┬── ResNet Encoder ──┬── Decoder ── Output
            │   (5 layers)       │   (4 stages)
            │                    │   ↑ Skip connections
            └── Original Size ───┘
                Processing
    ```
    
    ResNet Encoder Stages:
    - Layer0: Conv+BN+ReLU+MaxPool → H/2×W/2, 64 channels
    - Layer1: ResNet Block 1 → H/4×W/4, 64 channels  
    - Layer2: ResNet Block 2 → H/8×W/8, 128 channels
    - Layer3: ResNet Block 3 → H/16×W/16, 256 channels
    - Layer4: ResNet Block 4 → H/32×W/32, 512 channels
    
    Decoder Features:
    - Progressive 2× upsampling with bilinear interpolation
    - Skip connections from corresponding encoder layers
    - 1×1 convolutions for channel dimension adjustment
    - Feature concatenation followed by 3×3 convolutions
    
    Args:
        n_class (int): Number of output segmentation classes
        resnet_model (int): ResNet model variant (18 or 34)
    
    Attributes:
        resnet_model (int): ResNet backbone version
        base_model (torchvision.models.ResNet): Pretrained ResNet backbone
        layer0-4 (nn.Module): ResNet encoder stages
        layer0-4_1x1 (nn.Sequential): Channel adjustment layers
        conv_up0-3 (nn.Sequential): Decoder upsampling blocks
        conv_original_size0-2 (nn.Sequential): Original size processing blocks
        conv_last (nn.Conv2d): Final classification layer
    
    Example:
        >>> # ResNet-18 based model for 5-class segmentation
        >>> model = ResNetUNet(n_class=5, resnet_model=18)
        >>> input_tensor = torch.randn(4, 3, 256, 256)
        >>> output = model(input_tensor)
        >>> print(f"Output shape: {output.shape}")  # [4, 5, 256, 256]
        
        >>> # ResNet-34 for higher capacity
        >>> model = ResNetUNet(n_class=10, resnet_model=34)
        >>> high_res_input = torch.randn(2, 3, 512, 512)
        >>> segmentation = model(high_res_input)  # [2, 10, 512, 512]
    
    Key Advantages:
        - **Pretrained Features**: Benefits from ImageNet pretraining for robust representations
        - **Multi-Scale Processing**: Combines features from multiple resolution levels
        - **Detail Preservation**: Original size path maintains fine-grained spatial information
        - **Skip Connections**: U-Net style connections prevent information loss
        - **Transfer Learning**: Faster convergence compared to training from scratch
    
    Performance Characteristics:
        - **Memory Usage**: Higher than standard U-Net due to ResNet backbone
        - **Training Speed**: Moderate, benefits from pretrained initialization
        - **Accuracy**: Excellent for detailed segmentation tasks
        - **Generalization**: Strong due to pretrained features and multi-scale processing
    """
    
    def __init__(self, n_class, resnet_model):
        """
        Initialize ResNet-UNet with specified configuration.
        
        Args:
            n_class (int): Number of output segmentation classes
            resnet_model (int): ResNet variant (18 or 34) - determines backbone architecture
        
        Raises:
            ValueError: If resnet_model is not 18 or 34
            
        Example:
            >>> model = ResNetUNet(n_class=3, resnet_model=18)  # Binary + background
        """
        super().__init__()

        self.resnet_model = resnet_model
        
        # ===================== RESNET BACKBONE INITIALIZATION =====================
        
        # Load pretrained ResNet backbone
        if self.resnet_model == 18:
            self.base_model = models.resnet18(pretrained=True)
        elif self.resnet_model == 34:
            self.base_model = models.resnet34(pretrained=True)
        else:
            raise ValueError("Only ResNet-18 and ResNet-34 are supported")
            
        # Extract ResNet layers (excluding final classification layers)
        self.base_layers = list(self.base_model.children())

        # ===================== ENCODER LAYER DEFINITION =====================
        
        # Layer 0: Initial convolution + BN + ReLU + MaxPool
        # Input: (N, 3, H, W) → Output: (N, 64, H/2, W/2)
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)  # Channel adjustment
        
        # Layer 1: First ResNet block with MaxPool
        # Input: (N, 64, H/2, W/2) → Output: (N, 64, H/4, W/4)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        
        # Layer 2: Second ResNet block
        # Input: (N, 64, H/4, W/4) → Output: (N, 128, H/8, W/8)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        
        # Layer 3: Third ResNet block  
        # Input: (N, 128, H/8, W/8) → Output: (N, 256, H/16, W/16)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        
        # Layer 4: Fourth ResNet block (deepest features)
        # Input: (N, 256, H/16, W/16) → Output: (N, 512, H/32, W/32)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        # ===================== DECODER COMPONENTS =====================
        
        # Bilinear upsampling layer (2× scale factor)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Progressive decoder blocks with skip connection integration
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)  # Combine layer3 + upsampled layer4
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)  # Combine layer2 + upsampled features
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)   # Combine layer1 + upsampled features  
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)   # Combine layer0 + upsampled features

        # ===================== ORIGINAL SIZE PROCESSING PATH =====================
        
        # Parallel processing for input-resolution feature extraction
        self.conv_original_size0 = convrelu(3, 64, 3, 1)     # Initial feature extraction
        self.conv_original_size1 = convrelu(64, 64, 3, 1)    # Feature refinement
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)  # Fusion with decoder

        # ===================== FINAL CLASSIFICATION LAYER =====================
        
        # 1×1 convolution for final class prediction
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        """
        Forward pass through ResNet-UNet architecture.
        
        Processes input through three main pathways:
        1. ResNet encoder with progressive downsampling
        2. U-Net decoder with skip connections and progressive upsampling  
        3. Original size processing for fine detail preservation
        
        Args:
            input (torch.Tensor): Input images with shape (N, 3, H, W)
        
        Returns:
            torch.Tensor: Segmentation logits with shape (N, n_class, H, W)
                         Raw logits suitable for CrossEntropy loss
        
        Processing Flow:
            Input ── Original Size Path ────────────────────┐
              │                                            │
              └── ResNet Encoder ── Decoder ── Fusion ─────┼── Output
                  (5 layers)        (4 stages)             │
                  Skip connections ────────────────────────┘
        
        Example:
            >>> model = ResNetUNet(5, 18)
            >>> x = torch.randn(2, 3, 256, 256)
            >>> logits = model(x)  # [2, 5, 256, 256]
            >>> probabilities = torch.softmax(logits, dim=1)
        """
        # ===================== ORIGINAL SIZE PROCESSING PATH =====================
        
        # Process input at original resolution for fine detail preservation
        x_original = self.conv_original_size0(input)  # (N, 64, H, W)
        x_original = self.conv_original_size1(x_original)  # (N, 64, H, W)

        # ===================== RESNET ENCODER FORWARD PASS =====================
        
        # Progressive feature extraction with spatial downsampling
        layer0 = self.layer0(input)    # (N, 64, H/2, W/2)
        layer1 = self.layer1(layer0)   # (N, 64, H/4, W/4)
        layer2 = self.layer2(layer1)   # (N, 128, H/8, W/8)
        layer3 = self.layer3(layer2)   # (N, 256, H/16, W/16)
        layer4 = self.layer4(layer3)   # (N, 512, H/32, W/32) - deepest features

        # ===================== DECODER WITH SKIP CONNECTIONS =====================
        
        # Stage 1: Process deepest features and fuse with layer3
        layer4 = self.layer4_1x1(layer4)  # Channel adjustment
        x = self.upsample(layer4)          # (N, 512, H/16, W/16)
        layer3 = self.layer3_1x1(layer3)  # Channel adjustment  
        x = torch.cat([x, layer3], dim=1)  # (N, 768, H/16, W/16) - feature concatenation
        x = self.conv_up3(x)               # (N, 512, H/16, W/16) - process fused features

        # Stage 2: Upsample and fuse with layer2
        x = self.upsample(x)               # (N, 512, H/8, W/8)
        layer2 = self.layer2_1x1(layer2)  # Channel adjustment
        x = torch.cat([x, layer2], dim=1)  # (N, 640, H/8, W/8)
        x = self.conv_up2(x)               # (N, 256, H/8, W/8)

        # Stage 3: Upsample and fuse with layer1  
        x = self.upsample(x)               # (N, 256, H/4, W/4)
        layer1 = self.layer1_1x1(layer1)  # Channel adjustment
        x = torch.cat([x, layer1], dim=1)  # (N, 320, H/4, W/4)
        x = self.conv_up1(x)               # (N, 256, H/4, W/4)

        # Stage 4: Upsample and fuse with layer0
        x = self.upsample(x)               # (N, 256, H/2, W/2)
        layer0 = self.layer0_1x1(layer0)  # Channel adjustment
        x = torch.cat([x, layer0], dim=1)  # (N, 320, H/2, W/2)
        x = self.conv_up0(x)               # (N, 128, H/2, W/2)

        # ===================== FINAL UPSAMPLING AND ORIGINAL SIZE FUSION =====================
        
        # Final upsampling to original resolution
        x = self.upsample(x)               # (N, 128, H, W)
        
        # Fuse decoder features with original size processing
        x = torch.cat([x, x_original], dim=1)  # (N, 192, H, W)
        x = self.conv_original_size2(x)    # (N, 64, H, W)

        # ===================== FINAL CLASSIFICATION =====================
        
        # Generate per-pixel class predictions
        out = self.conv_last(x)            # (N, n_class, H, W)

        return out

    def __repr__(self):
        """String representation of the ResNet-UNet model."""
        return f"ResNetUNet(n_class={self.conv_last.out_channels}, resnet_model={self.resnet_model})"