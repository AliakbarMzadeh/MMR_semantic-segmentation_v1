"""
segnet.py - SegNet-Style Architecture for Semantic Segmentation

This module implements a SegNet-inspired encoder-decoder architecture for semantic 
segmentation. While named SegNet, this implementation uses a simplified approach
with transposed convolutions rather than the original SegNet's pooling indices
mechanism for upsampling.

Architecture Overview:
- Encoder: Progressive downsampling with increasing feature channels
- Decoder: Progressive upsampling with decreasing feature channels  
- No skip connections (unlike U-Net)
- Symmetric structure with batch normalization and dropout

Key Differences from Original SegNet:
- Uses transposed convolutions instead of pooling indices for upsampling
- Includes dropout layers in decoder for regularization
- Built-in softmax activation (note: may need removal for training)
- Simplified architecture without VGG backbone

Original SegNet Paper: "SegNet: A Deep Convolutional Encoder-Decoder Architecture 
for Image Segmentation" by Badrinarayanan et al. (2017)

Implementation adopted from:
https://github.com/salmanmaq/segmentationNetworks/blob/master/model/segnet.py

Author: Medical Robotics Research Team  
Compatible with: Multi-class semantic segmentation tasks
"""

# ================================ IMPORTS ================================

import torch.nn as nn

# ================================ ENCODER COMPONENT ================================

class encoder(nn.Module):
    """
    Encoder component of the SegNet-style architecture.
    
    The encoder progressively reduces spatial dimensions while increasing feature
    depth, capturing hierarchical representations of the input image. It uses
    strided convolutions for downsampling and batch normalization for training stability.
    
    Architecture Flow:
    Input(3) → Conv(64) → Conv(128) → Conv(256) → Conv(512) → Conv(1024)
    
    Spatial Reduction:
    H×W → H/2×W/2 → H/4×W/4 → H/8×W/8 → H/16×W/16 → H/16×W/16
    
    Args:
        batchNorm_momentum (float): Momentum parameter for batch normalization layers
                                   Controls the exponential moving average of batch statistics
        num_classes (int, optional): Number of output classes (default: 23)
                                    Note: Not used in encoder, kept for compatibility
    
    Attributes:
        batchNorm_momentum (float): Stored batch normalization momentum
        num_classes (int): Number of output classes
        main (nn.Sequential): Sequential container of encoder layers
        
    Example:
        >>> encoder_net = encoder(batchNorm_momentum=0.1, num_classes=5)
        >>> input_tensor = torch.randn(4, 3, 256, 256)
        >>> encoded = encoder_net(input_tensor)
        >>> print(f"Encoded shape: {encoded.shape}")  # [4, 1024, 16, 16]
    
    Notes:
        - First convolution has no batch normalization (common practice)
        - All convolutions use 4×4 kernels with stride 2 except the last one
        - Final layer uses stride 1 to maintain feature depth without spatial reduction
        - ReLU activation provides non-linearity after each convolution
    """
    
    def __init__(self, batchNorm_momentum, num_classes=23):
        """
        Initialize the encoder with specified batch normalization momentum.
        
        Args:
            batchNorm_momentum (float): Momentum for batch normalization (typically 0.1)
            num_classes (int, optional): Number of classes for compatibility (default: 23)
        """
        super(encoder, self).__init__()
        self.batchNorm_momentum = batchNorm_momentum
        self.num_classes = num_classes
        
        # Sequential encoder architecture
        self.main = nn.Sequential(
            # ===================== LAYER 1: INITIAL FEATURE EXTRACTION =====================
            # Input: (N, 3, H, W) → Output: (N, 64, H/2, W/2)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, dilation=1, bias=False),
            nn.ReLU(True),  # In-place operation for memory efficiency

            # ===================== LAYER 2: FEATURE DEPTH INCREASE =====================
            # Input: (N, 64, H/2, W/2) → Output: (N, 128, H/4, W/4)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(True),

            # ===================== LAYER 3: MID-LEVEL FEATURES =====================
            # Input: (N, 128, H/4, W/4) → Output: (N, 256, H/8, W/8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(True),

            # ===================== LAYER 4: HIGH-LEVEL FEATURES =====================  
            # Input: (N, 256, H/8, W/8) → Output: (N, 512, H/16, W/16)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(True),

            # ===================== LAYER 5: BOTTLENECK FEATURES =====================
            # Input: (N, 512, H/16, W/16) → Output: (N, 1024, H/16, W/16)
            # Note: stride=1, padding=0 maintains spatial dimensions while increasing depth
            nn.Conv2d(512, 1024, kernel_size=4, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(1024, momentum=batchNorm_momentum),
            nn.ReLU(True)
        )

    def forward(self, input):
        """
        Forward pass through the encoder.
        
        Args:
            input (torch.Tensor): Input image tensor with shape (N, 3, H, W)
        
        Returns:
            torch.Tensor: Encoded feature tensor with shape (N, 1024, H/16, W/16)
                         High-dimensional feature representation at reduced spatial resolution
        
        Example:
            >>> encoder_net = encoder(0.1)
            >>> x = torch.randn(2, 3, 256, 256)
            >>> features = encoder_net(x)  # Shape: [2, 1024, 13, 13]
        """
        output = self.main(input)
        return output

# ================================ DECODER COMPONENT ================================

class decoder(nn.Module):
    """
    Decoder component of the SegNet-style architecture.
    
    The decoder progressively increases spatial dimensions while decreasing feature
    depth, reconstructing the segmentation mask from encoded features. It uses
    transposed convolutions for upsampling and includes dropout for regularization.
    
    Architecture Flow:
    Input(1024) → TransConv(512) → TransConv(256) → TransConv(128) → TransConv(64) → TransConv(classes)
    
    Spatial Reconstruction:
    H/16×W/16 → H/16×W/16 → H/8×W/8 → H/4×W/4 → H/2×W/2 → H×W
    
    Args:
        batchNorm_momentum (float): Momentum parameter for batch normalization layers
        num_classes (int, optional): Number of output segmentation classes (default: 23)
    
    Attributes:
        main (nn.Sequential): Sequential container of decoder layers
        
    Example:
        >>> decoder_net = decoder(batchNorm_momentum=0.1, num_classes=5)
        >>> encoded_features = torch.randn(4, 1024, 16, 16)
        >>> segmentation = decoder_net(encoded_features)
        >>> print(f"Output shape: {segmentation.shape}")  # [4, 5, 256, 256]
    
    Notes:
        - Uses transposed convolutions for learnable upsampling
        - Dropout2d layers prevent overfitting during training
        - Built-in softmax activation (consider removing for training with CrossEntropy)
        - Final layer produces class probability maps
    """
    
    def __init__(self, batchNorm_momentum, num_classes=23):
        """
        Initialize the decoder with specified parameters.
        
        Args:
            batchNorm_momentum (float): Momentum for batch normalization
            num_classes (int, optional): Number of output classes (default: 23)
        """
        super(decoder, self).__init__()
        
        # Sequential decoder architecture
        self.main = nn.Sequential(
            # ===================== LAYER 1: INITIAL UPSAMPLING =====================
            # Input: (N, 1024, H/16, W/16) → Output: (N, 512, H/16, W/16)
            # Note: stride=1, padding=0 increases spatial dimensions slightly
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.Dropout2d(),  # 2D dropout for regularization
            nn.ReLU(True),

            # ===================== LAYER 2: SPATIAL UPSAMPLING =====================
            # Input: (N, 512, H/16, W/16) → Output: (N, 256, H/8, W/8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            # ===================== LAYER 3: CONTINUED UPSAMPLING =====================
            # Input: (N, 256, H/8, W/8) → Output: (N, 128, H/4, W/4)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            # ===================== LAYER 4: FINE-SCALE RECONSTRUCTION =====================
            # Input: (N, 128, H/4, W/4) → Output: (N, 64, H/2, W/2)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(True),  # No dropout in final layers

            # ===================== LAYER 5: FINAL CLASSIFICATION =====================
            # Input: (N, 64, H/2, W/2) → Output: (N, num_classes, H, W)
            nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1, bias=False),
            
            # WARNING: Built-in softmax may interfere with CrossEntropy loss training
            # Consider removing this line and applying softmax during inference only
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        """
        Forward pass through the decoder.
        
        Args:
            input (torch.Tensor): Encoded features with shape (N, 1024, H/16, W/16)
        
        Returns:
            torch.Tensor: Segmentation probabilities with shape (N, num_classes, H, W)
                         Each pixel contains class probability distribution
        
        Example:
            >>> decoder_net = decoder(0.1, num_classes=3)
            >>> features = torch.randn(1, 1024, 16, 16)
            >>> probabilities = decoder_net(features)  # Shape: [1, 3, 256, 256]
            >>> predictions = torch.argmax(probabilities, dim=1)  # Class indices
        
        Note:
            Output contains probabilities (due to softmax), not logits.
            For training with CrossEntropy loss, consider removing the softmax layer.
        """
        output = self.main(input)
        return output

# ================================ COMPLETE SEGNET ARCHITECTURE ================================

class SegNet(nn.Module):
    """
    Complete SegNet-style architecture for semantic segmentation.
    
    This implementation combines the encoder and decoder components into a complete
    segmentation network. The architecture follows an encoder-decoder paradigm
    without skip connections, making it simpler than U-Net but potentially less
    precise for fine-grained segmentation tasks.
    
    Architecture Characteristics:
    - **Encoder**: Progressive feature extraction with spatial downsampling
    - **Decoder**: Feature reconstruction with spatial upsampling
    - **No skip connections**: Unlike U-Net, relies solely on bottleneck features
    - **Symmetric structure**: Mirror image encoder-decoder design
    - **Built-in softmax**: Produces probabilities rather than logits
    
    Use Cases:
    - Semantic segmentation of natural images
    - Medical image segmentation (with appropriate preprocessing)
    - Scene understanding and object delineation
    - Applications where memory efficiency is important
    
    Args:
        batchNorm_momentum (float): Momentum for all batch normalization layers
                                   Typical values: 0.1 (default) to 0.01 (slower adaptation)
        num_classes (int, optional): Number of segmentation classes (default: 23)
                                    Should match your dataset's class count
    
    Attributes:
        batchNorm_momentum (float): Stored batch normalization momentum
        num_classes (int): Number of output classes
        encoder (encoder): Encoder component for feature extraction
        decoder (decoder): Decoder component for mask reconstruction
        
    Example:
        >>> # Multi-class segmentation
        >>> model = SegNet(batchNorm_momentum=0.1, num_classes=5)
        >>> input_images = torch.randn(4, 3, 256, 256)
        >>> segmentation_maps = model(input_images)
        >>> print(f"Output: {segmentation_maps.shape}")  # [4, 5, 256, 256]
        >>> 
        >>> # Binary segmentation (background + foreground)
        >>> binary_model = SegNet(batchNorm_momentum=0.05, num_classes=2)
        >>> predictions = binary_model(input_images)
        >>> 
        >>> # Medical imaging example
        >>> medical_model = SegNet(batchNorm_momentum=0.1, num_classes=8)
        >>> ct_scans = torch.randn(2, 3, 512, 512)  # High resolution
        >>> organ_segmentation = medical_model(ct_scans)
    
    Training Considerations:
        - **Loss Function**: May need to modify for CrossEntropy (remove softmax)
        - **Learning Rate**: Start with 1e-3, reduce if training is unstable  
        - **Batch Size**: Adjust based on memory constraints and image resolution
        - **Data Augmentation**: Important due to lack of skip connections
        
    Performance Characteristics:
        - **Memory Usage**: Lower than U-Net due to no skip connections
        - **Training Speed**: Faster than U-Net, moderate complexity
        - **Accuracy**: Good for coarse segmentation, may struggle with fine details
        - **Generalization**: Benefits from strong data augmentation
        
    Notes:
        - Consider removing softmax layer if training with CrossEntropy loss
        - May require careful initialization due to deep architecture
        - Dropout layers help prevent overfitting but may slow convergence
        - Architecture works best with sufficient training data
    """
    
    def __init__(self, batchNorm_momentum, num_classes=23):
        """
        Initialize the complete SegNet architecture.
        
        Creates encoder and decoder components with shared configuration parameters
        and connects them in a feed-forward manner.
        
        Args:
            batchNorm_momentum (float): Momentum for batch normalization layers
                                       Controls adaptation rate of running statistics
            num_classes (int, optional): Number of output segmentation classes
                                        Must match your dataset's class count
        
        Example:
            >>> # Standard configuration
            >>> model = SegNet(0.1, num_classes=10)
            >>> 
            >>> # Conservative batch norm (slower adaptation)
            >>> stable_model = SegNet(0.01, num_classes=5)
            >>> 
            >>> # Binary segmentation
            >>> binary_model = SegNet(0.1, num_classes=2)
        """
        super(SegNet, self).__init__()
        
        # Store configuration parameters
        self.batchNorm_momentum = batchNorm_momentum
        self.num_classes = num_classes
        
        # Initialize encoder and decoder components
        self.encoder = encoder(self.batchNorm_momentum, self.num_classes)
        self.decoder = decoder(self.batchNorm_momentum, self.num_classes)

    def forward(self, x):
        """
        Forward pass through the complete SegNet architecture.
        
        Processes input images through encoder-decoder pipeline to produce
        pixel-wise class probability predictions.
        
        Args:
            x (torch.Tensor): Input image batch with shape (N, 3, H, W) where:
                             - N: batch size
                             - 3: RGB channels (assuming color images)
                             - H, W: spatial dimensions (height, width)
        
        Returns:
            torch.Tensor: Segmentation probability maps with shape (N, num_classes, H, W)
                         Each pixel contains probability distribution over classes
                         Values sum to 1.0 across class dimension due to softmax
        
        Example:
            >>> model = SegNet(0.1, num_classes=3)
            >>> images = torch.randn(2, 3, 128, 128)
            >>> 
            >>> # Forward pass
            >>> probabilities = model(images)  # Shape: [2, 3, 128, 128]
            >>> 
            >>> # Get class predictions
            >>> predictions = torch.argmax(probabilities, dim=1)  # Shape: [2, 128, 128]
            >>> 
            >>> # Get confidence scores
            >>> confidence = torch.max(probabilities, dim=1)[0]  # Shape: [2, 128, 128]
        
        Processing Flow:
            Input → Encoder (feature extraction) → Latent Features → Decoder (reconstruction) → Output
        
        Notes:
            - Output contains probabilities (0-1 range) due to built-in softmax
            - For loss computation with CrossEntropy, consider modifying decoder
            - Spatial dimensions are preserved (input H,W = output H,W)
            - No intermediate feature access (unlike U-Net with skip connections)
        """
        # ===================== ENCODING PHASE =====================
        # Extract hierarchical features while reducing spatial resolution
        latent = self.encoder(x)
        
        # Optional: Uncomment for debugging feature shapes
        # print('Latent Shape:', latent.shape)
        
        # ===================== DECODING PHASE =====================  
        # Reconstruct segmentation mask from encoded features
        output = self.decoder(latent)
        
        # Optional: Uncomment for debugging output shapes
        # print('Output Shape:', output.shape)

        return output
    
    def __repr__(self):
        """
        String representation of the SegNet model.
        
        Returns:
            str: Human-readable description of the model configuration
        """
        return (f"SegNet(batchNorm_momentum={self.batchNorm_momentum}, "
                f"num_classes={self.num_classes})")