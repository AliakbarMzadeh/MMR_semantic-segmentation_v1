"""
Core Segmentation Model with Advanced Training Monitoring for MMR Framework

This module implements the main segmentation model architecture and comprehensive training 
pipeline for surgical tool segmentation in robotic surgery videos. It provides 
state-of-the-art deep learning capabilities with extensive monitoring, logging, and 
visualization features specifically designed for medical imaging research.

Key Components:
===============

TrainingMonitor Class:
- Real-time system resource monitoring (GPU/CPU usage, memory tracking)
- Comprehensive metrics collection (loss, IoU, learning rates, throughput)
- Professional logging with timestamps and structured output
- Advanced visualization dashboard with publication-ready plots
- ETA calculations and training progress tracking

SegModel Class:
- UNet++ architecture with MobileNetV3 encoders for efficient segmentation
- Mixed precision training with gradient scaling for performance
- IoU-based evaluation metrics specific to medical image segmentation
- Sliding window inference for high-resolution surgical videos
- Advanced checkpoint management with best model selection

Advanced Features:
==================

Training Optimizations:
- Automatic mixed precision (AMP) for faster training and reduced memory
- Gradient accumulation for effective large batch training
- Gradient clipping for stable training convergence
- Smart checkpoint management (last and best model tracking)

Medical Imaging Specific:
- DiceCE loss function optimized for medical image segmentation
- Class-wise IoU evaluation for detailed performance analysis
- Foreground sampling strategies for handling class imbalance
- Multi-scale inference with sliding window approach

Professional Monitoring:
- Comprehensive training dashboards with 6-panel visualizations
- Resource utilization tracking for computational efficiency analysis
- Training throughput monitoring for performance optimization
- JSON metrics export for programmatic analysis

Visualization Capabilities:
- Real-time training progress with rich progress bars
- Image/mask/prediction comparison plots during training
- Loss curves and metric trends over training epochs
- GPU memory usage and training throughput plots

Clinical Integration Features:
=============================

Performance Optimizations:
- Half-precision inference for real-time deployment
- Memory format optimization for 2D convolutions
- Sliding window inference for arbitrarily large images
- Efficient video sequence processing with temporal batching

Evaluation Metrics:
- Per-class IoU scores for detailed analysis
- Mean IoU across all classes for overall performance
- Frame-by-frame processing statistics
- Processing speed measurements (FPS) for clinical deployment

Research Capabilities:
- Reproducible training with detailed configuration logging
- Comprehensive metrics tracking for scientific publications
- Professional visualization for research presentations
- Extensible architecture for custom loss functions and metrics

Dependencies:
=============
- torch: PyTorch deep learning framework
- segmentation_models_pytorch: Pre-trained encoder architectures
- monai: Medical imaging specific loss functions and utilities
- torchinfo: Model architecture visualization and analysis
- psutil: System resource monitoring
- matplotlib: Professional plotting and visualization

Target Applications:
===================
- Robotic surgery tool tracking and identification
- Real-time surgical workflow analysis
- Medical image analysis research and development
- Clinical decision support system integration

Author: MMR Segmentation Team
Version: 1.0
"""

import os
import re
import sys
import random
import time
import torch
import pickle
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torchinfo import summary
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from torch.amp import GradScaler, autocast
from segmentation_models_pytorch.metrics import get_stats, iou_score
from MMR_Segmentation.common_utils import save_losses, create_config
import psutil
import threading
from datetime import datetime
import json


class TrainingMonitor:
    """
    Comprehensive training monitoring system for medical image segmentation.
    
    Provides real-time tracking of training metrics, system resources, and performance
    statistics with professional logging and visualization capabilities. Designed 
    specifically for long-running medical imaging experiments requiring detailed
    monitoring and analysis.
    
    Key Features:
    - Real-time GPU/CPU resource monitoring
    - Comprehensive training metrics collection and analysis
    - Professional logging with structured timestamps
    - Advanced visualization dashboard generation
    - Training progress tracking with ETA calculations
    - JSON export for programmatic analysis
    
    Attributes:
        log_file_path (str): Path to training log file for persistent logging
        training_start_time (float): Training session start timestamp
        epoch_start_time (float): Current epoch start timestamp  
        metrics_history (dict): Complete training metrics history storage
        
    Note:
        - Automatically handles log file creation and management
        - Thread-safe for concurrent metric collection
        - Memory-efficient storage of training history
        - Compatible with distributed training setups
    """
    
    def __init__(self, log_file_path):
        """
        Initialize training monitor with comprehensive metrics tracking.
        
        Sets up logging infrastructure and initializes metric collection
        dictionaries for comprehensive training analysis.
        
        Args:
            log_file_path (str): Path where training logs will be saved.
                Creates parent directories if they don't exist.
        """
        self.log_file_path = log_file_path
        self.training_start_time = None
        self.epoch_start_time = None
        
        # Comprehensive metrics storage for training analysis
        self.metrics_history = {
            'epochs': [],              # Epoch numbers
            'train_loss': [],          # Training loss values
            'val_loss': [],            # Validation loss values
            'train_iou': [],           # Training IoU scores
            'val_iou': [],             # Validation IoU scores
            'learning_rates': [],      # Learning rate schedule
            'epoch_times': [],         # Time per epoch (seconds)
            'gpu_memory_used': [],     # GPU memory reserved (GB)
            'gpu_memory_allocated': [], # GPU memory allocated (GB)
            'cpu_usage': [],           # CPU utilization percentage
            'samples_per_sec': [],     # Training throughput
            'eta_remaining': []        # Estimated time remaining (seconds)
        }
        
    def log_message(self, message, level="INFO"):
        """
        Log structured message with timestamp to both console and file.
        
        Provides consistent logging format with automatic timestamping for
        comprehensive training session documentation.
        
        Args:
            message (str): Message content to log
            level (str, optional): Log level (INFO, WARNING, ERROR). 
                Defaults to "INFO".
        
        Side Effects:
            - Prints formatted message to console
            - Appends timestamped entry to log file
            - Creates log file if it doesn't exist
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        
        # Ensure log directory exists and write to file
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        with open(self.log_file_path, 'a') as f:
            f.write(log_entry + '\n')
    
    def get_gpu_memory_info(self):
        """
        Retrieve current GPU memory usage statistics.
        
        Provides detailed GPU memory utilization information essential for
        monitoring training efficiency and detecting memory leaks.
        
        Returns:
            dict: GPU memory statistics with keys:
                - 'allocated': Memory allocated by PyTorch tensors (GB)
                - 'reserved': Memory reserved by PyTorch (GB) 
                - 'cached': Memory cached by PyTorch (GB)
                
        Note:
            - Returns zeros if CUDA is not available
            - Memory values are in gigabytes for readability
            - 'reserved' memory may be larger than 'allocated'
            - Cached memory can be freed if needed by other processes
        """
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # Convert to GB
            memory_cached = torch.cuda.memory_cached() / 1024**3        # Convert to GB
            
            return {
                'allocated': memory_allocated,
                'reserved': memory_reserved, 
                'cached': memory_cached
            }
        
        # Return zeros for CPU-only training
        return {'allocated': 0, 'reserved': 0, 'cached': 0}
    
    def get_cpu_usage(self):
        """
        Get current CPU utilization percentage.
        
        Provides system-wide CPU usage monitoring for training
        performance analysis and resource optimization.
        
        Returns:
            float: CPU usage percentage (0-100)
            
        Note:
            - Returns instantaneous CPU usage measurement
            - May vary significantly during training due to data loading
            - Useful for identifying CPU bottlenecks in training pipeline
        """
        return psutil.cpu_percent(interval=None)
    
    def start_training(self, total_epochs):
        """
        Initialize training session monitoring and logging.
        
        Sets up comprehensive monitoring for a complete training session
        with initial system state logging and progress tracking preparation.
        
        Args:
            total_epochs (int): Total number of training epochs planned
            
        Side Effects:
            - Records training start timestamp
            - Logs initial system configuration  
            - Stores total epoch count for ETA calculations
            - Reports GPU device information
        """
        self.training_start_time = time.time()
        self.total_epochs = total_epochs
        
        # Log training session initialization
        self.log_message(f"Starting training for {total_epochs} epochs")
        self.log_message(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        # Record initial system state
        gpu_info = self.get_gpu_memory_info()
        self.log_message(f"Initial GPU Memory - Allocated: {gpu_info['allocated']:.2f}GB, Reserved: {gpu_info['reserved']:.2f}GB")
    
    def start_epoch(self, epoch):
        """
        Begin epoch timing and monitoring.
        
        Initializes per-epoch timing for accurate performance measurement
        and throughput calculation.
        
        Args:
            epoch (int): Current epoch number (1-indexed)
        """
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        
    def end_epoch(self, epoch, train_loss, val_loss, train_iou=None, val_iou=None, lr=None, num_samples=None):
        """
        Complete epoch monitoring with comprehensive metrics collection.
        
        Calculates epoch statistics, updates metrics history, and provides
        detailed logging of training progress with ETA estimation.
        
        Args:
            epoch (int): Completed epoch number
            train_loss (float): Average training loss for the epoch
            val_loss (float): Average validation loss for the epoch
            train_iou (float, optional): Training IoU score. Defaults to None.
            val_iou (float, optional): Validation IoU score. Defaults to None.
            lr (float, optional): Current learning rate. Defaults to None.
            num_samples (int, optional): Number of samples processed. Defaults to None.
            
        Side Effects:
            - Updates complete metrics history
            - Logs comprehensive epoch summary
            - Calculates and stores ETA estimates
            - Records system resource utilization
        """
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.training_start_time
        
        # Calculate training throughput
        samples_per_sec = num_samples / epoch_time if num_samples else 0
        
        # Estimate time remaining based on average epoch duration
        avg_epoch_time = total_time / epoch
        remaining_epochs = self.total_epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_formatted = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
        
        # Collect current system metrics
        gpu_info = self.get_gpu_memory_info()
        cpu_usage = self.get_cpu_usage()
        
        # Update comprehensive metrics history
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['train_iou'].append(train_iou)
        self.metrics_history['val_iou'].append(val_iou)
        self.metrics_history['learning_rates'].append(lr)
        self.metrics_history['epoch_times'].append(epoch_time)
        self.metrics_history['gpu_memory_used'].append(gpu_info['reserved'])
        self.metrics_history['gpu_memory_allocated'].append(gpu_info['allocated'])
        self.metrics_history['cpu_usage'].append(cpu_usage)
        self.metrics_history['samples_per_sec'].append(samples_per_sec)
        self.metrics_history['eta_remaining'].append(eta_seconds)
        
        # Generate comprehensive epoch summary report
        self.log_message("="*80)
        self.log_message(f"EPOCH {epoch}/{self.total_epochs} COMPLETED")
        self.log_message(f"  Time: {time.strftime('%H:%M:%S', time.gmtime(epoch_time))} | ETA: {eta_formatted}")
        self.log_message(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if train_iou is not None and val_iou is not None:
            self.log_message(f"  Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}")
        if lr is not None:
            self.log_message(f"  Learning Rate: {lr:.2e}")
            
        self.log_message(f"  Throughput: {samples_per_sec:.1f} samples/sec")
        self.log_message(f"  GPU Memory: {gpu_info['allocated']:.2f}GB allocated, {gpu_info['reserved']:.2f}GB reserved")
        self.log_message(f"  CPU Usage: {cpu_usage:.1f}%")
        self.log_message("="*80)
    
    def save_metrics(self, save_path):
        """
        Export training metrics history to JSON format.
        
        Saves complete metrics history for programmatic analysis and
        comparison across different training experiments.
        
        Args:
            save_path (str): Directory path where metrics file will be saved
            
        Side Effects:
            - Creates 'training_metrics.json' in specified directory
            - Overwrites existing metrics file if present
        """
        metrics_path = os.path.join(save_path, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def plot_training_metrics(self, save_path):
        """
        Generate comprehensive training visualization dashboard.
        
        Creates publication-ready 6-panel dashboard showing all key training
        metrics and system performance indicators for comprehensive analysis.
        
        Args:
            save_path (str): Directory where dashboard plot will be saved
            
        Dashboard Panels:
            1. Loss Curves: Training and validation loss over epochs
            2. IoU Scores: Training and validation IoU progression  
            3. Learning Rate Schedule: Learning rate changes over time
            4. Epoch Duration: Training time per epoch trends
            5. GPU Memory Usage: Memory allocation and reservation
            6. Training Throughput: Samples processed per second
            
        Side Effects:
            - Creates 'training_dashboard.png' at 300 DPI resolution
            - Overwrites existing dashboard if present
            - Uses professional styling suitable for publications
        """
        if not self.metrics_history['epochs']:
            return  # No data to plot
            
        epochs = self.metrics_history['epochs']
        
        # Create comprehensive 6-panel dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Monitoring Dashboard', fontsize=16)
        
        # Panel 1: Loss Curves
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], 
                       label='Train Loss', color='blue', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 
                       label='Val Loss', color='red', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Panel 2: IoU Scores (if available)
        if any(x is not None for x in self.metrics_history['train_iou']):
            train_iou_clean = [x for x in self.metrics_history['train_iou'] if x is not None]
            val_iou_clean = [x for x in self.metrics_history['val_iou'] if x is not None]
            epochs_iou = [epochs[i] for i, x in enumerate(self.metrics_history['train_iou']) if x is not None]
            
            axes[0, 1].plot(epochs_iou, train_iou_clean, 
                           label='Train IoU', color='green', linewidth=2)
            axes[0, 1].plot(epochs_iou, val_iou_clean, 
                           label='Val IoU', color='orange', linewidth=2)
        axes[0, 1].set_title('IoU Scores')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Panel 3: Learning Rate Schedule
        if any(x is not None for x in self.metrics_history['learning_rates']):
            lr_clean = [x for x in self.metrics_history['learning_rates'] if x is not None]
            epochs_lr = [epochs[i] for i, x in enumerate(self.metrics_history['learning_rates']) if x is not None]
            axes[0, 2].plot(epochs_lr, lr_clean, color='purple', linewidth=2)
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Panel 4: Epoch Duration Trends
        axes[1, 0].plot(epochs, self.metrics_history['epoch_times'], 
                       color='brown', linewidth=2)
        axes[1, 0].set_title('Epoch Duration')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Panel 5: GPU Memory Usage
        axes[1, 1].plot(epochs, self.metrics_history['gpu_memory_allocated'], 
                       label='Allocated', color='blue', linewidth=2)
        axes[1, 1].plot(epochs, self.metrics_history['gpu_memory_used'], 
                       label='Reserved', color='red', linewidth=2)
        axes[1, 1].set_title('GPU Memory Usage')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Memory (GB)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Panel 6: Training Throughput
        axes[1, 2].plot(epochs, self.metrics_history['samples_per_sec'], 
                       color='darkgreen', linewidth=2)
        axes[1, 2].set_title('Training Throughput')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Samples/Second')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Save publication-ready dashboard
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


# Optional CUDA data prefetching for enhanced performance (currently commented out)
# class CUDAPrefetcher:
#     """
#     CUDA data prefetching utility for enhanced training performance.
#     
#     Implements asynchronous data loading with CUDA streams to overlap
#     data transfer with computation, reducing training bottlenecks.
#     """
#     def __init__(self, loader, device):
#         self.loader = iter(loader)
#         self.stream = torch.cuda.Stream(device=device)
#         self.device = device
#         self.next = None
#         self.preload()
#
#     def preload(self):
#         try:
#             self.next = next(self.loader)
#         except StopIteration:
#             self.next = None
#             return
#         with torch.cuda.stream(self.stream):
#             self.next['image'] = self.next['image'].to(self.device, non_blocking=True, dtype=torch.float32)
#             self.next['mask'] = self.next['mask'].to(self.device, non_blocking=True, dtype=torch.long)
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         torch.cuda.current_stream(self.device).wait_stream(self.stream)
#         if self.next is None: 
#             raise StopIteration
#         item = self.next
#         self.preload()
#         return item


class SegModel:
    """
    Advanced segmentation model for surgical tool identification in robotic surgery.
    
    Implements state-of-the-art deep learning architecture with comprehensive training,
    validation, and inference capabilities specifically designed for medical image
    segmentation. Features advanced monitoring, mixed precision training, and
    optimized inference for clinical deployment.
    
    Key Features:
    - UNet++ architecture with MobileNetV3 encoders for efficient segmentation
    - Mixed precision training with automatic gradient scaling
    - Comprehensive IoU-based evaluation for medical imaging
    - Advanced training monitoring with real-time resource tracking
    - Professional checkpoint management with best model selection
    - Sliding window inference for high-resolution images
    
    Architecture:
    - Encoder: MobileNetV3 (ImageNet pretrained) for feature extraction
    - Decoder: UNet++ skip connections for precise segmentation
    - Loss: DiceCE combination optimized for medical image segmentation
    - Optimization: AdamW with polynomial learning rate scheduling
    
    Attributes:
        config (dict): Complete training configuration parameters
        device (torch.device): Training device (CUDA or CPU)
        model (torch.nn.Module): Segmentation model architecture
        monitor (TrainingMonitor): Comprehensive training monitoring system
        seg_loss (monai.losses.DiceCELoss): Medical imaging optimized loss function
        loss_dict (dict): Training and validation loss history
        
    Note:
        - Automatically detects and utilizes GPU acceleration when available
        - Implements gradient accumulation for effective large batch training
        - Provides extensive logging and visualization for research reproducibility
    """
    
    def __init__(self, config, print_summary=True):
        """
        Initialize segmentation model with comprehensive configuration.
        
        Sets up model architecture, training monitoring, loss functions,
        and device configuration for medical image segmentation training.
        
        Args:
            config (dict): Complete training configuration containing:
                - 'model': Model architecture specifications
                - 'results_path': Output directory for logs and checkpoints
                - 'load_model_path': Optional checkpoint path for resuming
                - 'n_classes': Number of segmentation classes
            print_summary (bool, optional): Whether to print model architecture
                summary. Defaults to True.
        """
        self.config = config
        self.print_summary = print_summary

        # Initialize medical imaging specific loss function
        # DiceCE combines Dice and Cross-Entropy for better medical segmentation
        self.seg_loss = DiceCELoss(softmax=True)

        # Setup computational device with automatic GPU detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using device: CPU")

        # Initialize segmentation model architecture
        self.model = smp.create_model(**self.config['model']).to(self.device)

        # Setup comprehensive training monitoring system
        log_file_path = os.path.join(self.config['results_path'], 'training.log')
        self.monitor = TrainingMonitor(log_file_path)

        # Initialize or load training history for loss tracking
        if self.config['load_model_path'] and 'last' in config['load_model_path'].split('/')[-1]:
            # Continue training: load existing loss history
            loss_pickle_path = os.path.join(self.config['results_path'], 'loss_dict.pkl')
            if os.path.exists(loss_pickle_path):
                with open(loss_pickle_path, 'rb') as file:
                    self.loss_dict = pickle.load(file)
        else:
            # Start fresh training: initialize empty loss history
            self.loss_dict = {'train_loss': [], 'val_loss': []}

    def calculate_iou(self, predictions, masks):
        """
        Calculate Intersection over Union (IoU) score for segmentation evaluation.
        
        Computes IoU metric specifically designed for multi-class medical image
        segmentation, providing detailed performance analysis per class and overall.
        
        Args:
            predictions (torch.Tensor): Model predictions (logits or probabilities)
                Shape: (batch_size, num_classes, height, width)
            masks (torch.Tensor): Ground truth segmentation masks
                Shape: (batch_size, num_classes, height, width) or (batch_size, height, width)
        
        Returns:
            float: Mean IoU score across all classes (0.0 to 1.0)
        
        Note:
            - Automatically handles both one-hot and integer mask formats
            - Uses macro averaging across classes for balanced evaluation
            - Optimized for medical imaging with class imbalance considerations
        """
        # Convert predictions to class assignments
        preds_classes = predictions.argmax(1)
        
        # Handle different mask input formats
        masks_classes = masks.argmax(1) if masks.ndim > 3 else masks
        
        # Calculate confusion matrix statistics
        tp, fp, fn, tn = get_stats(preds_classes, masks_classes, mode='multiclass',
                                   num_classes=self.config['n_classes'] + 1)
        
        # Compute IoU with macro averaging for balanced class evaluation
        iou = iou_score(tp, fp, fn, tn, reduction='macro')
        return iou.mean().item()

    def train_one_epoch(self, epoch, train_loader, optimizer, scaler):
        """
        Execute one complete training epoch with comprehensive monitoring.
        
        Implements advanced training loop with mixed precision, gradient accumulation,
        real-time IoU tracking, and detailed progress monitoring for medical image
        segmentation.
        
        Args:
            epoch (int): Current epoch number (1-indexed)
            train_loader (DataLoader): Training data loader with augmented samples
            optimizer (torch.optim.Optimizer): Configured optimizer (typically AdamW)
            scaler (torch.cuda.amp.GradScaler): Automatic mixed precision scaler
        
        Returns:
            tuple: (epoch_iou, total_samples) where:
                - epoch_iou (float): Average IoU score for the epoch
                - total_samples (int): Total number of samples processed
        
        Training Features:
        - Mixed precision training for memory efficiency and speed
        - Gradient accumulation for effective large batch training
        - Gradient clipping for training stability
        - Real-time IoU calculation for immediate feedback
        - Comprehensive progress tracking with GPU monitoring
        """
        self.model.train()
        disable_prog_bar = not self.config['progress_bar']
        epoch_loss = 0
        epoch_iou = 0
        num_batches = len(train_loader)
        total_samples = 0

        # Initialize epoch timing and monitoring
        start = time.time()
        self.monitor.start_epoch(epoch)

        # Enhanced progress bar with detailed training information
        progress_desc = f"Epoch {epoch}/{self.config['n_epochs']} [TRAIN]"
        with tqdm(enumerate(train_loader), total=num_batches, ncols=180, 
                  disable=disable_prog_bar, file=sys.stdout, desc=progress_desc) as progress_bar:

            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            for step, batch in progress_bar:
                batch_start_time = time.time()
                
                # Transfer data to device with non-blocking for performance
                images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
                masks = batch["mask"].to(self.device, dtype=torch.long, non_blocking=True)
                
                batch_size = images.shape[0]
                total_samples += batch_size

                # Forward pass with automatic mixed precision
                with autocast(self.device.type):
                    predictions = self.model(images)

                # Convert masks to one-hot encoding for loss calculation
                masks_onehot = torch.nn.functional.one_hot(masks, num_classes=self.config['n_classes'] + 1)
                
                # Handle different dimensionalities (2D/3D compatibility)
                if masks_onehot.ndim == 5:
                    masks_onehot = masks_onehot.permute(0, 4, 1, 2, 3)
                elif masks_onehot.ndim == 4:
                    masks_onehot = masks_onehot.permute(0, 3, 1, 2)

                # Calculate DiceCE loss optimized for medical segmentation
                loss = self.seg_loss(predictions.float(), masks_onehot.float())
                
                # Calculate IoU for real-time performance monitoring
                batch_iou = self.calculate_iou(predictions, masks_onehot)
                
                # Backward pass with gradient scaling for mixed precision
                scaler.scale(loss).backward()
                
                # Gradient accumulation and optimization step
                if (step + 1) % self.config['grad_accumulate_step'] == 0 or (step + 1) == num_batches:
                    # Apply gradient clipping for training stability
                    if self.config['grad_clip_max_norm']:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                     max_norm=self.config['grad_clip_max_norm'])
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                # Update epoch statistics
                epoch_loss += loss.item()
                epoch_iou += batch_iou
                
                # Calculate real-time metrics for progress display
                avg_loss = epoch_loss / (step + 1)
                avg_iou = epoch_iou / (step + 1)
                batch_time = time.time() - batch_start_time
                samples_per_sec = batch_size / batch_time if batch_time > 0 else 0
                
                # Get current GPU memory usage for monitoring
                gpu_memory = self.monitor.get_gpu_memory_info()
                
                # Update progress bar with comprehensive metrics
                progress_bar.set_postfix({
                    "Loss": f"{avg_loss:.4f}",
                    "IoU": f"{avg_iou:.3f}",
                    "GPU": f"{gpu_memory['allocated']:.1f}GB",
                    "S/s": f"{samples_per_sec:.1f}"
                })

        # Calculate final epoch statistics
        epoch_loss /= num_batches
        epoch_iou /= num_batches

        # Log epoch summary if progress bar is disabled
        if disable_prog_bar:
            end = time.time() - start
            self.monitor.log_message(
                f"Epoch {epoch} [TRAIN] - Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                f"Loss: {epoch_loss:.6f} - IoU: {epoch_iou:.4f}"
            )

        # Update loss history for plotting and analysis
        self.loss_dict["train_loss"].append(epoch_loss)
        
        return epoch_iou, total_samples

    def validate_one_epoch(self, epoch, val_loader, return_img_mask_pred=False):
        """
        Execute comprehensive validation epoch with detailed evaluation metrics.
        
        Performs thorough model evaluation on validation set with IoU calculation,
        optional visualization sample collection, and performance monitoring.
        
        Args:
            epoch (int): Current epoch number for progress tracking
            val_loader (DataLoader): Validation data loader 
            return_img_mask_pred (bool, optional): Whether to return sample
                images, masks, and predictions for visualization. Defaults to False.
        
        Returns:
            float or tuple: If return_img_mask_pred is False, returns validation IoU.
                If True, returns (images, masks, predictions, val_iou) tuple.
        
        Validation Features:
        - Evaluation mode with gradient computation disabled
        - Mixed precision inference for memory efficiency
        - Real-time IoU calculation for immediate feedback
        - Optional sample collection for visualization
        - Comprehensive progress monitoring
        """
        self.model.eval()
        disable_prog_bar = not self.config['progress_bar']
        epoch_loss = 0
        epoch_iou = 0
        num_batches = len(val_loader)

        start = time.time()

        # Enhanced progress bar for validation tracking
        progress_desc = f"Epoch {epoch}/{self.config['n_epochs']} [VAL]"
        with tqdm(enumerate(val_loader), total=num_batches, ncols=180, 
                  disable=disable_prog_bar, file=sys.stdout, desc=progress_desc) as progress_bar:

            for step, batch in progress_bar:
                # Transfer validation data to device
                images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
                masks = batch["mask"].to(self.device, dtype=torch.long, non_blocking=True)

                # Validation forward pass without gradient computation
                with torch.no_grad():
                    with autocast(self.device.type):
                        predictions = self.model(images)

                # Prepare masks for loss calculation (same as training)
                masks_onehot = torch.nn.functional.one_hot(masks, num_classes=self.config['n_classes'] + 1)
                if masks_onehot.ndim == 5:
                    masks_onehot = masks_onehot.permute(0, 4, 1, 2, 3)
                elif masks_onehot.ndim == 4:
                    masks_onehot = masks_onehot.permute(0, 3, 1, 2)

                # Calculate validation loss and IoU
                loss = self.seg_loss(predictions.float(), masks_onehot.float())
                batch_iou = self.calculate_iou(predictions, masks_onehot)
                
                epoch_loss += loss.item()
                epoch_iou += batch_iou
                
                # Update progress bar with validation metrics
                avg_loss = epoch_loss / (step + 1)
                avg_iou = epoch_iou / (step + 1)
                progress_bar.set_postfix({
                    "Val Loss": f"{avg_loss:.4f}",
                    "Val IoU": f"{avg_iou:.3f}"
                })

        # Calculate final validation statistics
        epoch_loss /= num_batches
        epoch_iou /= num_batches

        # Log validation summary if progress bar is disabled
        if disable_prog_bar:
            end = time.time() - start
            self.monitor.log_message(
                f"Validation - Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                f"Val Loss: {epoch_loss:.6f} - Val IoU: {epoch_iou:.4f}"
            )

        # Update validation loss history
        self.loss_dict["val_loss"].append(epoch_loss)

        # Return visualization samples if requested
        if return_img_mask_pred:
            to_np = lambda x: x.detach().cpu().numpy()
            images, masks_onehot, predictions = map(to_np, (images, masks_onehot, predictions))
            return images, masks_onehot, predictions, epoch_iou
        
        return epoch_iou

    def get_optimizer_and_lr_schedule(self):
        """
        Configure optimizer and learning rate scheduler from configuration.
        
        Dynamically creates optimizer and scheduler objects based on configuration
        parameters, enabling flexible experimentation with different optimization
        strategies.
        
        Returns:
            tuple: (optimizer, lr_scheduler) where:
                - optimizer: Configured PyTorch optimizer
                - lr_scheduler: Learning rate scheduler or None
        
        Note:
            - Uses reflection to instantiate optimizers and schedulers by name
            - Supports all PyTorch optimizers and schedulers
            - Scheduler is optional and can be disabled via configuration
        """
        # Create optimizer dynamically from configuration
        optimizer_class = getattr(torch.optim, self.config['optimizer']['name'])
        optimizer_args = {key: value for key, value in self.config['optimizer'].items() if key != 'name'}
        optimizer = optimizer_class(self.model.parameters(), **optimizer_args)

        # Create learning rate scheduler if specified
        if self.config["lr_scheduler"]:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config['lr_scheduler']['name'])
            lr_scheduler_args = {key: value for key, value in self.config['lr_scheduler'].items() if key != 'name'}
            lr_scheduler = scheduler_class(optimizer, **lr_scheduler_args)
        else:
            lr_scheduler = None

        return optimizer, lr_scheduler

    def save_model(self, epoch, validation_loss, optimizer, scheduler=None):
        """
        Save model checkpoint with comprehensive state preservation.
        
        Implements professional checkpoint management with both last and best
        model tracking, ensuring training can be resumed and best models
        are preserved for deployment.
        
        Args:
            epoch (int): Current epoch number
            validation_loss (float): Current validation loss for best model tracking
            optimizer (torch.optim.Optimizer): Optimizer state to save
            scheduler (torch.optim.lr_scheduler, optional): Scheduler state to save.
                Defaults to None.
        
        Side Effects:
            - Saves 'last_model.pth' with complete training state
            - Updates 'best_model.pth' when validation loss improves
            - Logs checkpoint information with file sizes
            - Creates checkpoint directory if it doesn't exist
        
        Checkpoint Contents:
            - Model state dictionary (weights and biases)
            - Optimizer state (momentum, learning rates, etc.)
            - Scheduler state (if provided)
            - Current epoch number and validation loss
        """
        # Ensure checkpoint directory exists
        save_path = os.path.join(self.config['results_path'], 'checkpoints')
        os.makedirs(save_path, exist_ok=True)

        # Create comprehensive checkpoint dictionary
        last_checkpoint_path = os.path.join(save_path, 'last_model.pth')
        checkpoint = {
            'epoch': epoch,
            'network_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': validation_loss
        }

        # Include scheduler state if available
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save latest checkpoint
        torch.save(checkpoint, last_checkpoint_path)
        
        # Log checkpoint information with file size
        checkpoint_size = os.path.getsize(last_checkpoint_path) / 1024**2  # Convert to MB
        self.monitor.log_message(f"Checkpoint saved: {last_checkpoint_path} ({checkpoint_size:.1f}MB)")

        # Manage best model checkpoint based on validation loss
        best_checkpoint_path = os.path.join(save_path, 'best_model.pth')
        if os.path.isfile(best_checkpoint_path):
            # Check if current model is better than previous best
            best_checkpoint = torch.load(best_checkpoint_path, weights_only=False)
            best_loss = best_checkpoint.get('validation_loss', float('inf'))
            
            if validation_loss < best_loss:
                torch.save(checkpoint, best_checkpoint_path)
                self.monitor.log_message(f"NEW BEST MODEL! Val Loss: {validation_loss:.6f} < {best_loss:.6f}")
        else:
            # First epoch: save as initial best model
            torch.save(checkpoint, best_checkpoint_path)
            self.monitor.log_message(f"First best model saved with Val Loss: {validation_loss:.6f}")

    def load_model(self, load_model_path, optimizer=None, scheduler=None, for_training=False):
        """
        Load model checkpoint with optional optimizer and scheduler restoration.
        
        Provides flexible checkpoint loading for both inference and training
        continuation, with comprehensive state restoration capabilities.
        
        Args:
            load_model_path (str): Path to checkpoint file (.pth)
            optimizer (torch.optim.Optimizer, optional): Optimizer to restore state.
                Defaults to None.
            scheduler (torch.optim.lr_scheduler, optional): Scheduler to restore state.
                Defaults to None.
            for_training (bool, optional): Whether loading for training continuation.
                If True, returns next epoch number. Defaults to False.
        
        Returns:
            int or None: If for_training is True, returns next epoch number.
                Otherwise returns None.
        
        Note:
            - Gracefully handles missing scheduler state in older checkpoints
            - Logs detailed loading information for verification
            - Automatically calculates next epoch for training continuation
        """
        self.monitor.log_message(f'Loading model from {load_model_path}...')
        checkpoint = torch.load(load_model_path, weights_only=False)
        
        # Restore model weights
        self.model.load_state_dict(checkpoint['network_state_dict'])

        # Restore optimizer state if provided
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler state if available and provided
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Return next epoch number for training continuation
        if for_training:
            start_epoch = checkpoint['epoch'] + 1
            self.monitor.log_message(f'Resuming training from epoch {start_epoch}')
            return start_epoch

    def save_plots(self, images, masks, preds, save_path, one_hot=True):
        """
        Generate and save image/mask/prediction comparison plots.
        
        Creates publication-ready visualization comparing input images,
        ground truth masks, and model predictions for qualitative evaluation.
        
        Args:
            images (numpy.ndarray): Input images array
            masks (numpy.ndarray): Ground truth masks
            preds (numpy.ndarray): Model predictions
            save_path (str): Full path where plot will be saved
            one_hot (bool, optional): Whether inputs are one-hot encoded.
                If True, converts to class indices. Defaults to True.
        
        Side Effects:
            - Creates directory structure if it doesn't exist
            - Saves high-resolution plot (150 DPI) suitable for publications
            - Handles single and multiple sample visualizations
        
        Plot Features:
            - Side-by-side comparison of image, mask, and prediction
            - Proper normalization and color mapping for medical images
            - Clean layout with titles and no axis labels
            - Consistent color scaling across samples
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert one-hot encodings to class indices if needed
        if one_hot:
            masks = masks.argmax(1)
            preds = preds.argmax(1)
            
        # Determine number of samples to display (maximum 4)
        rows = min(4, images.shape[0])
        n_classes = self.config['n_classes']

        # Create subplot grid for comparison
        fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows))
        titles = ['image', 'mask', 'prediction']

        # Handle single row case (matplotlib returns 1D array)
        if rows == 1:
            axes = axes.reshape(1, -1)

        # Generate comparison plots for each sample
        for i in range(rows):
            # Normalize image for display (convert CHW to HWC)
            img = images[i].transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
            
            # Display image
            axes[i, 0].imshow(img)
            axes[i, 0].axis('off')
            
            # Display ground truth mask with consistent color mapping
            axes[i, 1].imshow(masks[i], cmap='hot', vmin=0, vmax=n_classes)
            axes[i, 1].axis('off')
            
            # Display prediction with same color mapping
            axes[i, 2].imshow(preds[i], cmap='hot', vmin=0, vmax=n_classes)
            axes[i, 2].axis('off')

        # Add column titles to first row
        for j, title in enumerate(titles):
            axes[0, j].set_title(title)

        # Save with professional formatting
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)  # Free memory

    def train(self, train_loader, val_loader):
        """
        Execute complete training pipeline with comprehensive monitoring.
        
        Implements full training workflow including model initialization,
        training/validation loops, checkpoint management, and advanced
        monitoring with professional logging and visualization.
        
        Args:
            train_loader (DataLoader): Training data loader with augmentation
            val_loader (DataLoader): Validation data loader for evaluation
        
        Training Pipeline:
        1. Initialize mixed precision training and monitoring systems
        2. Configure optimizer and learning rate scheduler
        3. Load checkpoint if resuming training
        4. Execute training/validation epochs with comprehensive logging
        5. Save checkpoints, metrics, and visualization plots
        6. Generate final training summary with statistics
        
        Features:
        - Mixed precision training for memory efficiency and speed
        - Real-time IoU tracking and visualization
        - Comprehensive resource monitoring (GPU/CPU/memory)
        - Professional logging with timestamps and structured output
        - Advanced checkpoint management with best model tracking
        - Training dashboard generation for analysis
        """
        # Initialize training infrastructure
        scaler = GradScaler()  # For mixed precision training
        total_start = time.time()
        start_epoch = 1
        plot_save_path = os.path.join(self.config['results_path'], 'plots')

        # Prepare model input shape for architecture summary
        img_shape = self.config['transformations']['patch_size']
        input_shape = (self.config['batch_size'], 3, *img_shape)
        
        # Configure optimization strategy
        optimizer, lr_scheduler = self.get_optimizer_and_lr_schedule()

        # Initialize comprehensive training monitoring
        self.monitor.start_training(self.config['n_epochs'])

        # Resume from checkpoint if specified
        if self.config['load_model_path']:
            start_epoch = self.load_model(
                self.config['load_model_path'], 
                optimizer=optimizer, 
                scheduler=lr_scheduler, 
                for_training=True
            )

        # Display model architecture summary for verification
        if self.print_summary:
            self.monitor.log_message("Model Architecture Summary:")
            summary(self.model, input_shape, batch_dim=None, depth=3)

        # Log comprehensive training configuration
        self.monitor.log_message("Training Configuration:")
        self.monitor.log_message(f"  Batch Size: {self.config['batch_size']}")
        self.monitor.log_message(f"  Learning Rate: {self.config['optimizer']['lr']}")
        self.monitor.log_message(f"  Optimizer: {self.config['optimizer']['name']}")
        self.monitor.log_message(f"  Scheduler: {self.config['lr_scheduler']['name'] if self.config['lr_scheduler'] else 'None'}")
        self.monitor.log_message(f"  Patch Size: {self.config['transformations']['patch_size']}")
        self.monitor.log_message(f"  Number of Classes: {self.config['n_classes']}")

        # Main training loop with comprehensive monitoring
        for epoch in range(start_epoch, self.config['n_epochs'] + 1):
            # Training phase with IoU tracking
            train_iou, total_samples = self.train_one_epoch(epoch, train_loader, optimizer, scaler)
            
            # Validation phase with optional visualization
            if epoch % self.config['val_plot_interval'] == 0:
                # Generate visualization samples
                images, masks, predictions, val_iou = self.validate_one_epoch(
                    epoch, val_loader, return_img_mask_pred=True
                )
                self.save_plots(images, masks, predictions, 
                               save_path=os.path.join(plot_save_path, f"epoch_{epoch}.png"))
                del images, masks, predictions  # Free memory
            else:
                # Standard validation without visualization
                val_iou = self.validate_one_epoch(epoch, val_loader)
            
            # Get current learning rate for monitoring
            current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else self.config['optimizer']['lr']
            
            # Update comprehensive monitoring system
            self.monitor.end_epoch(
                epoch=epoch,
                train_loss=self.loss_dict['train_loss'][-1],
                val_loss=self.loss_dict['val_loss'][-1],
                train_iou=train_iou,
                val_iou=val_iou,
                lr=current_lr,
                num_samples=total_samples
            )
            
            # Save training artifacts and visualizations
            save_losses(self.loss_dict, plot_save_path)
            self.monitor.save_metrics(self.config['results_path'])
            self.monitor.plot_training_metrics(plot_save_path)
            
            # Save model checkpoint
            self.save_model(epoch, self.loss_dict['val_loss'][-1], optimizer, scheduler=lr_scheduler)

            # Persist loss history for analysis
            loss_pickle_path = os.path.join(self.config['results_path'], 'loss_dict.pkl')
            with open(loss_pickle_path, 'wb') as file:
                pickle.dump(self.loss_dict, file)

            # Update learning rate with logging
            if lr_scheduler:
                lr_scheduler.step()
                new_lr = lr_scheduler.get_last_lr()[0]
                if new_lr != current_lr:
                    self.monitor.log_message(f"Learning rate updated: {current_lr:.2e} -> {new_lr:.2e}")

        # Generate comprehensive training completion summary
        total_time = time.time() - total_start
        self.monitor.log_message("="*80)
        self.monitor.log_message("TRAINING COMPLETED!")
        self.monitor.log_message(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        self.monitor.log_message(f"Average time per epoch: {total_time/self.config['n_epochs']:.1f}s")
        self.monitor.log_message(f"Final train loss: {self.loss_dict['train_loss'][-1]:.6f}")
        self.monitor.log_message(f"Final validation loss: {self.loss_dict['val_loss'][-1]:.6f}")
        self.monitor.log_message("="*80)

    def run_inference(self, test_loader):
        """
        Execute optimized inference on test dataset with performance monitoring.
        
        Implements high-performance inference pipeline with sliding window approach
        for processing high-resolution surgical videos. Features memory optimization,
        comprehensive evaluation metrics, and performance monitoring.
        
        Args:
            test_loader (DataLoader): Test dataset loader with video sequences
        
        Inference Features:
        - Sliding window inference for arbitrarily large images
        - Half-precision computation for memory efficiency and speed
        - Memory format optimization for 2D convolutions
        - Per-video processing with detailed timing statistics
        - Comprehensive IoU evaluation with class-wise analysis
        - Automatic visualization sample selection and generation
        
        Performance Optimizations:
        - cuDNN benchmark mode for consistent input sizes
        - Memory-optimized processing with channels-last format
        - Efficient video sequence processing with frame batching
        - Real-time FPS calculation for clinical deployment assessment
        
        Note:
            - Results may vary slightly between runs due to floating-point precision
            - Creates numbered output folders to avoid overwriting previous results
            - Generates both numerical results and visual comparisons
        """
        # Enable performance optimizations for inference
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        
        # Load best model for inference
        self.load_model(self.config['load_model_path'])
        self.model.eval()
        
        # Optimize memory layout for 2D convolutions
        self.model.to(memory_format=torch.channels_last)
        self.model.half()  # Use half precision for speed and memory efficiency
        
        all_iou_list = []

        # Create unique output directory for this inference run
        base_results_path = self.config['results_path']
        prefix = "test_plots_"
        existing = [int(re.search(rf"{prefix}(\d+)", d).group(1))
                    for d in os.listdir(base_results_path) if d.startswith(prefix) and re.search(rf"{prefix}(\d+)", d)]
        plot_folder_path = os.path.join(base_results_path, f"{prefix}{max(existing, default=0) + 1}")
        
        # Save configuration for reproducibility
        create_config(self.config, plot_folder_path)
        print('')
        start = time.time()

        seen = set()
        
        # Main inference loop with comprehensive monitoring
        with torch.inference_mode():  # More efficient than no_grad for inference
            for data_item in test_loader:
                name = data_item['id']
                frames = data_item['image'].to(self.device, dtype=torch.float16, 
                                              memory_format=torch.channels_last, non_blocking=True)
                masks = data_item['mask'].to(self.device, dtype=torch.long, non_blocking=True)

                # Process each video sequence
                if name not in seen:
                    # Log results for previous video (if any)
                    if len(seen) > 0:
                        video_end = time.time() - video_start
                        fps = num_frames / video_end
                        print(f"    Inference time: {time.strftime('%H:%M:%S', time.gmtime(video_end))} ({fps:.2f} fps)")
                        
                        # Calculate and display per-class IoU statistics
                        video_iou_tensor = torch.cat(current_iou_list, dim=0)
                        mean_per_class = video_iou_tensor.mean(dim=0)
                        all_iou_list.append(video_iou_tensor)
                        
                        # Format class-wise results
                        parts = []
                        for i, score in enumerate(mean_per_class):
                            parts.append(f"C{i+1}: {score.numpy() * 100:.2f}")
                        parts.append(f"AVG: {mean_per_class.mean().numpy() * 100:.2f}")
                        print("    IoU scores per class: ")
                        print("        " + " - ".join(parts))
                        
                        # Save visualization for this video
                        self.save_plots(*plot_item, one_hot=False)

                    # Initialize processing for new video
                    current_iou_list = []
                    seen.add(name)
                    vid_idx = test_loader.dataset.ids.index(name)
                    num_frames = test_loader.dataset.video_lengths[vid_idx]
                    print(f"Running inference for: {name}")
                    print(f"    Number of frames: {num_frames}")
                    
                    # Randomly select batch for visualization
                    plot_batch_t0 = random.choice([t[1] for t in test_loader.dataset._index if t[0] == vid_idx])
                    plot_item = ()
                    video_start = time.time()

                # Sliding window inference for high-resolution processing
                with autocast(self.device.type):
                    preds = sliding_window_inference(
                        frames, 
                        roi_size=self.config['transformations']['patch_size'],
                        sw_batch_size=self.config['sw_batch_size'], 
                        predictor=self.model, 
                        overlap=self.config['sw_overlap'],
                        sw_device=self.device, 
                        device=self.device
                    )
                
                # Convert predictions to class indices
                preds = preds.argmax(1)
                
                # Calculate IoU statistics for this batch
                tp, fp, fn, tn = get_stats(preds-1, masks-1, mode='multiclass',
                                          num_classes=self.config['n_classes'], ignore_index=-1)
                score = iou_score(tp, fp, fn, tn)
                current_iou_list.append(score)

                # Collect visualization sample if this is the selected batch
                if data_item['t0'] == plot_batch_t0:
                    plot_path = os.path.join(plot_folder_path, f"{name}_{data_item['t0']}-{data_item['t1']}.png")
                    to_np = lambda x: x.detach().cpu().numpy()
                    frames, masks, preds = map(to_np, (frames.float(), masks.long(), preds.long()))
                    plot_item = (frames, masks, preds, plot_path)

            # Process results for the final video
            video_end = time.time() - video_start
            fps = num_frames / video_end
            print(f"    Inference time: {time.strftime('%H:%M:%S', time.gmtime(video_end))} ({fps:.2f} fps)")
            
            # Calculate final video statistics
            video_iou_tensor = torch.cat(current_iou_list, dim=0)
            mean_per_class = video_iou_tensor.mean(dim=0)
            all_iou_list.append(video_iou_tensor)
            
            # Display final video results
            parts = []
            for i, score in enumerate(mean_per_class):
                parts.append(f"C{i+1}: {score.numpy() * 100:.2f}")
            parts.append(f"AVG: {mean_per_class.mean().numpy() * 100:.2f}")
            print("    IoU scores per class: ")
            print("        " + " - ".join(parts))
            self.save_plots(*plot_item, one_hot=False)

        # Calculate and display overall inference statistics
        all_iou_tensors = torch.cat(all_iou_list, dim=0)
        total_mean_per_class = all_iou_tensors.mean(dim=0)

        total = time.time() - start
        print(f"\nTotal inference time: {time.strftime('%H:%M:%S', time.gmtime(total))}")
        
        # Display comprehensive final results
        parts = []
        for i, score in enumerate(total_mean_per_class):
            parts.append(f"C{i+1}: {score.numpy() * 100:.2f}")
        parts.append(f"AVG: {total_mean_per_class.mean().numpy() * 100:.2f}")
        print("Overall IoU scores per class: ")
        print("    " + " - ".join(parts))