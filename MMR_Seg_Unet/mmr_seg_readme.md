# MMR_Seg_Unet: Semantic Segmentation for Surgical Images

Data: SAR-RARP50 dataset. This project implements multiple state-of-the-art segmentation architectures including U-Net++, ResNet-UNet( and other Unet family architecture)
## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Dataset Support](#dataset-support)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Parameters](#configuration-parameters)
- [Training](#training)
- [Evaluation](#evaluation)
- [Customization](#customization)
- [Results and Outputs](#results-and-outputs)
- [Troubleshooting](#troubleshooting)

## Project Overview

MMR_Seg_Unet is a PyTorch-based semantic segmentation framework that supports multiple datasets and architectures for medical/surgical image segmentation. The project is specifically configured for the SAR-RARP50 dataset but can be adapted for other surgical datasets.

**Key Features:**
- Multiple segmentation architectures (U-Net++, Unet family)
- Mixed loss functions (Cross-Entropy + Dice Loss)
- Automated evaluation metrics (IoU, Precision, Recall, F1-Score)
- Full validation and test set support

## Project Structure

```
MMR_Seg_Unet/
├── MMR_Core_ModelData/                 # Core source code directory
│   ├── SurgicalDataClass/             # Data handling and processing
│   │   ├── classes/                   # Dataset class definitions (JSON files)
│   │   │   └── sarrarp50SegClasses.json
│   │   ├── dataloaders/              # Data loading implementations  
│   │   │   └── SegNetDataLoaderV1_SAR.py
│   │   └── datasets/                 # Dataset storage
│   │       └── sarrarp50/           # SAR-RARP50 dataset
│   │           ├── train/           # Training data
│   │           ├── val/             # Validation data
│   │           └── test/            # Test data
│   ├── UArchModel/                   # Model architectures
│   │   ├── segnet.py               # SegNet implementation
│   │   ├── unet.py                 # U-Net implementation  
│   │   ├── unet_parts.py           # U-Net components
│   │   └── resnet_unet.py          # ResNet-UNet implementation
│   ├── ModelTraining.py            # Main training script
│   ├── ModelEval.py               # Evaluation script
│   ├── utils.py                   # Utility functions
│   └── dice_loss.py              # Dice loss implementation
├── ConfigModelUnetPlus/              # Training configuration scripts
│   └── train_sarrarp50.sh          # Training configuration for SAR-RARP50
├── results/                         # Training outputs and results
├── notebooks/                      # Jupyter notebooks for analysis
├── data_preprocessing.py           # Data preprocessing utilities
└── README.md                      # This file
```

### Directory Details

#### MMR_Core_ModelData/
Contains all core source code including models, data loaders, training, and evaluation scripts.

#### SurgicalDataClass/
- **classes/**: JSON files defining segmentation classes for different datasets
- **dataloaders/**: PyTorch Dataset implementations for loading and preprocessing data
- **datasets/**: Actual dataset storage (images and ground truth masks)

#### UArchModel/
Contains implementations of various segmentation architectures:
- SegNet with configurable batch normalization
- U-Net with bilinear upsampling options
- ResNet-UNet with different ResNet backbones
- Integration with segmentation_models_pytorch for additional architectures

#### ConfigModelUnetPlus/
Shell scripts containing training configurations and hyperparameters for different experiments.

## Model Architectures

### Supported Models

1. **SegNet**
   - Encoder-decoder architecture with symmetric structure
   
2. **U-Net**
   - Classic U-Net with skip connections


3. **ResNet-UNet**
   - U-Net architecture with ResNet encoder
   - Pre-trained ResNet backbones (ResNet-18)


4. **Segmentation Models PyTorch Integration**
   - **U-Net++**: Enhanced U-Net with nested skip connections


### Model Selection

Models are selected via the `--model` parameter in the training configuration:
- `segnet`: Standard SegNet
- `unet`: Classic U-Net
- `resnet18`: ResNet-18 based U-Net
- `smp_UNet++`: U-Net++ from segmentation_models_pytorch
- `smp_unet18`: U-Net with ResNet-18 encoder
- `smp_DeepLabV3+`: DeepLabV3+ architecture
- `smp_MANet`: Multi-scale Attention Network

## Dataset Support

### Primary Dataset: SAR-RARP50
The project is optimized for the SAR-RARP50 surgical dataset:
- **Format**: RGB images with corresponding grayscale segmentation masks
- **Structure**: Separate train/validation/test splits
- **Classes**: Defined in `sarrarp50SegClasses.json`

### Dataset Structure Expected
```
datasets/sarrarp50/
├── train/
│   ├── images/           # Training RGB images (.jpg, .png)
│   └── groundtruth/      # Training masks (.png)
├── val/
│   ├── images/           # Validation RGB images
│   └── groundtruth/      # Validation masks  
└── test/
    ├── images/           # Test RGB images
    └── groundtruth/      # Test masks
```

### Adding New Datasets
1. Create dataset folder structure following the above pattern
2. Add class definitions JSON file in `SurgicalDataClass/classes/`
3. Update dataloader if needed for different file naming conventions
4. Modify training script to include new dataset name

## Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended)
- Google Colab environment (current configuration)

### Dependencies
```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install torchsummary
pip install scikit-image
pip install tqdm
pip install matplotlib
pip install Pillow
pip install numpy
```

### Setup in Colab
The project is pre-configured for Google Colab. Simply upload the project directory and run the training script.

## Usage

### Quick Start
```bash
# Navigate to the project directory
cd /content/MMR_Seg_Unet

# Run training with default SAR-RARP50 configuration
!bash ConfigModelUnetPlus/train_sarrarp50.sh
```

### Training Command Structure
```bash
python MMR_Core_ModelData/ModelTraining.py \
    --model smp_UNet++ \
    --dataset sarrarp50 \
    --epochs 20 \
    --trainBatchSize 8 \
    --lr 0.001 \
    --data_dir /path/to/dataset \
    --json_path /path/to/classes.json \
    --save_dir /path/to/results
```

## Configuration Parameters

### Model Parameters
- `--model`: Architecture selection (segnet, unet, smp_UNet++, etc.)
- `--batchnorm_momentum`: Batch normalization momentum (default: 0.1)
- `--resnetModel`: ResNet version for ResNet-based models (18, 34, 50)

### Training Parameters
- `--epochs`: Number of training epochs (default: 20)
- `--trainBatchSize`: Training batch size (default: 8)
- `--valBatchSize`: Validation batch size (default: 4)
- `--lr`: Learning rate (default: 1e-3)
- `--optimizer`: Optimizer choice (Adam, AdamW, SGD)
- `--wd`: Weight decay factor (default: 0.00001)
- `--lr_steps`: Number of learning rate decay steps (default: 2)
- `--step_gamma`: Learning rate decay factor (default: 0.1)

### Loss Configuration
- `--dice_loss_factor`: Weight for Dice loss vs CrossEntropy (0.0-1.0, default: 0.5)
  - 0.0: CrossEntropy only
  - 1.0: Dice loss only
  - 0.5: Equal weighting

### Image Parameters
- `--resizedHeight`: Input image height (default: 256)
- `--resizedWidth`: Input image width (default: 256) 
- `--cropSize`: Random crop size, -1 for no cropping (default: -1)
- `--full_res_validation`: Validate on full resolution images ("True"/"False")





## Training

### Training Process
1. **Data Loading**:  loading 
2. **Model Initialization**: Architecture-specific setup with optional pre-trained weights
3. **Loss Computation**: Combined CrossEntropy and Dice loss
4. **Optimization**: Adam/AdamW/SGD with learning rate scheduling
5. **Validation**: Every epoch with comprehensive metrics
6. **Checkpointing**: Automatic saving of best model based on F1-score

### Training Outputs
- **Loss curves**: Training and validation loss plots
- **Accuracy curves**: IoU, Precision, Recall, F1-score plots  
- **Model checkpoints**: Best model saved automatically
- **Training logs**: Detailed training progress logs
- **Sample predictions**: Visual results from final epoch

### Monitoring Training
Monitor training progress through:
- Console output with epoch-by-epoch metrics
- Generated loss and accuracy curve plots
- Saved training logs in `{save_dir}/train.log`
- Debug information in `{save_dir}/debug.log`

## Evaluation

### Metrics Computed
- **IoU (Intersection over Union)**: Class-wise and mean IoU
- **Precision**: Class-wise and mean precision
- **Recall**: Class-wise and mean recall  
- **F1-Score**: Class-wise and mean F1-score
- **Dice Coefficient**: Every 25 epochs


### Evaluation Script
```bash
python MMR_Core_ModelData/ModelEval.py \
    --model /path/to/checkpoint.tar \
    --data_dir /path/to/test/data \
    --json_path /path/to/classes.json
```

## Customization

### Adding New Models
1. Implement model class in `UArchModel/your_model.py`
2. Add import statement in `ModelTraining.py`
3. Add model selection logic in the training script
4. Update this README with model details

### Modifying Data Loading
1. Edit `SurgicalDataClass/dataloaders/SegNetDataLoaderV1_SAR.py`
2. Add support for new file formats or naming conventions


### Custom Loss Functions
1. Implement loss in separate file (see `dice_loss.py` example)
2. Import in `ModelTraining.py`


### Hyperparameter Tuning
Key parameters to tune:
- **Learning rate**: Start with 1e-3, try 1e-4 for fine-tuning
- **Batch size**: Limited by GPU memory, reduce if out-of-memory
- **Dice loss factor**: 0.3-0.7 range typically works well
- **Weight decay**: 1e-5 to 1e-4 range

## Results and Outputs

### Generated Files
- `{model}_{dataset}_checkpoint.tar`: Best model checkpoint
- `loss_{model}_{dataset}_bs{batch}lr{lr}e{epochs}.png`: Loss curves
- `acc_{model}_{dataset}_bs{batch}lr{lr}e{epochs}.png`: Accuracy curves
- `train.log`: Detailed training log
- `debug.log`: Debug information and console output
- `seg_results/`: Sample segmentation visualizations

### Model Checkpoint Contents
- Model state dictionary
- Optimizer state
- Training epoch information
- Best validation metrics

## Troubleshooting

### Common Issues

#### Out of Memory Errors
- Reduce `--trainBatchSize` and `--valBatchSize`
- Decrease image size (`--resizedHeight`, `--resizedWidth`)
- Use gradient accumulation for effective larger batch sizes

#### Import Errors
- Ensure all dependencies are installed
- Check that folder structure matches expectations
- Verify Python path includes project root

#### Dataset Loading Issues
- Check dataset folder structure matches expected format
- Verify JSON class file format and path
- Ensure image and mask file extensions are supported (.jpg, .png)

#### Training Convergence Issues
- Try different learning rates (1e-4, 5e-4, 1e-3)
- Adjust dice loss factor (try 0.3 or 0.7)
- Increase number of epochs
- Check for class imbalance in dataset

#### CUDA Issues
- Ensure PyTorch is installed with CUDA support
- Check GPU memory usage
- Verify CUDA version compatibility

### Performance Optimization
- Use mixed precision training for faster training
- Enable pin_memory in DataLoader for faster GPU transfer
- Use multiple workers for data loading (adjust `--workers`)
- Profile code to identify bottlenecks



---

## License and Citation

MMR Company Maybe. :)

**Contact**: a.mahmood.zadeh@student.tue.nl