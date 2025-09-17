# Segmentation 

Hi, you need firstly run the notebook(be sure that you are in the right Environment and also pip install all reqs. the MINI_dataset can be creat by notebook and convert to.zaar
then put all file in one /contents root, then run the code as a bash ot .py
you can use: 



# MMR Segmentation 
This implementation uses UNet++ architecture with pre-trained en/dCoder weight, I used freezing and then updating the weight



**Key Components:**
- UNet++ with MobileNetV3 encoder for balanced speed/accuracy
- Zarr-based data format for pre-trained deep model
- Mixed precision training with gradient accumulation


## Installation

```bash
# Clone repository
git clone **link**
cd MMR_EN/DE_CODER

# Install PyTorch (adjust CUDA version as needed)
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Install framework
pip install -e .
```

## Dataset Structure

Organize your data as follows:

```
dataset/
├── data/          
│   ├── case_001.zarr
│   └── case_002.zarr
└── labels/       
    ├── case_001.zarr
    └── case_002.zarr
```

Images should be RGB format, labels should be single-channel with class indices (0=background, 1-9=classes).

## Training

### Basic Training

```bash
mmr_train_model "/path/to/dataset" "/path/to/results" "train-val-test" 9 "[512,640]" -p
```


### Common Training Options

```bash
# Custom batch size and epochs
mmr_train_model "/data/instruments" "/results/exp1" "train-val-test" 9 "[512,640]" \
  -p --batch_size 16 --n_epochs 150

# Continue from checkpoint
mmr_train_model "/data/instruments" "/results/exp1" "train-val-test" 9 "[512,640]" \
  -p -c


```

### Hyperparameter Modification

You can modify any training parameter:

```bash
# Learning rate and optimizer settings
--optimizer_lr 0.001 --optimizer_weight_decay 1e-5

# Data augmentation control
--transformations_rotation false --transformations_brightness true

# Training schedule
--val_plot_interval 5 --grad_accumulate_step 4
```

## Inference

```bash

# Test on original dataset's test split
mmr_run_inference "/path/to/original/dataset" "/path/to/trained/model"


```


## Model Architecture and Files

### Core Architecture
- **UNet++**: Advanced U-Net variant with nested skip connections
- **MobileNetV3**: Lightweight encoder pretrained on ImageNet
- **DiceCE Loss**: Combined Dice and Cross-Entropy loss for medical segmentation
- **Mixed Precision**: Automatic mixed precision for memory efficiency

###  Files Structure

**Training Pipeline:**
- `MMR_Model_Training.py`: Main training interface and argument parsing
- `Main_MMR_SegModel.py`: Model implementation with training/validation loops
- `Pre_Process.py`: Data loading, augmentation, and preprocessing
- `common_utils.py`: Configuration management and utility functions



**Outputs:**
- `checkpoints/best_model.pth`: Best performing model
- `checkpoints/last_model.pth`: Latest checkpoint for resuming
- `loss.png`: Training/validation loss curves
Find them in the plot section

### Data Processing
See the Jupyter Notebook, you can select the dataset size(mini) and then run the algorithm much faster




```bash
# Model changes
--model_arch DeepLabV3Plus --model_encoder_name efficientnet-b0

`



## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU recommended
- 16GB+ RAM for typical datasets




this is the Architecture: 
[2025-09-16 22:31:22] [INFO] Model Architecture Summary:
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
UnetPlusPlus                                            [2, 10, 512, 640]         --
├─TimmUniversalEncoder: 1-1                             [2, 3, 512, 640]          --
│    └─MobileNetV3Features: 2-1                         [2, 16, 256, 320]         --
│    │    └─Conv2d: 3-1                                 [2, 16, 256, 320]         432
│    │    └─BatchNorm2d: 3-2                            [2, 16, 256, 320]         32
│    │    └─Hardswish: 3-3                              [2, 16, 256, 320]         --
│    │    └─Sequential: 3-4                             --                        926,544
├─UnetPlusPlusDecoder: 1-2                              [2, 16, 512, 640]         --
│    └─ModuleDict: 2-2                                  --                        --
│    │    └─DecoderBlock: 3-5                           [2, 256, 32, 40]          2,028,544
│    │    └─DecoderBlock: 3-6                           [2, 24, 64, 80]           20,832
│    │    └─DecoderBlock: 3-7                           [2, 16, 128, 160]         8,128
│    │    └─DecoderBlock: 3-8                           [2, 16, 256, 320]         6,976
│    │    └─DecoderBlock: 3-9                           [2, 128, 64, 80]          498,176
│    │    └─DecoderBlock: 3-10                          [2, 16, 128, 160]         10,432
│    │    └─DecoderBlock: 3-11                          [2, 16, 256, 320]         9,280
│    │    └─DecoderBlock: 3-12                          [2, 64, 128, 160]         138,496
│    │    └─DecoderBlock: 3-13                          [2, 16, 256, 320]         11,584
│    │    └─DecoderBlock: 3-14                          [2, 32, 256, 320]         46,208
│    │    └─DecoderBlock: 3-15                          [2, 16, 512, 640]         6,976
├─SegmentationHead: 1-3                                 [2, 10, 512, 640]         --
│    └─Conv2d: 2-3                                      [2, 10, 512, 640]         1,450
│    └─Identity: 2-4                                    [2, 10, 512, 640]         --
│    └─Activation: 2-5                                  [2, 10, 512, 640]         --
│    │    └─Identity: 3-16                              [2, 10, 512, 640]         --
=========================================================================================================
Total params: 3,714,090
Trainable params: 3,714,090
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 35.19
=========================================================================================================
Input size (MB): 7.86
Forward/backward pass size (MB): 1172.65
Params size (MB): 14.81
Estimated Total Size (MB): 1195.32
=========================================================================================================