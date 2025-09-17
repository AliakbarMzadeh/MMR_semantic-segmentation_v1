# MMR Surgical Tool Segmentation

This repo solves **semantic segmentation of surgical tools** with **two methods** implemented side-by-side. I’ll keep the flow exactly as I used while building it: **data → task → methods → why two → ViT note → constraints → where details live → results**.

---

## Data (brief)
- **Inputs:** endoscopic RGB frames  
- **Targets:** pixel-wise masks for surgical tools (binary or multiclass by split)  
- **Typical artifacts:** illumination changes, motion blur, specular highlights, **noise**

Folder layout, preprocessing, and splits are documented in each method’s README.

---

## Task
Semantic segmentation of instruments in endoscopic frames. Primary metrics are **Dice** and **IoU** with an emphasis on **robustness to noise and OR artifacts**.

---

## Methods

### 1) `MMR_Seg_Unet` — U-Net family (default **U-Net++**)
- **What I implemented:** a U-Net family pipeline; **U-Net++** is the **default**.  
- **Switchability:** other U-Net family variants are available; you can swap the backbone without changing the training loop.  
- **Core idea:** use a **state-of-the-art U-Net-style architecture** (dense skips / deep supervision options) to deliver stable medical segmentation under noise.  
- **Where details ?:** the folder has multiple files and READMEs with configs, scripts, and options.

### 2) `MMR_EN/DE_CODER` — Encoder/Decoder focused on **pretrained** weights
- **What I implemented:** an encoder/decoder stack that **prioritizes pretrained encoders** (e.g., **ResNet** pretrained on **ImageNet**).  
- **Modes:** multiple backbones are supported; select a backbone and the decoder adapts.  
- **Why this path:** I wanted to **show the strength of high-capacity pretrained encoders** versus “plain” EN/DE stacks. The goal here is feature quality from the encoder; it’s not limited to U-Net-style long skips.  
- **Where details?:** the folder lists supported encoders, flags, configs, and training commands.

---

## Why two methods?
There is a **trade-off** I wanted to expose clearly:
- **U-Net++ path:** very **stable and predictable** on noisy frames; strong medical baseline.
- **Pretrained EN/DE path:** can produce **sharp masks** when the backbone aligns with the domain(like when you use image-net weight and Checkmark); behavior depends on the chosen encoder and augmentations.

I implemented **both** to compare results fairly and decide which direction to push in the next phase.

---

## Note on ViT / ViT-U-Net hybrids
Vit is intriduced near 2022, after U-net, I’m familiar with **Vision Transformers (ViT)** and the ViT+U-Net hybrids (e.g., MICCAI winners around 2022–2024; see for example: https://arxiv.org/abs/2401.00496).  
From my own work on **ViT robustness (2022–2023)** and tests here (public **pretrained ViT** weights + **added noise**), ViT behaved **poorly under noise**: attention latched onto small patches and segmentation degraded, even though ViT often **outperforms U-Net++ on clean benchmarks**. Because **noise robustness** is essential in this dataset, I did not use ViT as a primary baseline at this stage. **But** in our next step we can use ViT that has better optimisation function and attention( I am familiar with ViT adversarial attack, so in next stage I belive we can robust the Vit and link them to the U-net++, and it would be one of the greatest architecture for demantic segmentation, but it is too heavy to run, I belive).

---

## Training constraints (transparent)
- Limited hardware; I couldn’t train long with very high settings.  
- I had some credits on a **NVIDIA A100 40GB** server, but the model is ~**15M parameters**.  
- Even with fast optimizers like **Adam**, I trained for **~60–70 epochs**.  
- **Checkpoints** and **train.log** are on **Google Drive**.
**Link**: https://drive.google.com/file/d/1OCVpfnYouCpc3qmnV2HrsNlJ6XGESMRW/view?usp=sharing

---

## Where details?
Each method folder (`MMR_Seg_Unet/` and `MMR_EN/DE_CODER/`) includes:
- A focused **README** (setup, configs, commands)
- Training/eval scripts and configuration files
- Notes on expected data paths and directory structure

---

## Results

### Qualitative PNGs
<p align="center">
  <img src="assets/Loss_F1_IOU copy.png" alt="Pipeline overview" width="900">
</p>

<p align="center">
  <img src="assets/U-net++_Results copy.png" alt="U-Net++ qualitative predictions" width="440">
  <img src="assets/epoch_48 copy.png"  alt="Pretrained EN/DE qualitative predictions" width="440">
</p>

<p align="center">
  <img src="assets/download copy.png" alt="Pipeline overview" width="900">
</p>

### What to notice
- **U-Net++** preserves tool contours under glare and moderate noise.  
- **Pretrained EN/DE** variants give crisp masks when the encoder matches the domain; sensitivity varies by backbone and augmentation strength.

### Reproducibility notes
- Trained **~60–70 epochs** per method given constraints.  
- Exact hyperparameters, augmentations, and chosen backbones are in the method READMEs and configs.  
- **Checkpoints** and **training logs** are available on **Google Drive**.
