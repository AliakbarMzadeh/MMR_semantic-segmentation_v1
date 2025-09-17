"""
DataLoader for Semantic Segmentation on Surgical Datasets
NOTE: Does not load all data into memory. Use if you don't have enough RAM for storing data.
Modified for SAR-RARP50 dataset
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image

import os
import json
import random

class SegNetDataset(Dataset):
    """
    Dataset Class for Semantic Segmentation on Surgical Data
    Modified for SAR-RARP50 with lazy loading
    """

    def __init__(self, root_dir, crop_size=-1, json_path=None, sample=None, 
                 dataset=None, image_size=[256, 256], horizontal_flip=True, brightness=True, contrast=True,
                 rotate=True, vertical_flip=True, full_res_validation="False"):
        """
        args:

        root_dir (str) = File Directory with Input Surgical Images
        json_path (str) = File with Semantic Segmentation Class information
        sample (str) = Specify whether the sample is from train, test, or validation set
        dataset (str) = Specify whether the Segmentation dataset is from Synapse, Cholec, Miccai, or SAR-RARP50
        """
        
        # General Parameters
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.gt_dir = os.path.join(root_dir, 'groundtruth')
        self.image_list = [f for f in os.listdir(self.img_dir) if (f.endswith(".png") or f.endswith(".jpg"))]
        self.crop_size = crop_size
        self.sample = sample
        self.dataset = dataset
        self.full_res_validation = full_res_validation

        # Data Augmentation Parameters
        self.resizedHeight = image_size[0]
        self.resizedWidth = image_size[1]
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate = rotate
        self.brightness = brightness
        self.contrast = contrast

        if json_path:
            self.classes = json.load(open(json_path))["classes"]
        
        self.key = self.generateKey(self.classes)

        print(f"Initialized dataset with {len(self.image_list)} images (lazy loading)")

    def convert_grayscale_to_rgb(self, grayscale_image):
        """
        Convert grayscale segmentation mask to RGB format for SAR-RARP50
        Each pixel value corresponds to a class ID
        """
        # Convert to numpy for processing
        mask_array = np.array(grayscale_image)
        
        # Create RGB image with same values in all channels
        rgb_array = np.stack([mask_array, mask_array, mask_array], axis=-1)
        
        # Convert back to PIL Image
        return Image.fromarray(rgb_array.astype(np.uint8))

    def generateKey(self, key):
        """
            Disentangles the key for class and labels obtained from the
            JSON file
            Returns a python dictionary of the form:
                {Class Id: RGB Color Code as numpy array}
        """
        dKey = {}
        for i in range(len(key)):
            class_id = int(key[i]['id'])
            c = key[i]['color']
            c = c.split(',')
            c0 = int(c[0][1:])
            c1 = int(c[1])
            c2 = int(c[2][:-1])
            color_array = np.asarray([c0,c1,c2])
            dKey[class_id] = color_array

        return dKey

    def random_crop(self, img, mask, label, width, height):
        assert img.shape[1] >= height, f"img.shape[0]: {img.shape[0]} is not >= height: {height}"
        assert img.shape[2] >= width, f"img.shape[2]: {img.shape[2]} is not >= width: {width}"
        assert img.shape[1] == mask.shape[1], f"img.shape[1] {img.shape[1]} != mask.shape[1]: {mask.shape[1]}"
        assert img.shape[2] == mask.shape[2], f"img.shape[2] {img.shape[2]} != mask.shape[2]: {mask.shape[2]}"
        x = random.randint(0, img.shape[2] - width)
        y = random.randint(0, img.shape[1] - height)
        img = img[:, y:y+height, x:x+width]
        mask = mask[:, y:y+height, x:x+width]
        label = label[y:y+height, x:x+width]
        return img, mask, label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load images on-demand (lazy loading)
        img_name = os.path.join(self.img_dir, self.image_list[idx])

        # Handle different ground truth naming conventions
        if self.dataset == "synapse":
            gt_file_name = self.image_list[idx][0:-4] + ".png"
        elif self.dataset == "cholec":
            gt_file_name = self.image_list[idx][0:-4] + "_color_mask.png"
        elif self.dataset == "miccai":
            gt_file_name = self.image_list[idx][0:-4] + "_gt.png"
        elif self.dataset == "sarrarp50":
            # For SAR-RARP50, ground truth has same name as image
            gt_file_name = self.image_list[idx]
        else:
            raise ValueError("Ground Truth File Name Does Not Exist")

        gt_name = os.path.join(self.gt_dir, gt_file_name)

        # Check if ground truth file exists
        if not os.path.exists(gt_name):
            raise FileNotFoundError(f"Ground truth file not found: {gt_name}")

        image = Image.open(img_name)
        image = image.convert("RGB")

        gt_image = Image.open(gt_name)
        
        # Handle different ground truth formats
        if self.dataset == "sarrarp50":
            # SAR-RARP50 masks are grayscale, convert to RGB for consistency
            if gt_image.mode == 'L':  # Grayscale
                gt_image = self.convert_grayscale_to_rgb(gt_image)
            else:
                gt_image = gt_image.convert("RGB")
        else:
            gt_image = gt_image.convert("RGB")

        to_tensor = ToTensor()
        image, gt_image = to_tensor(image), to_tensor(gt_image)

        if self.sample == 'train':
            # Resize to Half-HD Resolution to do half-crop or five-crop
            image = TF.resize(image, [540, 960], interpolation=Image.BILINEAR)
            gt = TF.resize(gt_image, [540, 960], interpolation=Image.NEAREST)
        elif self.sample == 'test' or self.sample == 'val':
            # NOTE: Typically set to "False" unless you want to validate your network on Full-Resolution Images
            if self.full_res_validation == "True":
                image = TF.resize(image, [1080, 1920], interpolation=Image.BILINEAR)
                gt = TF.resize(gt_image, [1080, 1920], interpolation=Image.NEAREST)
            else:
                image = TF.resize(image, [self.resizedHeight, self.resizedWidth], interpolation=Image.BILINEAR)
                gt = TF.resize(gt_image, [self.resizedHeight, self.resizedWidth], interpolation=Image.NEAREST)

        # Process GT label for loss calculation
        gt_label = gt.permute(1, 2, 0)
        gt_label = (gt_label * 255).long()
        catMask = torch.zeros((gt_label.shape[0], gt_label.shape[1]))
        
        # Handle SAR-RARP50 grayscale masks differently
        if self.dataset == "sarrarp50":
            # For SAR-RARP50, the mask values directly correspond to class IDs
            catMask = gt_label[:, :, 0]  # Take first channel since all channels are the same
            # Ensure class IDs are within valid range
            catMask = torch.clamp(catMask, 0, len(self.key) - 1)
        else:
            # Iterate over all the key-value pairs in the class Key dict
            for k in range(len(self.key)):
                rgb = torch.Tensor(self.key[k])
                mask = torch.all(gt_label == rgb, axis=2)
                assert mask.shape == catMask.shape, f"mask shape {mask.shape} unequal to catMask shape {catMask.shape}"
                catMask[mask] = k

        if self.sample == "train":
            # Random Horizontal Flip
            if self.horizontal_flip and random.random() > 0.5:
                image, gt = TF.hflip(image), TF.hflip(gt)
                catMask = TF.hflip(catMask.unsqueeze(0)).squeeze(0)  # Add/remove channel dimension
            
            # Random Vertical Flip
            if self.vertical_flip and random.random() > 0.5:
                image, gt = TF.vflip(image), TF.vflip(gt)
                catMask = TF.vflip(catMask.unsqueeze(0)).squeeze(0)  # Add/remove channel dimension
            
            # Random Rotate
            if self.rotate and random.random() > 0.5:
                image, gt = TF.rotate(image, 90), TF.rotate(gt, 90)
                catMask = TF.rotate(catMask.unsqueeze(0), 90).squeeze(0)  # Add/remove channel dimension for rotation

            # Brightness Adjustment
            if self.brightness and random.random() > 0.5:
                bright_factor = random.uniform(0.9, 1.1)
                image = TF.adjust_brightness(image, bright_factor)

            # Contrast Adjustment
            if self.contrast and random.random() > 0.5:
                cont_factor = random.uniform(0.9, 1.1)
                image = TF.adjust_contrast(image, cont_factor)
            
            # Random Crop
            if self.crop_size == -1:
                image, gt, catMask = self.random_crop(image, gt, catMask, self.resizedWidth, self.resizedHeight)

        gt = gt * 255

        return image.type(torch.float32), gt.type(torch.int64), catMask.type(torch.int64)
