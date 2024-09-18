#!/usr/bin/env python
# coding: utf-8


# Install required packages
!pip install -q torch torchvision
!pip install -q opencv-python
!pip install -q Pillow

# Import Libraries
import os
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class VITONDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        self.cloth_dir = os.path.join(root_dir, mode, 'cloth')
        self.cloth_mask_dir = os.path.join(root_dir, mode, 'cloth-mask')
        self.image_dir = os.path.join(root_dir, mode, 'image')
        self.image_parse_dir = os.path.join(root_dir, mode, 'image-parse')
        self.pose_dir = os.path.join(root_dir, mode, 'pose')

        self.image_names = [f.split('_')[0] for f in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # Update paths based on actual dataset structure
        cloth_path = os.path.join(self.cloth_dir, f"{image_name}_1.jpg")
        cloth_mask_path = os.path.join(self.cloth_mask_dir, f"{image_name}_1.jpg")
        image_path = os.path.join(self.image_dir, f"{image_name}_0.jpg")
        image_parse_path = os.path.join(self.image_parse_dir, f"{image_name}_0.png")
        pose_path = os.path.join(self.pose_dir, f"{image_name}_0_keypoints.json")

        # Open and convert images
        cloth_image = Image.open(cloth_path).convert('RGB')
        cloth_mask = Image.open(cloth_mask_path).convert('L')  # Convert to single-channel grayscale
        body_image = Image.open(image_path).convert('RGB')
        image_parse = Image.open(image_parse_path).convert('L')  # Convert segmentation mask to single-channel
        # # if wanna to load image_parse as 3 channel
        # image_parse = Image.open(image_parse_path).convert('RGB')

        # Load and parse pose keypoints
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
            pose_keypoints = torch.tensor(pose_data['people'][0]['pose_keypoints']).view(-1, 3)

        # Apply transformations
        if self.transform:
            cloth_image = self.transform(cloth_image)
            cloth_mask = transforms.ToTensor()(cloth_mask)  # Convert mask to tensor
            cloth_mask = (cloth_mask > 0).float()  # Ensure mask values are 0 or 1
            body_image = self.transform(body_image)
            image_parse = transforms.ToTensor()(image_parse)  # Convert parse mask to tensor
            image_parse = (image_parse > 0).float()  # Binary segmentation mask
            # # if wanna to load image_parse as 3 channel
            # image_parse = self.transform(image_parse)

        return {
            'cloth_image': cloth_image,
            'cloth_mask': cloth_mask,
            'body_image': body_image,
            'image_parse': image_parse,
            'pose_keypoints': pose_keypoints
        }

# Prepare the DataLoader and Transformations
# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create dataset and dataloader
viton_dataset_path = './viton_resize'

train_dataset = VITONDataset(root_dir=viton_dataset_path, mode='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = VITONDataset(root_dir=viton_dataset_path, mode='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True)

def debug_dataloader(dataloader):
    for i, batch in enumerate(dataloader):
        print(f"Batch {i} shapes:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        if i == 0:  # Only check the first batch
            break

debug_dataloader(train_loader)

"""## Show Sample of dataset"""

def show_sample(dataset, idx):
    sample = dataset[idx]

    cloth_image = sample['cloth_image'].permute(1, 2, 0).numpy()  # [C, H, W] to [H, W, C]
    cloth_mask = sample['cloth_mask'].squeeze(0).numpy()  # Remove channel dimension
    body_image = sample['body_image'].permute(1, 2, 0).numpy()  # [C, H, W] to [H, W, C]
    image_parse = sample['image_parse'].squeeze(0).numpy()  # Remove channel dimension for mask
    # # if wanna to load image_parse as 3 channel
    # image_parse = sample['image_parse'].permute(1, 2, 0).numpy()  # [C, H, W] to [H, W, C]
    pose_keypoints = sample['pose_keypoints']  # Pose keypoints are tensors

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(cloth_image)
    axs[0].set_title('Cloth Image')

    axs[1].imshow(cloth_mask, cmap='gray')
    axs[1].set_title('Cloth Mask')

    axs[2].imshow(body_image)
    axs[2].set_title('Body Image')

    axs[3].imshow(image_parse, cmap='gray')
    # # if wanna to load image_parse as 3 channel
    # axs[3].imshow(image_parse)
    axs[3].set_title('Image Parse')

    for ax in axs:
        ax.axis('off')

    plt.show()

    print("Pose Keypoints:", pose_keypoints)

# Visualize a sample from the train dataset
show_sample(train_dataset, idx=0)

"""## Model Architecture Design"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""### Geometric Matching Module (GMM)"""

class GMM(nn.Module):
    """
    Geometric Matching Module (GMM) that warps the cloth image to match the body image
    using a Spatial Transformer Network (STN) with Thin-Plate Spline (TPS) transformation.

    Args:
        None

    Input:
        body_image (Tensor): The body image of the person (batch_size, 3, H, W)
        cloth_image (Tensor): The cloth image to be warped (batch_size, 3, H, W)

    Output:
        warped_cloth (Tensor): The cloth image warped to fit the body (batch_size, 3, H, W)
    """
    def __init__(self):
        super(GMM, self).__init__()

        # Feature extraction layers for body and cloth images
        # Shared convolutional layers for both inputs
        self.extraction = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),  # Output: (batch_size, 64, H, W)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # Output: (batch_size, 128, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),  # Output: (batch_size, 256, H/4, W/4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample
        )

        # Thin-Plate Spline (TPS) transformation module
        # Fully connected layers to predict affine transformation matrix
        self.tps = nn.Sequential(
            nn.Linear(256 * 32 * 24, 512),  # Input size depends on the image size and feature map size
            nn.ReLU(),
            nn.Linear(512, 6)  # Predicts 6 parameters for affine transformation
        )

    def forward(self, body_image, cloth_image):
        # Extract features from body and cloth images
        body_features = self.extraction(body_image)
        cloth_features = self.extraction(cloth_image)

        # Flatten the feature maps to feed into the fully connected layers
        body_flatten = body_features.view(body_features.size(0), -1)
        cloth_flatten = cloth_features.view(cloth_features.size(0), -1)

        # Predict transformation matrix (theta)
        theta = self.tps(body_flatten)

        # Reshape theta to a 2x3 affine transformation matrix
        theta = theta.view(-1, 2, 3)

        # Create a grid for warping the cloth image
        grid = F.affine_grid(theta, cloth_image.size())

        # Warp the cloth image using the grid
        warped_cloth = F.grid_sample(cloth_image, grid)

        return warped_cloth

"""### UNet for Segmentation"""

class UNet(nn.Module):
    """
    UNet-based segmentation network that generates a segmentation mask for the body.
    The input consists of the body image and the warped cloth image, and the output is a binary mask
    indicating where the clothing should appear on the person.

    Args:
        input_channels (int): Number of input channels, including the body image and warped cloth
        output_channels (int): Number of output channels, typically 1 for binary mask

    Input:
        x (Tensor): Concatenated input tensor of the body image and warped cloth (batch_size, input_channels, H, W)

    Output:
        mask (Tensor): Segmentation mask for the body (batch_size, 1, H, W)
    """
    def __init__(self, input_channels=7, output_channels=1):
        super(UNet, self).__init__()

        # Encoder layers (downsampling path)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # (batch_size, 64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (batch_size, 256, H/8, W/8)
            nn.ReLU()
        )

        # Decoder layers (upsampling path)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (batch_size, 64, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),  # (batch_size, 1, H, W)
            nn.Sigmoid()  # Sigmoid to normalize output to [0, 1] for mask
        )

    def forward(self, x):
        # Pass input through encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x

"""### Composition Network"""

class CompositionNetwork(nn.Module):
    """
    Composition Network that combines the body image, warped cloth, and segmentation mask
    to produce the final try-on result (person wearing the new cloth).

    Args:
        None

    Input:
        body_image (Tensor): The body image of the person (batch_size, 3, H, W)
        warped_cloth (Tensor): The warped cloth image (batch_size, 3, H, W)
        segmentation_mask (Tensor): The segmentation mask indicating clothing regions (batch_size, 1, H, W)

    Output:
        output (Tensor): The final composite image (batch_size, 3, H, W)
    """
    def __init__(self):
        super(CompositionNetwork, self).__init__()

        # Composition layers that combine body image, warped cloth, and mask
        self.network = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=4, stride=2, padding=1),  # (batch_size, 64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (batch_size, 64, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # (batch_size, 3, H, W)
            nn.Tanh()  # Final output in range [-1, 1] for RGB image
        )

    def forward(self, body_image, warped_cloth, segmentation_mask):
        # Concatenate body image, warped cloth, and segmentation mask along the channel dimension
        input_tensor = torch.cat([body_image, warped_cloth, segmentation_mask], dim=1)

        # Pass through the composition network
        output = self.network(input_tensor)
        return output