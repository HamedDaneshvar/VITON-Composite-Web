import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SegNet(nn.Module):
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
    def __init__(self, input_channels=6, output_channels=1):
        super(SegNet, self).__init__()

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


class CompNet(nn.Module):
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
        super(CompNet, self).__init__()

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
