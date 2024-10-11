import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from scipy.interpolate import Rbf


class PoseKeypointsExtractor(nn.Module):
    """
    PoseKeypointsExtractor: Extracts keypoints from the body image using CNN layers and fully connected layers.
    """
    def __init__(self):
        super(PoseKeypointsExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)

        # Adjust the size based on the feature map shape after conv layers
        # The shape before flattening is [batch_size, 256, 32, 24], so:
        self.fc1 = nn.Linear(256 * 32 * 24, 512)  # Adjusted input size for fc1 layer
        self.fc2 = nn.Linear(512, 18 * 2)  # 18 keypoints, each with (x, y) coordinates

    def forward(self, x):
        """
        Forward pass to extract keypoints from the body image.
        :param x: Input body image (tensor).
        :return: Predicted keypoints with shape (batch_size, 18, 2).
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the feature map and adjust the size for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the tensor to (batch_size, features)
        x = F.relu(self.fc1(x))
        keypoints = self.fc2(x).view(-1, 18, 2)  # (batch_size, 18 keypoints, 2 coordinates)

        return keypoints


class SpatialTransformerNetwork(nn.Module):
    """
    Spatial Transformer Network (STN) that warps the clothing image based on keypoints
    using a Thin-Plate Spline (TPS) transformation.
    """

    def __init__(self, num_points=18):
        """
        Initialize the SpatialTransformerNetwork.

        Args:
            num_points (int): Number of keypoints (control points) used for TPS transformation.
        """
        super(SpatialTransformerNetwork, self).__init__()
        self.num_points = num_points
        # TPS control points are initialized randomly or using some fixed strategy.
        self.control_points = Parameter(torch.randn(num_points, 2))  # Random control points for cloth image

    def tps_warp(self, src_points, dst_points, cloth_image):
        """
        Apply Thin-Plate Spline (TPS) transformation to warp the clothing image based on keypoints.

        Args:
            src_points (Tensor): Source keypoints on the clothing image (batch_size, num_points, 2).
            dst_points (Tensor): Destination keypoints from the body image (batch_size, num_points, 2).
            cloth_image (Tensor): Clothing image to be warped (batch_size, C, H, W).

        Returns:
            warped_cloth (Tensor): Warped clothing image (batch_size, C, H, W).
        """
        batch_size, _, height, width = cloth_image.size()

        # Ensure source and destination points are of the same length
        if src_points.size(1) != self.num_points or dst_points.size(1) != self.num_points:
            raise ValueError(f"Expected {self.num_points} points, got {src_points.size(1)} and {dst_points.size(1)}.")

        # Convert PyTorch tensors to numpy arrays for RBF interpolation
        src_points_np = src_points.detach().cpu().numpy()  # Clothing keypoints (detached from computation graph)
        dst_points_np = dst_points.detach().cpu().numpy()  # Body keypoints (detached from computation graph)

        warped_images = []

        for i in range(batch_size):
            # Create meshgrid for the image (pixel coordinates)
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            flat_grid = np.vstack([grid_x.flatten(), grid_y.flatten()]).T

            # Add small noise to avoid singular matrix error
            epsilon = 1e-5
            dst_points_np[i] += epsilon * np.random.randn(*dst_points_np[i].shape)

            # Apply TPS using Rbf (Radial Basis Function) with smooth regularization to avoid singular matrix
            rbf_x = Rbf(dst_points_np[i, :, 0], dst_points_np[i, :, 1], src_points_np[i, :, 0], function='thin_plate', smooth=epsilon)
            rbf_y = Rbf(dst_points_np[i, :, 0], dst_points_np[i, :, 1], src_points_np[i, :, 1], function='thin_plate', smooth=epsilon)

            warped_grid_x = rbf_x(flat_grid[:, 0], flat_grid[:, 1])
            warped_grid_y = rbf_y(flat_grid[:, 0], flat_grid[:, 1])

            warped_grid_x = np.clip(warped_grid_x, 0, width - 1).reshape(height, width)
            warped_grid_y = np.clip(warped_grid_y, 0, height - 1).reshape(height, width)

            # Warp each channel of the clothing image using bilinear interpolation
            warped_image = np.zeros_like(cloth_image[i].cpu().numpy())

            for c in range(cloth_image.size(1)):  # For each color channel
                warped_image[c] = self.bilinear_interpolate(cloth_image[i, c].cpu().numpy(), warped_grid_x, warped_grid_y)

            warped_images.append(warped_image)

        # Convert warped images back to a PyTorch tensor
        warped_images = torch.tensor(warped_images).float().to(cloth_image.device)

        return warped_images

    @staticmethod
    def bilinear_interpolate(im, x, y):
        """
        Bilinear interpolation to sample pixel values from the original image.

        Args:
            im (ndarray): Original image (H, W).
            x (ndarray): X-coordinates of the grid to sample from.
            y (ndarray): Y-coordinates of the grid to sample from.

        Returns:
            ndarray: Warped image based on the interpolated pixel values.
        """
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1] - 1)
        x1 = np.clip(x1, 0, im.shape[1] - 1)
        y0 = np.clip(y0, 0, im.shape[0] - 1)
        y1 = np.clip(y1, 0, im.shape[0] - 1)

        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    def forward(self, cloth_image, keypoints):
        """
        Forward pass for the STN.

        Args:
            cloth_image (Tensor): Input clothing image (batch_size, C, H, W).
            keypoints (Tensor): Destination keypoints from the body image (batch_size, num_points, 2).

        Returns:
            warped_cloth (Tensor): Warped clothing image.
        """
        # Source points for the cloth: control points initialized or learned
        src_points = self.control_points.unsqueeze(0).repeat(cloth_image.size(0), 1, 1).to(cloth_image.device)
        dst_points = keypoints

        # Perform TPS warping
        warped_cloth = self.tps_warp(src_points, dst_points, cloth_image)

        return warped_cloth


class GMM(nn.Module):
    """
    GMM: Geometric Matching Module that warps the clothing image based
    on pose keypoints.

    The module uses a Spatial Transformer Network (STN) to warp the clothing
    image according to the body pose represented by keypoints.
    """
    def __init__(self):
        super(GMM, self).__init__()
        self.stn = SpatialTransformerNetwork()

    def forward(self, cloth_image, body_image):
        """
        Forward pass for GMM.
        :param cloth_image: Clothing image tensor.
        :param body_image: Body image tensor.
        :return: Warped clothing image tensor.
        """
        # Step 1: Extract keypoints from the body image
        keypoints_extractor = PoseKeypointsExtractor().to(next(self.parameters()).device)
        body_keypoints = keypoints_extractor(body_image)

        # Step 2: Use the keypoints in the STN to warp the clothing image
        warped_cloth = self.stn(cloth_image, body_keypoints)

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
        mask (Tensor): Segmentation mask for the body (batch_size, 1, H, W), values between 0 and 1
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
            nn.Sigmoid()  # Sigmoid ensures output is in range [0, 1]
        )

    def forward(self, x):
        # Pass input through encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # Final output is already sigmoid-activated


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


class VirtualTryOnModel(nn.Module):
    """
    Virtual Try-On model using pose keypoints for training and predicted keypoints for inference.
    """
    def __init__(self, use_keypoints=True):
        super(VirtualTryOnModel, self).__init__()
        self.use_keypoints = use_keypoints
        self.tps_net = SpatialTransformerNetwork()
        self.gmm = GMM()  # GMM includes the Spatial Transformer Network
        self.segnet = SegNet()
        self.compnet = CompNet()
        if not self.use_keypoints:
            self.keypoint_predictor = PoseKeypointsExtractor()  # Predicts keypoints during inference
            self.keypoint_predictor.to(next(self.parameters()).device)

    def forward(self, cloth_image, body_image, body_keypoints=None):
        """
        Forward pass: uses keypoints for training and predicts keypoints during inference.
        """
        if self.use_keypoints:
            # Training: Use provided keypoints
            warped_cloth = self.tps_net(cloth_image, body_keypoints)
        else:
            # Inference: Predict keypoints
            predicted_keypoints = self.keypoint_predictor(body_image)
            warped_cloth = self.tps_net(cloth_image, predicted_keypoints)

        refined_cloth = self.gmm(body_image, warped_cloth)
        seg_input = torch.cat([body_image, refined_cloth], dim=1)
        segmentation_mask = self.segnet(seg_input)
        final_output = self.compnet(body_image, refined_cloth, segmentation_mask)

        return final_output, warped_cloth, refined_cloth, segmentation_mask
