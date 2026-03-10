from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class STN3d(nn.Module):
    """
    Spatial Transformer Network for 3D point clouds.
    Predicts a 3x3 transformation matrix to canonicalize input geometry.
    """

    def __init__(self, num_points: int = 2500):
        super().__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = Variable(
            torch.from_numpy(np.eye(3).flatten().astype(np.float32))
        ).view(1, 9).repeat(batchsize, 1).to(x.device)
        x = (x + iden).view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    """PointNet global feature extractor."""

    def __init__(self, num_points: int = 8192):
        super().__init__()
        self.stn = STN3d(num_points)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        return x.view(-1, 1024)


class PointGenCon(nn.Module):
    """Point cloud generation decoder (folding network)."""

    def __init__(self, bottleneck_size: int = 1026):
        super().__init__()
        self.conv1 = nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = nn.Conv1d(bottleneck_size, bottleneck_size // 2, 1)
        self.conv3 = nn.Conv1d(bottleneck_size // 2, bottleneck_size // 4, 1)
        self.conv4 = nn.Conv1d(bottleneck_size // 4, 3, 1)
        self.bn1 = nn.BatchNorm1d(bottleneck_size)
        self.bn2 = nn.BatchNorm1d(bottleneck_size // 2)
        self.bn3 = nn.BatchNorm1d(bottleneck_size // 4)
        self.th = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.th(self.conv4(x))


class PointNetRes(nn.Module):
    """Residual PointNet for point cloud refinement."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)
        self.th = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        npoints = x.size(2)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        return self.th(self.conv7(x))


class MSN(nn.Module):
    """
    Morphing and Sampling Network (MSN) for point cloud completion.
    Takes a partial point cloud as input and produces a dense, complete point cloud.

    Architecture: PointNet encoder -> multi-primitive folding decoder -> PointNet residual refinement.
    """

    def __init__(self, num_points: int = 8192, bottleneck_size: int = 1024, n_primitives: int = 16):
        super().__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives

        self.encoder = nn.Sequential(
            PointNetfeat(num_points),
            nn.Linear(1024, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            PointGenCon(bottleneck_size=2 + bottleneck_size)
            for _ in range(n_primitives)
        ])
        self.res = PointNetRes()

    def forward(self, x: torch.Tensor):
        """
        Forward pass for point cloud completion.

        Args:
            x: Partial point cloud, shape (B, 3, N).

        Returns:
            Tuple of (coarse_output, refined_output) both shape (B, num_points, 3).
        """
        partial = x
        latent = self.encoder(x)
        outs = []

        pts_per_prim = self.num_points // self.n_primitives
        for decoder in self.decoder:
            rand_grid = torch.rand(latent.size(0), 2, pts_per_prim, device=x.device)
            y = latent.unsqueeze(2).expand(-1, -1, pts_per_prim)
            y = torch.cat([rand_grid, y], 1)
            outs.append(decoder(y))

        outs = torch.cat(outs, 2)
        coarse = outs.transpose(1, 2)

        # Refinement via residual network
        id0 = torch.zeros(outs.size(0), 1, outs.size(2), device=x.device)
        id1 = torch.ones(partial.size(0), 1, partial.size(2), device=x.device)
        merged = torch.cat([torch.cat([outs, id0], 1), torch.cat([partial, id1], 1)], 2)

        delta = self.res(merged)
        refined = (merged[:, :3, :] + delta).transpose(2, 1)
        return coarse, refined
