import torch
from torch import nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward_with_intermediates(self, x: torch.Tensor):
        """Return logits and activations needed for LRP."""
        activations = []

        a0 = x
        z1 = self.conv1(a0)
        activations.append(("conv", self.conv1, a0.detach(), z1.detach()))
        a1 = F.relu(z1)
        activations.append(("relu", a1.detach()))

        z2_input = a1
        z2 = self.conv2(z2_input)
        activations.append(("conv", self.conv2, z2_input.detach(), z2.detach()))
        a2 = F.relu(z2)
        activations.append(("relu", a2.detach()))

        z3_input = a2
        z3 = self.conv3(z3_input)
        activations.append(("conv", self.conv3, z3_input.detach(), z3.detach()))
        a3 = F.relu(z3)
        activations.append(("relu", a3.detach()))

        flat_input = a3
        activations.append(("flatten", flat_input.shape))
        a4 = torch.flatten(flat_input, 1)

        z4_input = a4
        z4 = self.fc1(z4_input)
        activations.append(("linear", self.fc1, z4_input.detach(), z4.detach()))
        a5 = F.relu(z4)
        activations.append(("relu", a5.detach()))

        z5_input = a5
        z5 = self.fc2(z5_input)
        activations.append(("linear", self.fc2, z5_input.detach(), z5.detach()))
        logits = z5
        return logits, activations
