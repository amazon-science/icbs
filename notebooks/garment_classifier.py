# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms


class GarmentClassifier(nn.Module):
    """Simple model for benchmarking."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(28 * 28, 512)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(512, 512)),
                    ("relu2", nn.ReLU()),
                    ("linear3", nn.Linear(512, 10)),
                ]
            )
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def load_model_from_file(model, model_filename, device):
    """Loads state_dict from model_filename into the model.

    Note: it is recommended to pickle just the state_dict (as assumed here) and not the
    full model.
    """
    with torch.inference_mode():
        state_dict = torch.load(model_filename)
        model.load_state_dict(state_dict)
        model.to(device, non_blocking=True)
        model.eval()
    return model


def load_garment_classifier(device):
    """Loads trained GarmentClassifier for our experiments."""
    model = GarmentClassifier()
    parent_path = Path(__file__).resolve().parent
    model_filename = parent_path / Path("garment_classifier.pt")
    return load_model_from_file(model, model_filename, device)


def load_fashion_mnist():
    """Loads the train and valid datasets for FashionMNIST."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    valid_dataset = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )
    return train_dataset, valid_dataset
