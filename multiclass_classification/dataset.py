import glob
import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


class CarlaDataset(Dataset):
    def __init__(self, data_dir, labels, transform=None, image_size=(96, 96)):
        self.data_dir = data_dir
        self.image_filenames = os.listdir(data_dir)
        self.labels = pd.read_csv(labels)
        self.delta = 1e-3 # Defines the interval (+/-) where sensor margin of error is considered

        # Assert that the number of images are greater than or equal to the number of labels
        # assert len(self.image_filenames) <= len(self.labels), "There are more labels than images. This means there are missing images."

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.Normalize(mean=0, std=1)
            ])

        self.idx_to_action = {
            0: "throttle",
            1: "left",
            2: "right",
            3: "brake",
            4: "OOD"
        }

        self.action_to_idx = {
            "throttle": 0,
            "left": 1,
            "right": 2,
            "brake": 3,
            "OOD": 4
        }

    def __len__(self):
        """
        __len__ Get the length of the dataset

        Returns:
            int: length of the dataset
        """
        return min(len(self.image_filenames), len(self.labels))

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        data = self.labels.iloc[idx]
        image = torchvision.io.read_image(os.path.join(self.data_dir, data.filename)) / 255.
        image = self.transform(image) if self.transform else image

        left = 0
        right = 0

        # Steering
        if data.steer < -self.delta:
            left = 1
        elif data.steer > self.delta:
            right = 1
        
        # Throttle
        throttle = 1 if data.throttle > self.delta else 0

        # Brake
        brake = 1 if data.brake > self.delta else 0

        label = torch.tensor([throttle, left, right, brake]).type(torch.float)

        return (image, label)


"""Not using this function because I prefer to directly call the dataloader, makes it easier to pass arguments"""
def get_dataloader(data_dir, labels, batch_size, num_workers=32, shuffle=True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir=data_dir, labels=labels),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )
    