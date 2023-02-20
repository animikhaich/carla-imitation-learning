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
    def __init__(self, data_dir, labels, transform=None, image_size=(224, 224)):
        self.data_dir = data_dir
        self.image_filenames = os.listdir(data_dir)
        self.labels = pd.read_csv(labels)
        self.delta = 1e-3 # Defines the interval (+/-) where sensor margin of error is considered
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.Normalize(mean=0, std=1)
            ])
        
        self.idx_to_action = {
            0: "throttle",
            1: "throttle_left",
            2: "throttle_right",
            3: "brake",
            4: "brake_left",
            5: "brake_right",
            6: "OOD"
        }

        self.action_to_idx = {
            "throttle": 0,
            "throttle_left": 1,
            "throttle_right": 2,
            "brake": 3,
            "brake_left": 4,
            "brake_right": 5,
            "OOD": 6
        }

    def __len__(self):
        return len(self.image_filenames)

    def actions_to_classes(self, throttle, steer, brake):
        # Discretize Steering 
        if steer < -self.delta:
            steer = -1
        elif steer > self.delta:
            steer = 1
        else:
            steer = 0
        
        # Discretize Throttle
        throttle = 1 if throttle > self.delta else 0

        # Conditions
        if brake and steer == 0:
            return "brake"
        elif brake and steer < 0:
            return "brake_left"
        elif brake and steer > 0:
            return "brake_right"
        elif throttle and steer == 0:
            return "throttle"
        elif throttle and steer < 0:
            return "throttle_left"
        elif throttle and steer > 0:
            return "throttle_right"
        else:
            return "OOD"            


    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        data = self.labels.iloc[idx]
        image = torchvision.io.read_image(os.path.join(self.data_dir, data.filename)) / 255.0
        image = self.transform(image) if self.transform else image

        action = self.actions_to_classes(data.throttle, data.steer, data.brake)
        label = torch.tensor(self.action_to_idx[action])
        label = torch.nn.functional.one_hot(label, num_classes=len(self.action_to_idx))

        return (image, label.type(torch.DoubleTensor))




def get_dataloader(data_dir, labels, batch_size, num_workers=32, shuffle=True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir=data_dir, labels=labels),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )
    