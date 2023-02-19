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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

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
        image = torchvision.io.read_image(os.path.join(self.data_dir, data.filename))
        image = image.type(torch.float)
        image = self.transform(image) if self.transform else image

        # We use tanh activation for the steering and brake, so we need to normalize the data
        throttle = (data.throttle - 0.5) * 2 # Range is now (-1, 1)
        steer = data.steer # Range is already (-1, 1)
        brake = (data.brake - 0.5) * 2 # Range is now (-1, 1)

        label = torch.tensor([throttle, steer, brake])

        return (image, label.type(torch.float))


"""Not using this function because I prefer to directly call the dataloader, makes it easier to pass arguments"""
def get_dataloader(data_dir, labels, batch_size, num_workers=32, shuffle=True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir=data_dir, labels=labels),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )
    