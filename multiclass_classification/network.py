import torch
import torchvision
import torch.nn as nn

class MultiClassClassificationModel(torch.nn.Module):
    def __init__(self, input_size=(96, 96, 3)):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_size = input_size
        self.init_model()

    def init_model(self):
        self.model = torchvision.models.resnet34(weights='DEFAULT')
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(self.model.fc.in_features, 64),
            nn.ReLU(),

            nn.Linear(64, 4),
            nn.Sigmoid(),
        )



    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        return self.model(observation)

        

    def scores_to_action(self, scores, scale_down=True):
        """
        The scores come in as [throttle, steer, brake] in the range (-1, 1).
        We need to map them back to the action space, where throttle and brake
        is in the range of (0, 1) and steer is in the range (-1, 1).
        We also convert the tensors back to numpy arrays.
        
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        scores = scores.detach().cpu().numpy()
        throttle = scores[0]
        left = scores[1]
        right = scores [2]
        brake = scores[3]
        if left > right:
            steer = -1
        else:
            steer = 1

        if scale_down:
            throttle = throttle / 2.5
            steer = steer / 2.5

        return (throttle, steer, brake)
