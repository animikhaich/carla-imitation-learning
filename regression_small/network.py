import torch
import torch.nn as nn

class RegressionNetwork(torch.nn.Module):
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
        num_filters = 32

        self.fe = nn.Sequential(
            nn.Conv2d(min(self.input_size), num_filters, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.clf = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),
            
            nn.Linear(64, 3),
            nn.Tanh()
        )



    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        x = self.fe(observation)
        x = torch.flatten(x, 1)
        x = self.clf(x)
        return x


        

    def scores_to_action(self, scores):
        """
        The scores come in as [throttle, steer, brake] in the range (-1, 1).
        We need to map them back to the action space, where throttle and brake
        is in the range of (0, 1) and steer is in the range (-1, 1).
        We also convert the tensors back to numpy arrays.
        
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        scores = scores.detach().cpu().numpy()
        throttle = (scores[0] + 1) / 2
        steer = scores[1]
        brake = (scores[2] + 1) / 2
        return (throttle, steer, brake)
