import torch
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
        num_filters = 32

        self.fe = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=5, stride=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(),
        )

        self.clf = nn.Sequential(
            nn.Linear(225792, 2048),
            nn.ReLU(),

            nn.Linear(2048, 4),
            nn.Sigmoid()
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
