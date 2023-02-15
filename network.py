import torch
import torch.nn as nn
import torchvision

class ClassificationNetwork(torch.nn.Module):
    def __init__(self, input_size=(96, 96, 3)):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_size = input_size
        
        self.idx_to_action = {
            0: (1., 0., 0.), # Throttle
            1: (1., -1., 0.), # Throttle Left
            2: (1., 1., 0.), # Throttle Right
            3: (0., 0., 1.), # Brake
            4: (0., -1., 1.), # Brake Left
            5: (0., 1., 1.), # Brake Left
            6: (0., 0., 0.) # OOD
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

        self.num_classes = len(self.idx_to_action)

        self.init_model()

    

    def init_model(self):
        self.model = torchvision.models.resnet18(weights='DEFAULT')
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(self.model.fc.in_features, 2048),
            nn.ReLU(),

            torch.nn.Linear(2048, 256),
            nn.ReLU(),

            nn.Linear(256, self.num_classes),
            nn.LeakyReLU(negative_slope=0.2),
        )




    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        
        return self.model(observation)

    def actions_to_classes(self, actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are C different classes, then
        every action is represented by a C-dim vector which has exactly one
        non-zero entry (one-hot encoding). That index corresponds to the class
        number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size C
        """
        throttle, steer, brake = actions

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
            action = "brake"
        elif brake and steer < 0:
            action = "brake_left"
        elif brake and steer > 0:
            action = "brake_right"
        elif throttle and steer == 0:
            action = "throttle"
        elif throttle and steer < 0:
            action = "throttle_left"
        elif throttle and steer > 0:
            action = "throttle_right"
        else:
            action = "OOD"

        label = torch.tensor(self.action_to_idx[action])
        encoded_action = torch.nn.functional.one_hot(label, num_classes=len(self.action_to_idx))
        return encoded_action
        

    def scores_to_action(self, scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [accelaration, steering, braking].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        idx = torch.argmax(scores, axis=1).item()
        return self.idx_to_action[idx]
