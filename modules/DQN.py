import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from .utils import Variable


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.fc = nn.Linear(128, num_actions)
        self.num_actions = num_actions

    def forward(self, x, return_latent = 'last'):
        """Args:
        	 return_latent: 'last': return last hidden vector
           								'state': return the state
        """
        hidden = self.layers(x)
        out = self.fc(F.relu(hidden))
        if return_latent == "state":
          	return out, state
        return out, hidden 

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value,_  = self.forward(state)
            action  = q_value.max(1)[1].data[0]
            action = int(action)
        else:
            action = random.randrange(self.num_actions)
        return action


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
