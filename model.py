import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

class Actor(nn.Module):
    def __init__(self, n_inp, n_output, n_hidden):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.mean = nn.Linear(n_hidden, n_output)
        self.mean.weight.data.mul_(0.1)
        self.mean.bias.data.mul_(0.0)

        self.log_stddev = nn.Parameter(torch.zeros(n_output))

        self.module_list = [self.fc1, self.fc2, self.mean, self.log_stddev]
        self.module_list_old = [None] * 4

        # required so that start of episode does not throw error
        self.backup()

    def backup(self):
        for i in range(len(self.module_list)):
            self.module_list_old[i] = deepcopy(self.module_list[i])

    def forward(self, x, old=False):
        if not old:
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            mu = self.mean(x)
            log_stddev = self.log_stddev.expand_as(mu)
            return mu.squeeze(), log_stddev.squeeze()
        else:
            x = torch.tanh(self.module_list_old[0](x))
            x = torch.tanh(self.module_list_old[1](x))
            mu = self.module_list_old[2](x)
            log_stddev = self.module_list_old[3].expand_as(mu)
            return mu.squeeze(), log_stddev.squeeze()


class Critic(nn.Module):
    def __init__(self, n_inp, n_hidden):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.state_val = nn.Linear(n_hidden, 1)
        self.state_val.weight.data.mul_(0.1)
        self.state_val.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        state_val = self.state_val(x)
        return state_val


"""
Model definitions.
These are modified versions of the models from
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py
"""
import numpy as np



def init(module, weight_init, bias_init, gain=1):
    """Helper to initialize a layer weight and bias."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MLPBase(nn.Module):
    """Basic multi-layer linear model."""
    def __init__(self, num_inputs, num_outputs, dist, hidden_size=64):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        init2_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init2_(nn.Linear(hidden_size, num_outputs)))

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        self.dist = dist

    def forward(self, x):
        value = self.critic(x)
        action_logits = self.actor(x)
        return value, self.dist(action_logits)


class Discrete(nn.Module):
    """A module that builds a Categorical distribution from logits."""
    def __init__(self, num_outputs):
        super().__init__()

    def forward(self, x):
        # Do softmax on the proper dimesion with either batched or non
        # batched inputs
        if len(x.shape) == 3:
            probs = nn.functional.softmax(x, dim=2)
        elif len(x.shape) == 2:
            probs = nn.functional.softmax(x, dim=1)
        else:
            print(x.shape)
            raise
        dist = torch.distributions.Categorical(probs=probs)
        return dist


class Normal(nn.Module):
    """A module that builds a Diagonal Gaussian distribution from means.
    Standard deviations are learned parameters in this module.
    """
    def __init__(self, num_outputs):
        super().__init__()
        # initial variance is e^0 = 1
        self.stds = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        dist = torch.distributions.Normal(loc=x, scale=self.stds.exp())

        # By default we get the probability of sampling each dimension of the
        # distribution. The full probability is the product of these, or
        # the sum since we're working with log probabilities.
        # So overwrite the log_prob function to handle this for us
        dist.old_log_prob = dist.log_prob
        dist.log_prob = lambda x: dist.old_log_prob(x).sum(-1)

        return dist