import random
from collections import namedtuple
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'reward'))

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, state, action, mask, reward):
        """Saves a transition."""
        self.memory.append(Transition(state, action, mask, reward))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
    
class ReplayBuffer(object):
    """
    Replay buffer for training with RNN
    """
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.observations = np.zeros((capacity, observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)
        self.mask = np.zeros((capacity, 1), dtype=bool)

        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done, mask):
        """
        Add experience to replay buffer
        NOTE: observation should be transformed to np.uint8 before push
        """
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done
        self.mask[self.index] = mask

        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        """
        Sample experiences from replay buffer (almost) uniformly
        The resulting array will be of the form (batch_size, chunk_length)
        and each batch is consecutive sequence
        NOTE: too large chunk_length for the length of episode will cause problems
        """
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            cross_border = True

     
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1) # len(self) is the length of the buffer full so far
                
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(initial_index <= episode_borders,
                                              episode_borders < final_index).any()
                '''
            for i in range(6):

                initial_index = np.random.randint(len(self) - chunk_length + 1)
                
                final_index = initial_index + chunk_length - 1

                cross_border = np.logical_and(initial_index <= episode_borders,
                                              episode_borders < final_index).any()
            '''
            sampled_indexes += list(range(initial_index, final_index + 1))                              
        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, self.observations.shape[1])
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1])
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        sampled_mask = self.mask[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        return sampled_observations, sampled_actions, sampled_rewards, sampled_done, sampled_mask

    def __len__(self):
        return self.capacity if self.is_filled else self.index
