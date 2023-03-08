import random
from collections import namedtuple
import numpy as np
import torch

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb


## buffertype 1
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'mask'))
class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, state, action, mask, reward):
        """Saves a transition."""
        self.memory.append(Transition(state, action, reward, reward, mask))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

## buffertype 2 (currently used but I am not sure what happens when memory is complete)    
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


## buffertype 3

class Buffer(object):
    def __init__(
        self,
        state_size,
        action_size,
        ensemble_size,
        signal_noise=None,
        buffer_size=10 ** 6,
        device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size
        self.buffer_size = buffer_size
        self.signal_noise = signal_noise
        self.device = device

        self.states = np.zeros((buffer_size, state_size))
        self.actions = np.zeros((buffer_size, action_size))
        self.rewards = np.zeros((buffer_size, 1))
        self.state_deltas = np.zeros((buffer_size, state_size)) 
        self.dones = np.zeros((buffer_size, 1), dtype=bool)
        self.masks = np.zeros((buffer_size, 1), dtype=bool)


        self._total_steps = 0

    def add(self, state, action, reward, next_state, done, mask):
        idx = self._total_steps % self.buffer_size # count steps in order, fills in from beg then all 0
        state_delta = next_state - state

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.state_deltas[idx] = state_delta
        self.dones[idx] =  done
        self.masks[idx] = mask
        self._total_steps += 1


    def get_train_batches(self, batch_size):
        size = len(self) # 50 max episode len?
        indices = [
            np.random.permutation(range(size)) for _ in range(self.ensemble_size)
        ]
        indices = np.stack(indices).T

        for i in range(0, size, batch_size):
            j = min(size, i + batch_size)

            if (j - i) < batch_size and i != 0:
                return

            batch_size = j - i
            print("batch_size",batch_size)

            batch_indices = indices[i:j]
            batch_indices = batch_indices.flatten()
            print("batch_indices",batch_indices.shape)

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            rewards = self.rewards[batch_indices]
            state_deltas = self.state_deltas[batch_indices]
            dones = self.dones[batch_indices]
            masks = self.masks[batch_indices]

            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            state_deltas = torch.from_numpy(state_deltas).float().to(self.device)
            dones = torch.from_numpy(dones).float().to(self.device)
            masks = torch.from_numpy(masks).float().to(self.device)

            if self.signal_noise is not None:
                states = states + self.signal_noise * torch.randn_like(states)

            states = states.reshape(self.ensemble_size, batch_size, self.state_size)
            actions = actions.reshape(self.ensemble_size, batch_size, self.action_size)
            rewards = rewards.reshape(self.ensemble_size, batch_size, 1)
            state_deltas = state_deltas.reshape(self.ensemble_size, batch_size, self.state_size)
            dones = dones.reshape(self.ensemble_size, batch_size, 1)
            masks = masks.reshape(self.ensemble_size, batch_size, 1)

            yield states, actions, rewards, state_deltas, dones, masks

    def __len__(self):
        return min(self._total_steps, self.buffer_size)

    @property
    def total_steps(self):
        return self._total_steps


    