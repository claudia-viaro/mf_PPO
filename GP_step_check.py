import torch
import os
import time
import pickle
import numpy as np
from datetime import datetime
import json
import itertools
import more_itertools as mit
from sklearn import preprocessing
from constants import *
from model import Actor, Critic, MLPBase
from ppo import PPO

from running_state import *
from replay_memory import *
import pandas as pd
from sklearn.gaussian_process.kernels import PairwiseKernel, Exponentiation, WhiteKernel, RBF, ConstantKernel as C
from scipy.special import erf, expit
from scipy.linalg import cholesky, cho_solve, solve
from sklearn.metrics.pairwise import polynomial_kernel

import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_environment')
from wrapper import BasicWrapper

## suppose in the exploration part, the intervention funciton is deterministic


# make df to store X_a values
df = pd.DataFrame()


args = get_args()
env = BasicWrapper()
actor = Actor(env.observation_size, env.action_size, args.n_hidden)
critic = Critic(env.observation_size, args.n_hidden)  
MLPBase_model = MLPBase(env.observation_size, env.action_size, env.action_size) #what 3rd arg?
replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                observation_shape= env.observation_size,
                                action_dim=env.action_size)
running_state = ZFilter((env.observation_size,), clip=5)
ppo_agent = PPO(env, args, actor, critic, MLPBase_model) 

''''''
start_time = time.time()
total_reward = 0; total_rewardLR = 0
patients, S = env.reset() # S tensorg
A = env.sample_random_action()
S_prime, R, pat, s_LogReg, r_LogReg, Xa, Xa_prime, outcome, done = env.step(A, S.detach().numpy())
Xa_pre = Xa_prime
Y = outcome
A = env.sample_random_action()
S_prime, R, pat, rho_LogReg, r_LogReg, Xa_prime, outcome, is_done = env.GPstep_wrapper(A, S.detach().numpy(), Y, Xa_pre, pat[:, 1])
total_reward += R; total_rewardLR += r_LogReg

print("tot reward", total_reward, total_rewardLR)
print("time it took- ", time.time()-start_time)
'''
# take a single episode
for episode in range(2): #15 episodes
            
            start_time = time.time()
            patients, S = env.reset() # S tensorg
            A = env.sample_random_action()
            S_prime, R, pat, s_LogReg, r_LogReg, Xa, Xa_prime, outcome, done = env.step(A, S.detach().numpy())  
            done = False
            total_reward = 0; total_rewardLR = 0; count_iter = 0
            while not done:
                Xa_pre = Xa_prime
                Y = outcome
                A = env.sample_random_action()
                S_prime, R, pat, rho_LogReg, r_LogReg, Xa_prime, outcome, is_done = env.GPstep_wrapper(A, S.detach().numpy(), Y, Xa_pre, pat[:, 1])
                # reward is actually the mean of 4 steps
                mask = 1 - int(is_done)
                replay_buffer.push(S, A, R, is_done, mask)
                if is_done:
                    done = True                    
                    break
                S = running_state(S_prime)
                total_reward += R; total_rewardLR += r_LogReg
            mean_rew = total_reward/count_iter; mean_rewardLR = total_rewardLR/count_iter        
            print('episodes [%4d/%4d] are collected for experience.' % (args.seed_episodes, args.all_episodes))
            print('episode [%4d/%4d] is collected. Mean reward is %f' % (episode+1, args.all_episodes, mean_rew))
'''

            



