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
sys.path.append('C:/Users/cvcla/my_py_projects/toy_game')
from wrapper import BasicWrapper
sys.path.append('C:/Users/cvcla/my_py_projects/GP_transition')
from GP_param_train import GaussianProcessClassifierLaplace

## suppose in the exploration part, the intervention funciton is deterministic


# make df to store X_a values
df = pd.DataFrame()


args = get_args()
env = BasicWrapper()
actor = Actor(env.observation_size, env.action_size, args.n_hidden)
critic = Critic(env.observation_size, args.n_hidden)  
MLPBase_model = MLPBase(env.observation_size, env.action_size, env.action_size) #what 3rd arg?
#GP_transition = StepGP(args, kernel_choice = linear) 

replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                observation_shape= env.observation_size,
                                action_dim=env.action_size)
running_state = ZFilter((env.observation_size,), clip=5)

ppo_agent = PPO(env, args, actor, critic, MLPBase_model) 

patients, S = env.reset() # S tensor
A = env.sample_random_action()
S_prime, R, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, done = env.multi_step(A, S.detach().numpy())
Xa_initial = patients[:, 1]
Xs_initial = patients[:, 0]
levels = ["L", "M", "H"]


kernel = RBF() + WhiteKernel(noise_level=0.5)
GPc = GaussianProcessClassifierLaplace(kernel = kernel)

# makea single episode
for episode in range(args.seed_episodes): #15 episodes
        start_time = time.time(); count_iter = 0
        patients, S = env.reset() # S tensor
        done = False
        while not done:
            count_iter +=1 # count transitions in a trajectory/episode
            A = env.sample_random_action()
            S_prime, R, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, done = env.multi_step(A, S.detach().numpy())
            mask = 1 - int(done)
            #replay_buffer.push(S, A, R, done, mask)

            S = running_state(S_prime)

            



df = pd.DataFrame(data={'Xa_pre': Xa_pre, 'Xa_post':  pat[:, 1], 'states': S_prime, "Outcome": outcome, 'states_old': S})
df = (df.assign(risk= lambda x: pd.cut(df['states'], 
                                            bins=[0, 0.4, 0.8, 1],
                                            labels=levels),
                risk_old= lambda x: pd.cut(df['states_old'], 
                                            bins=[0, 0.4, 0.8, 1],
                                            labels=levels)                            ))


#GPc.fit(preprocessing.normalize(Xa_pre.reshape(-1,1), norm='l2'), outcome.reshape(-1,1))
GPc.fit(Xa_pre.reshape(-1,1), outcome.reshape(-1,1))

mean, var = GPc.post_parameters(Xa_post.reshape(-1,1))
foc = GPc.FOC_i(Xa_post.reshape(-1,1))


def find_min(gaussian_process):
     
     values = np.zeros_like(Xa_post.reshape(-1,1))
     
     i = 0
     for i in range(3):
          trial = np.random.uniform(-4, 4,200)
          #values = np.column_stack((values, trial))
          b = gaussian_process.FOC_i(trial.reshape(-1, 1)).squeeze()

          values = np.column_stack((values,b))
          i += 1
     values = np.delete(values, 0,1)
     print("shape grid", np.array(values).shape) 
 
     return np.array(values).min(axis=1)

minim = find_min(GPc) 
#print("minimium", minim)  
print("min shape", minim.shape)
print("xapost shape", Xa_post.shape)

print("shape bounds", GPc.kernel_.bounds.shape)
optim_Xa = GPc.fit_Xa()

'''



import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import torch

def MCSelector(func, model, mc_search_num = 1000):
    xspace = model.XspaceGenerate(mc_search_num)

    utilitymat = np.zeros(mc_search_num)+float('-Inf')

    if hasattr(model, 'multi_hyper') and model.multi_hyper:
            for i, x in enumerate(xspace):
                if hasattr(model, 'is_real_data') and model.is_real_data:
                    if i in model.dataidx:
                        continue
                x = xspace[i:i+1]
                for m in model.modelset:
                    utilitymat[i]+= func(x, m)
    else:
        for i, x in enumerate(xspace):
            if hasattr(model, 'is_real_data') and model.is_real_data:
                if i in model.dataidx:
                    continue
            x = xspace[i:i+1]# all the inputs should take 2d array 
            # if version == 'pytorch':
            #     x = torch.tensor(x, requires_grad=True)
            utilitymat[i] = func(x, model)
    
    max_value = np.max(utilitymat, axis = None)
    max_index = np.random.choice(np.flatnonzero(utilitymat == max_value))

    if hasattr(model, 'is_real_data') and model.is_real_data:
        model.dataidx = np.append(model.dataidx, max_index)

    # plt.figure()
    # plt.plot(xspace, utilitymat, 'ro')
    # plt.show()
    
    x = xspace[max_index]

    # plt.figure()
    # plt.plot(xspace, utilitymat)
    # plt.show()

    return x, max_value

def RandomSampling(model):
    x = model.XspaceGenerate(1)
    max_value = 0
    return x, max_value

def SGD(func, model, mc_search_num = 1000, learning_rate = 0.001):
    #for mm in range(100):
    random_num = round(0.7*mc_search_num)
    #x11, value11 = MCSelector(func, model, mc_search_num)
    x1, value1 = MCSelector(func, model, random_num)
    #x0 = model.XspaceGenerate(1).reshape(-1)
    x0 = torch.tensor(x1, requires_grad= True)
    optimizer = torch.optim.SGD([x0], lr=learning_rate)
    

    # for _ in range(round(0.3*mc_search_num)):
    #     loss = -func(x0, model, version='pytorch')

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     print("loss: {}".format(loss))
    
    # x0 = torch.tensor(x1, requires_grad= True)
    # optimizer = torch.optim.Adam([x0], lr=learning_rate)
    

    for _ in range(round(0.3*mc_search_num)):
        loss = -func(x0, model, version='pytorch')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       # print("loss: {}".format(loss))

    return x0.detach().numpy(), -loss

    # func2 = lambda x: -1.0*func(x, model)
    # bounds = np.array([model.xinterval[0], model.xinterval[1]])*np.ones((model.f_num, 2))
    # res = minimize(func2, x0, method='TNC', options={'disp':False}, bounds = bounds)
    # xstar = res.x
    # max_value = -res.fun
    # return xstar, max_value
    # max_value = float('-Inf')
    # for mm in range(50):
    #     x0 = model.XspaceGenerate(1).item()
    #     func2 = lambda x: -1.0*func(x, model)
    #     bounds = [(model.xinterval[0], model.xinterval[1])]
    #     res = minimize(func2, x0, method='TNC', 
    #                     options={ 'disp':False}, bounds = bounds)
    #     xstar22 = res.x
    #     max_value22 = -res.fun
    #     print(res)
    #     if max_value22.item() > max_value:
    #         max_value = max_value22.item()
    #         xstar = xstar22



    # # x0 = model.XspaceGenerate(1).item()
    # # func2 = lambda x: -1.0*func(x, model)
    # # bounds = [(-4, 4)]
    # # res = minimize(func2, x0, method='trust-constr', 
    # #                 options={#'xatol':1e-8, 
    # #                 'disp':True}, bounds = bounds)
    # # x = res.x
    # # max_value = -res.fun
    # return xstar, max_value

'''
