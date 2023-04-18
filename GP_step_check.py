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
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.gaussian_process.kernels import PairwiseKernel, Exponentiation, WhiteKernel, RBF, ConstantKernel as C
from scipy.special import erf, expit
from scipy.linalg import cholesky, cho_solve, solve
from sklearn.metrics.pairwise import polynomial_kernel
from utils import *
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_environment')
from wrapper import BasicWrapper

RAW_DATA_CSV_1 = os.path.join(os.path.dirname(__file__), "plots/raw_results1.csv")
RAW_DATA_CSV_2 = os.path.join(os.path.dirname(__file__), "plots/raw_results2.csv")


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
def one_random_run():
    episode = 1
    start = time.time()
    print('---episode [%4d/%4d] experience' % (episode, args.seed_episodes))

    patients, S = env.reset() # S tensor
    plot_histogram(S, "Histogram_reset", episode=episode, iter=0)
    A = env.sample_random_action()
    S_prime, R, pat, s_LogReg, r_LogReg, Xa, Xa_prime, outcome, is_done = env.multi_step(A, S.detach().numpy())
    done = False; max_done = 0
    total_reward = 0; total_rewardLR = 0; count_iter = 0
    while max_done <= 2: #not done:
        max_done += 1
        count_iter +=1 # count transitions in a trajectory
        
        Xa_pre = Xa_prime
        Y = outcome
        A = env.sample_random_action()
        S_prime, R, pat, rho_LogReg, r_LogReg, Xa_prime, outcome, is_done = env.GPstep_wrapper(A, S.detach().numpy(), Y, Xa_pre, pat[:, 1])
        mask = 1 - int(is_done)
        replay_buffer.push(S, A, R, is_done, mask)
        S = running_state(S_prime)
        total_reward += R; total_rewardLR += r_LogReg 
        print("iter [{:.0f}]: Reward {:.2f}, LR Rewards {:.2f}".format(count_iter, R, r_LogReg)             )
        export_data_trajectory = [episode, max_done, R, r_LogReg, S, Xa_prime, outcome]
        export_to_csv(RAW_DATA_CSV_1, export_data_trajectory)
        if max_done ==2:
             plot_Xa_risk(S, "Plot", episode, iter, directory = results_dir)
             
    mean_rew = total_reward/count_iter; mean_rewardLR = total_rewardLR/count_iter 
    print("episode [{:.0f}/{:.0f}] is collected. Mean Rewards {:.2f}, Mean LR Rewards {:.2f} over {:.0f} transitions, it took {:.2f} min".format(episode+1, args.all_episodes, mean_rew, mean_rewardLR, count_iter, (time.time() - start))             )
    export_data_episode = [episode, mean_rew, mean_rewardLR, count_iter]
    export_to_csv(RAW_DATA_CSV_2, export_data_episode)

 

def summarise_results():
	"""
	Read in the csv file of results and summarise the results
	"""
	# Import the results to a dataframe
	trajectory_output = pd.read_csv(RAW_DATA_CSV_1, sep=";", header=None, names=["episode", "iters", "reward", "LRreward", "states", "Xa", "outcome"])
	pd.set_option('display.max_columns', 10)
	#summary = trajectory_output.groupby(["student"])["win"].agg(["mean"]).reset_index()	
	
    #summary.to_csv(SUMMARY_CSV_FILE_b, sep=";")
	episode_output = pd.read_csv(RAW_DATA_CSV_2, sep=";", header=None, names=["episode", "mean_reward", "mean_rewardLR", "iter_count"])


if __name__=="__main__":
    args = get_args() 
    one_random_run()
    #summarise_results()

