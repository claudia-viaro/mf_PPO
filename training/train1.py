from datetime import datetime
import json
from pprint import pprint
import time
import torch
import os
import pickle
import numpy as np
from constants import *
from environment_game import Game
from model import Actor, Critic
from ppo import PPO
from utils import plot1, plot2, save_model, save_plots, SaveBestModel, get_count
from running_state import *
from replay_memory import *
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_game')
from wrapper import BasicWrapper
sys.path.append('C:/Users/cvcla/my_py_projects/ModelFree/PPO_2/utils')
from logger import Logger
sys.path.append('C:/Users/cvcla/my_py_projects/GP_intervention')
from derivate import Derivate


def main(args):
    env = BasicWrapper()

    logger = Logger(args.logdir, args.seed)
    logger.log(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    actor = Actor(env.observation_size, env.action_size, args.n_hidden)
    critic = Critic(env.observation_size, args.n_hidden)    
    
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                 observation_shape= env.observation_size,
                                 action_dim=env.action_size)
    running_state = ZFilter((env.observation_size,), clip=5)
    statistics = {
        'reward': [],
        'val_loss': [],
        'policy_loss': [],
    }
    ppo_agent = PPO(env, args, actor, critic) 
    # collect initial experience with random action
    for episode in range(args.seed_episodes): #5 episodes
            print("collecting experience", episode)
            patients, S = env.reset() # S tensorg
            done = False
            while not done:
                A = env.sample_random_action()
                S_prime, R, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, is_done = env.multi_step(A, S.detach().numpy())
                # reward is actually the mean of 4 steps
                mask = 1 - int(is_done)
                replay_buffer.push(S, A, R, is_done, mask)
                if is_done:
                    done = True                    
                    break
                S = running_state(S_prime)

    # main training
    for episode in range(args.seed_episodes, args.all_episodes):
        print("main training", episode)
        start = time.time()         
        patients, S = env.reset() 
        
        done = False
        total_reward = 0
        count_iter = 0
        while not done:
            count_iter +=1 # count transitions in a trajectory
            A = ppo_agent.select_best_action(S)
            A = A.detach().numpy()
            A += np.random.normal(0, np.sqrt(args.action_noise_var),
                                        env.action_size)
            S_prime, R, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, is_done = env.step(A, S.detach().numpy())
            replay_buffer.push(S, A, R, is_done, mask)
            if is_done:
                    done = True                    
                    break
            S = running_state(S_prime)
            total_reward += R # summing rewards in a traj 
            mean_rew = total_reward/count_iter # mean reward in the trajectory

        print('episode [%4d/%4d] is collected. Mean reward is %f' % (episode+1, args.all_episodes, mean_rew))
        print('elasped time for interaction: %.2fs' % (time.time() - start))
        

        # update model parameters
        start = time.time()
        for update_step in range(args.collect_interval): #100 steps
            observations, actions, rewards, sampled_done, sampled_mask = \
                replay_buffer.sample(args.batch_size, args.chunk_length)
            actions = torch.as_tensor(actions).transpose(0, 1) #torch.Size([10, 10, 3])
            rewards = torch.as_tensor(rewards).transpose(0, 1)
            masks = torch.as_tensor(sampled_mask).transpose(0, 1)
            print("actions", actions.shape)
            print("rewards", rewards.shape)
            print("observations", observations.shape)

            
            embedded_observations = torch.tensor(observations, dtype=torch.float32)

            policy_loss, val_loss = ppo_agent.update_params_unstacked(embedded_observations, actions, rewards, masks) 
            # print losses
            print('update_step: %3d loss: %.5f, kl_loss: %.5f, obs_loss: %.5f, reward_loss: % .5f'
                    % (update_step+1,
                        policy_loss.item(), val_loss.item()))
            total_update_step = episode * args.collect_interval + update_step
        print('elasped time for update: %.2fs' % (time.time() - start))
        print("(episode + 1) % args.test_interval", (episode + 1) % args.test_interval)

if __name__ == "__main__":

    args = get_args()
    main(args)