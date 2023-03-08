'''
1 episode 1 trajectory 
'''

import torch
import os
import time
import pickle
import numpy as np
from datetime import datetime
import json
from constants import *
from model import Actor, Critic, MLPBase
from ppo import PPO
from utils import plot1, plot2, save_model, save_plots, SaveBestModel, get_count
from running_state import *
from replay_memory import *
from GP_step import StepGP
from logger.logger import Logger

import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_game')
from wrapper import BasicWrapper


def main(args, logger):

    


    env = BasicWrapper()
    logger.log(args)


    actor = Actor(env.observation_size, env.action_size, args.n_hidden)
    critic = Critic(env.observation_size, args.n_hidden)  
    MLPBase_model = MLPBase(env.observation_size, env.action_size, env.action_size) #what 3rd arg?
    #GP_transition = StepGP(args, kernel_choice = linear) 
    
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                 observation_shape= env.observation_size,
                                 action_dim=env.action_size)
    running_state = ZFilter((env.observation_size,), clip=5)
    
    ppo_agent = PPO(env, args, actor, critic, MLPBase_model) 
    # collect initial experience with random action
    for episode in range(args.seed_episodes): #15 episodes
            start_time = time.time()
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
    print('episodes [%4d/%4d] are collected for experience.' % (args.seed_episodes, args.all_episodes))

    # main training
    for episode in range(args.seed_episodes, args.all_episodes):
        #start = time.time()         
        patients, S = env.reset()         
        done = False
        total_reward = 0; total_rewardLR = 0; count_iter = 0
        while not done:
            count_iter +=1 # count transitions in a trajectory
            
            A = ppo_agent.select_best_action(S)
            A = A.detach().numpy()
            A += np.random.normal(0, np.sqrt(args.action_noise_var),
                                        env.action_size)
            S_prime, R, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, done = env.step(A, S.detach().numpy())


            replay_buffer.push(S, A, R, is_done, mask)
            
            S = running_state(S_prime)
            total_reward += R; total_rewardLR += r_LogReg
            
            
        mean_rew = total_reward/count_iter; mean_rewardLR = total_rewardLR/count_iter    
        logger.log_episode(episode+1, args.all_episodes, mean_rew, mean_rewardLR, count_iter)             

        #print('episode [%4d/%4d] is collected. Mean reward is %f' % (episode+1, args.all_episodes, mean_rew))
        #print('elasped time for interaction: %.2fs' % (time.time() - start))
        

        # update model parameters
        start = time.time()
        total_Ploss = 0; total_Vloss = 0
        for update_step in range(args.collect_interval): #100 steps
            observations, actions, rewards, sampled_done, sampled_mask = \
                replay_buffer.sample(args.batch_size, args.chunk_length)
            actions = torch.as_tensor(actions).transpose(0, 1) #torch.Size([10, 10, 3])
            rewards = torch.as_tensor(rewards).transpose(0, 1)
            masks = torch.as_tensor(sampled_mask).transpose(0, 1)            
            embedded_observations = torch.tensor(observations, dtype=torch.float32).transpose(0, 1)
            
            policy_loss, val_loss = ppo_agent.update_params_unstacked(embedded_observations, actions, rewards, masks) 
            total_Ploss += val_loss.detach().numpy(); total_Vloss += policy_loss.detach().numpy()
            
            
            # print losses
            if (update_step + 1) % 20 == 0:
                print('update_step: %3d policy loss: %.5f, value loss: %.5f'% (update_step+1,policy_loss.item(), val_loss.item()))
                total_update_step = episode * args.collect_interval + update_step
                
        # logging mean loss values
        logger.log_update(total_Ploss, total_Vloss)    
        #print('elasped time for update: %.2fs' % (time.time() - start))

        # test to get score without exploration noise
        if (episode + 1) % args.test_interval == 0:
            start = time.time()
            ppo_agent = PPO(env, args, actor, critic, MLPBase_model) 
            pat, obs = env.reset()
            done = False
            total_reward = 0
            count_iter_test = 0
            while not done:
                count_iter_test += 1
                action = ppo_agent.select_best_action(obs)
                obs, reward, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, done = env.step(action, obs.detach().numpy())
        
                total_reward += reward

            print('Total test reward at episode [%4d/%4d] is %f' %
                    (episode+1, args.all_episodes, total_reward/count_iter_test))
            #print('elapsed time for test: %.2fs' % (time.time() - start))


    # save learned model parameters
    torch.save(ppo_agent.actor.state_dict(), os.path.join(args.log_dir, 'actor.pth'))
    torch.save(ppo_agent.critic.state_dict(), os.path.join(args.log_dir, 'critic.pth'))
    logger.log_time(time.time() - start_time)


if __name__ == "__main__":

    args = get_args()
    args.log_dir = "train_1"
    logger = Logger(args.log_dir, args.seed)    
    
    main(args, logger)
    
    logger.save()