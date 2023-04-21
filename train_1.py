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
from utils import *
from running_state import *
from replay_memory import *
from logger.logger import Logger

import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_environment')
from wrapper import BasicWrapper

PPOfix_DATA_CSV_1 = os.path.join(os.path.dirname(__file__), "plots/PPOfix_results1.csv")
PPOfix_DATA_CSV_2 = os.path.join(os.path.dirname(__file__), "plots/PPOfix_results2.csv")

def main(args, logger):   

    env = BasicWrapper()
    logger.log(args)

    actor = Actor(env.observation_size, env.action_size, args.n_hidden)
    critic = Critic(env.observation_size, args.n_hidden)  
    MLPBase_model = MLPBase(env.observation_size, env.action_size, env.action_size) #what 3rd arg?
    
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                 observation_shape= env.observation_size,
                                 action_dim=env.action_size)
    running_state = ZFilter((env.observation_size,), clip=5)
    
    ppo_agent = PPO(env, args, actor, critic, MLPBase_model) 
    
    
    # collect initial experience with random action
    for episode in range(args.seed_episodes): #15 episodes
            start_time = time.time()
            print('---episode [%4d/%4d] experience' % (episode, args.seed_episodes))
            patients, S = env.reset() # S tensor
            plot_Xa_risk(S, "Plot_reset", episode=episode, iter=0)
            done = False
            total_reward = 0; total_rewardLR = 0; count_iter = 0
            while not done:
                count_iter +=1

                A = env.sample_random_action()
                S_prime, R, pat, rho_LogReg, r_LogReg, Xa_prime, outcome, is_done = env.multi_step(A, S.detach().numpy(), patients)
                mask = 1 - int(is_done)
                replay_buffer.push(S, A, R, is_done, mask)
                
                # link values 
                S = running_state(S_prime); patients = pat            
                total_reward += R; total_rewardLR += r_LogReg

                # save data
                logger.log_trajectory(count_iter, R, r_LogReg, S_prime, Xa_prime, outcome, A)
                export_data_trajectory = [episode, count_iter, R, r_LogReg, S, Xa_prime, outcome]
                export_to_csv(PPOfix_DATA_CSV_1, export_data_trajectory)

                if is_done:
                    done = True   
                    plot_Xa_risk(S_prime, "Plot", episode, count_iter, directory = results_dir)
                    plot_classification(S_prime, outcome, pat, episode, count_iter, "classification")
                    plot_classification_1(S_prime, outcome, pat, episode, count_iter, "joint_plot_classification")                 
                    break

            mean_rew = total_reward/count_iter; mean_rewardLR = total_rewardLR/count_iter    
            logger.log_episode(episode+1, args.all_episodes, mean_rew, mean_rewardLR, count_iter, (time.time() - start_time))
            export_data_episode = [episode, mean_rew, mean_rewardLR, count_iter]
            export_to_csv(PPOfix_DATA_CSV_2, export_data_episode)        
    print('episodes [%4d/%4d] are collected for experience.' % (args.seed_episodes, args.all_episodes))

    # main training
    for episode in range(args.seed_episodes, args.all_episodes):
        start = time.time() 
        print('---episode [%4d/%4d] experience' % (episode, args.seed_episodes))        
        patients, S = env.reset() 
        plot_Xa_risk(S, "Plot_reset", episode=episode, iter=0)        
        done = False
        total_reward = 0; total_rewardLR = 0; count_iter = 0
        while not done:            
            count_iter +=1

            A = ppo_agent.select_best_action(S)
            A = A.detach().numpy()
            A += np.random.normal(0, np.sqrt(args.action_noise_var), env.action_size)

            S_prime, R, pat, rho_LogReg, r_LogReg, Xa_prime, outcome, is_done = env.step(A, S.detach().numpy(), patients)
            replay_buffer.push(S, A, R, is_done, mask)
            # link values 
            S = running_state(S_prime); patients = pat  
            total_reward += R; total_rewardLR += r_LogReg   
            logger.log_trajectory(count_iter, R, r_LogReg, S_prime, Xa_prime, outcome, A)   
            export_data_trajectory = [episode, count_iter, R, r_LogReg, S, Xa_prime, outcome]
            export_to_csv(PPOfix_DATA_CSV_1, export_data_trajectory)       

            if is_done:
                done = True   
                plot_Xa_risk(S_prime, "Plot", episode, count_iter, directory = results_dir)
                plot_classification(S_prime, outcome, pat, episode, count_iter, "classification")
                plot_classification_1(S_prime, outcome, pat, episode, count_iter, "joint_plot_classification")                 
                break
            
        mean_rew = total_reward/count_iter; mean_rewardLR = total_rewardLR/count_iter    
        logger.log_episode(episode+1, args.all_episodes, mean_rew, mean_rewardLR, count_iter, (time.time() - start))         
        export_data_episode = [episode, mean_rew, mean_rewardLR, count_iter]
        export_to_csv(PPOfix_DATA_CSV_2, export_data_episode)
        print('episode [%4d/%4d] is collected. Mean reward is %f' % (episode+1, args.all_episodes, mean_rew))             
        
        

        # update model parameters
        start_update = time.time()
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
            if (update_step + 1) % 2 == 0:
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
    logger.log_time(time.time() - start_update)


if __name__ == "__main__":

    args = get_args()
    args.log_dir = "train_PPOfix"
    logger = Logger(args.log_dir, args.seed)    
    main(args, logger)    
    logger.save(); logger.save_m() #logger.save_csv()