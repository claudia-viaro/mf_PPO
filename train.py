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


# https://github.com/Abhipanda4/PPO-PyTorch/blob/master/environment.py
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_game')
from wrapper import BasicWrapper
sys.path.append('C:/Users/cvcla/my_py_projects/ModelFree/PPO_2/utils')
from logger import Logger


def main(args):
    logger = Logger(args.logdir, args.seed)
    logger.log(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # directory to save plots
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'plots_PPO/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)       


    env = BasicWrapper()
    actor = Actor(env.observation_size, env.action_size, args.n_hidden)
    critic = Critic(env.observation_size, args.n_hidden)    
    ppo_agent = PPO(env, args, actor, critic)
    
    '''
    # retrieve previous saved model if exists
    if os.path.exists(args.actor_save_path):
        print("Loading saved actor model...")
        ppo_agent.actor.load_state_dict(torch.load(args.actor_save_path))
    else: os.makedirs(args.actor_save_path)    
    if os.path.exists(args.critic_save_path):
        print("Loading saved critic model...")
        ppo_agent.critic.load_state_dict(torch.load(args.critic_save_path))
    else: os.makedirs(args.critic_save_path)      

    save_best_model = SaveBestModel()
    '''

    running_state = ZFilter((env.observation_size,), clip=5)
    statistics = {
        'reward': [],
        'val_loss': [],
        'policy_loss': [],
    }

    best_reward = 0
    list_rew = []
    for i in range(0, args.n_episodes):
        memory = Memory()
        num_steps = 0
        num_ep = 0
        reward_batch = 0
        rew_batch = []
        rew_batch_LR = []
        Xa_pre_batch = []
        Xa_post_batch = []
        st_batch = []

        while num_steps < args.batch_size:
            patients, S = env.reset() # S tensor
            st_batch.append(S.detach().numpy())
            #S = running_state(S)
            t = 0
            if num_steps ==0 and t==0:
                logger.log("-----{} E, New pop draw - initial risk: {:.3f}, min: {:.3f}, max: {:.3f}".format(i, np.mean(np.float64(S.detach().cpu().numpy())), np.min(np.float64(S.detach().cpu().numpy())), (np.max(np.float64(S.detach().cpu().numpy())))))
                logger.log("-----Count risks levels: {}".format(get_count(np.float64(S.detach().cpu().numpy())))
                )

            reward_sum = 0  
            done = False
            while not done:        
                t += 1
                A = ppo_agent.select_best_action(S)
                S_prime, R, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, is_done = env.step(A, S.detach().numpy())
                reward_sum += R
                rew_batch.append(R)
                rew_batch_LR.append(r_LogReg)
                Xa_pre_batch.append(Xa_pre)
                Xa_post_batch.append(Xa_post)
                st_batch.append(S_prime)

                mask = 1 - int(is_done)
                memory.push(S, A.detach().numpy(), mask, R)

                
                
                if is_done:
                    done = True                    
                    break

                S = running_state(S_prime)


            num_steps += t
            num_ep += 1
            reward_batch += reward_sum
            rew1 = reward_sum/num_steps
            
            logger.log_reward(reward_sum)

        reward_batch /= num_ep
        
        # The memory is now full of rollouts. Sample from memory and optimize
        batch = memory.sample()
        if i == 0:
            policy_loss, val_loss = ppo_agent.update_params(batch, print_StateDict = False)
        else: policy_loss, val_loss = ppo_agent.update_params(batch)    
        logger.log_losses(policy_loss, val_loss)

        # log data onto stdout
        if i == 0 or i % args.log_steps == 0:
            print("Episode: %d, Reward: %.3f, Value loss: [%.3f], Policy Loss: [%.3f]" %(i, np.mean(rew_batch), val_loss, policy_loss))
            message = "> Train epoch {} [losses - value {:.2f} policy {:.2f} | Reward {:.2f}]"
            logger.log(message.format(i,  val_loss, policy_loss, np.mean(rew_batch)))

   
            
        # save statistics per episode
        statistics['reward'].append(np.mean(rew_batch))
        statistics['val_loss'].append(val_loss.detach().numpy().astype(float))
        statistics['policy_loss'].append(policy_loss.detach().numpy().astype(float))
        
        '''
        # save models and statistics
        if reward_batch > best_reward:
            best_reward = reward_batch
            torch.save(ppo_agent.actor.state_dict(), args.actor_save_path)
            torch.save(ppo_agent.critic.state_dict(), args.critic_save_path)
        '''
    #plot2(statistics, results_dir, "plots")


if __name__ == "__main__":

    args = get_args()
    main(args)
