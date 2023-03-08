import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from constants import *
from utils import *


class PPO:
    def __init__(self, env, args, actor, critic, MLPBase):
        self.env = env
        self.args = args
        self.actor = actor
        self.critic = critic
        self.MLPBase_model = MLPBase
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.a_learning_rate, eps=args.epsilon)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.c_learning_rate, eps=args.epsilon)
        
        self.optimizer_MLP = torch.optim.Adam(self.MLPBase_model.parameters(), lr=self.args.a_learning_rate) 

    def select_best_action(self, S):
        S = S.to(dtype=torch.float32) 
        S = torch.FloatTensor(S)
        mu, log_sigma = self.actor(Variable(S))
        action = torch.normal(mu, torch.exp(log_sigma))
        return action

    def compute_advantage(self, values, batch_R, batch_mask):
        batch_size = len(batch_R)

        v_target = torch.FloatTensor(batch_size)
        advantages = torch.FloatTensor(batch_size)

        prev_v_target = 0
        prev_v = 0
        prev_A = 0
        
        
        for i in reversed(range(batch_size)):
            v_target[i] = batch_R[i] + self.args.gamma * prev_v_target * batch_mask[i]
            delta = batch_R[i] + self.args.gamma * prev_v * batch_mask[i] - values.data[i]
            advantages[i] = delta + self.args.gamma * self.args.tau * prev_A * batch_mask[i]

            prev_v_target = v_target[i]
            prev_v = values.data[i]
            prev_A = advantages[i]
        

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        return advantages, v_target
    
    def gae(self, rewards, values, episode_ends):
        """Compute generalized advantage estimate.
            rewards: a list of rewards at each step.
            values: the value estimate of the state at each step.
            episode_ends: an array of the same shape as rewards, with a 1 if the
                episode ended at that step and a 0 otherwise.
            gamma: the discount factor.
            lam: the GAE lambda parameter.
        """
        # Invert episode_ends to have 0 if the episode ended and 1 otherwise
        episode_ends = (episode_ends * -1) + 1

        N = rewards.shape[0]
        T = rewards.shape[1]
        gae_step = np.zeros((N, ))
        advantages = np.zeros((N, T))
        for t in reversed(range(T - 1)):
            # First compute delta, which is the one-step TD error
            delta = rewards[:, t] + self.args.gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
            # Then compute the current step's GAE by discounting the previous step
            # of GAE, resetting it to zero if the episode ended, and adding this
            # step's delta
            gae_step = delta + self.args.gamma * self.args.tau * episode_ends[:, t] * gae_step
            # And store it
            advantages[:, t] = gae_step
        return advantages
    
    def train_step(self, batch_data):
        self.MLPBase_model.train()
        self.optimizer_MLP.zero_grad()

        advantages, rewards_to_go, values, actions, obs, \
            selected_prob = batch_data

        values_new, dist_new = self.MLPBase_model(obs)
        values_new = values_new.flatten()
        selected_prob_new = dist_new.log_prob(actions)

        # Compute the PPO loss
        prob_ratio = torch.exp(selected_prob_new) / torch.exp(selected_prob)

        a = prob_ratio * advantages
        b = torch.clamp(prob_ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantages
        ppo_loss = -1 * torch.mean(torch.min(a, b))

        # Compute the value function loss
        # Clipped loss - same idea as PPO loss, don't allow value to move too
        # far from where it was previously
        value_pred_clipped = values + (values_new - values).clamp(-self.args.epsilon, self.args.epsilon)
        value_losses = (values_new - rewards_to_go) ** 2
        value_losses_clipped = (value_pred_clipped - rewards_to_go) ** 2
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        value_loss = value_loss.mean()

        entropy_loss = torch.mean(dist_new.entropy())

        loss = ppo_loss + self.args.value_loss_coef * value_loss - self.args.entropy_coef * entropy_loss
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.MLPBase_model.parameters(), .5)
        self.optimizer_MLP.step()

        return loss, ppo_loss


    def update_params(self, batch, print_StateDict = False):
        '''
        S = torch.FloatTensor(batch.state)
        masks = torch.FloatTensor(batch.mask)
        A = torch.FloatTensor(np.concatenate(batch.action, 0))
        R = torch.FloatTensor(batch.reward)
        '''
        S = torch.stack(list(batch.state), dim=0).to(dtype=torch.float32) 
        masks = torch.Tensor(batch.mask)
        A = torch.Tensor(batch.action) 
        
        R = torch.Tensor(batch.reward)

        V_S = self.critic(Variable(S))
        advantages, v_target = self.compute_advantage(V_S, R, masks)
        advantages =  advantages[:, None]

        # loss function for value net
        L_vf = torch.mean(torch.pow(V_S - Variable(v_target), 2))

        # optimize the critic net
        self.critic_optimizer.zero_grad()
        L_vf.backward(retain_graph=True)
        self.critic_optimizer.step()

        # cast into variable
        A = Variable(A)

        # new log probability of the actions
        means, log_stddevs = self.actor(Variable(S))
        new_log_prob = get_gaussian_log(A, means, log_stddevs)

        # old log probability of the actions
        with torch.no_grad():
            old_means, old_log_stddevs = self.actor(Variable(S), old=True)
            old_log_prob = get_gaussian_log(A, old_means, old_log_stddevs)

        # save the old actor
        self.actor.backup()

        # ratio of new and old policies
        ratio = torch.exp(new_log_prob - old_log_prob)

        # find clipped loss
        advantages = Variable(advantages)
        L_cpi = ratio * advantages
        clip_factor = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantages
        L_clip = -torch.mean(torch.min(L_cpi, clip_factor))
        actor_loss = L_clip

        # optimize actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
        self.actor_optimizer.step()
        if print_StateDict == True:
            print("Model's state_dict:")
            for param_tensor in self.actor.state_dict():
                print(param_tensor, "\t", self.actor.state_dict()[param_tensor].size())
        return actor_loss, L_vf, advantages, clip_factor, v_target, V_S
    

    def update_params_unstacked(self, S, A, R, M, print_StateDict = False):
        L_vf = 0
        L_clip = 0
        S = S.clone().detach().requires_grad_(True)
        A = A.clone().detach().requires_grad_(True)

        for l in range(self.args.chunk_length - 1):
            print("l", l)
            V_S = self.critic(Variable(S[l]))
            advantages, v_target = self.compute_advantage(V_S, R[l], M[l])
            
            advantages =  advantages[:, None]
            # loss function for value net
            L_vf += torch.mean(torch.pow(V_S - Variable(v_target), 2)) 
             
            # cast into variable
            A_s = Variable(A[l])
            # new log probability of the actions
            means, log_stddevs = self.actor(Variable(S[l]))
            new_log_prob = get_gaussian_log(A_s, means, log_stddevs)    
            # old log probability of the actions
            with torch.no_grad():
                old_means, old_log_stddevs = self.actor(Variable(S[l]), old=True)
                old_log_prob = get_gaussian_log(A_s, old_means, old_log_stddevs)

            # save the old actor
            self.actor.backup()
            # ratio of new and old policies
            ratio = torch.exp(new_log_prob - old_log_prob)
            # find clipped loss
            advantages = Variable(advantages)
            L_cpi = ratio * advantages
            clip_factor = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantages
            L_clip += -torch.mean(torch.min(L_cpi, clip_factor))
            
            if self.args.chunk_length == 2:
                print("advantages", advantages.shape, advantages)

        
        L_vf /= (self.args.chunk_length - 1) 
        # optimize the critic net
        self.critic_optimizer.zero_grad()
        L_vf.backward(retain_graph=True)
        self.critic_optimizer.step()
        L_clip /= (self.args.chunk_length - 1)
        # optimize actor network
        self.actor_optimizer.zero_grad()
        L_clip.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
        self.actor_optimizer.step()
        if print_StateDict == True:
            print("Model's state_dict:")
            for param_tensor in self.actor.state_dict():
                print(param_tensor, "\t", self.actor.state_dict()[param_tensor].size())
        
        

        return L_clip, L_vf
    
    