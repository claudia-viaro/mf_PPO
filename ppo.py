import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from constants import *
from utils import *

class PPO:
    def __init__(self, env, args, actor, critic):
        self.env = env
        self.args = args
        self.actor = actor
        self.critic = critic

        self.all_params = (list(self.actor.parameters()) +
                list(self.critic.parameters()) )
        
        self.optimizer = optim.Adam(self.all_params, lr=self.args.a_learning_rate, eps=args.epsilon)


    def select_best_action(self, S):
        S = S.to(dtype=torch.float32) 
        S = torch.FloatTensor(S)
        mu, log_sigma = self.actor(Variable(S))
        action = torch.normal(mu, torch.exp(log_sigma))
        print("action selected", action)
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
        self.optimizer.zero_grad()
        L_vf.backward()
        self.optimizer.step()

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
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
        self.optimizer.step()
        if print_StateDict == True:
            print("Model's state_dict:")
            for param_tensor in self.actor.state_dict():
                print(param_tensor, "\t", self.actor.state_dict()[param_tensor].size())
        return L_clip, L_vf