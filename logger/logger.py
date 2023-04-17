import os
import json
from datetime import datetime
import pprint
import csv


class Logger(object):
    def __init__(self, logdir, seed):
        self.logdir = logdir
        self.seed = seed
        self.path = "log_" + logdir + "_" + str(seed) + "/"
        self.print_path = self.path + "out.txt"
        self.metrics_path = self.path + "metrics.json"
        self.saved_models_paths = self.path + "models.json"
        self.path_csv = os.path.join(os.path.dirname(__file__), "/output.csv")

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.path_csv, exist_ok=True)
        self.metrics = {}
        self.saved_models= {}
        self._init_print()
        self._setup_metrics()
        self._setup_list()

    def log(self, string):
        f = open(self.print_path, "a")
        f.write("\n")
        f.write(str(string))
        f.close()
        print(string)

  
    # logging trajectory info, all transitions
    def log_trajectory(self, steps, reward, LR_reward, S_prime, Xa_prime, outcome, action):
        self.metrics["t_rewards"].append(reward)
        self.metrics["t_rewardsLR"].append(LR_reward)
        self.metrics["steps"].append(steps)
        self.metrics["t_states"].append(S_prime)
        self.metrics["t_Xa"].append(Xa_prime)
        self.metrics["t_outcome"].append(outcome)
        self.metrics["t_actions"].append(action)
        msg = "step [{:.0f}] Reward {:.2f}, LR Rewards {:.2f}" 
        self.log(msg.format(steps, reward, LR_reward))
        
    def log_trajectory_update(self, policy_loss, value_loss):

        self.metrics["e_Plosses"].append(policy_loss)
        self.metrics["e_Vlosses"].append(value_loss)


    # log at the end of each episode, number of steps in an episode
    def log_episode(self, ep, tot_episodes, reward, LR_reward, steps, time):
        self.metrics["mean_rewards"].append(reward)
        self.metrics["mean_LR_reward"].append(LR_reward)

        self.metrics["steps"].append(steps)
        msg = "episode [{:.0f}/{:.0f}] is collected. Mean Rewards {:.2f}, Mean LR Rewards {:.2f} over {:.0f} transitions, it took {:.2f} min" 
        self.log(msg.format(ep, tot_episodes, reward, LR_reward, steps, time/60))
    
    # log at the end of each episode and end of update (losses), number of update steps 
    def log_update(self, policy_loss, value_loss):
        self.metrics["mean_Plosses"].append(policy_loss)
        self.metrics["mean_Vlosses"].append(value_loss)

        msg = "Update steps / Mean Policy and Value Losses {:.2f}, {:.4f}" 
        self.log(msg.format(policy_loss, value_loss))
        

    def log_time(self, time):
        self.metrics["times"].append(time)
        self.log("Episode time {:.2f}".format(time))

    def log_stats(self, stats):
        reward_stats, info_stats = stats
        self.metrics["reward_stats"].append(reward_stats)
        self.metrics["info_stats"].append(info_stats)
        for key in reward_stats:
            reward_stats[key] = "{:.2f}".format(reward_stats[key])
        for key in info_stats:
            info_stats[key] = "{:.2f}".format(info_stats[key])
        self.log("Reward stats:\n {}".format(pprint.pformat(reward_stats)))
        self.log("Information gain stats:\n {}".format(pprint.pformat(info_stats)))

    def save(self):
        self._save_json(self.metrics_path, self.metrics)
        self.log("Saved metrics")
    
    def log_models(self, actor_state_dict, critic_state_dict):
        self.m_metrics["actor"].append(actor_state_dict)
        self.m_metrics["critic"].append(critic_state_dict)
     
    def save_m(self):
        self._save_json(self.saved_models_paths, self.saved_models)
        self.log("Saved _models_")

    def _init_print(self):
        f = open(self.print_path, "w")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f.write(current_time)
        f.close()

    def _setup_metrics(self):
        self.m_metrics = {
            "actor":[],
            "critic":[]
        }
        self.metrics = {
            # record losses - mean episode and the entire sequence in an episode
            "e_Plosses": [],
            "e_Vlosses": [],
            "mean_Plosses": [],
            "mean_Vlosses": [],
            
            # record rewards - mean reward in an episode (mean over until done iters)
            "e_rewards": [],
            "e_rewardsLR": [],
            "mean_rewards": [],
            "mean_LR_reward": [],

            # others
            "steps": [], # until done iters in an episode
            "times": [],
            "reward_stats": [],
            "info_stats": [],

            # record objects in trajectory
            "t_rewards":[],
            "t_rewardsLR":[],
            "t_states":[],
            "t_Xa":[],
            "t_outcome":[],
            "t_actions":[],

            # covariates info, perhaps too many
            # think of getting mean/sd per group 
            "Xa":[],
            "Xs":[], 
            "state":[]
        }

    def _save_json(self, path, obj):
        with open(path, "w") as file:
            json.dump(obj, file)
    

    def save_csv(self):
        self.export_to_csv(self.path_csv, self.metrics_list)
        self.log("Saved metrics in csv")

    def export_to_csv(self, path, obj):
        with open(path, 'a', newline='') as fp:
            a= csv.writer(fp, delimiter=';')
            data = obj
            a.writerows(data)

    def _setup_list(self):
        self.metrics_list = {
            "episode":[],
            "total_episodes":[],
            "reward":[],
            "lr_reward":[],
            "steps":[]
        }
   
    
    def log_episode_list(self, ep, tot_episodes, reward, LR_reward, steps):
        self.metrics_list["episode"].append(ep)
        self.metrics_list["total_episodes"].append(tot_episodes)
        self.metrics_list["reward"].append(reward)
        self.metrics_list["lr_reward"].append(LR_reward)
        self.metrics_list["steps"].append(steps)
    

# write a list in train and use export_data
