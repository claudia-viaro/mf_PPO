import os
import json
from datetime import datetime
import pprint


class Logger(object):
    def __init__(self, logdir, seed):
        self.logdir = logdir
        self.seed = seed
        self.path = "log_" + logdir + "_" + str(seed) + "/"
        self.print_path = self.path + "out.txt"
        self.metrics_path = self.path + "metrics.json"
        self.saved_models_paths = self.path + "models.json"

        os.makedirs(self.path, exist_ok=True)
        self.metrics = {}
        self.saved_models= {}
        self._init_print()
        self._setup_metrics()

    def log(self, string):
        f = open(self.print_path, "a")
        f.write("\n")
        f.write(str(string))
        f.close()
        print(string)

  

    # logging trajectory info, all transitions
    def log_trajectory(self, reward, LR_reward, critic_loss, actor_loss):
        self.metrics["e_rewards"].extend(reward)
        self.metrics["e_rewardsLR"].extend(LR_reward)

        self.metrics["e_Closses"].extend(critic_loss)
        self.metrics["e_Alosses"].extend(actor_loss)

    # log at the end of each episode, number of steps in an episode
    def log_episode(self, reward, LR_reward, steps):
        self.metrics["mean_rewards"].append(reward)
        self.metrics["mean_LR_reward"].append(LR_reward)

        self.metrics["steps"].append(steps)
        msg = "Steps {:.2f} / Mean Rewards {:.2f}, Mean LR Rewards {:.2f}" 
        self.log(msg.format(steps, reward, LR_reward))
    
    # log at the end of each episode and end of update (losses), number of update steps 
    def log_update(self, policy_loss, value_loss):
        self.metrics["mean_Plosses"].append(policy_loss)
        self.metrics["mean_Vlosses"].append(value_loss)

        msg = "Update steps / Mean Policy and Value Losses {:.2f}, {:.2f}" 
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
        self.log("Saved _metrics_")



    def _init_print(self):
        f = open(self.print_path, "w")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f.write(current_time)
        f.close()

    def _setup_metrics(self):
        self.metrics = {
            # record losses - mean episode and the entire sequence in an episode
            "e_Closses": [],
            "e_Alosses": [],
            "mean_Closses": [],
            "mean_Alosses": [],
            
            # record rewards - mean episode and the entire sequence in an episode
            "e_rewards": [],
            "e_rewardsLR": [],
            "mean_rewards": [],
            "mean_rewardsLR": [],

            # others
            "steps": [],
            "times": [],
            "reward_stats": [],
            "info_stats": [],

            # covariates info, perhaps too many
            # think of getting mean/sd per group 
            "Xa":[],
            "Xs":[], 
            "state":[]
        }

    def _save_json(self, path, obj):
        with open(path, "w") as file:
            json.dump(obj, file)

          