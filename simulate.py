import os

from model import *
from environment_game import Game
from ppo import PPO
from constants import *
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/ModelFree/utilities')
from environment import Env

env = Game()
n_input = env.state_dim

actor = Actor(n_input, N_HIDDEN)
critic = Critic(n_input, N_HIDDEN)

# retirieve previous saved model if exists
if os.path.exists(ACTOR_SAVE_PATH):
    print("Loading saved actor model...")
    actor.load_state_dict(torch.load(ACTOR_SAVE_PATH))
if os.path.exists(CRITIC_SAVE_PATH):
    print("Loading saved critic model...")
    critic.load_state_dict(torch.load(CRITIC_SAVE_PATH))

agent = PPO(env, actor, critic)
S = env.reset()
while True:
    A = agent.select_best_action(S)
    S_p, R, is_done = env.take_one_step(A.item())
    S = S_p

    if is_done:
        break