# discount rate
GAMMA = 0.995

# for clipping ratios
EPSILON = 0.1

# lambda constant for GAE
TAU = 0.97

# number of episodes to train
N_EPISODES = 30  #300

# number of frames to store in memory
BATCH_SIZE = 500 # 5000

# number of hidden units in models of actor & critic
N_HIDDEN = 64

# learning rate for adam optimizer
A_LEARNING_RATE = 0.001
C_LEARNING_RATE = 0.001

# interval of steps after which statistics should be printed
LOG_STEPS = 10

# interval of steps after which models should be saved
SAVE_STEPS = 20

# path to save actor model
ACTOR_SAVE_PATH = "PPO_2/saved_models/actor_ppo.pth"

# path to sace critic model
CRITIC_SAVE_PATH = "PPO_2/saved_models/critic_ppo.pth"

import argparse

def get_args():
    """
    Utility for getting the arguments from the user for running the experiment
    :return: parsed arguments
    """

    # Env
    parser = argparse.ArgumentParser(description='collect arguments')


    
    parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
    parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon")
    parser.add_argument('--c-learning-rate', type=float, default=1e-3, help="critic learning rate")
    parser.add_argument('--a-learning-rate', type=float, default=1e-3, help="critic learning rate")
    parser.add_argument('--tau', type=float, default=0.97, help='soft update rule for target netwrok(default: 0.001)')

    parser.add_argument('--n-episodes', type=int, default=30, help="number of episodes to train")
    parser.add_argument('--batch-size', type=int, default=100, help="number of hidden units in models of actor & critic")
    parser.add_argument('--log-steps', type=int, default=10, help="interval of steps after which statistics should be printed")
    parser.add_argument('--save-steps', type=int, default=20, help="interval of steps after which models should be saved")
    parser.add_argument('--n-hidden', type=int, default=64, help="number of hidden units in models of actor & critic")
    
    parser.add_argument('--actor-save-path', type=str, default="saved/actor_ppo.pth", help="path to save actor model")
    parser.add_argument('--critic-save-path', type=str, default="saved/critic_ppo.pth", help="path to save critic model")


    parser.add_argument('--checkpoint-interval', type=int, default=1e5, help="when to save the models") #1e5
    parser.add_argument('--out', type=str, default='/tmp/ppo/models/')
    #parser.add_argument('--log-dir', type=str, default="/tmp/ppo/logs/")
    parser.add_argument('--reset-dir', action='store_true', help="give this argument to delete the existing logs for the current set of parameters")

    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--config_name", type=str, default="model_update")
    parser.add_argument("--strategy", type=str, default="information")
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument('--buffer-capacity', type=int, default=8) # 1000000 64
    parser.add_argument('-S', '--seed-episodes', type=int, default=5)
    parser.add_argument('--all-episodes', type=int, default=7) # 1000
    parser.add_argument('--action-noise-var', type=float, default=0.3)
    parser.add_argument('-C', '--collect-interval', type=int, default=2) # 100
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')


    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')    
    args = parser.parse_args()

    return args    