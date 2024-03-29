import sys
import time
import pathlib
import argparse
import numpy as np
import torch
import os
import pandas as pd
from replay_memory import *
from logger.logger import Logger
import json
import seaborn as sns
import matplotlib.pyplot as plt

# directory to save plots
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'plots_/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


# Plot or not
plot = True

# load metrics files
# different json files for each train type

# without GP
f1 = open('log_train_1_0\metrics.json')
data_log = json.load(f1)
for key, value in data_log.items() :
    print (key)

# Closing file
f1.close()

# with GP
fgp = open('log_train_2_0\metrics.json')
data_log = json.load(fgp)
for key, value in data_log.items() :
    print (key)

# Closing file
fgp.close()