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




f = open('log_train_1_0\metrics.json')
data_log = json.load(f)
for key, value in data_log.items() :
    print (key)




'''
# extract objects and turn them into dataframes for plotting purposes
# 5 epochs, each has 300 transitions

# Xa, Xs
# for each episode we have n transitions. at each transitins, covariates have length of the population


# return dataset for one epoch, 31x1, where the colummn is a list of 200 values
def get_1epoch(json_load, t):
    df_obj = pd.DataFrame(data = {'variable': json_load[t]} ) 
    df_Xa2 = pd.DataFrame(df_obj)
    return df_Xa2

Xa_e0 = get_1epoch(json_load = data_log['Xa'], t=0) 
state_e0 = get_1epoch(json_load = data_log['state'], t=0) # epoch 0   
Xa_e25 = get_1epoch(json_load=data_log['Xa'], t= 24)
state_e25 = get_1epoch(json_load = data_log['state'], t=24)
Xa_e75 = get_1epoch(json_load=data_log['Xa'], t= 24)
state_e75 = get_1epoch(json_load = data_log['state'], t=74)
Xa_e100 = get_1epoch(json_load = data_log['Xa'], t=len(data_log['Xa'])-1) 
state_e100 = get_1epoch(json_load = data_log['state'], t=len(data_log['Xa'])-1)
'''
plot states in the form of histogram, distinguish risk levels

# dataset is state_e0, t is which step in the trajectory, e is epoch (assume 0)
'''






df_Xs = pd.DataFrame(data_log['Xs'])
df_Xs = df_Xs.T
#print("df_Xs", df_Xs)



# MEAN reward, reward log reg (5, 300)
# for each episode we have n transitions. reward is the mean reward for each epoch, across all transitions

df_MeanR = pd.DataFrame(data = {'M_reward': data_log['mean_rewards'], 'M_rewardLR': data_log['mean_LR_reward']})

# log reg reward, reward
df_RLogReg = pd.DataFrame(data_log['reward_logreg'])
df_RLogReg = df_RLogReg.T
df_R = pd.DataFrame(data_log['reward'])
df_R = df_R.T

figure, ax = plt.subplots(figsize=(4,5))
sample_file_name = "subplots_"+  ".png" 
# Data Coordinates

x = np.linspace(-3, 3, 200)
y=Xa_e0['variable'][0]
sns.kdeplot(np.sort(y), fill = True, color='blue', label = "Xa_pre")
plt.ion()
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")

for value in range(len(Xa_e0['variable'])):
    update_y_value = np.sort(Xa_e0['variable'][value])
    
    sns.kdeplot(update_y_value, fill = True, color='gray')
    
    figure.canvas.draw()
    figure.canvas.flush_events()
    time.sleep(1)


# Display
#plt.savefig(results_dir + sample_file_name, dpi=300, bbox_inches='tight')
#plt.show()



from matplotlib.animation import FuncAnimation
from celluloid import Camera

'''
# create figure object
fig = plt.figure()
# load axis box
ax = plt.axes()
# set axis limit
#ax.set_ylim(0, 1)
ax.set_xlim(-5, 5)
sns.kdeplot(Xa_e0['variable'][0], fill = True, color='blue')
camera = Camera(fig)
for i in range(len(Xa_e0['variable'])):
    update_y_value = np.sort(Xa_e0['variable'][i])
    
    sns.kdeplot(update_y_value, fill = True, color='gray')
    
    plt.pause(0.1)
    camera.snap()

animation = camera.animate()
animation.save('distribution_Xa.gif', writer='Pillow', fps=2)
'''

iterate_over = np.linspace(2, len(Xa_e0['variable']), 10)

if plot == True:
    simple_plot(df_MeanR['M_reward'], df_MeanR['M_rewardLR'], title = 'Mean Reward', axis = 'Epochs', dir=results_dir, file_name="Mean_Reward")
    simple_plot(df_R[0], df_RLogReg[0], title = 'Episodic Rewards', axis = 'Transitions', dir=results_dir, file_name="Reward")
    plot_distr(Xa_e0, data_log['Xs'], e=0, dir=results_dir, file_name="Distr1")
    plot_distr(Xa_e25, data_log['Xs'], e=25, dir=results_dir, file_name="Distr2")
    plot_distr(Xa_e75, data_log['Xs'], e=75, dir=results_dir, file_name="Distr3")
    plot_distr(Xa_e100, data_log['Xs'], e=100, dir=results_dir, file_name="Distr4")
    plot_StateHist(state_e0,  len(state_e0)-1, dir=results_dir, file_name="Histogram")


'''
# Closing file
#f.close()
