from ast import arg
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
import pandas as pd
from torch.distributions import MultivariateNormal, Categorical, Dirichlet
import sys
from wrapper import BasicWrapper
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

env = BasicWrapper()
state_dim = env.observation_size
action_dim = env.action_size
size = env._size

patients, S = env.reset() # S tensor
A = env.sample_random_action()
S_prime, R, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, is_done = env.step(A, S.detach().numpy())
patients = patients[:,1:3]
start_state = S
Xa_initial = patients[:, 1]
Xs_initial = patients[:, 0]

df = pd.DataFrame(data={'Xa_reset':  Xa_initial, 'Xa_post':  Xa_post, 'states': start_state})
df = (df.assign(risk= lambda x: pd.cut(df['states'], 
                                            bins=[0, 0.4, 0.8, 1],
                                            labels=["L", "M", "H"])))
levels = ["L", "M", "H"]
count_groups = df.groupby(['risk']).size().reset_index(name='counts')
indexesH = df['risk'].loc[df['risk'] == 'H'].index.tolist()
indexesM = df['risk'].loc[df['risk'] == 'M'].index.tolist()
indexesL = df['risk'].loc[df['risk'] == 'L'].index.tolist()
dfL = df.loc[indexesL]
dfM = df.loc[indexesM]
dfH = df.loc[indexesH]


def intervention(Xa, rho, which=1, add_noise = False, rho_bar=0.25, l = 0.8):
    # Xa is Xa_e(0))
    # rho is rho_e-1(Xs_e(0), Xa_e(0))
    if which ==1 :
        g = ((Xa) + 0.5*(Xa+np.sqrt(1+(Xa)**2)))*(1-rho) + ((Xa) - 0.5*(Xa+np.sqrt(1+(Xa)**2)))*rho
    elif which ==2:
        g = ((Xa) + 0.5*(Xa+np.sqrt(1+(Xa)**2)))*(1-rho**2) + ((Xa) - 0.5*(Xa+np.sqrt(1+(Xa)**2)))*(rho**2)
    elif which ==3:
        g = 0.5*((3-2*rho)*Xa+(1-2*rho)*(np.sqrt(1+(Xa)**2)))
    elif which ==4:
        pre_g = ((Xa) + 0.5*(Xa+np.sqrt(1+(Xa)**2)))*(1-rho**2) + ((Xa) - 0.5*(Xa+np.sqrt(1+(Xa)**2)))*(rho**2)
        g = (1-(rho_bar**l))*pre_g + (rho_bar**l)*Xa
    noise = np.random.normal(0, 1, 1)
    if add_noise == True:
        return g + noise
    else: return g    

def set_outcome(data, rho, Xa_post): # data is pat_e1
    df = pd.DataFrame(data={'Xs': data[:, 1], 'Xa': Xa_post})
    df = df.assign(rho = rho, Xa_post = Xa_post)
    df = (df.assign(risk= lambda x: pd.cut(df['rho'], 
                                                bins=[0, 0.25, 0.5, 1],
                                                labels=["L", "M", "H"])))
                                                
    quartile_post_80 = np.percentile(df.Xa_post + df.Xs, 80)
    Y_array = np.zeros(size)
        
    for i in range(len(Y_array)):
        if df.risk[i] == "H":
            Y_array[i] = 1
        if (df.Xa_post[i]+ df.Xs[i]) >= quartile_post_80:
          Y_array[i] = 1

      
        else: Y_array[i] = Y_array[i]  
    
    return Y_array


# directory to save plots
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'plots_data/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

def plot_interv(name, directory):
    fig = plt.figure()
    sample_file_name = name+  ".png" 
    x_axis = np.sort(np.random.uniform(-4, 4, 2000))
    rho_values = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    for i in range(0, len(rho_values)):
        plt.plot(x_axis, intervention(x_axis, rho_values[i], 1, add_noise = False), label=rho_values[i])


    plt.legend(title=r'$\rho$')
    plt.xlabel("pre-intervention Xa")
    plt.ylabel("post-intervention Xa")
    plt.title(r'$g_1$')
    plt.grid(True)
    plt.savefig(directory + sample_file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
#plot_interv("rho", results_dir)


def interventions(name, directory):  
    fig = plt.figure()  
    x_axis = np.sort(np.random.uniform(-4, 4, 2000))
    rho_values = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    #plt.show()
    file_name_multi = name +  ".png" 
    fig, axs = plt.subplots(2, 2)

    for i in range(0, len(rho_values)):
        axs[0, 0].plot(x_axis, intervention(x_axis, rho_values[i], 1, add_noise = False), label=rho_values[i])
        axs[0, 1].plot(x_axis, intervention(x_axis, rho_values[i], 2, add_noise = False), label=rho_values[i])
        axs[1, 0].plot(x_axis, intervention(x_axis, rho_values[i], 3, add_noise = False), label=rho_values[i])
        axs[1, 1].plot(x_axis, intervention(x_axis, rho_values[i], 4, add_noise = False), label=rho_values[i])
    axs[0, 0].set_title(r'$g_1$')
    axs[0, 1].set_title(r'$g_2$')
    axs[1, 0].set_title(r'$g_3$')
    axs[1, 1].set_title(r'$g_4$')

    for ax in axs.flat:
        ax.set(xlabel=r'$X_a$ pre', ylabel=r'$X_a$ post')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.savefig(directory + file_name_multi, dpi=300, bbox_inches='tight')
    plt.close(fig)
#interventions("rho_multi", results_dir)



#------- plot pre post intervention Xa

def plot_datashift(patients0, intervened, name, directory):   
    # intervened is Xa_post
    fig = plt.figure() 
    file_name_multiplot = name +  ".png" 
    covariates_label = np.array(["Xa_start", "Xa_post"])
    df = pd.DataFrame(data={'Xa_reset':  np.sort(patients0), 'Xa_post':  np.sort(intervened)})

    median_reset = np.median(patients0)
    median_post = np.median(intervened)

    for i in range(0, df.shape[1]):
        # Draw the density plot
        sns.kdeplot(df.iloc[:, i], fill = True,
        
                    label=covariates_label[i])
        
    plt.axvline(median_reset, linestyle = '--')
    plt.axvline(median_post, linestyle = '--', color='g' )
    # Plot formatting

    plt.legend(title = 'Covariates')
    plt.title('Covariate after 1 intervention') 
    plt.xlabel(r'$X_a$')
    plt.ylabel('Density')
    plt.grid(True)
    fig.savefig(directory + file_name_multiplot, dpi=300, bbox_inches='tight')
    plt.close(fig)



#plot_datashift(patients[:, 1], Xa_post, "dataset_shift", results_dir)

'''
plot_datashift(dfM[:, 0], dfM[:, 1], "dataset_shiftM", results_dir)
plot_datashift(dfH[:, 0], dfH[:, 1], "dataset_shiftH", results_dir)
'''

def plot_datashifts(pre, post, name, title, directory):   
    # intervened is Xa_post
    fig = plt.figure() 
    file_name_multiplot = name +  ".png" 
    covariates_label = np.array(["Xa_reset", "Xa_post"])
    df = pd.DataFrame(data={'Xa_reset':  np.sort(pre), 'Xa_post':  np.sort(post)})
    

    for i in covariates_label:
        # Draw the density plot
        
        sns.kdeplot(df[i], fill = True,
        
                    label=i)
        
    # Plot formatting

    plt.legend(title = 'Covariates')
    plt.title('Covariate shift group' + name) 
    plt.xlabel(r'$X_a$')
    plt.ylabel('Density')
    plt.grid(True)
    fig.savefig(directory + file_name_multiplot, dpi=300, bbox_inches='tight')
    plt.close(fig)


#plot_datashifts(dfM['Xa_reset'], dfM['Xa_post'], "dataset_shift_M", "M", results_dir)
#plot_datashifts(dfL['Xa_reset'], dfL['Xa_post'], "dataset_shift_L", "L", results_dir)
#plot_datashifts(dfH['Xa_reset'], dfH['Xa_post'], "dataset_shift_H", "H", results_dir)

