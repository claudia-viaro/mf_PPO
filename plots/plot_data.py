from ast import arg
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
import pandas as pd
from torch.distributions import MultivariateNormal, Categorical, Dirichlet
import sys
from wrapper import BasicWrapper
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()




def plot_datashift(patients0, Xa_post, time):   

    covariates_label = np.array(["Xs", "Xa_start", "Xa_post"])
    df = pd.DataFrame(data={'Xs':  np.sort(patients0[:, 1]), 'Xa_reset':  np.sort(patients0[:, 2]), 'Xa_post':  np.sort(Xa_post)})

    median_reset = np.median(patients0[:, 2])
    median_post = np.median(Xa_post)

    for i in range(0, df.shape[1]):
        # Draw the density plot
        sns.kdeplot(df.iloc[:, i], fill = True,
        
                    label=covariates_label[i])
        
    plt.axvline(median_reset, linestyle = '--')
    plt.axvline(median_post, linestyle = '--', color='g' )
    # Plot formatting

    plt.legend(title = 'Covariates')
    plt.title('Covariate shift after ' +str(time)+' transitions') 
    plt.xlabel('Density Plot of Patients Covariates')
    plt.ylabel('Density')

    return plt.show()   



