import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from constants import *
import csv
import os

#--------------------------------------------------------------------------------------------------------------------------------------------------------
# other functions

def export_to_csv(output_csv_file, data_list):

	with open(output_csv_file, 'a', newline='') as fp:
		a = csv.writer(fp, delimiter=';')
		data = [data_list]
		a.writerows(data)

def get_counts(state):
    
    df = pd.DataFrame(data={'states': state})
    #print(df)

    df = (df.assign(risk= lambda x: pd.cut(df['states'], 
                                                bins=[0, 0.4, 0.8, 1],
                                                labels=["L", "M", "H"])))

    count_groups = df.groupby(['risk']).size().reset_index(name='counts')
    df['Xa'] = np.nan
    for i in range(0, df.shape[0]):
        if df['risk'][i] == "L":
            df['Xa'][i] = np.random.normal(0,1, size=(1,1))
        elif df['risk'][i] == "M":    
            df['Xa'][i] = np.random.normal(3,1, size=(1,1))
        else:  df['Xa'][i] = np.random.normal(6,1, size=(1,1))    

    
    return count_groups, df['Xa'], df

#--------------------------------------------------------------------------------------------------------------------------------------------------------
# directory to save plots
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'plots/')


if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


#--------------------------------------------------------------------------------------------------------------------------------------------------------
# plots

def plot_Xa_risk(states, name, episode, iter, directory = results_dir):
    count_states, df_Xa, df = get_counts(states)
    sns.set(style="darkgrid")
    figure, axes = plt.subplots(1, 2, sharex=False, figsize=(10,5))
    figure.suptitle('Episode' + str(episode) + '.iter' + str(iter))
    axes[0].set_title('Density plot Xa')
    axes[1].set_title('Histogram risk - rho')
    sample_file_name = name+ str(episode) + "."+ str(iter) + ".png" 
    #sample_file_name = name+ str(episode) + "." + str(iter) + ".png" 
    sns.kdeplot(ax=axes[0], data=df, x="Xa", hue="risk", label = "risk")
    sns.histplot(ax=axes[1], data= df, x="states", hue="risk", kde=False, label = "risk")
    plt.grid(True)
    plt.savefig(directory + sample_file_name, dpi=300, bbox_inches='tight')
    #plt.show()
    #plt.close(fig)

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


def plot_classification(state, Y, patients_df, episode, iter, name, directory = results_dir):
    # use values from env.step
    print("Y shape", Y.shape)
    df = pd.DataFrame(data={'states': state, 'Xa': patients_df[:, 2], 'Y': Y.squeeze(), 'Xs':patients_df[:, 1]})
    df = (df.assign(risk= lambda x: pd.cut(df['states'], 
                                                    bins=[0, 0.4, 0.8, 1],
                                                    labels=["L", "M", "H"])))
    fig = plt.figure()
    sns.scatterplot(data=df, x="Xa", y="Xs", hue="risk", style='Y', markers=['o', 's'], s=10)
    
    sample_file_name = name + str(episode) + "." + str(iter) + ".png" 
    plt.title(name + str(episode) + "." + str(iter))
    plt.grid(True)
    plt.savefig(directory + sample_file_name, dpi=300, bbox_inches='tight')

def plot_classification_1(state, Y, patients_df, episode, iter, name, directory = results_dir):
    # use values from env.step
    df = pd.DataFrame(data={'states': state, 'Xa': patients_df[:, 2], 'Y': Y, 'Xs':patients_df[:, 1]})
    df = (df.assign(risk= lambda x: pd.cut(df['states'], 
                                                    bins=[0, 0.4, 0.8, 1],
                                                    labels=["L", "M", "H"])))
    fig = plt.figure()
    colormap = {"L": "purple", "M": "orange", "H": "green"}

    sns.jointplot(
        data=df,
        x="Xa",
        y="Xs",
        hue="risk",
        palette=colormap,
        ec=df["risk"].map(colormap),
        fc="none",
        s=10,

    )

    sample_file_name = name + str(episode) + "." + str(iter) + ".png" 
    plt.title(name + str(episode) + "." + str(iter))
    plt.grid(True)
    plt.savefig(directory + sample_file_name, dpi=300, bbox_inches='tight')

#--------------------------------------------------------------------------------------------------------------------------------------------------------

def get_count(state):

    df = pd.DataFrame(data={'states': state})
    #print(df)

    df = (df.assign(risk= lambda x: pd.cut(df['states'], 
                                                bins=[0, 0.4, 0.8, 1],
                                                labels=["L", "M", "H"])))


    count_groups = df.groupby(['risk']).size().reset_index(name='counts')
    return count_groups      


def get_gaussian_log(x, mu, log_stddev):
    '''
    returns log probability of picking x
    from a gaussian distribution N(mu, stddev)
    '''
    # ignore constant since it will be cancelled while taking ratios
    log_prob = -log_stddev - (x - mu)**2 / (2 * torch.exp(log_stddev)**2)
    return log_prob

def plot_data(statistics):
    '''
    plots reward and loss graph for entire training
    '''
    x_axis = np.linspace(0, N_EPISODES, N_EPISODES // LOG_STEPS)
    plt.plot(x_axis, statistics["reward"])
    plt.title("Variation of mean rewards")
    plt.show()

    plt.plot(x_axis, statistics["val_loss"])
    plt.title("Variation of Critic Loss")
    plt.show()

    plt.plot(x_axis, statistics["policy_loss"])
    plt.title("Variation of Actor loss")
    plt.show()

def plot1(statistics, results_dir, sample_file_name):
    sns.set(style="darkgrid")
    x_axis = np.linspace(0, N_EPISODES, N_EPISODES // LOG_STEPS)
    f, axs = plt.subplots(1,3,
                      figsize=(9,5),
                      sharex=True)
    f.tight_layout()

    sns.lineplot(x=x_axis, y=statistics["reward"],  ax=axs[0])
    sns.lineplot(x=x_axis, y=statistics["val_loss"], ax=axs[1])
    sns.lineplot( x=x_axis, y=statistics["policy_loss"],  ax=axs[2])

 
    axs[0].set_title('Variation of mean rewards')
    axs[1].set_title('Variation of Critic Loss')
    axs[2].set_title('Variation of Actor loss')

    
    plt.savefig(results_dir + sample_file_name, dpi=300, bbox_inches='tight')    
    


def plot2(statistics, results_dir, sample_file_name):
    x_axis = np.linspace(0, N_EPISODES, num=N_EPISODES)
    fig, (ax1,ax2, ax3) = plt.subplots(nrows=3, sharex=True) # frameon=False removes frames

    plt.subplots_adjust(hspace=.25)
    ax1.grid()
    ax2.grid()
    ax3.grid()

    #ax1.xaxis.set_tick_params(labelbottom=True)
    ax1.set_title('Variation of mean rewards', fontsize=9)
    ax2.set_title('Variation of Critic Loss', fontsize=9)
    ax3.set_title('Variation of Actor loss', fontsize=9)

    ax1.plot(x_axis, statistics["reward"], color='r')
    ax2.plot(x_axis, statistics["val_loss"], color='b')
    ax3.plot(x_axis, statistics["policy_loss"], color='g')
    plt.savefig(results_dir + sample_file_name, dpi=300, bbox_inches='tight')  

# save the best model


plt.style.use('ggplot')
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/best_model.pth')

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/final_model.pth')


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')