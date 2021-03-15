#%%  
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
from scipy.special import logit
import pandas as pd
from matplotlib.axes import Axes, Subplot
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm



#%% 

def reoccuring_drift(length=50000,width=10,rate=0.1,plot=True,filename="reoccuring_drift.pdf"):
    length = length / 2
    probability_drift = np.array([])
    time = np.array([])

    part_length = rate*length
    for part in range(int(length/part_length)):
        t = np.arange(time.size, time.size+part_length, 1)
        x = np.array([1.0 / (1.0 + np.exp(-4.0 * float(i - int(time.size+part_length-part_length/2)) / float(width))) for i in t])
        y = np.array([1 - p for p in x])
        probability_drift = np.append(probability_drift,x)
        probability_drift = np.append(probability_drift,y)
        time = np.append(time,t)

    probability_drift = (probability_drift-.5)*2
    
    t = np.arange(1, probability_drift.size+1, 1)
    

    signal = probability_drift
    
    pos_signal = signal.copy()
    neg_signal = signal.copy()

    pos_signal[pos_signal <= 0] = np.nan
    neg_signal[neg_signal > 0] = np.nan


    plot_attributes(pos_signal,neg_signal,filename=filename,plot=plot)


def incremental_drift(length=50000,width=10000,plot=True,filename="incremental_drift.pdf"):
    probability_drift = np.array([])
    time = np.array([])

    t = np.arange(time.size, length, 1)
    x = np.array([1.0 / (1.0 + np.exp(-4.0 * float(i - int(length/2)) / float(width))) for i in t])
    probability_drift = np.append(probability_drift,x)
    # probability_drift = np.append(probability_drift,y)
    time = np.append(time,t)

    probability_drift = (probability_drift-.5)*2

    t = np.arange(1, probability_drift.size+1, 1)

    signal = probability_drift
    pos_signal = signal.copy()
    neg_signal = signal.copy()

    pos_signal[pos_signal <= 0] = np.nan
    neg_signal[neg_signal > 0] = np.nan

    plot_attributes(pos_signal,neg_signal,filename=filename,plot=plot)


def gradual_drift(length=50000,width=10,rate=0.4,plot=True,filename="gradual_drift.pdf"):
    length = length / 2
    probability_drift = np.array([])
    time = np.array([])

    part_length = rate*length
    for part in range(int(length/part_length)):

        t = np.arange(time.size, time.size+part_length, 1)
        x = np.array([1.0 / (1.0 + np.exp(-4.0 * float(i - int(time.size+part_length-part_length/2)) / float(width))) for i in t])
        y = np.array([1 - p for p in x])

        if 0 == part:
            probability_drift = np.append(probability_drift,np.zeros(10000))
        if int(length/part_length)-1 == part:
            probability_drift = np.append(probability_drift,x)   
            probability_drift = np.append(probability_drift,np.ones(10000))
        else:
            probability_drift = np.append(probability_drift,x)
            probability_drift = np.append(probability_drift,y)

        time = np.append(time,t)

    probability_drift = (probability_drift-.5)*2
    t = np.arange(1, probability_drift.size+1, 1)

    signal = probability_drift
    pos_signal = signal.copy()
    neg_signal = signal.copy()

    pos_signal[pos_signal <= 0] = np.nan
    neg_signal[neg_signal > 0] = np.nan

    plot_attributes(pos_signal,neg_signal,filename=filename,plot=plot)

def plot_attributes(pos_signal,neg_signal,filename,plot):
    fig, ax = plt.subplots()
    # plt.figure(0)
    # plt.set_size_inces(6.4,4.8)
    fig.set_size_inches(6.4, 4.8)
    #plotting
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Data Mean')
    plt.style.use('seaborn-paper')
    
    SMALL = 18
    SIZE = 22
    plt.rc('font', size=SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL)  # legend fontsize
    plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title
    plt.rc('lines', lw=4)

    plt.xticks(np.arange(0, 45000, step=20000))
    plt.plot(neg_signal,label="Concept 1 (C1)")
    plt.plot(pos_signal,label="Concept 2 (C2)", linestyle='dotted')
    plt.legend(bbox_to_anchor=(-0.2,1.02,1.2,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2,frameon=False)
    plt.yticks([-1,1.0],["C1","C2"],rotation='vertical')
    ticks = ax.yaxis.get_majorticklabels()
    ticks[0].set_verticalalignment("center")
    ticks[1].set_verticalalignment("center")

    plt.show() if plot else ""
    fig.savefig(filename,dpi=400, transparent=True,bbox_inches='tight',pad_inches = 0.05)
    # ax1 = ax.twinx()
    # plt.yticks([-1,0,1],["","",""],rotation='vertical')


reoccuring_drift(width=600,filename="frequent_reoccuing_drift.pdf") # Frequent Reoccurring
reoccuring_drift(length=60000,width=2000,rate=0.4) # Reoccurring
incremental_drift(width=15000) # Incremental
incremental_drift(width=2500,filename="abrupt_drift.pdf") # Abrupt
gradual_drift(length=45000,width=1000,rate=0.3) #Gradual

# %%
