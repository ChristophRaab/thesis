
import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../"))
import pandas as pd
import numpy as np
import glob
from scipy.interpolate import UnivariateSpline,InterpolatedUnivariateSpline
from numpy import linspace,exp
import matplotlib.cm as mpl_color_map
import matplotlib.pyplot as plt#

def filter_results(df,learning_rate = 0.001):
    first = df[df.iloc[:,1] == learning_rate]
    final = first.sort_values(by=[0]) # last one has no parameters
    return final

def plot_hyperparamters(paramters,name):
    # plt.gca().set_color_cycle(['red', 'blue'])
    xs = linspace(0, 20, 20)


    plt.figure()
    fig_acc = plt.gcf()
    axes = plt.gca()
    plots = []
    results = []
    for paras,n in zip(paramters,name):
        x = paras.iloc[:,0].values
        k = paras.iloc[:,-2].values if len(paras.columns)== 4 else paras.iloc[:,-1].values


        us1 = UnivariateSpline(x, k)
        y1 = us1(xs)
        results.append(y1)
        a, = plt.plot(xs,y1,'^-',linewidth = 2, markersize=5, label=n)
        plots.append(a)

    results = np.array(results)
    mean = np.mean(results,axis=0)
    best_k = round(xs[np.argmax(mean)])

    a = plt.axvline(x=best_k,label="Best mean acc. at k="+str(best_k), c = "grey")

    plt.legend(handles=plots+[a])
    plt.xlabel("k Value")
    plt.ylabel("Accuracy in %")

    fig_acc.savefig('plots/plot_hyperparmeter.png', dpi=400)
    fig_acc.savefig('plots/plot_hyperparmeter.pdf', dpi=400, bbox_inches = 'tight',pad_inches = 0.05)


if __name__ == '__main__':
    files = glob.glob("results/_optimize*.csv")

    parameter_results  = []
    task_name = ["P vs I","Amazon vs Webcam","Art vs Clipart","Dslr vs Amazon"]
    for f,name in zip(files,task_name):
        df = pd.read_csv(f,header=None,index_col=None,delimiter=",")
        parameter_results.append(filter_results(df))

    plot_hyperparamters(parameter_results,task_name)