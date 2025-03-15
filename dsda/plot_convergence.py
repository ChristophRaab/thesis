import os
import copy
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import sys
import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../"))
import torch
from dsda.data_list import ImageList
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dsda import pre_process as prep
from torch import nn
from sklearn.manifold import TSNE
from scipy.interpolate import UnivariateSpline,InterpolatedUnivariateSpline
from numpy import linspace,exp

def plot_a_distance(x,transfer,a_distance,model_names,name=None):


    # plt.gca().set_color_cycle(['red', 'blue'])
    xs = linspace(0, 74999, 50)


    plt.figure()
    fig_acc = plt.gcf()
    axes = plt.gca()
    plots = []
    us1 = UnivariateSpline(x, transfer[1])
    y1 = us1(xs)

    us2 = UnivariateSpline(x, a_distance[0])
    y2 = us2(xs)

    us3 = UnivariateSpline(x, a_distance[1])
    y3 = us3(xs)


    e, = plt.plot(x,transfer[1],'-',linewidth = 1,alpha=0.5, markersize=1,   label="RSL")
    f, = plt.plot(x,a_distance[0],'-',linewidth = 1,alpha=0.5, markersize=1, label="A-Distance BSP")
    g, = plt.plot(x,a_distance[1],'-',linewidth = 1, alpha=0.5,markersize=1, label="A-Distance ASAN")

    b, = plt.plot(xs,y1,'^-',linewidth = 2, markersize=5, label="Trend RSL")
    c, = plt.plot(xs,y2,'.-',linewidth = 2, markersize=5, label="Trend A-Dist. BSP")
    d, = plt.plot(xs,y3,'^-',linewidth = 2, markersize=5, label="Trend A-Dist. ASAN")
    plt.legend(handles=[e,b,f,c,g,d])
    plt.xlabel("No. Iterations")

    fig_acc.savefig('plots/plot_transfer.png', dpi=400)
    fig_acc.savefig('plots/plot_transfer.pdf', dpi=400, bbox_inches = 'tight',
    pad_inches = 0.05)

    # plt.show()

def plot_a_distance_fill(x,transfer,a_distance,model_names,name=None):

    # plt.gca().set_color_cycle(['red', 'blue'])
    sampling_size = 50
    xs = linspace(0, 74999, sampling_size)

    fill_size = int(len(transfer[0])/sampling_size)

    plt.figure()
    fig_acc = plt.gcf()
    axes = plt.gca()
    plots = []
    us1 = UnivariateSpline(x, transfer[1])
    y1 = us1(xs)

    us2 = UnivariateSpline(x, a_distance[0])
    y2 = us2(xs)

    us3 = UnivariateSpline(x, a_distance[1])
    y3 = us3(xs)

    b, = plt.plot(xs,y1,'^-',linewidth = 2, markersize=5,color="#d62728", label="Trend RSL")
    c, = plt.plot(xs,y2,'.-',linewidth = 2, markersize=5,color="#9467bd", label="Trend A-Dist. BSP")
    d, = plt.plot(xs,y3,'^-',linewidth = 2, markersize=5,color="#8c564b", label="Trend A-Dist. ASAN")

    transfer[1] = np.array(transfer[1])[0:-1:fill_size]
    a_distance[0] = np.array(a_distance[0])[0:-1:fill_size]
    a_distance[1] = np.array(a_distance[1])[0:-1:fill_size]

    e = plt.fill_between(xs,y1 - np.abs(y1 - transfer[1]), y1 + np.abs(y1 - transfer[1])   ,alpha=0.2 , label="Spread RSL")
    f = plt.fill_between(xs,y2 - np.abs(y2 - a_distance[0]), y2 + np.abs(y2 - a_distance[0]),alpha=0.2 ,label="Spread A-Dist. BSP")
    g = plt.fill_between(xs,y3 - np.abs(y3 - a_distance[1]), y3 + np.abs(y3 - a_distance[1]),alpha=0.2 ,label="Spread A-Dist. ASAN")

    plt.legend(handles=[e,b,f,c,g,d])
    plt.xlabel("No. Iterations")
    # plt.ylim(0,1.8)
    fig_acc.savefig('plots/plot_transfer.png', dpi=400)
    fig_acc.savefig('plots/plot_transfer.pdf', dpi=400, bbox_inches = 'tight',
    pad_inches = 0.05)

    # plt.show()

def plot_error(x,data,model_names):
    # plt.gca().set_color_cycle(['red', 'blue'])
    xs = linspace(0, 74999, 25)


    # us1 = UnivariateSpline(x, data[0])
    # y1 = us1(xs)

    # us2 = UnivariateSpline(x, data[1])
    # y2 = us2(xs)

    plt.figure()
    fig_acc = plt.gcf()
    axes = plt.gca()
    plots = []
    for name,d in zip(model_names,data):
        a, = plt.plot(x,d,'.-',linewidth = 1, markersize=5, label=name)
        plots.append(a)

    plt.legend(handles=plots)
    plt.xlabel("No. Iterations")

    fig_acc.savefig('plots/plot_target_acc.png', dpi=400)
    fig_acc.savefig('plots/plot_target_acc.pdf', dpi=400, bbox_inches = 'tight',
    pad_inches = 0.05)
    # plt.show()

def plot_train_test(x,train,test,model_names):
    # plt.gca().set_color_cycle(['red', 'blue'])
    xs = linspace(0, 74999, 25)


    us1 = UnivariateSpline(x, train[0])
    y1 = us1(xs)

    us2 = UnivariateSpline(x, train[1])
    y2 = us2(xs)

    plt.figure()
    fig_acc = plt.gcf()
    axes = plt.gca()
    plots = []

    a, = plt.plot(x,train[0],'-',linewidth = 1, markersize=1, alpha=0.5, label="ASAN w/o SN train loss.")
    b, = plt.plot(x,train[1],'-',linewidth = 1, markersize=1,alpha=0.5, label="ASAN train loss.")
    c, = plt.plot(x,test[0],'x-',linewidth = 1, markersize=3, label="ASAN w/o SN Test Acc.")
    d, = plt.plot(x,test[1],'.-',linewidth = 1, markersize=3, label="ASAN Test Acc.")
    e, = plt.plot(xs,y1,'x-',linewidth = 2, markersize=7,  label="Trend ASAN w/o SN train loss.")
    f, = plt.plot(xs,y2,'.-',linewidth = 2, markersize=7, label="Trend ASAN train loss.")

    plt.legend(handles=[a,b,c,d,e,f])
    plt.xlabel("No. Iterations")

    fig_acc.savefig('plots/plot_train_test.png', dpi=400)
    fig_acc.savefig('plots/plot_train_test.pdf', dpi=400, bbox_inches = 'tight',
    pad_inches = 0.05)
    # plt.show()

def plot_train_test_fill(x,train,test,model_names):
    # plt.gca().set_color_cycle(['red', 'blue'])
    sampling_size = 25

    xs = linspace(0, 74999, sampling_size)


    us1 = UnivariateSpline(x, train[0])
    y1 = us1(xs)

    us2 = UnivariateSpline(x, train[1])
    y2 = us2(xs)

    fill_size = int(len(train[0])/sampling_size)
    plt.figure()
    fig_acc = plt.gcf()
    axes = plt.gca()
    plots = []
    train[0] = np.array(train[0])[0:-1:fill_size]
    train[1] = np.array(train[1])[0:-1:fill_size]

    so = y1 - np.abs(y1 - train[0])
    st = y2 -  np.abs(y2 - train[1])

    so[so < 0] = 0
    st[st < 0] = 0
    c, = plt.plot(x,test[0],'x-',linewidth = 1, markersize=3, color="#2ca02c",label="ASAN w/o SN Test Acc.")
    d, = plt.plot(x,test[1],'.-',linewidth = 1, markersize=3, color="#d62728", label="ASAN Test Acc")
    a = plt.fill_between(xs,so,y1 + np.abs(y1 - train[0]),color="#1f77b4",alpha=0.2,label="Spread ASAN w/o SN train loss")
    b = plt.fill_between(xs,st,y2 + np.abs(y2 - train[1]),color="#ff7f0e",alpha=0.2,label="Spread ASAN train loss")
    e, = plt.plot(xs,y1,'x-',linewidth = 2, markersize=7,color="#9467bd", label="Trend ASAN w/o SN train loss")
    f, = plt.plot(xs,y2,'.-',linewidth = 2, markersize=7, color="#8c564b",label="Trend ASAN train loss")

    plt.legend(handles=[c,d,a,e,b,f])
    plt.xlabel("No. Iterations")
    fig_acc.savefig('plots/plot_train_test.png', dpi=400)
    fig_acc.savefig('plots/plot_train_test.pdf', dpi=400, bbox_inches = 'tight',
    pad_inches = 0.05)
    # plt.show()

def ad_net_prediction(predictions,features,ad_net,domain_label=0):
    softmax_out = nn.Softmax(dim=1)(predictions)
    op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
    ad_out = ad_net(op_out.view(-1, softmax_out.size(1) * features.size(1)))
    return ad_out

if __name__ == '__main__':
    # log_path = ["snapshot/san/AW_Trained_Models/_log_no_on_amazon_vs_webcam_CDAN.txt","snapshot/san/AW_Trained_Models/_log_no_on_amazon_vs_webcam_DANN.txt","snapshot/san/AW_Trained_Models/_log_BSP_on_amazon_vs_webcam.txt","snapshot/san/AW_Trained_Models/_log_ASAN_on_amazon_vs_webcam.txt","snapshot/san/AW_Trained_Models/_log_ASAN+E_on_amazon_vs_webcam.txt"]
    log_path = ["snapshot/san/ImageClef_PI_Trained_Models/log_no_da_on_image-clef_vs_image-clef_with_False_11_0.0001_CDAN.txt","snapshot/san/ImageClef_PI_Trained_Models/log_no_da_on_image-clef_vs_image-clef_with_False_11_0.0001_DANN.txt","snapshot/san/ImageClef_PI_Trained_Models/log_BSP_on_image-clef_vs_image-clef_with_False_11_0.0001_CDAN.txt","snapshot/san/ImageClef_PI_Trained_Models/log_RSL_on_image-clef_vs_image-clef_with_False_11_0.0001_CDAN+E.txt","snapshot/san/ImageClef_PI_Trained_Models/log_RSL_on_image-clef_vs_image-clef_with_True_11_0.0001_CDAN+E.txt"]
    all_target_acc ,all_a_distance , all_source_loss , all_d_loss , all_transfer =  [],[],[],[],[]
    model_names = ["CDAN","DANN","BSP","RSL","ASAN"]
    for name,file in zip(model_names,log_path):

        data = pd.read_csv(file, sep=",|:", header=None,skiprows=1)

        iters = data.iloc[:,1].values.tolist()

        target_acc = data.iloc[:,3].values.tolist()
        all_target_acc.append(target_acc)

        a_distance = data.iloc[:,5].values.tolist()
        all_a_distance.append(a_distance)

        source_loss = data.iloc[:,9].values.tolist()
        all_source_loss.append(source_loss)

        d_loss = data.iloc[:,11].values.tolist()
        all_d_loss.append(d_loss)

        if name == "ASAN" or name == "BSP" or "RSL":
            rsl = data.iloc[:,7].values.tolist()
            all_transfer.append(rsl)
    plt.rcParams.update({'font.size': 14})


    #plot_train_test(iters,[all_source_loss[-2],all_source_loss[-1]],[all_target_acc[-2],all_target_acc[-1]],model_names)
    plot_train_test_fill(iters,[all_source_loss[-2],all_source_loss[-1]],[all_target_acc[-2],all_target_acc[-1]],model_names)

    plot_a_distance_fill(iters,[all_transfer[0],all_transfer[-1]],[all_a_distance[2],all_a_distance[-1]],[model_names[2],model_names[-1]])
    # plot_a_distance(iters,[all_transfer[0],all_transfer[-1]],[all_a_distance[2],all_a_distance[-1]],[model_names[2],model_names[-1]])
    plot_error(iters, all_target_acc[:3] + all_target_acc[4:],model_names[:3] + model_names[4:])
