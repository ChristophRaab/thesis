import argparse
import os.path as osp
import numpy as np
import torch
import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../"))
import torch.nn as nn
import torch.optim as optim
from dsda import network
from dsda import loss
from dsda import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
import itertools
from dsda.data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
import pandas as pd
import re
from dsda.study_asan import study_iteration
from dsda.train_asan import train,make_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='dataset/Office-31/images/amazon/',, help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='dataset/Office-31/images/webcam/',, help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--sn', type=bool, default=True, help="whether to use spectral normalization")
    parser.add_argument('--tl', type=str, default="RSL", help="transfer_loss")
    parser.add_argument('--k', type=int, default=1, help="K Parameter of RSL")
    parser.add_argument('--tllr', type=float, default=0.001, help="transfer_loss learning rate")
    parser.add_argument('--num_workers', type=int, default=12, help="Number of data loader workers")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    shrinkage_parameters = np.arange(1,21).tolist()
    tl_lrs = [0.001,0.0001,0.00001]
    tl_lrs = [0.001]
    paramter_space = list(itertools.product(shrinkage_parameters, tl_lrs))
    selection = random.sample(paramter_space,5)
    results = []
    for paras in selection:
            args.k = paras[0]
            args.tllr = paras[1]
            args,tmp_acc = study_iteration(args,args.s_dset_path,args.t_dset_path)
            results.append([args.k,args.tllr, np.mean(tmp_acc),np.std(tmp_acc)])
    results = pd.DataFrame(results)
    results.to_csv("results/optimize_"+str(args.k)+"_"+str(args.tllr)+"_"+str(args.s_dset_path.split("/")[2].split(".")[0])+"_"+str(args.t_dset_path.split("/")[2].split(".")[0])+"_"+args.tl+"_"+str(args.sn)+".csv")

#optimize_image.py --dset image-clef --s_dset_path data/image-clef/p_list.txt --t_dset_path data/image-clef/i_list.txt --gpu_id 1