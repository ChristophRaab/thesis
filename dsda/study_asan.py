import argparse
import os.path as osp
import numpy as np
import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../"))
os.chdir(os.path.abspath(__file__ + "/.."))
import torch
import torch.nn as nn
import torch.optim as optim
from dsda import network
from dsda import loss
from dsda import pre_process as prep
from torch.utils.data import DataLoader
import itertools
from dsda.data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
import pandas as pd
import re
from dsda.train_asan import train,make_config

oh = "data/office-home/"
ic = "data/image-clef/"
o31 = "data/office/"
office_home =[oh+"Art.txt",oh+"Clipart.txt",oh+"Product.txt",oh+"Real_World.txt"] # ,
image_clef = [ic+"p_list.txt",ic+"i_list.txt",ic+"c_list.txt"]
office31 = [o31+"amazon.txt",o31+"webcam.txt",o31+"dslr.txt"]
visda = ["data/visda/train.txt","data/visda/validation.txt"]

def study_iteration(args,source,target):
    args.s_dset_path = source
    args.t_dset_path = target
    tmp_acc = []
    for i in range(3):

        # train config
        config = make_config(args)
        acc = train(config)
        tmp_acc.append(acc)
    return args,tmp_acc

def save_results(args,results):
        results = pd.DataFrame(results)
        results.to_csv("results/study_"+str(args.s_dset_path.split("/")[2].split(".")[0])+"_"+str(args.t_dset_path.split("/")[2].split(".")[0])+"_"+args.tl+"_"+str(args.sn)+".csv")

def do_study(args):
    results = []
    if args.dset == "office-home":
        dataset = office_home
    elif args.dset =="image-clef":
        dataset = image_clef
    elif args.dset =="office":
        dataset = office31
    elif args.dset =="visda":
        dataset = visda
    else:
        print("dataset not found")

    if args.dset == "visda":
        args,tmp_acc = study_iteration(args,dataset[0],dataset[1])

        results.append([np.mean(tmp_acc),np.std(tmp_acc)])
        save_results(args,results)

    else:
        for source in dataset:
            for target in dataset:
                if source != target:
                    args,tmp_acc = study_iteration(args,source,target)

                    results.append([np.mean(tmp_acc),np.std(tmp_acc)])
                    save_results(args,results)

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
    parser.add_argument('--k', type=int, default=11, help="K Parameter of RSL")
    parser.add_argument('--tllr', type=float, default=0.001, help="transfer_loss learning rate")
    parser.add_argument('--num_workers', type=int, default=12, help="Number of data loader workers")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    do_study(args)



#study.py --dset office --gpu_id 0 --sn False
#study.py --dset visda --gpu_id 1
#study.py --dset vsda --net ResNet101