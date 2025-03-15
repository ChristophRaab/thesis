from __future__ import print_function, division
from numpy.lib.utils import source
import sys,os
from torch.optim import optimizer
sys.path.append(os.path.abspath(__file__ + "/../../"))
os.chdir(os.path.abspath(__file__ + "/.."))
from torchvision.datasets.folder import ImageFolder
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
import time
import copy                   
from dsda.network import init_weights
from dsda.pre_process import image_train,image_test
from torch.utils.data import DataLoader
import numpy as np
#from dsda.loss import Entropy_Regularization
from itertools import cycle

available_datasets = {
        'o':'dataset/Office-31/images/',
        'oh':'dataset/Office-Home/images/',
        'ic': 'dataset/imageCLEF/images/',
            }
subsets = {'o':["amazon","webcam","dslr"],
            'oh': ["Clipart","Art","Product","Real World",],
            'ic': ["b","c","i","p"],
            }

def SL(source, target):
    ns, nt = source.size(0), target.size(0)
    d = source.size(1)

    # Compute singular values of layers output
    _, u,_ = torch.svd(source)
    _, l,_ = torch.svd(target)

    # Compute Spectral loss
    loss = torch.norm(u-l)
    loss = loss / ns

    return loss

def create_resnet50_features():
    model_resnet = models.resnet50(pretrained=True)
    conv1 = model_resnet.conv1
    bn1 = model_resnet.bn1
    relu = model_resnet.relu
    maxpool = model_resnet.maxpool
    layer1 = model_resnet.layer1
    layer2 = model_resnet.layer2
    layer3 = model_resnet.layer3
    layer4 = model_resnet.layer4
    avgpool = model_resnet.avgpool
    feature_layers = nn.Sequential(conv1, bn1, relu, maxpool, \
                            layer1, layer2, layer3, layer4, avgpool)
    return feature_layers
    

def init_train_eval(source_dir,target_dir):
    source_dataset = ImageFolder(source_dir,transform=image_train())
    target_dataset = ImageFolder(target_dir,transform=image_train())
    validation_dataset = ImageFolder(target_dir,transform=image_test())
    source_loader = DataLoader(source_dataset,shuffle=True,num_workers=num_workers,batch_size=batch_size,drop_last=True)
    target_loader = DataLoader(target_dataset,shuffle=True,num_workers=num_workers,batch_size=batch_size,drop_last=True)
    validation_loader = DataLoader(validation_dataset,shuffle=False,num_workers=num_workers,batch_size=4)

    source_loader_size,target_loader_size,validation_loader_size = len(source_loader),len(target_loader),len(validation_loader)
    source_dataset_size, target_dataset_size = len(source_dataset),len(target_dataset)
    num_classes = len(source_dataset.classes)

    features_extractor = nn.Sequential(create_resnet50_features(),nn.Flatten(),nn.Linear(2048,bottleneck_dim)).cuda()
    classifier = nn.Sequential(nn.Linear(bottleneck_dim,num_classes)).cuda()
    classifier.apply(init_weights)
    features_extractor[-1].apply(init_weights)

    optimizer = optim.SGD(
        [{'params': features_extractor[:-1].parameters(),"lr_mult":1,'decay_mult':2},
        {'params': features_extractor[-1].parameters(),"lr_mult":10,'decay_mult':2},
        {'params': classifier.parameters(),"lr_mult":10,'decay_mult':2}],
        lr=lr,nesterov=True,momentum=0.9,weight_decay=0.0005)

# summary(features_extractor, (3, 224, 224))
# summary(classifier,(72,256))

    best_acc = 0
    for i in range(num_epochs):
        with torch.set_grad_enabled(True):
            avg_loss = avg_acc = cls_loss = avg_dc = dc_loss = classifier_loss = discriminator_loss = loss = 0.0
            training_list = zip(source_loader, cycle(target_loader)) if len(source_loader) > len(target_loader) else zip(cycle(source_loader), target_loader)
            for (xs,ys),(xt,yt) in training_list:
                
                xs,ys,xt,yt = xs.cuda(),ys.cuda(),xt.cuda(),yt.cuda()
                features_extractor.train(),classifier.train()
                optimizer.zero_grad() 

                fes = features_extractor(xs)
                fet = features_extractor(xt)
                ls = classifier(fes)
                lt = classifier(fet)

                classifier_loss = nn.CrossEntropyLoss()(ls,ys)
                spectral_loss = SL(ls,lt)
                loss = classifier_loss + 0.5 * spectral_loss
                loss.backward()
                optimizer.step()
    
                _,preds = nn.Softmax(1)(ls).max(1)
                avg_loss = avg_loss + loss
                avg_acc  = avg_acc + (preds == ys).sum()

        if i % 1 == 0:
            with torch.set_grad_enabled(False):
                vavg_loss,vavg_acc,best_acc = 0.0,0.0,0.0
                for xt,yt in validation_loader:
                    
                    xt,yt = xt.cuda(),yt.cuda()
                    features_extractor.eval(),classifier.eval()

                    lt = classifier(features_extractor(xt))

                    _,preds = nn.Softmax(1)(lt).max(1)
                    classifier_loss = nn.CrossEntropyLoss()(lt,yt)
                    
                    loss = classifier_loss

                    vavg_loss = vavg_loss + loss
                    vavg_acc  = vavg_acc + (preds == yt).sum()
            tmp_best = round((vavg_acc/target_dataset_size).item(),4)*100
            best_acc = tmp_best if tmp_best > best_acc else best_acc
            best_model = copy.deepcopy([features_extractor,classifier]) if tmp_best > best_acc else best_acc
            print("Progress " + str(i) +  " Mean Validation Loss: "+str(round((vavg_loss/validation_loader_size).item(),3))+ " Acc "+str(round((vavg_acc/target_dataset_size).item(),3))
                    + " --- Mean Training Loss: "+str(round((avg_loss/source_loader_size).item(),3))+ " Acc "+str(round((avg_acc/source_dataset_size).item(),3)))
    return best_acc,best_model


bottleneck_dim = 256
lr = 0.001
batch_size = 36
num_workers = 4
cuda = "cuda:0"
num_epochs = 50
dataset_name = 'oh' # Office-home
dataset = available_datasets[dataset_name]
testsize = 3
results_path = "results_"+datetime.today().strftime('%Y-%m-%d-%H:%M:%S')+"_dsn_"+dataset_name+".csv"
results = []
pd.DataFrame(results).to_csv(results_path)
for s1 in subsets[dataset_name]:
    for s2 in subsets[dataset_name]:  
        if s1 != s2:
            tmp_results = []      
            for _ in range(testsize):
                acc,_ = init_train_eval(dataset+s1, dataset+s2)
                tmp_results.append(acc)
            results.append([s1,s2,np.mean(tmp_results),np.std(tmp_results)])
            pd.DataFrame(results).to_csv(results_path,mode="a")