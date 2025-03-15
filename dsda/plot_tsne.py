import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../"))
os.chdir(os.path.abspath(__file__ + "/.."))
import torch
from dsda.data_list import ImageList
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dsda import pre_process as prep
from torch import nn
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
def plot_tsne(x,y,name):
    plt.figure()
    tsne = TSNE(n_components=2)

    data = tsne.fit_transform(x)
    data_max, data_min = np.max(
        data, 0), np.min(data, 0)
    d = (data-data_min) / (data_max - data_min)


    if 'classification' in name:
        plt.scatter(d[:, 0], d[:, 1], s=2, c=y.flatten())#cmap=plt.get_cmap("tab20"))
    else:
        colors = ['dodgerblue' if label is 1 else 'darkred' for  label in y.ravel().tolist() ]
        plt.scatter(d[:, 0], d[:, 1], s=2, color=colors)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("plots/tsne_"+name+"_.jpg", transparent=True,dpi=400,)
    plt.savefig("plots/tsne_"+name+"_.pdf", transparent=True,dpi=400,bbox_inches = 'tight',
    pad_inches = 0.05)
    # plt.show(block=False)
    # plt.show(block=False)

def plot_scatter(d,y,name):
    plt.figure()

    if 'classification' in name:
        plt.scatter(d[:, 0], d[:, 1], s=2, c=y.flatten())#cmap=plt.get_cmap("tab20"))
    else:
        colors = ['dodgerblue' if label is 1 else 'darkred' for  label in y.ravel().tolist() ]
        plt.scatter(d[:, 0], d[:, 1], s=2, color=colors)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("plots/tsne_"+name+"_.jpg", transparent=True,dpi=400,)
    plt.savefig("plots/tsne_"+name+"_.pdf", transparent=True,dpi=400,bbox_inches = 'tight',
    pad_inches = 0.05)
    # plt.show(block=False)
    # plt.show(block=False)

def cdan_domain_prediction(predictions,features,ad_net,domain_label=0):
    softmax_out = nn.Softmax(dim=1)(predictions)
    op_out = torch.bmm(softmax_out.unsqueeze(2), features.unsqueeze(1))
    ad_out = ad_net(op_out.view(-1, softmax_out.size(1) * features.size(1)))
    return ad_out

def dann_domain_prediction(features,ad_net,domain_label=0):
    ad_out = ad_net(features)
    return ad_out

def calc_vectors(bottleneck_features, source_size, target_size,name):
    s  = bottleneck_features[:source_size,:]
    t = bottleneck_features[:target_size,:]
    cov_s = (s.T @ s)
    cov_t = (t.T @ t)


    ds,us = np.linalg.eig(cov_s,)
    dt,ut = np.linalg.eig(cov_t)
    us = us.T
    ut = ut.T
    K = cosine_similarity(us,ut)
    rads = np.arccos(np.diag(K))
    np.save(name+"rads.npy",rads)
    print(np.mean(ds-dt))
def plot_subspace_angle(name):
    print(name)

    rads = np.load(name+"rads.npy")
    rads = np.sqrt(rads)
    i = 100
    x = np.arange(i)
    plt.figure()

    m = np.mean(rads[:i])
    s = np.std(rads[:i])
    plt.ylim(0,3.3)
    p1 = plt.bar(x,rads[:i],label="Singular Vector Cosine Angle")
    p2, = plt.plot(x,m*np.ones(i),"r",label="Mean")
    # p3 = plt.fill_between(x,m-s,m+s,alpha=0.2,label="Std")
    # fig = plt.gcf()
    plt.legend(handles =[p1,p2])
    plt.xlabel("No.")
    plt.ylabel("Rad")

    plt.savefig("plots/"+name+"_plot_angle.png")
    plt.savefig("plots/"+name+"_plot_angle.pdf",transparent=True,dpi=400,bbox_inches = 'tight',
    pad_inches = 0.05)

def min_max_scaling(X):

    mx = np.max(X)
    mn = np.min(X)

    X = (X - mn) / (mx -mn)
    return X

def plot_spectra(source,target):

    i = 10
    x = np.arange(i)
    plt.figure()


    source =  min_max_scaling(source[1:i+1])
    target =  min_max_scaling(target[1:i+1])
    # plt.ylim(0,3.3)
    p1, = plt.plot(x,source,label="Source Spectrum")
    p2, = plt.plot(x,target,label="Target Spectrum")
    # p3 = plt.fill_between(x,m-s,m+s,alpha=0.2,label="Std")
    # fig = plt.gcf()
    plt.legend(handles =[p1,p2])
    plt.xlabel("No.")
    plt.ylabel("Singular Value")

    plt.savefig("plots/"+name+"_plot_both_spectra.png")
    plt.savefig("plots/"+name+"_plot_both_spectra.pdf",dpi=400)

if __name__ == '__main__':
    # base_models = ["snapshot/san/AW_Trained_Models/_ASAN+E_on_amazon_vs_webcam.pth.tar","snapshot/san/AW_Trained_Models/_CDAN_on_amazon_vs_webcam.pth.tar","snapshot/san/AW_Trained_Models/_DANN_on_amazon_vs_webcam.pth.tar"]
    # ad_nets = ["snapshot/san/AW_Trained_Models/_ASAN+E_ad_net_on_amazon_vs_webcam.pth.tar","snapshot/san/AW_Trained_Models/_CDAN_ad_net_on_amazon_vs_webcam.pth.tar","snapshot/san/AW_Trained_Models/_DANN_ad_net_on_amazon_vs_webcam.pth.tar"]
    base_models = ["snapshot/san/best_model_RSL_on_image-clef_vs_image-clef_SN_True_CDAN+E.pth.tar","snapshot/san/best_model_no_da_on_image-clef_vs_image-clef_SN_False_CDAN.pth.tar","snapshot/san/best_model_no_da_on_image-clef_vs_image-clef_SN_False_DANN.pth.tar"]
    ad_nets = ["snapshot/san/best_ad_net_RSL_on_image-clef_vs_image-clef_SN_True_CDAN+E.pth.tar","snapshot/san/best_ad_net_no_da_on_image-clef_vs_image-clef_SN_False_CDAN.pth.tar","snapshot/san/best_ad_net_no_da_on_image-clef_vs_image-clef_SN_False_DANN.pth.tar"]


    ## Plot Subspace similarities
    model_names = ["ASAN","CDAN","DANN"]
    for name in model_names:
        data  = np.load(name+"featuers.npz",allow_pickle=True)
        calc_vectors( data["bottleneck_features"], data["source_size"],data["target_size"],name)
        plot_subspace_angle(name)



    # Plot Tsne representation
    for name,file,ad_net_file in zip(model_names,base_models,ad_nets):
        print(name)
        ### Load trained model
        pretrained_model = torch.load(file,map_location="cpu")
        pretrained_model.eval()

        # Load trained domain discriminator
        ad_net = torch.load(ad_net_file,map_location="cpu")
        ad_net.eval()

        # Preprocessing and dataset config
        config = {}
        config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
        config["dataset"] = "image-clef"
        # config["data"] = {"source":{"list_path":"data/amazon.txt", "batch_size":36}, \
        #                         "target":{"list_path":"data/webcam.txt", "batch_size":36}, \
        #                         "test":{"list_path":"data/webcam.txt", "batch_size":4}}
        config["data"] = {"source":{"list_path":"data/image-clef/p_list.txt", "batch_size":36}, \
                                "target":{"list_path":"data/image-clef/i_list.txt", "batch_size":36}, \
                                "test":{"list_path":"data/image-clef/i_list.txt", "batch_size":4}}
        # Associate prepocessing to datasets
        prep_dict = {}
        prep_config = config["prep"]
        prep_dict["source"] = prep.image_train(**config["prep"]['params'])
        prep_dict["target"] = prep.image_train(**config["prep"]['params'])
        if prep_config["test_10crop"]:
            prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
        else:
            prep_dict["test"] = prep.image_test(**config["prep"]['params'])

        # Configure dataloader
        dsets = {}
        dset_loaders = {}
        data_config = config["data"]
        train_bs = data_config["source"]["batch_size"]
        test_bs = data_config["test"]["batch_size"]
        dsets["source"] = ImageFolder(data_config["source"]["list_path"],transform=prep_dict["source"])
        dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                shuffle=True, num_workers=config["num_loader"], drop_last=True)
        dsets["target"] = ImageFolder(data_config["target"]["list_path"],transform=prep_dict["target"])
        dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                shuffle=True, num_workers=config["num_loader"], drop_last=True)


        # Set up iterators and load data
        iter_source = iter(dset_loaders["source"])
        iter_target = iter(dset_loaders["target"])

        # Set up data placeholder and domain labels
        class_predictions,domain_predictions,truth_labels,bottleneck_features = np.array([]),np.array([]),np.array([]),np.array([])
        domain_labels = np.array([[1]] * (config["data"]["source"]["batch_size"] *len(dset_loaders["source"])) + [[0]] *(config["data"]["target"]["batch_size"] *  len(dset_loaders["target"])))
        classes = np.arange(12)
        # featuer extraction of features, predictions and domain predictions
        for iter_data in [iter_source,iter_target]:
            for inputs,labels in iter_data:
                # inputs, labels = iter_source.next()

                features,predictions = pretrained_model(inputs)

                if "DANN" in name:
                    d_pred = dann_domain_prediction(features,ad_net,0)
                else:
                    d_pred = cdan_domain_prediction(predictions,features,ad_net,0)

                class_predictions = np.vstack([class_predictions, predictions.cpu().detach().numpy()]) if class_predictions.size else predictions.cpu().detach().numpy()

                truth_labels = np.hstack([truth_labels, labels.cpu().detach().numpy()]) if truth_labels.size else labels.cpu().detach().numpy()

                domain_predictions = np.vstack([domain_predictions, d_pred.cpu().detach().numpy()]) if domain_predictions.size else d_pred.cpu().detach().numpy()

                bottleneck_features = np.vstack([bottleneck_features, features.cpu().detach().numpy()]) if bottleneck_features.size else features.cpu().detach().numpy()
                plot_deep_view(inputs,truth_labels,classes,pretrained_model)
        # plot tsne
        source_size = len(np.array([[1]] * (config["data"]["source"]["batch_size"] *len(dset_loaders["source"]))))
        target_size = len(np.array([[0]] *(config["data"]["target"]["batch_size"] *  len(dset_loaders["target"]))))


        np.savez("features/"+name+"features.npz", bottleneck_features=bottleneck_features, source_size=source_size, target_size=target_size,truth_labels=truth_labels,domain_predictions=domain_predictions,class_predictions=class_predictions)
        # plot_subspace_angle(bottleneck_features,source_size,target_size)

        # data = np.load(name+"features.npz")
        # bottleneck_features = data["bottleneck_features"]
        # source_size = data["source_size"]
        # target_size = data["target_size"]

        # tsne = TSNE(n_components=2)
        # data = tsne.fit_transform(bottleneck_features)
        # data_max, data_min = np.max(
        # data, 0), np.min(data, 0)
        # d = (data-data_min) / (data_max - data_min)

        # plot_scatter(d,truth_labels,name+"_classification")
        # plot_scatter(d,domain_labels,name+"_features_domain")

        # plot_tsne(bottleneck_features,truth_labels,name+"_classification")
        # plot_tsne(bottleneck_features,domain_labels,name+"_features_domain")