import argparse
import os
import os.path as osp
import sys

from torch.nn.functional import softmax
import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../"))
os.chdir(os.path.abspath(__file__ + "/.."))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from dsda import network
from dsda import loss
from dsda import pre_process as prep 
from torch.utils.data import DataLoader
from dsda import lr_schedule
from dsda.data_list import ImageList
import re
def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
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

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageFolder(data_config["target"]["list_path"],transform=prep_dict["test"][i]) \
                                for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=config["num_loader"]) for dset in dsets['test']]
    else:
        dsets["test"] = ImageFolder(data_config["target"]["list_path"],transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=config["num_loader"])

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        random_layer.cuda()
        if config["sn"]:
            ad_net = network.SNAdversarialNetwork(config["loss"]["random_dim"], 1024)
        else:
            ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        out_dim = base_network.output_num() * class_num if "CDAN" in config["method"] else config["network"]["params"]["bottleneck_dim"]
        if config["sn"]:
            ad_net = network.SNAdversarialNetwork(out_dim, 1024)
        else:
            ad_net = network.AdversarialNetwork(out_dim,  1024)

    
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters() 

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])


    ## train

    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value =source_entropy = target_entropy= 0.0
    best_acc = 0.0
    config_description = config["tl"]+"_on_"+re.split(r'[.,/]', config["data"]["source"]["list_path"])[1]+"_vs_"+re.split(r'[.,/]', config["data"]["target"]["list_path"])[1]+"_SN_"+str(config["sn"])+"_"+str(config['method'])

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1 or i == 1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
                best_ad_net = ad_net
            with torch.no_grad():
                a_distance = loss.compute_a_distance(config['method'],[features, softmax_out], ad_net,random_layer)
                log_str = "iter: {:05d}, precision: {:.5f}, a_distance: {:.5f}, rsl: {:.5f}, classifier_loss: {:.5f},target_entropy: {:.5f}, source_entropy: {:.5f},  discriminator_loss: {:.5f},".format(i, temp_acc, a_distance, rsl.item(),classifier_loss.item(),target_entropy.item(), source_entropy.item(), transfer_loss.item())
                config["out_file"].write(log_str+"\n")
                config["out_file"].flush()
                print(log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "log_iter_{:05d}_"+config_description+".pth.tar".format(i)))

        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source) # Bootleneck Features and Classification featuers
        features_target, outputs_target = base_network(inputs_target) # Bootleneck Features and Classification featuers
        features = torch.cat((features_source, features_target), dim=0) # Concat features for domain classification
        outputs = torch.cat((outputs_source, outputs_target), dim=0) # Required for CDAN
        softmax_out = nn.Softmax(dim=1)(outputs) #(outputs - outputs.min()) / (outputs.max() - outputs.min())# # Required for CDAN
        if config['method'] == 'CDAN+E': # Calculate Domain Confusion Loss
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['method']  == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        elif config['method']  == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')

        if config['tl'] =="RSL":
            rsl = loss.RSL(features_source,features_target,config['k'])
        elif config["tl"] =="RSLc":
             rsl = loss.RSL(outputs_source,outputs_target,config['k'])
        elif config["tl"] =="BSP":
             rsl = loss.BSP(outputs_source,outputs_target )
        else:
            rsl = torch.tensor([0]).cuda()
        source_entropy = loss.Entropy_Regularization(softmax_out[:softmax_out.size(0)//2,:])
        target_entropy = loss.Entropy_Regularization(softmax_out[softmax_out.size(0)//2:,:])


        rsl_loss = config["tllr"] * rsl
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + rsl_loss + target_entropy - source_entropy
        total_loss.backward()
        optimizer.step()
    torch.save(best_model, osp.join(config["output_path"], "best_model_"+config_description+".pth.tar"))
    torch.save(best_ad_net, osp.join(config["output_path"], "best_ad_net_"+config_description+".pth.tar"))
    return best_acc

def make_config(args):
    config = {}
    config['tl'] = args.tl
    config["sn"] = args.sn
    config["k"] = args.k
    config["tllr"] = args.tllr
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 75000#100004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log_"+config["tl"]+"_on_"+re.split(r'/|\.', args.s_dset_path)[1]+
    "_vs_"+re.split(r'/|\.', args.t_dset_path)[1]+"_with_"+str(config["sn"])+"_"+str(config["k"])+"_"+str(config["tllr"])+"_"+config["method"]+".txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                        "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                    "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                    "test":{"list_path":args.t_dset_path, "batch_size":4}}
    config["num_loader"] = args.num_workers
    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
        ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
        ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
        ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
            ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='dataset/Office-31/images/amazon/', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='dataset/Office-31/images/webcam/', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of data loader workers")
    parser.add_argument('--sn', type=bool, default=True, help="whether to use spectral normalization")
    parser.add_argument('--tl', type=str, default="RSL", help="transfer_loss")
    parser.add_argument('--k', type=int, default=11, help="K Parameter of RSL")
    parser.add_argument('--tllr', type=float, default=0.001, help="transfer_loss learning rate")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    print(args.tl)
    config = make_config(args)
    acc = train(config)

#train_image.py --tl RSL --s_dset_path data/amazon.txt --t_dset_path data/webcam.txt --test_interval 100 --num_workers 2 --sn True

