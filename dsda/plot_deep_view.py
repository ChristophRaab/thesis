import os
import copy
from deepview import DeepView
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
import sys
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

file = np.load("features/ASANfeatures.npz",allow_pickle=True)

data = file["bottleneck_features"]
source_size = file["source_size"]
target_size = file["target_size"]


labels,domain_predictions,class_predictions =file["truth_labels"],file["domain_predictions"],file["class_predictions"]
classes = np.unique(labels)

# def torch_wrapper(x):
#     with torch.no_grad():
#         x = np.array(x, dtype=np.float32)
#         tensor = torch.from_numpy(x).to(device)
#         pred = model(tensor).cpu().numpy()
#     return pred

n_trees = 100
model = RandomForestClassifier(n_trees)
model = model.fit(data, labels)
test_score = model.score(data, labels)
print('Created random forest')
print(' * No. of Estimators:\t', n_trees)
print(' * Train score:\t\t', test_score)

# create a wrapper function for deepview
# Here, because random forest can handle numpy lists and doesn't
# need further preprocessing or conversion into a tensor datatype
pred_wrapper = DeepView.create_simple_wrapper(model.predict_proba)


# --- Deep View Parameters ----
batch_size = 32
max_samples = 500
data_shape = (64,)
resolution = 100
N = 10
lam = 0.64
data_samples = 32
data_shape = (data.shape[1],)
cmap = 'tab10'
# to make shure deepview.show is blocking,
# disable interactive mode
interactive = False
title = 'Forest - MNIST'

deepview = DeepView(pred_wrapper, classes, max_samples, batch_size, data_shape,
	N, lam, resolution, cmap, interactive, title)

deepview.add_samples(data[:data_samples], labels[:data_samples])
deepview.show()