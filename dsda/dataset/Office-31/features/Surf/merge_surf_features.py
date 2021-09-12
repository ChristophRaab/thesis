import os
import glob
import re
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
domains = ["dslr","amazon","webcam"]
for d in domains:
    print(d)
    files = glob.glob(d+'\\**/*800*.mat', recursive=True)
    data = []
    labels = []
    for f in files:
        classname = f.split("\\")[2]
        mat = sio.loadmat(f)
        mat =mat["histogram"][0].tolist()
        data.append(mat)
        labels.append(classname)


    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    data = np.array(data)
    sio.savemat(d+'_surf.mat', {'X': data, 'Y': labels})