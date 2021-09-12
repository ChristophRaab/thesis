import os
import glob
import re
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
domains = ["amazon","dslr","webcam"]
decaf_level = ["fc6","fc7","fc8"]
for dl in decaf_level:
    for d in domains:
        files = glob.glob(d+'\\**/*.mat', recursive=True)
        data = []
        labels = []
        for f in files:
            classname = f.split("\\")[1]
            mat = sio.loadmat(f)
            mat = mat[dl]
            data.append(mat)
            labels.append(classname)

        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = np.array(data)
        sio.savemat(d+'_decaf_'+dl+'.mat', {'X': data, 'Y': labels})