import numpy as np
import errno
import glob,os
from numpy import loadtxt
listpath = "list/pList.txt"

folderpath = "image_folders/p/"

idx = []
clss = []
with open(listpath) as f:
    for line in f:
        line = line.split("\n")[0]
        line = line.split("p/")[1]
        folder = line.split(" ")[1]
        name = line.split(" ")[0]
        idx.append(str(name))
        clss.append(str(folder))

for f in glob.glob(folderpath + "*.jpg"):
    name = f.split("\\")[1]
    for i,listname  in enumerate(idx):
        if listname==name:
            if not os.path.exists(os.path.dirname(folderpath + clss[i]+"/")):
                os.makedirs(os.path.dirname(folderpath + clss[i]+"/"))

            os.rename(folderpath + name ,folderpath + clss[i] +"/"+ name)

