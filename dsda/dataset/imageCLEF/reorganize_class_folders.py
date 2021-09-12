import numpy as np
import errno
import glob,os
from numpy import loadtxt
import shutil

for char in ["i","c","b","p"]:

    listpath = "list/"+char+"List.txt"

    folderpath = "image_folders/"+char+"/"
    imagepath = char+"/"
    idx = []
    clss = []
    with open(listpath) as f:
        for line in f:
            line = line.split("\n")[0]
            line = line.split(char+"/")[1]
            folder = line.split(" ")[1]
            name = line.split(" ")[0]
            idx.append(str(name))
            clss.append(str(folder))

    for f in glob.glob(imagepath + "*.jp*"):
        name = f.split("\\")[1]
        for i,listname  in enumerate(idx):
            if listname==name:
                if not os.path.exists(os.path.dirname(folderpath + clss[i]+"/")):
                    os.makedirs(os.path.dirname(folderpath + clss[i]+"/"))

                shutil.copy2(imagepath + name ,folderpath + clss[i] +"/"+ name)



