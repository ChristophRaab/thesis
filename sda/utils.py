
#Imports
import os
import scipy.io as sio
from sklearn import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from os import walk
from sklearn.svm import SVC
from NBT import NBT
from keras import regularizers
from sklearn_lvq import GlvqModel


def train_network(Xs,Ys,X,Yt):
    idx = Ys < 0 
    Ys[idx]= 0
    idx = Yt < 0 
    Yt[idx]= 0
    model = Sequential()

    model.add(Dense(units=1000, activation='relu', input_dim=Xs.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(units=750, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=250, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(units=np.unique(Ys).size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
    model.fit(Xs,Ys, epochs=30, batch_size=16,verbose=0)

    score = model.evaluate(X, Yt,
                        batch_size=32, verbose=1)
    y_prob = model.predict(X) 
    y_classes = y_prob.argmax(axis=-1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def train_glvq(Xs,Ys,X,Yt):
    clf = SVC(C=10)
    clf.fit(Xs,Ys)
    print("SVM ACC: "+str(clf.score(X,Yt)))

    clf = GlvqModel(4)
    clf.fit(Xs, Ys)
    print('GLVQ accuracy:', clf.score(X, Yt))



def load_data(filename):
    errors = sio.loadmat(filename)
    X = np.asarray(errors["Xt"].T.todense())
    Xs = np.asarray(errors["Xs"].T.todense())
    Ys = np.ravel(np.asarray(errors["Ys"].todense()))
    Yt = np.ravel(np.asarray(errors["Yt"].todense()))
    X = preprocessing.scale(X)
    Xs = preprocessing.scale(Xs)
    return Xs,Ys,X,Yt


def test_reuters():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.join("datasets","domain_adaptation","features","reuters"))
    filename = next(walk(os.curdir))[2][0]
    Xs, Ys, X, Yt = load_data(filename)

    clf = SVC(gamma=2, C=10)
    clf.fit(Xs, Ys)
    print("SVM: " + str(clf.score(X, Yt)))

    nbt = NBT()
    Ys, Xs = nbt.data_augmentation(Xs, X.shape[0], Ys)
    X, Xs = nbt.fit(X, Xs, Ys.flatten(), landmarks=500)
    clf = SVC(gamma=2, C=10)
    clf.fit(Xs, Ys)
    print("SVM + NBT: " + str(clf.score(X, Yt)))

def test_office_chaltech():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.join("datasets", "domain_adaptation", "features", "OfficeCaltech_Surf"))
    amazon = sio.loadmat("amazon_SURF_L10.mat")
    X = preprocessing.scale(np.asarray(amazon["fts"]))
    Yt = np.asarray(amazon["labels"])

    dslr = sio.loadmat("dslr_SURF_L10.mat")

    Xs = preprocessing.scale(np.asarray(dslr["fts"]))
    Ys = np.asarray(dslr["labels"])

    clf = SVC(gamma=1, C=10)
    clf.fit(Xs, Ys)
    print("SVM: " + str(clf.score(X, Yt)))

    nbt = NBT()
    Ys, Xs = nbt.data_augmentation(Xs, X.shape[0], Ys)
    X, Xs = nbt.fit(X, Xs, Ys.flatten(), landmarks=100)
    clf = SVC(gamma=1, C=10)
    clf.fit(Xs, Ys)
    print("SVM + NBT: " + str(clf.score(X, Yt)))

if __name__ == "__main__":
    test_reuters()
    test_office_chaltech()