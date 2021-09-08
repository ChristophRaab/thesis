import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../../"))
class GDA():
    """
    Geometric Transfer Transfer Service Class
    
    Functions
    ----------
    geometric transfer: Transfer Basis from Target to Source Domain.
    data_augmentation: Augmentation of data by removing or upsampling of source data

    Example:
    --------
    >>> import os
    >>> import scipy.io as sio
    >>> from sklearn.svm import SVC
    >>> 
    >>> os.chdir("dataset/OfficeCaltech/features/surf")
    >>> 
    >>> # Load and preprocessing of data. Note normalization to N(0,1) is necessary.
    >>> dslr = sio.loadmat(os.path.abspath(__file__ + "/..")+"/dataset/OfficeCaltech/features/surf/dslr_SURF_L10.mat")
    >>> Xs = preprocessing.scale(np.asarray(dslr["fts"]))
    >>> Ys = np.asarray(dslr["labels"])
    >>> 
    >>> amazon = sio.loadmat(os.path.abspath(__file__ + "/..")+"/dataset/OfficeCaltech/features/surf/amazon_SURF_L10.mat")
    >>> Xt = preprocessing.scale(np.asarray(amazon["fts"]))
    >>> Yt = np.asarray(amazon["labels"])
    >>> 
    >>> # Applying SVM without transfer learning. Accuracy should be about 10%
    >>> clf = SVC(gamma=1,C=10)
    >>> clf.fit(Xs,Ys)
    >>> print("SVM without transfer "+str(clf.score(Xt,Yt.ravel())))
    >>> 
    >>> # Initialization of GDA. Accuracy of SVM + GDA should be about 90%
    >>> GDA = GDA(landmarks=100)
    >>> # Compute domain invariant space data directly by 
    >>> Xt,Xs,Ys = GDA.fit_transform(Xt,Xs,Ys)
    >>> 
    >>> # Or use two steps
    >>> # GDA.fit(Xt)
    >>> # Xt,Xs,Ys = GDA.transform(Xs,Ys)
    >>> 
    >>> clf = SVC(gamma=1,C=10)
    >>> clf.fit(Xs,Ys)
    >>> print("SVM + GDA: "+str(clf.score(Xt,Yt.ravel())))
    >>> 
    >>> model = KNeighborsClassifier(n_neighbors=1)
    >>> model.fit(Xs, Ys.ravel())
    >>> 
    >>> score = model.score(Xt, Yt)
    >>> print("KNN + GDA: "+str(score))
    >>> """

    def __init__(self):
        pass

    def fit_transform(self,Xt,Xs,Ys=None):
        """
        Geometric Transfer
        Transfers Basis of X to Xs obtained by  SVD
        Applications in domain adaptation or transfer learning
        Parameters.
        Note target,source are order sensitiv.

        ----------
        Parameters 
        X   : Target Matrix, where classifier is trained on
        Xs  : Source Matrix, where classifier is trained on
        Ys  : Source data label, if none, classwise sampling is not applied.
        
        ----------
        Returns
        Xt : Reduced Target Matrix
        Xs : Reduced approximated Source Matrix
        Ys : Augmented source label matrix
        """
        Ys,Xs = self.data_augmentation(Xs,Xt.shape[0],Ys)
        self.n_xt = Xt.shape[0]
        if type(Xt) is not np.ndarray or type(Xs) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")

        # Correct landmarks if user enters impossible value
        L,T,R = np.linalg.svd(Xt,compute_uv=True,full_matrices=False)
        S = np.linalg.svd(Xs,compute_uv=False,full_matrices=False)
        
        self.Xt = L @ np.diag(T) @ R
        self.Xs = L @ np.diag(S) @ R
        self.Ys = Ys
        return self.Xt,self.Xs,self.Ys


    def fit(self,Xt):
        '''
        Applies SVD to obtain basis matrices approximation to  Xt. 
        ----------
        Parameters
        :param Xt: ns * n_feature, source feature
        '''
        if type(Xt) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")

        #V_k = np.concatenate([H, np.matmul(np.matmul(B.T,U),np.linalg.pinv(S))])
        L,T,R = np.linalg.svd(Xt,compute_uv=True,full_matrices=False)
        self.L = L
        self.R = R
        self.Xt = Xt
        self.n_xt = Xt.shape[0]

    def transform(self,Xs,Ys):
        '''
        Augments Xs and Ys to fit sample sizes. Projects Xs into the space of Xt
       
        ----------
        Parameters
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        
        ----------
        Returns
        Xt : Augment and Projected
        Xs : augmented and projected
        Ys : Augmented source labels 

        '''
        if type(Xs) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")
        
        Ys,Xs = self.data_augmentation(Xs,self.n_xt,Ys)
        
        D = np.linalg.svd(Xs,compute_uv=False,full_matrices=False)
        Xs = self.L @ np.diag(D) @ self.R.T
        self.Ys = Ys
        return self.Xt,self.Xs,self.Ys

    def data_augmentation(self,Xs,required_size,Y):
        """
        Data Augmentation
        Upsampling if Xs smaller as required_size via multivariate gaussian mixture
        Downsampling if Xs greater as required_size via uniform removal

        Note both are class-wise with goal to harmonize class counts
        
        ----------
        Parameters
        Xs : Matrix, where classifier is trained on
        required_size : Size to which Xs is reduced or extended
        Y : Label vector, which is reduced or extended like Xs

        ----------
        Returns
        Ys : Augmented 
        Xs : Augmented

        """
        if type(Xs) is not np.ndarray or type(required_size) is not int or type(Y) is not np.ndarray:
            raise ValueError("Numpy Arrays must be given!")
        if Xs.shape[0] == required_size:
            return Y,Xs
        
        _, idx = np.unique(Y, return_index=True)
        C = Y[np.sort(idx)].flatten().tolist()
        size_c = len(C)
        if Xs.shape[0] < required_size:
            print("Source smaller target")
            data = np.empty((0,Xs.shape[1]))
            label = np.empty((0,1))
            diff = required_size - Xs.shape[0]
            sample_size = int(np.floor(diff/size_c))
            for c in C:
                #indexes = np.where(Y[Y==c])
                indexes =  np.where(Y==c)
                class_data = Xs[indexes,:][0]
                m = np.mean(class_data,0) 
                sd = np.var(class_data,0)
                sample_size = sample_size if c !=C[-1] else sample_size+np.mod(diff,size_c)
                augmentation_data =np.vstack([np.random.normal(m, sd, size=len(m)) for i in range(sample_size)])
                data =np.concatenate([data,class_data,augmentation_data])
                label = np.concatenate([label,np.ones((class_data.shape[0]+sample_size,1))*c])
            
        if Xs.shape[0] > required_size:
            print("Source greater target")
            data = np.empty((0,Xs.shape[1]))
            label = np.empty((0,1))
            sample_size = int(np.floor(required_size/size_c))
            for c in C:
                indexes = np.where(Y[Y==c])[0]
                class_data = Xs[indexes,:]
                if len(indexes) > sample_size:
                    sample_size = sample_size if c !=C[-1] else np.abs(data.shape[0]-required_size)
                    y = np.random.choice(class_data.shape[0],sample_size)
                    class_data = class_data[y,:]
                data =np.concatenate([data,class_data])
                label = np.concatenate([label,np.ones((class_data.shape[0],1))*c])
        self.Xs = data
        self.Ys = label
        return self.Ys,self.Xs

if __name__ == "__main__":

    import os
    import scipy.io as sio
    from sklearn.svm import SVC




    # Load and preprocessing of data. Note normalization to N(0,1) is necessary.
    dslr = sio.loadmat(os.path.abspath(__file__ + "/..")+"/dataset/OfficeCaltech/features/surf/dslr_SURF_L10.mat")
    Xs = preprocessing.scale(np.asarray(dslr["fts"]))
    Ys = np.asarray(dslr["labels"])

    amazon = sio.loadmat(os.path.abspath(__file__ + "/..")+"/dataset/OfficeCaltech/features/surf/amazon_SURF_L10.mat")
    Xt = preprocessing.scale(np.asarray(amazon["fts"]))
    Yt = np.asarray(amazon["labels"])

    # Applying SVM without transfer learning. Accuracy should be about 10%
    clf = SVC(gamma=1,C=10)
    clf.fit(Xs,Ys)
    print("SVM without transfer "+str(clf.score(Xt,Yt.ravel())))

    # Initialization of GDA. Accuracy of SVM + GDA should be about 90%
    GDA = GDA()
    # Compute domain invariant space data directly by 
    Xt,Xs,Ys = GDA.fit_transform(Xt,Xs,Ys)
    
    # Or use two steps
    # GDA.fit(Xt)
    # Xt,Xs,Ys = GDA.transform(Xs,Ys)

    clf = SVC(gamma=1,C=10,kernel="linear")
    clf.fit(Xs,Ys)
    print("SVM + GDA: "+str(clf.score(Xt,Yt.ravel())))

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(Xs, Ys.ravel())

    score = model.score(Xt, Yt)
    print("KNN + GDA: "+str(score))