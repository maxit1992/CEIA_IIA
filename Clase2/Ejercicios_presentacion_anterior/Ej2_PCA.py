# -*- coding: utf-8 -*-
"""
Created on Sat May  8 21:13:41 2021

@author: MaxiT
"""
import numpy as  np
from sklearn.decomposition import PCA

def my_pca(X):
    """
    Transformacion de vectores X segun PCA considerando componente principal
    
    Args:
        X (numpy array): vectores
        
    Returns:
        numpy array: vecotres transformados
    """
    X_norm= X - np.mean(X,axis=0)
    #X_norm= X_norm / np.std(X_norm)
    S= np.cov(X_norm.transpose())/(X_norm.shape[0])
    w,v=np.linalg.eig(S)
    v_max=v[:,np.argmax(w)]

    X_pca= np.matmul(v_max.transpose(),X_norm.transpose())
    
    return X_pca



def Test():
    """
    Test de metodo de PCA y comparacion contra metodo de scikit
    """
    X=np.array([[0.8,0.7],[0.1,-0.1]])
    X_pca_1=my_pca(X)
    X_norm= X - np.mean(X,axis=0)
    pca = PCA(n_components=1)
    X_pca_2 = pca.fit_transform(X_norm)
    print("Mi PCA:{}".format(X_pca_1))
    print("PCA Scikit:{}".format(X_pca_2))


Test()