# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:32:39 2021

@author: MaxiT
"""
import numpy as np

def z_score(x):
    """
    Normalización por z-score.
    
    Args:
        X (numpy array): matriz de m features x n muestras
        
    Returns:
        numpy array: matirz normalizada
    """
    den=np.std(x,axis=0)
    #Avoid zero division
    den[den==0]=1
    x_norm=(x-np.mean(x,axis=0))/den
    return x_norm


def Test():
    """
    Test de metodo de normalizacion por z-score
    """
    x=np.array([[4,-100],[1,50],[4,50]])
    x_norm=z_score(x)
    print(x_norm)
    return


Test()

