# -*- coding: utf-8 -*-
"""
Created on Sat May  8 18:11:26 2021

@author: MaxiT
"""
import numpy as np

def NormL2(arr):
    """
    Calcula la norma L2 de cada vector fila de una matriz
    
    Args:
        arr (numpy array): matriz de entrada
        
    Returns:
        numpy array: vector con normas L2
    """
    l2= np.power(np.sum(np.power(np.abs(arr),2),axis=2),1/2)
    return l2

def nearest_centroid(x,centroid):
    """
    Calcula centroide mas cercano a vectores x
    
    Args:
        x (numpy array): vectores x
        centroid (numpy array): matriz de centroides
        
    Returns:
        numpy array: indice de centroide mas cercano
    """
    centroid=centroid[:,np.newaxis]
    distances=NormL2(centroid-x)
    min_distances=np.argmin(distances,axis=0)
    return min_distances


def Test():
    """
    Test metodo de centroide mas cercano
    
    Args:
        
    Returns:
        AssertionError: en caso de falla
    """
    x=np.array([[1,0,0],[1,2,3],[4,5,6],[7,8,9]])
    c=np.array([[1,0,0],[0,1,1]])
    dist=nearest_centroid(x,c)
    dist=np.round(dist)
    expected = np.array([0,1,1,1])
    np.testing.assert_equal(expected, dist)
    
    return


Test()
