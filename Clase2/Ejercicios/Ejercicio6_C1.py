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

def centroid_distance(x,centroid):
    """
    Calcula la distancia de vectores x a los centroides
    
    Args:
        x (numpy array): vectores x
        centroid (numpy array): matriz de centroides
        
    Returns:
        numpy array: vector con distancias
    """
    centroid=centroid[:,np.newaxis]
    distances=NormL2(centroid-x)
    
    return distances


def Test():
    """
    Test metodo de distancia a centroides
    
    Args:
        
    Returns:
        AssertionError: en caso de falla
    """
    x=np.array([[1,2,3],[4,5,6],[7,8,9]])
    c=np.array([[1,0,0],[0,1,1]])
    dist=centroid_distance(x,c)
    dist=np.round(dist)
    expected = np.array([[4, 8,13],[2,8,13]])
    np.testing.assert_equal(expected, dist)
    
    return

Test()
