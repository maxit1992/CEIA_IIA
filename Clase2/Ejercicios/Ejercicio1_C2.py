# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:48:59 2021

@author: MaxiT
"""
import numpy as np


def synthetic_dataset(centroids,n_samples=200,distancing=1):
    """
    Genera un dataset sintetico superponiendo ruido gaussiano de varianza 0.1
    
    Args:
        centroids (numpy array): matriz de centroides (pueden ser mas de 2)
        n_samples (int): cantidad de puntos en cluster
        distancing (int): distanciamiento opcional entre centroides
        
    Returns:
        numpy array: cluster de puntos
        numpy_array: vector de ids de pertenencia puntos
    """
    centroids=centroids*distancing
    data = np.repeat(centroids, n_samples / centroids.shape[0], axis=0)
    stddev=0.1
    noise=np.random.normal(0,stddev,data.shape)
    data=data+noise
    
    cluster_ids = np.repeat(np.arange(0,centroids.shape[0]), \
                            n_samples / centroids.shape[0], axis=0)
        
    return data, cluster_ids
    

def Test():
    """
    Test generacion dataset por dimension resultante
    
    Args:
        
    Returns:
        AssertionError: en caso de falla
    """
    centroids=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    cluster, cluster_ids=synthetic_dataset(centroids,n_samples=300)
    expected_cluster_dim = np.array([300, 4])
    expected_clusterid_dim= np.array([300])
    np.testing.assert_equal(expected_cluster_dim, cluster.shape)
    np.testing.assert_equal(expected_clusterid_dim, cluster_ids.shape)
    return


Test()