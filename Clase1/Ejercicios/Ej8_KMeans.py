# -*- coding: utf-8 -*-
"""
Created on Sat May  8 18:18:41 2021

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

def k_means(x,n):
    """
    Clusteriza vectores x en n grupos
    
    Args:
        x (numpy array): vectores x
        n (int): cantidad de grupos
        
    Returns:
        numpy array: centroides
        numpy array: indice de grupos de vectores x
    """
    centroids_idx=np.random.randint(0,x.shape[0],n)
    centroids=x[centroids_idx,:]
    i=0
    prev_centroid=centroids*-1
    
    while i < 500 and np.array_equal(prev_centroid,centroids)==False:
        
        prev_centroid=centroids
        
        cluster_ids=nearest_centroid(x,centroids)
        x2=x[:,np.newaxis,:]
        x2=np.repeat(x2, n , axis=1)
        
        cluster_ids2=cluster_ids[:,np.newaxis,np.newaxis]
        cluster_ids2=np.repeat(cluster_ids2, n , axis=1)
        
        a=np.unique(cluster_ids2)
        a=a[np.newaxis,:,np.newaxis]
        a=(cluster_ids2==a)
        
        new_centroids_num=a*x2
        new_centroids_num = np.sum(new_centroids_num ,axis=0)
        
        new_centroids_den = np.sum(a, axis=0)
        centroids=new_centroids_num/new_centroids_den
        i=i+1
        
    return centroids,cluster_ids


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
    Test funcion de calculo de precision promedio de query
    """
    centroids=np.array([[7,0,0,0],[0,25,0,3],[0,0,16,0]])
    x, cluster_ids=synthetic_dataset(centroids,n_samples=300)
    
    centroids_kmeans,ids_kmeans=k_means(x,3)
    print('Original centroids:')
    print(centroids)
    print('Founded centroids:')
    print(centroids_kmeans)
    
    return

Test()

