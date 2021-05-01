# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:23:22 2021

@author: MaxiT
"""
import numpy as np
import sys

def NormL2_sort(arr):
    """
    Ordena los vectores fila de una matriz según mayor Norma L2
    
    Args:
        arr (numpy array): matriz de entrada
        
    Returns:
        numpy_array: matriz ordenada
    """
    l2= np.power(np.sum(np.power(np.abs(arr),2),axis=1),1/2)
    arr_sort=arr[np.argsort(-l2),:]
    return arr_sort
    
def Test():
    """
    Test de metodos de ordenamiento segun norma L2
    
    Args:
        
    Returns:
        AssertionError: en caso de falla
    """
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    sorted_a = NormL2_sort(a)
    np.testing.assert_equal(np.array([[5, 6, 7, 8], [1, 2, 3, 4]]), sorted_a)

    a = np.array([[1, 2, 3, 4], [10, 11, 12, 13], [5, 6, 7, 8]])
    sorted_a = NormL2_sort(a)
    np.testing.assert_equal(np.array([[10, 11, 12, 13], [5, 6, 7, 8], \
                                      [1, 2, 3, 4]]), sorted_a)
    return
