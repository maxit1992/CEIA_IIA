# -*- coding: utf-8 -*-
"""
Created on Sun May  9 00:32:06 2021

@author: MaxiT
"""

import numpy as np

def replace_nan(x):
    """
    Reemplaza NaN por la media de la columna
    
    Args:
        X (numpy array): matriz de entrada
        
    Returns:
        numpy array: matriz con NaN reemplazados
    """
    valid=np.logical_not(np.isnan(x))
    col_avg=x.sum(axis=0,where=valid) /valid.sum(axis=0,where=valid) 
    col_avg=col_avg[np.newaxis,:]
    col_avg= np.repeat(col_avg,x.shape[0],axis=0)
    x[np.isnan(x)]=col_avg[np.isnan(x)]
    return x


def Test():
    """
    Test de metodo de reemplazo Nan por media columna
    """
    x=np.array([[1,9,np.nan],[np.nan,3,10],[np.nan,3,2]])
    x=replace_nan(x)
    expected=np.array([[1,9,6],[1,3,10],[1,3,2]])
    np.testing.assert_equal(x, expected)
    return

Test()
