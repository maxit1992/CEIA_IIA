# -*- coding: utf-8 -*-
"""
Created on Sun May  9 00:59:42 2021

@author: MaxiT
"""
import numpy as np

def filter_nan(x):
    """
    Elimina filas y columnas con NaN
    
    Args:
        X (numpy array): matriz de entrada
        
    Returns:
        numpy array: matriz con columnas y filas con NaN eliminadas
    """
    valid_cols=np.argwhere(\
                       np.logical_not(\
                          np.any(np.isnan(x),axis=0))).transpose()
        
    valid_rows=np.argwhere(np.logical_not(np.any(np.isnan(x),axis=1)))
    x_filtered=x[valid_rows,valid_cols]
    return x_filtered


def Test():
    """
    Test de metodo de reemplazo Nan por media columna
    """
    x=np.array([[1,9,10,14],[np.nan,3,10,20],[50,3,2,np.nan],[5,6,7,8]])
    #x=replace_nan(x)
    expected=np.array([[9,10],[6,7]])
    np.testing.assert_equal(x, expected)
    return




