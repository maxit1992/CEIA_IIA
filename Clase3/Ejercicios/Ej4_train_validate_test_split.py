# -*- coding: utf-8 -*-
"""
Created on Sun May  9 01:19:02 2021

@author: MaxiT
"""
import numpy as np

def train_validate_test_split(x,y):
    """
    Divide datasets entre entrenamiento, validacion y testeo (70/20/10)
    
    Args:
        x (numpy array): dataset a dividir
        y (numpy array): indices a dividir
        
    Returns:
        numpy array: dataset de entrenamiento
        numpy array: indices de entrenamiento
        numpy array: dataset de validacion
        numpy array: indices de validacion
        numpy array: dataset de testeo
        numpy array: indices de testeo
    """
    perm_idx=np.random.permutation(x.shape[0])
    train_x=x[perm_idx[:round(x.shape[0]*0.7)]]
    train_y=y[perm_idx[:round(y.shape[0]*0.7)]]
    validate_x=x[perm_idx[round(x.shape[0]*0.7):round(x.shape[0]*0.9)]]
    validate_y=y[perm_idx[round(y.shape[0]*0.7):round(y.shape[0]*0.9)]]
    test_x=x[perm_idx[round(x.shape[0]*0.9):]]
    test_y=y[perm_idx[round(y.shape[0]*0.9):]]
    
    return train_x, train_y , \
        validate_x, validate_y, \
            test_x, test_y

def test():
    """
    Test de metodo de division de dataset según dimensión resultante
    """
    x=np.random.normal(0,1,[300,10])
    y= np.random.uniform(size=300)>0.5
    tr_data,tr_label,va_data, val_label,te_data, te_label = \
        train_validate_test_split(x,y)
    np.testing.assert_equal(tr_data.shape, np.array([210,10]))
    np.testing.assert_equal(va_data.shape, np.array([60,10]))
    np.testing.assert_equal(te_data.shape, np.array([30,10]))
    return

test()
