# -*- coding: utf-8 -*-
"""
Created on Sun May  9 01:19:02 2021

@author: MaxiT
"""
import numpy as np

def train_validate_test_split(x):
    """
    Divide datasets entre entrenamiento, validacion y testeo (70/20/10)
    
    Args:
        X (numpy array): dataset a dividir
        
    Returns:
        numpy array: dataset de entrenamiento
        numpy array: dataset de validacion
        numpy array: dataset de testeo
    """
    perm_idx=np.random.permutation(x.shape[0])
    train_dataset=x[perm_idx[:round(x.shape[0]*0.7)]]
    validate_dataset=x[perm_idx[round(x.shape[0]*0.7):round(x.shape[0]*0.9)]]
    test_dataset=x[perm_idx[round(x.shape[0]*0.9):]]
    
    return train_dataset,validate_dataset,test_dataset

def test():
    """
    Test de metodo de division de dataset según dimensión resultante
    """
    x=np.random.normal(0,1,[300,10])
    tr_data,va_data,te_data=train_validate_test_split(x)
    np.testing.assert_equal(tr_data.shape, np.array([210,10]))
    np.testing.assert_equal(va_data.shape, np.array([60,10]))
    np.testing.assert_equal(te_data.shape, np.array([30,10]))
    return

test()
