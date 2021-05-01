# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:35:33 2021

@author: MaxiT
"""

import numpy as np

def Performance(truth,prediction):
    """
    Calcula la precision, recall y exactitud
    
    Args:
        truth (numpy array): Array con valores verdaderos
        prediction (numpy array): Array con valores de predicción
        
    Returns:
        float: precision
        float: recall
        float: exactitud
        
    """
    TP = np.sum(truth & prediction)
    TN = np.sum((1-truth) & (1-prediction))
    FP = np.sum((1-truth) & (prediction))
    FN = np.sum((truth) & (1-prediction))
    precision= TP / (TP+FP)
    recall= TP / (TP+FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return precision, recall, accuracy

    
def Test():
    """
    Test metodo de calculo de metricas
    """
    verdad=np.array([1,1,0,1,1,1,0,0,0,1])
    pred=np.array([1,1,1,1,0,0,1,1,0,0])
    prec,rec,acc=Performance(verdad,pred)
    np.testing.assert_equal(prec,0.5)
    np.testing.assert_equal(rec,0.5)
    np.testing.assert_equal(acc,0.4)
    
    return