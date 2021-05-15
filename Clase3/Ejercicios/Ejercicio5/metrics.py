# -*- coding: utf-8 -*-
"""
Created on Sat May 15 18:14:22 2021

@author: MaxiT
"""

import numpy as np

class BaseMetric:
    """
    Clase base métricas
    """
    def __init__(self):
        pass

    def __call__(self,true, predicted):
        pass
    
    
class MSE(BaseMetric):
    """
    Metrica MSE
    """
    def __init__(self):
        pass

    def __call__(self,true, predicted):
        """
        Calculo de MSE
        
        Args:
            true (numpy array): array de valores verdaderos
            predicted (numpy array): array de valores predichos

        Returns
            float: error cuadrático medio

        """
        mse= np.sum(np.power((predicted - true),2))/ predicted.shape[0]
        return mse
    
    
def testMSE():
    """
    Funcion de testeo de la clase MSE.
    """
    mse=MSE()
    error=mse(np.array([1,1,1]),np.array([0,1,2]))
    np.testing.assert_equal(error,2/3)