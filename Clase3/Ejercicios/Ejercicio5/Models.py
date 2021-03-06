# -*- coding: utf-8 -*-
"""
Created on Sat May 15 18:32:52 2021

@author: MaxiT
"""
import numpy as np

class Model:
    """
    Clase base modelos
    """
    def __init__(self):
        pass
    
    def fit(x,y):
        return None
    
    def predict(x):
        return None
    


class LinearRegression(Model):
    """
    Modelo de aproximación con regresión lineal
    """
    _W=None
    
    def __init__(self):
        pass
    
    def fit(self, x, y):
        """
        Función de ajuste de W
        
        Args:
            x (numpy array): array de valores de entrada
            y (numpy array): array de valores de salida

        """
        x=x[:,np.newaxis]
        y=y[:,np.newaxis]
        self._W= np.linalg.inv(np.transpose(x)@x)@np.transpose(x)@y

        return y
    
    def predict(self,x):
        """
        Función de predicción
        
        Args:
            x (numpy array): array de valores de entrada
            
        Returns
            numpy array: array de predicciones

        """
        x=x[np.newaxis,:]
        y=np.transpose(self._W)@x
        return y
    
    
class AffineLinearRegression(Model):
    """
    Modelo de aproximación con regresión afín a la lineal
    """
    _W=None
    
    def __init__(self):
        pass
    
    def fit(self, x, y):
        """
        Función de ajuste de W
        
        Args:
            x (numpy array): array de valores de entrada
            y (numpy array): array de valores de salida

        """
        x=x[:,np.newaxis]
        x=np.append(x, np.ones(shape=(x.shape[0],1)), axis = 1)
        y=y[:,np.newaxis]
        self._W= np.linalg.inv(np.transpose(x)@x)@np.transpose(x)@y

        return y
    
    def predict(self,x):
        """
        Función de predicción
        
        Args:
            x (numpy array): array de valores de entrada
            
        Returns
            numpy array: array de predicciones

        """
        x=x[np.newaxis,:]
        x=np.append(x,np.ones(shape=(1,x.shape[1])), axis = 0)
        y=np.transpose(self._W)@x
        return y
    
    
def testModel():
    """
    Funcion de testeo de modelos de regresión lineal y afín a la lineal
    """
    model_linear = LinearRegression()
    model_linear.fit(np.array([1,2,3]),np.array([2,4,6]))
    y=model_linear.predict(np.array([8]))
    np.testing.assert_almost_equal(y,16)
    
    model_affine = AffineLinearRegression()
    model_affine.fit(np.array([1,2,3]),np.array([4,7,10]))
    y2=model_affine.predict(np.array([8]))
    np.testing.assert_almost_equal(y2,25)
    