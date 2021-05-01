# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:35:51 2021

@author: MaxiT
"""
import numpy as np
import math
import sys

def NormL0(arr):
    """
    Calcula la norma L0 de cada vector fila de una matriz
    
    Args:
        array (numpy array): matriz de entrada
        
    Returns:
        numpy array: vector con normas L0
    """
    l0=np.sum(arr!=0,axis=1)
    
    return l0

def NormL1(arr):
    """
    Calcula la norma L1 de cada vector fila de una matriz
    
    Args:
        arr (numpy array): matriz de entrada
        
    Returns:
        numpy array: vector con normas L1
    """
    l1= np.sum(np.abs(arr),axis=1)
    return l1

def NormL2(arr):
    """
    Calcula la norma L2 de cada vector fila de una matriz
    
    Args:
        arr (numpy array): matriz de entrada
        
    Returns:
        numpy array: vector con normas L2
    """
    l2= np.power(np.sum(np.power(np.abs(arr),2),axis=1),1/2)
    return l2


def NormLp(arr,p):
    """
    Calcula la norma L2 de cada vector fila de una matriz
    
    Args:
        arr (numpy array): matriz de entrada
        p (int): norma a calcular
        
    Returns:
        numpy array: vector con normas Lp
    """
    lp= np.power(np.sum(np.power(np.abs(arr),p),axis=1),1/p)
    return lp


def NormLinf(arr):
    """
    Calcula la norma L infinito de cada vector fila de una matriz
    
    Args:
        arr (numpy array): matriz de entrada
        
    Returns:
        numpy array: vector con normas L inifnito
    """
    linf= np.max(np.abs(arr),axis=1)
    return linf


def Test():
    """
    Test de todos los metodos de normas
    
    Args:
        
    Returns:
        AssertionError: en caso de falla
    """
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected = np.array([4, 4])
    result = NormL0(a)
    np.testing.assert_equal(expected, result)

    a = np.array([[1, 0, 0, 4], [5, 6, 0, 8]])
    expected = np.array([2, 3])
    result = NormL0(a)
    np.testing.assert_equal(expected, result)


    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected = np.array([10, 26])
    result = NormL1(a)
    np.testing.assert_equal(expected, result)

    a = np.array([[-1, -2, -3, -4], [-5, -6, -7, -8]])
    expected = np.array([10, 26])
    result = NormL1(a)
    np.testing.assert_equal(expected, result)

    a = np.array([[1, 2], [3, 4]])
    expected = np.array([math.sqrt(5), math.sqrt(25)])
    result = NormL2(a)
    np.testing.assert_allclose(expected, result)

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected = np.array([4, 8])
    result = NormLinf(a)
    np.testing.assert_equal(expected, result)
    
    return
