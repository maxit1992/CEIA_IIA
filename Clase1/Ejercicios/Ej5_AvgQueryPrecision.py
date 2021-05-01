# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:56:02 2021

@author: MaxiT
"""

import numpy as np

def avg_q_precision(q_id, truth_relevance):
    """ 
    Calculo de average query precision
    
    Args:
        q_id (numpy array): array con los queries id
        truth_relevance (numpy array): array de relevancia de los resultados
        
    Returns:
        float: precision promedio de la query
        
    """
    a = np.unique(q_id)
    
    q_id_expanded= np.zeros((q_id.size,a.size))
    q_id_expanded=q_id_expanded+q_id[:,np.newaxis]
    
    truth_expanded=np.zeros((truth_relevance.size,a.size))
    truth_expanded=truth_expanded+truth_relevance[:,np.newaxis]
    
    numerators= np.sum((q_id_expanded==a) & (truth_expanded==1),axis=0)
    dividers= np.sum(q_id_expanded==a,axis=0)
    avg_q=np.sum(numerators/dividers)/a.size
    
    return avg_q


def Test():
    """ 
    Test funcion de calculo de precision promedio de query
    """    
    q_id = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
    truth_relevance=np.array([True, False, True, False, True, True, True, \
                              False, False, False, False, False, True, False, \
                                  False, True] )
    np.testing.assert_equal(avg_q_precision(q_id,truth_relevance),0.5)
    
    return