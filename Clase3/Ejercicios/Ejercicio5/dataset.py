# -*- coding: utf-8 -*-
"""
Created on Sat May  1 18:03:02 2021

@author: MaxiT
"""

import numpy as np
import errno
import os
import sys

class Dataset:
    """
    Clase para cargar los datos de income
    """
    _instance=None
    _dataset=None
    
    def __new__(cls,path):
        if Dataset._instance is None:
            Dataset._instance=super(Dataset,cls).__new__(cls)
            return Dataset._instance
        else:
            return Dataset._instance
    
    def __init__(self,path):
        if os.path.isfile(path):
            structure = [('income', np.float64),('hapiness', np.float64)]
            with open(path, encoding="utf8") as data_csv:
                data_gen = ((float(line.split(',')[1]), \
                             float(line.split(',')[2])) \
                            for i, line in enumerate(data_csv) if i != 0)
                
                self._dataset=np.fromiter(data_gen,structure)
        else:
            raise FileNotFoundError( errno.ENOENT, \
                                     os.strerror(errno.ENOENT), \
                                     path)
        pass
        
    def get_dataset(self):
        """
        Funcion para obtener el dataset

        Returns
            numpy array: dataset.

        """
        return self._dataset
        
    def train_test_split(self, percentage):
        """
        Función para obtener una partición del dataset en entrenamiento y 
        testeo
        
        Args:
            percentage (uint): porcentaje de dataset para entrenamiento

        Returns
            numpy array: dataset entrenamiento.
            numpy array: dataset testeo.

        """
        dataset_length=self._dataset.shape[0]
        train_length=round(dataset_length*percentage)
    
        perm_idx=np.random.permutation(dataset_length)
        
        train_dataset=self._dataset[perm_idx[:train_length]]
        test_dataset=self._dataset[perm_idx[train_length:]]
    
        return train_dataset, test_dataset

def testDataset():
    """
    Funcion de testeo de la clase Dataset.
    """
    dataset=Dataset('./data/income.csv') 
    print(dataset.get_dataset()[5])
    data_train,data_test = dataset.train_test_split(0.9)
    print(data_train.shape)
    del dataset
    try:
        dataset=Dataset('./data/not_existing.csv') 
    except:
        print("Unexpected error:", sys.exc_info()[0])
    
    return





