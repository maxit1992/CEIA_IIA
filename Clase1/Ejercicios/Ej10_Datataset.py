# -*- coding: utf-8 -*-
"""
Created on Sat May  1 18:03:02 2021

@author: MaxiT
"""

import numpy as np
import pandas as pd
import pickle as pkl
import os



class rating_dataset:
    """
    Clase que inicializa los datos del dataset ratings desde pkl o csv
    """
    
    instance=None
    dataset=None
    
    def __new__(cls,path):
        if rating_dataset.instance is None:
            rating_dataset.instance=super(rating_dataset,cls).__new__(cls)
            return rating_dataset.instance
        else:
            return rating_dataset.instance
    
    def __init__(self,path):
        if os.path.isfile(path+'ratings.pkl'):
            
            with open(path+'ratings.pkl','rb') as file:
                self.dataset=pkl.load(file)
                
        else:
            structure = [('userId', np.uint32),
                         ('movieId', np.uint32),
                         ('rating', np.float32),
                         ('timestmap', np.uint32)]
            data=pd.read_csv('./datos/ratings.csv', delimiter=',')
            #Codigo obtenido de stackoverflow para pasar pandas Dataframe
            #a numpy array estructurado
            s = data.dtypes
            res2 = np.array([tuple(x) for x in data.values], \
                            dtype=list(zip(s.index, s)))
            
            self.dataset=np.array(res2,dtype=structure)
            
            with open(path+'ratings.pkl','wb') as file:
                pkl.dump(self.dataset,file,protocol=pkl.HIGHEST_PROTOCOL)
                
            pass
        
    def get_dataset(self):
        """
        Funcion para obtener el dataset de ratings

        Returns
        -------
        numpy array: dataset de ratings.

        """
        return self.dataset
        
    

def Test():
    """
    Funcion de testeo de la clase rating_dataset.
    Imprime la 6ta fila
    """
    rating=rating_dataset('./datos/') 
    print(rating.get_dataset()[5])
    return





