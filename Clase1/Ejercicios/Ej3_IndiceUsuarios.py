# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:51:30 2021

@author: MaxiT
"""
import numpy as np

class Indexing:
    """ 
    Clase que implementa los indices para identificadores de ususarios
    
    """
    
    id2idx = None
    """ Array de indices / identificadores """
    
    def __init__(self, id_usuarios):
       
        self.id2idx=np.ones(id_usuarios.max()+1,dtype=np.int32)*-1
        
        indexes = np.unique(id_usuarios, return_index=True)[1]
        id_usuarios_unique=np.array([id_usuarios[index] for index in \
                                      sorted(indexes)])
        
        self.id2idx[id_usuarios_unique]=np.arange(0,id_usuarios_unique.size)
        
        pass
    
    def get_users_id(self, idx):
        """
        Obtiene el id de un usuario a partir de su indice
        
        Args:
            idx (int): indice del usuario
            
        Returns:
            int: id del usuario. -1 si no existe
        """
        usuario_id = np.argwhere(self.id2idx==idx)
        if(usuario_id.size==0):
            usuario_id=-1
        return int(usuario_id)
    
    def get_users_idx(self, user_id):
        """
        Obtiene el indice de un usuario a partir de su id
        
        Args:
            user_id (int): id del usuario
            
        Returns:
            int: indice del usuario. -1 si no existe
        """
        if((user_id+1)>self.id2idx.size):
            idx=-1
        else:
            idx=self.id2idx[user_id]
        return idx
        

def Test():
    """ 
    Test de clase Indexing
    
    """
    a = np.array([15, 12, 14, 10, 1, 2, 1])
    a_index=Indexing(a)
    np.testing.assert_equal(a_index.get_users_idx(15),0)
    np.testing.assert_equal(a_index.get_users_idx(1),4)
    np.testing.assert_equal(a_index.get_users_idx(7),-1)
    np.testing.assert_equal(a_index.get_users_idx(16),-1)
    np.testing.assert_equal(a_index.get_users_id(0),15)
    np.testing.assert_equal(a_index.get_users_id(4),1)
    np.testing.assert_equal(a_index.get_users_id(7),-1)
    return


