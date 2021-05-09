# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:17:49 2021

@author: MaxiT
"""

import numpy as np
import matplotlib.pyplot as plt

# si pdf(x)=3*x^2 => cdf(x)=F(x)=x^3 => (F-1)(x)=raiz3(x)
# Si aplico y=(F-1)(u) con u~U[0,1] => y tiene pdf 3*x^2

n=10000
x=np.random.uniform(0,1,n)
y=np.power(x,1/3)

#Grafico histograma para chequear PDF
n_bins=100
bins = np.arange(0,1 + 1/n_bins,1/n_bins)
plt.hist(y, bins = bins,density=True) 
plt.title("histogram") 
plt.show()
#Se observa que efectivamente la VA Y tiene pdf 3*x^2
