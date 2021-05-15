# -*- coding: utf-8 -*-
"""
Created on Sat May 15 20:35:58 2021

@author: MaxiT
"""

import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset
from Models import LinearRegression,AffineLinearRegression
from metrics import MSE


#Dataset load and partition
dataset = Dataset('data\income.csv')
dataset_train, dataset_test = dataset.train_test_split(0.9)

#Separate x and y
x_train = dataset_train['income']
y_train = dataset_train['hapiness']

x_test = dataset_test['income']
y_test = dataset_test['hapiness']


##############################################################################

#Linear model aproximation
linear_model=LinearRegression()
linear_model.fit(x_train,y_train)
y_predicted=linear_model.predict(x_test)

mse=MSE()

linear_mse=mse(y_test,y_predicted)
print("Error cuadratico medio para regresión lineal: {}".format(linear_mse))

x_predicted=x_test[np.argsort(x_test)]
y_predicted=y_predicted[0]
y_predicted=y_predicted[np.argsort(x_test)]

fig, axs = plt.subplots(1)
axs.scatter(x_test,y_test,color='green',label='Real')
axs.plot(x_predicted,y_predicted,color='red',label='Predicted')
axs.legend()
axs.set_title("Linear Regression aproximation")
axs.set_ylabel("Hapiness")
axs.set_xlabel("Income")
plt.xlim([0, 8])
plt.ylim([0, 8])

##############################################################################


#Affine linear aproximation
affine_model=AffineLinearRegression()
affine_model.fit(x_train,y_train)
y_predicted=affine_model.predict(x_test)

mse=MSE()

affine_mse=mse(y_test,y_predicted)
print("Error cuadratico medio para regresión afín lineal: {}".format(affine_mse))

x_predicted=x_test[np.argsort(x_test)]
y_predicted=y_predicted[0]
y_predicted=y_predicted[np.argsort(x_test)]

fig, axs = plt.subplots(1)
axs.scatter(x_test,y_test,color='green',label='Real')
axs.plot(x_predicted,y_predicted,color='red',label='Predicted')
axs.legend()
axs.set_title("Linear Regression aproximation")
axs.set_ylabel("Hapiness")
axs.set_xlabel("Income")
plt.xlim([0,8])
plt.ylim([0,8])