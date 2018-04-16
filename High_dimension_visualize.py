# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 11:30:03 2017

@author: VANLOI
"""
import matplotlib.pyplot as plt


import numpy as np
from sklearn.manifold import TSNE
X = np.array([[0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 1,0], [0,1, 1, 1]])
def visualize_n_dimension(X, Y):  
  Z = np.concatenate((X, Y))
  Z_embedded = TSNE(n_components=2).fit_transform(Z)
  Z_embedded.shape
  
  X_embedded = Z_embedded[:100]
  Y_embedded = Z_embedded[100:]
  
  plt.plot(X_embedded[:,0], X_embedded[:,-1], 'o') 
  
  plt.plot(Y_embedded[:,0], Y_embedded[:,-1], 'x')
  
  plt.show()