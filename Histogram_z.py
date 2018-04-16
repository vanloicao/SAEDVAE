# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 10:55:40 2017

@author: VANLOI
"""
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab 
import numpy as np

"******************** Histogram z, z_mu and z_var ************************"
"""This program is to plot some subfigure together. For plotting each individual,
we put it in Plot_Curves and call in Sda_VAE2"""

path = "D:/Python_code/SDA-02/Results/Exp_Hidden/"
   
def Plot_histogram(alpha):
    #list_ep = [0, 5, 15, 30]
    list_ep = [0, 10, 20]
    plt.subplots(ncols=1, nrows = 3, figsize=(12, 3))
    
    num = 1
    for ep in list_ep:
      
      z_train = np.genfromtxt(path + "Visualize_histogram/" + "z_train_" + str(ep) + "_" + str(alpha) + ".csv", delimiter=",")
      x = z_train[:,0]
      mu    = np.mean(x)
      sigma = np.std(x)
       
      fig = plt.subplot(1, 3, num) 
       # the histogram of the data
      n, bins, patches = plt.hist(x, 20, normed=1, facecolor='green', alpha=0.5)
      # add a 'best fit' line
      y = mlab.normpdf( bins, mu, sigma)    
      
#      title = r'$\mathrm{Histogram\ of\ z}\ (\mathrm{\alpha\ = ' + str(alpha) + ',}\ \mathrm{epoch\ = }'+ str(ep) + ')$'      plt.plot(bins, y, 'r--', linewidth=1)
      title = r'$\mathrm{epoch\ = }'+ str(ep) + '$'
      xlabel = r'$\mathrm{z_{0}}$'
      
      
      #plt.text(1.0, 0.35, r'$\mathrm{epoch\ = }'+ str(ep) + '$', fontsize = 20)
      plt.xlabel(xlabel, fontsize=24)
      if (num == 1):
        plt.ylabel('Probability', fontsize=18)
    
      plt.title(title, fontsize=22)
      plt.axis([-3, 3, 0, max(y)+ 0.1*max(y)])
#      plt.grid(True)
      plt.yticks(fontsize=13)
      plt.xticks(fontsize=13)
      num = num + 1
      
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.0) 
#    plt.savefig(path + "Visualize_histogram/" + "his_z_" + str(alpha) + ".pdf" )
    plt.show()      
      
Plot_histogram(1)      
      

