# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 21:32:56 2017

@author: VANLOI
"""
import matplotlib.pyplot as plt
import numpy as np

def Plotting_querytime(times, data, name, path):
    x    = times[:,0]
    lof  = times[:,1]
    cen  = times[:,2] 
    dis  = times[:,3]
    kde  = times[:,4]
    svm05 = times[:,5]
    svm01 = times[:,6]
    re   = times[:,7]
    
    
    plt.figure(figsize=(6,5.5))
    ax = plt.subplot(111)
    #plt.title(name, fontsize=16)
    plt.xlim([200, 10200])
    plt.ylim([0.0, 1.2])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #{'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'};
    plt.plot(x, lof,  'm-o', ms=6, label = 'SAE-LOF')
    plt.plot(x, cen,  'g-^', ms=6, label = 'SAE-CEN')

    plt.plot(x, dis,  'r--', ms=6, label = 'SAE-MDIS')
    plt.plot(x, kde,  'c-d', ms=6, label = 'SAE-KDE')

    plt.plot(x, svm05,  'b-s', ms=6, label = r'SAE-OCSVM$_{\nu = 0.5}$')
    plt.plot(x, svm01,  'y->', ms=6, label = r'SAE-OCSVM$_{\nu = 0.1}$')    
    
    plt.plot(x, re,  'c--', ms=6, label = 'RE-Based SAE\nClassifier')
    
    plt.legend(frameon=False,bbox_to_anchor=(0.4, 1.02), ncol = 1, fontsize = 'x-large')
    plt.legend(loc='upper left', fontsize = 'large')

#    new_tick_locations = np.array([0.000, 0.0002, 0.0005, 0.001])
#    new_tick = ["%1.1e" % z for z in new_tick_locations]
#    ax.set_yticks(new_tick_locations)
#    ax.set_yticklabels(new_tick, fontsize=12)

    plt.ylabel(r'Query time per example (in milliseconds)', fontsize=16)
    plt.xlabel('Size of training set ', fontsize=16)
    plt.tight_layout()
    plt.savefig(path + data + "_querytime.pdf")
    plt.show() 
    
sizes = [500, 1000, 2000, 5000, 10000]  
#list_data =  ["CTU13_08", "UNSW", "NSLKDD"]
#list_name =  ["CTU13-08", "UNSW-NB15", "NSL-KDD"]

list_data =  ["CTU13_08", "CTU13_09", "CTU13_13", "UNSW", "NSLKDD"]
list_name =  ["CTU13-08", "CTU13-09", "CTU13-13", "UNSW-NB15", "NSL-KDD"]


def plotting_QT(list_data, list_name, sizes):
    path = "D:/Python_code/SDA-02/Results/Exp_Hidden/Time_Size_SAE_5000/"
    for data, name in zip(list_data, list_name):
        times = np.genfromtxt(path + data +  "_query_time.csv", delimiter=",")
        Plotting_querytime(times, data, name, path)    
plotting_QT(list_data, list_name, sizes)        