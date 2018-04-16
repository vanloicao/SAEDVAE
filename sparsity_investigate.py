# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 12:53:55 2017

@author: VANLOI
"""
from ProcessingData import load_data
from Plot_Curves import plot_sparsity_auc_bar, plot_sparsity_auc, plot_dimension_auc
import numpy as np
import matplotlib.pyplot as plt


#datasets =  ["PageBlocks", "WPBC", "PenDigits", "GLASS", "Shuttle", "Arrhythmia",\
#            "CTU13_10", "CTU13_08","CTU13_09","CTU13_13",\
#            "Spambase", "UNSW", "NSLKDD", "InternetAds"]          
#Compute the sparsity of datasets     
def sparsity_measurement(list_data):
    
    sparsity_data = dict()
    for data in list_data:  
        train_set,_,_ = load_data(data)
        m = train_set.shape[1]
        n = train_set.shape[0]
        d = (train_set == 0).sum()
        sparsity = round((float(d)/(m*n)),2)
        
        print(data + ":    %0.2f" %sparsity)
        sparsity_data[data] = sparsity
        print(sparsity_data)
    return sparsity_data   
    
#sparsity_measurement(datasets)  

#plot AUC vs the sparsity of datasets
datasets = np.asarray(["PageBlocks", "WPBC", "PenDigits", "GLASS", "Shuttle", "Arrhythmia",\
                       "CTU13-10", "CTU13-08","CTU13-09","CTU13-13",\
                       "Spambase", "UNSW-NB15", "NSL-KDD", "InternetAds"]) 

spa_dic = {'PageBlocks': [0.0, 10], 'WPBC': [0.02, 32], 'PenDigits': [0.13, 16],\
           'GLASS': [0.18, 9], 'Shuttle': [0.22, 9], 'Arrhythmia': [0.5, 259],\
           'CTU13-10': [0.71,38], 'CTU13-08': [0.73, 40],'CTU13-09': [0.73, 41],'CTU13-13':[0.73, 40],\
           'Spambase': [0.81,57], 'UNSW-NB15':[0.84, 196], 'NSL-KDD':[0.88,122], 'InternetAds':[0.99,1558]}
           
           
spa_dim  =  np.asarray([[0,  0.0,  10],
                        [1,  0.02, 32],
                        [2,  0.13, 16],
                        [3,  0.18, 9 ],
                        [4,  0.22, 9],
                        [5,  0.5,  259],
                        [6,  0.71, 38],
                        [7,  0.73, 40],
                        [8,  0.73, 41],
                        [9,  0.73, 40],
                        [10, 0.81, 57],
                        [11, 0.84, 196],
                        [12, 0.88, 122],
                        [13, 0.99, 1558]])           
           
                        
#path = "D:/Python_code/SDA-02/Results/Exp_Hidden/Sparsity_dimension_auc/"
#method = ["DAE", "Shrink_DAE", "Shrink" ,"VAE"]
#
#for m in method:
#    auc_input   = np.genfromtxt(path + "AUC_Input.csv", delimiter=",")
#    auc_hidden  = np.genfromtxt(path + "AUC_Hidden_"+ m +".csv", delimiter=",")
#    improve_auc = auc_hidden - auc_input
#    #plot_sparsity_auc_bar(datasets, improve_auc, spa_dim, m, path)
#    plot_sparsity_auc(datasets, improve_auc, spa_dim, m, path)
#    plot_dimension_auc(datasets, improve_auc, spa_dim, m, path)
    
    
#from matplotlib import rcParams
##rcParams['mathtext.default'] = 'regular'
#rcParams['text.usetex']=True    
"*************************** Plot 3 methods together ***********************"    
path = "D:/Python_code/SDA-02/Results/Exp_Hidden/Sparsity_dimension_auc/LoF_10/"
method = ["DAE", "Shrink" ,"VAE"]
label = ["DAE", "SAE" ,"DVAE"]

def plot_sparsity_all(data, spa_score, method, label, path):
  plt.subplots(ncols=3, nrows = 1, figsize=(8, 9))

  num = 0
  for m, l in zip(method, label):
    auc_input   = np.genfromtxt(path + "AUC_Input.csv", delimiter=",")
    auc_hidden  = np.genfromtxt(path + "AUC_Hidden_"+ m +".csv", delimiter=",")
    improve_auc = auc_hidden - auc_input
    
    
    #Using CTU13-13 to demonstrate for four CTU13 datasets, [6,7,8,9]
    id_data = [0, 1, 2, 4, 5, 9, 10, 11, 12, 13]
    improve_auc = improve_auc[id_data]
    spa_score   = spa_dim[id_data]        #sparsity score
    labels = data[id_data]  


    LOF     = improve_auc[:,2:3]
    CEN     = improve_auc[:,3:4]
    DIS     = improve_auc[:,4:5]
    KDE     = improve_auc[:,5:6]
    SVM05   = improve_auc[:,6:7]
    SVM01   = improve_auc[:,7:8]    
    
    
    fig = plt.subplot(3, 1, num+1)  

    
    plt.ylim([-0.38, 0.38])
    plt.xlim([-0.02, max(spa_score[:,1]) + 0.01])
    plt.xticks(spa_score[:,1], spa_score[:,1], rotation=90)
    
  
    #'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
    plt.plot(spa_score[:,1], LOF,   'b-s', ms=5, mec="b", label ='LOF', markevery = 1)
    plt.plot(spa_score[:,1], CEN,   'r-p', ms=5, mec="r", label ='CEN', markevery = 1)
    plt.plot(spa_score[:,1], DIS,   'g-^', ms=5, mec="g", label= 'MDIS', markevery = 1)
    plt.plot(spa_score[:,1], KDE,   'y-d', ms=5, mec="y", label ='KDE', markevery = 1)
    plt.plot(spa_score[:,1], SVM05, 'm-o', ms=5, mec="m", label =r'OCSVM$_{\nu = 0.5}$', markevery = 1)
    plt.plot(spa_score[:,1], SVM01, 'c-x', ms=5, mec="c", label= r'OCSVM$_{\nu = 0.1}$', markevery = 1)
    
    #xx-small, x-small, small, medium, large, x-large, xx-large  
    
    #plt.yticks(fontsize=12)
    fig.yaxis.grid(True)    
    fig.axes.get_xaxis().set_visible(False)   
    
    if (num == 2):
      fig.axes.get_xaxis().set_visible(True) 
      for xtick in fig.xaxis.get_major_ticks():
        xtick.label.set_fontsize(14)
      plt.xlabel('Sparsity of data', fontsize=17)
      
#    if num == 1:
#      plt.ylabel('$\mathrm{AUC}_{\mathrm{hidden}}$' + ' - ' + '$\mathrm{AUC}_{\mathrm{input}}$\n' + '$' + l + '$', fontsize=15)
#    else:
#      plt.ylabel('$' + l + '$' , fontsize=15)
      
#    plt.ylabel('AUC-DIFF'+ r'$_{'  + l + '}$' , fontsize=16)
    plt.ylabel('AUC-DIFF (' + l + ')' , fontsize=17)          
    yticks = fig.yaxis.get_major_ticks() 
    yticks[0].label1.set_visible(False)   #Disable the last yticks
    yticks[-1].label1.set_visible(False)   #Disable the last yticks
    for ytick in yticks:
      ytick.label.set_fontsize(14)  
        
    
    if num == 0:
      plt.legend(bbox_to_anchor=(1.005, 1.02), ncol=3, fontsize = 'x-large')
      fig.twiny()
      plt.xlabel('Datasets', fontsize=17)
      plt.xlim([-0.02, max(spa_score[:,1])+0.01])
      plt.xticks(spa_score[:,1], labels, fontsize = 13, rotation=90)
      
    num = num + 1
      
  plt.tight_layout()
  plt.subplots_adjust(wspace=0.025, hspace=0.02) 
  plt.savefig(path + "auc_sparsity" + "_line10.pdf")
  plt.show() 



"****************** Plot dimension versus auc on three method ****************"
def plot_dimension_all(data, spa_score, method, label, path):
  
  plt.subplots(ncols=3, nrows = 1, figsize=(8, 9))

  num = 0
  for m, l in zip(method, label):
    auc_input   = np.genfromtxt(path + "AUC_Input.csv", delimiter=",")
    auc_hidden  = np.genfromtxt(path + "AUC_Hidden_"+ m +".csv", delimiter=",")
    improve_auc = auc_hidden - auc_input
    

    #Using CTU13-13 to demonstrate for four CTU13 datasets, [6,7,8,9]
    id_data = [0, 1, 2, 4, 5, 9, 10, 11, 12, 13] #list id datasets
    improve_auc = improve_auc[id_data]
    spa_dim     = spa_score[id_data]             
    #insert id, sparsity and dimension
    improve_auc = np.insert(improve_auc, [0], spa_dim, axis=1) 
    #sort by number of dimension
    improve_auc = improve_auc[improve_auc[:,2].argsort()]
    #get dimension
    dim     = np.asanyarray(improve_auc[:,2], dtype = int)
    #get id of data
    idx     = np.asanyarray(improve_auc[:,0], dtype = int)    
    labels1 = []    
    for d in idx:
      #select data for visulizing
      labels1 = np.append(labels1, data[d]) 


    LOF     = improve_auc[:,5:6]
    CEN     = improve_auc[:,6:7]
    DIS     = improve_auc[:,7:8]
    KDE     = improve_auc[:,8:9]
    SVM05   = improve_auc[:,9:10]
    SVM01   = improve_auc[:,10:11]   
    
    
    fig = plt.subplot(3, 1, num+1)  
    
    plt.ylim([-0.38, 0.38])
    log_dim = np.round(np.log(dim+1-9), 2)
    plt.xlim([-0.1, max(log_dim)+0.1])
    plt.xticks(log_dim, dim, rotation=90)    

    #'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
    plt.plot(log_dim, LOF,   'b-s', ms=5, mec="b", label ='LOF', markevery = 1)
    plt.plot(log_dim, CEN,   'r-p', ms=5, mec="r", label ='CEN', markevery = 1)
    plt.plot(log_dim, DIS,   'g-^', ms=5, mec="g", label= 'MDIS', markevery = 1)
    plt.plot(log_dim, KDE,   'y-d', ms=5, mec="y", label ='KDE', markevery = 1)
    plt.plot(log_dim, SVM05, 'm-o', ms=5, mec="m", label =r'OCSVM$_{\nu = 0.5}$', markevery = 1)
    plt.plot(log_dim, SVM01, 'c-x', ms=5, mec="c", label= r'OCSVM$_{\nu = 0.1}$', markevery = 1)
    
    fig.yaxis.grid(True)    
    fig.axes.get_xaxis().set_visible(False)   
    
    if (num == 2):
      fig.axes.get_xaxis().set_visible(True) 
      for xtick in fig.xaxis.get_major_ticks():
        xtick.label.set_fontsize(14)
      plt.xlabel('Dimension (in log scale) of data', fontsize=17)
      
#    if num == 1:
#      plt.ylabel('$\mathrm{AUC}_{\mathrm{hidden}}$' + ' - ' + '$\mathrm{AUC}_{\mathrm{input}}$\n' + '$' + l + '$', fontsize=15)
#    else:
#      plt.ylabel('$' + l + '$' , fontsize=15)
      
#    plt.ylabel('AUC-DIFF'+ r'$_{'  + l + '}$' , fontsize=16)
    plt.ylabel('AUC-DIFF (' + l + ')' , fontsize=17)        
    yticks = fig.yaxis.get_major_ticks() 
    yticks[0].label1.set_visible(False)   #Disable the last yticks
    yticks[-1].label1.set_visible(False)   #Disable the last yticks
    for ytick in yticks:
      ytick.label.set_fontsize(14)  
    
    if num == 0:
      #xx-small, x-small, small, medium, large, x-large, xx-large 
      plt.legend(bbox_to_anchor=(1.005, 1.02), ncol=3, fontsize = 'x-large')
      fig.twiny()
      plt.xlabel('Datasets', fontsize=17)
      plt.xlim([-0.1, max(log_dim)+0.1])
      plt.xticks(log_dim, labels1, fontsize = 13, rotation=90)
    num = num + 1      

  plt.tight_layout()
  plt.subplots_adjust(wspace=0.025, hspace=0.02) 
  plt.savefig(path + "auc_dimension" + "_line10.pdf")
  plt.show() 


plot_sparsity_all(datasets, spa_dim, method, label, path)   
plot_dimension_all(datasets, spa_dim, method, label, path)   