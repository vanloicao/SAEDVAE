# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:47:04 2017

@author: VANLOI
"""
import numpy as np
from BaseOneClass import DensityBasedOneClassClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from ProcessingData import load_data, normalize_data 
from Plot_Curves import Plot_AUC_Bandwidth
from LOF import LocalOutlierFactor

import matplotlib.pyplot as plt
import matplotlib.font_manager

def investigate_svm(train_set, test_set, actual, scale, gamma, nu):
    train_set, test_set = normalize_data(train_set, test_set, scale)   
    clf_svm = svm.OneClassSVM( nu=nu, kernel="rbf", gamma=gamma)
    clf_svm.fit(train_set)    
    
    predictions_svm  = clf_svm.decision_function(test_set)
    FPR_svm, TPR_svm, thresholds_svm = roc_curve(actual, predictions_svm)
    auc_svm = auc(FPR_svm, TPR_svm)
    return auc_svm
    
def investigate_kde(train_set, test_set, actual, scale, bw):
    #  ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
    KDE = DensityBasedOneClassClassifier(bandwidth = bw, 
                                         kernel="gaussian", 
                                         metric="euclidean",
                                         scale = scale)
    KDE.fit(train_set)
    predictions_kde = KDE.get_density(test_set)
    FPR_kde, TPR_kde, thresholds_kde = roc_curve(actual, predictions_kde)
    auc_kde = auc(FPR_kde, TPR_kde)
    return auc_kde

def Investigate_lof(train_set, test_set, actual, scale, k):
  
    train_set, test_set =  normalize_data(train_set, test_set, scale)
    neighbors = (int)(len(train_set)*k)
    clf_lof = LocalOutlierFactor(n_neighbors=neighbors)
    clf_lof.fit(train_set)
    predict = clf_lof._decision_function(test_set)
    FPR, TPR, thresholds = roc_curve(actual, predict)
    lof = auc(FPR, TPR)    
    return lof



"************ Investigate SVM and KDE with many bandwidth values *************"    
def investigate_bandwidth(norm, data, method, path):
    for m in method:
      train_z = np.genfromtxt(path + m + data + "_train_z.csv", delimiter=",")
      test_z  = np.genfromtxt(path + m + data + "_test_z.csv", delimiter=",")

      _, _, actual = load_data(data)
      #gamma = 1/features =  1/(2*bw*bw) 
      #bw = sqrt(features/2)
      n_features = train_z.shape[1]
      AUC = np.empty([0,4])    
      steps = 10
      n     = 5.0
    
      bw = np.asarray([i for i in np.linspace(0.0, n, steps+1)])
      gamma = 1.0/(2.0*bw*bw)
      
      for b, g in zip(bw, gamma):
          if b> 0:
              svm05 = investigate_svm(train_z, test_z, actual, norm, g, 0.5)
              svm01 = investigate_svm(train_z, test_z, actual, norm, g, 0.1)
              kde   = investigate_kde(train_z, test_z, actual, norm, b)
        
              AUC = np.append(AUC, [b, kde, svm05, svm01]) 
    
      AUC = np.reshape(AUC, (-1,4))  
      Plot_AUC_Bandwidth(AUC, data, n, n_features, path)      
"*****************************************************************************"






"**** Investigate Bandwidth, gamma of KDE and SVM on DAE, SAE, DVAE 3 sub figure *****"    
def bandwidth_auc(norm, data, label, method, path, load):
    
    plt.subplots(ncols=3, nrows = 1, figsize=(6, 6))    
    num = 0
    _, _, actual = load_data(data)
    for m, l in zip(method, label):
      AUC = np.empty([0,4])
      
      if (load == 0):
        train_z = np.genfromtxt(path + 'NEW_STOPPING_' + m  + '/'+ data + "_train_z.csv", delimiter=",")
        test_z  = np.genfromtxt(path + 'NEW_STOPPING_' + m  + '/'+ data + "_test_z.csv", delimiter=",")
        
        #gamma = 1/features =  1/(2*bw*bw) 
        #bw = sqrt(features/2)
        #n_features = train_z.shape[1]
          
        steps = 50
        n     = 5.0
        bw = np.asarray([i for i in np.linspace(0.0, n, steps+1)])
        gamma = 1.0/(2.0*bw*bw)
        for b, g in zip(bw, gamma):
          if b> 0:
              svm05 = investigate_svm(train_z, test_z, actual, norm, g, 0.5)
              svm01 = investigate_svm(train_z, test_z, actual, norm, g, 0.1)
              kde   = investigate_kde(train_z, test_z, actual, norm, b)
              AUC = np.append(AUC, [b, kde, svm05, svm01])              
        AUC = np.reshape(AUC, (-1,4))  
      else:
        AUC = np.genfromtxt(path + "Paramter_h/" + data + "_" + m + "_auc_bw.csv", delimiter=",")
        
      fig= plt.subplot(3, 1, num+1)
      
      plt.xlim([0.0, max(AUC[:,0]) + 0.1])
      plt.ylim([0.40,1.0])
      
     # plt.plot(AUC[:,0], AUC[:,2],  'r-o', ms=6, mec="r", label = r'$\mathrm{SVM}_{\nu = 0.5}$', markevery = 3)
      plt.plot(AUC[:,0], AUC[:,3],  'g-^', ms=6, mec="g", label = r'$\mathrm{SVM}_{\nu = 0.1}$', markevery = 3)
      #plt.plot(AUC[:,0], AUC[:,1],  'b-x', ms=6, mec="b", label = 'KDE', markevery = 3)
      np.savetxt(path + "Paramter_h/" + data + "_" + m + "_auc_bw.csv", AUC, delimiter=",", fmt='%f' )

      fig.axes.get_xaxis().set_visible(False)   
      if (num == 2):
         plt.legend(bbox_to_anchor=(1.0, 0.29), ncol = 3, fontsize = 'large')
         
         fig.axes.get_xaxis().set_visible(True) 
         for xtick in fig.xaxis.get_major_ticks():
           xtick.label.set_fontsize(14)
         plt.xlabel('Bandwidth $(h)$', fontsize=16)
      
      plt.ylabel('AUC$_{' + label[num] + '}$' , fontsize=16)
      yticks = fig.yaxis.get_major_ticks() 
      for ytick in yticks:
        ytick.label.set_fontsize(14)  
      yticks[0].label1.set_visible(False)   #Disable the last yticks
      

      if num == 0:
        ax2 = fig.twiny()
        new_tick_locations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        def tick_function(bw):
          gamma = 1.0/(2.0*bw*bw)
          return ["%.3f" % z for z in gamma]
        
        ax2.set_xlim(fig.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations), fontsize=13)
        ax2.set_xlabel(r"$\gamma$ =  $1/(2*h^{2})$", fontsize=16 )
      
        #fig.set_xticklabels(tick_function(new_tick_locations), fontsize=13)


      num = num + 1
      
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.025, hspace=0.025) 
    plt.savefig(path + "Paramter_h/" + data + "_BW.pdf")
    plt.show() 







"************ Investigate gamma of SVM on DAE, SAE, DVAE together *************"    
def parameter_gamma_svm(norm, data, label, method, path, load):    
    num = 0
    AUC = np.empty([0,1])
    if (load == 0):   
      _, _, actual = load_data(data)
      for m, l in zip(method, label):
        auc = np.empty([0,1])
      
        train_z = np.genfromtxt(path + 'NEW_STOPPING_' + m  + '/'+ data + "_train_z.csv", delimiter=",")
        test_z  = np.genfromtxt(path + 'NEW_STOPPING_' + m  + '/'+ data + "_test_z.csv", delimiter=",")
       
       
        h = [0.05, 0.1,0.2,0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22\
        ,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
        h = np.reshape(h, (-1,1))
#        g1 = np.asarray([i for i in np.linspace(1e-5, 1e-4, steps+1)])
#        g2 = np.asarray([i for i in np.linspace(1e-4, 1e-3, steps+1)])
#        g3 = np.asarray([i for i in np.linspace(1e-3, 1e-2, steps+1)])
#        g4 = np.asarray([i for i in np.linspace(1e-2, 1e-1, steps+1)])
#        g5 = np.asarray([i for i in np.linspace(1e-1, 1e-0, steps+1)])
#        g6 = np.asarray([i for i in np.linspace(1e-0, 1e+1, steps+1)])  
#        g7 = np.asarray([i for i in np.linspace(1e+1, 1e+2, steps+1)])
#        g8 = np.asarray([i for i in np.linspace(1e+2, 1e+3, steps+1)])           
#        gamma = np.concatenate((g3[:steps], g4[:steps], g5[:steps], g6[:steps], g7[:steps]))
#        print(gamma)
#        steps = 50
#        gamma = np.asarray([i for i in np.linspace(1e-5, 1e+0, steps+1)])
        if (num == 0):
          AUC = np.reshape(h, (-1,1))
        gamma = 1.0/(2*h)
        for gamma1 in gamma:
          if gamma1> 0:
              svm = investigate_svm(train_z, test_z, actual, norm, gamma1, 0.1)
              auc = np.append(auc, [svm]) 
              
        auc = np.reshape(auc, (-1,1))  
        AUC  = np.insert(AUC , [num+1], auc, axis=1)
        num = num+1
        print(AUC)
    else:
      AUC = np.genfromtxt(path + "Parameter_gamma/" + data +  "_auc_gamma.csv", delimiter=",")
   
    fig, ax = plt.subplots(figsize=(6.4,3.6))     
#    plt.title(data , fontsize=16)
    plt.xlim([0.0, max(AUC[:,0]) + 0.1])
    plt.ylim([0.25,1.0])
      
    plt.plot(AUC[:,0], AUC[:,1],  'r-o', ms=6, mec="r", label = 'DAE-OCSVM', markevery = 5)
    plt.plot(AUC[:,0], AUC[:,2],  'g-^', ms=6, mec="g", label = 'SAE-OCSVM', markevery = 5)
    plt.plot(AUC[:,0], AUC[:,3],  'b-x', ms=6, mec="b", label = 'DVAE-OCSVM', markevery = 5)
    np.savetxt(path + "Parameter_gamma/" + data + "_auc_gamma.csv", AUC, delimiter=",", fmt='%f' )
    #xx-small, x-small, small, medium, large, x-large, xx-large
    plt.legend(bbox_to_anchor=(1.0, 0.95), ncol = 2, fontsize = 'large')
    plt.xlabel(r"$\gamma$ =  $1/(2*h^{2})$", fontsize=20)
    plt.ylabel('AUC' , fontsize=16)
    plt.yticks(fontsize=12)  
      
    new_tick_locations = np.array([0.05, 10, 20, 30, 40, 50])
    def tick_function(bw):
      gamma = 1.0/(2.0*bw*bw)
      return ["%1.1e" % z for z in gamma]
    ax.set_xticklabels(tick_function(new_tick_locations), fontsize=12)
#      
#
    ax2 = ax.twiny()
    new_tick_locations = np.array([0.05, 10, 20, 30, 40, 50])
        
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(new_tick_locations, fontsize=12)
    ax2.set_xlabel("$h$", fontsize=20)      
      
    plt.tight_layout()
    plt.savefig(path + "Parameter_gamma/" + data + "_gamma.pdf")
    plt.show() 
    

"************ Investigate nu of SVM on DAE, SAE, DVAE together *************"    
def parameter_nu_svm(norm, data, label, method, path, load):    
    num = 0
    AUC = np.empty([0,1])

    if (load == 0):
      _, _, actual = load_data(data)
      for m, l in zip(method, label):
        auc = np.empty([0,1])
        train_z = np.genfromtxt(path + 'NEW_STOPPING_' + m  + '/'+ data + "_train_z.csv", delimiter=",")
        test_z  = np.genfromtxt(path + 'NEW_STOPPING_' + m  + '/'+ data + "_test_z.csv", delimiter=",")
       
        n_features = train_z.shape[1]
        bw = (n_features/2.0)**0.5        #default value in One-class SVM
        gamma = 1/(2*bw*bw)       
          
        steps = 50
        nu = np.asarray([i for i in np.linspace(0.0, 0.5, steps+1)])
        if (num == 0):
          AUC = np.reshape(nu[1:], (-1,1))
        for n in nu:
          if n> 0:
              svm = investigate_svm(train_z, test_z, actual, norm, gamma, n)
              auc = np.append(auc, [svm]) 
              
        auc = np.reshape(auc, (-1,1))  
        AUC  = np.insert(AUC , [num+1], auc, axis=1)
        num = num+1
        print(AUC)
    else:
      AUC = np.genfromtxt(path + "Parameter_nu/" + data + "_auc_nu.csv", delimiter=",")
               
    plt.figure(figsize=(6,3)) 
#    plt.title(data , fontsize=16)
    plt.xlim([0.0, max(AUC[:,0]) + 0.01])
    plt.ylim([0.25,1.0])
    plt.yticks(fontsize=12)   
    
    plt.plot(AUC[:,0], AUC[:,1],  'r-o', ms=6, mec="r", label = 'DAE-OCSVM', markevery = 5)
    plt.plot(AUC[:,0], AUC[:,2],  'g-^', ms=6, mec="g", label = 'SAE-OCSVM', markevery = 5)
    plt.plot(AUC[:,0], AUC[:,3],  'b-x', ms=6, mec="b", label = 'DVAE-OCSVM', markevery = 5)
    np.savetxt(path + "Parameter_nu/" + data + "_auc_nu.csv", AUC, delimiter=",", fmt='%f' )
      
    plt.legend(bbox_to_anchor=(1.0, 0.4), ncol = 2, fontsize = 'large')
    plt.xlabel(r"$\nu$", fontsize=20)
    plt.ylabel('AUC' , fontsize=16)
      
    plt.tight_layout()
    plt.savefig(path + "Parameter_nu/" + data + "_nu.pdf")
    plt.show() 
    
    
"************ Inverstigate K of LOF on DAE, SAE, DVAE together *************"    
def parameter_k_lof(norm, data, label, method, path, load):    
    num = 0
    AUC = np.empty([0,1])
    
    if (load == 0):
      _, _, actual = load_data(data)
      for m, l in zip(method, label):
        auc = np.empty([0,1])
      
      
        train_z = np.genfromtxt(path + 'NEW_STOPPING_' + m  + '/'+ data + "_train_z.csv", delimiter=",")
        test_z  = np.genfromtxt(path + 'NEW_STOPPING_' + m  + '/'+ data + "_test_z.csv", delimiter=",")
                      
        steps = 50
        nu = np.asarray([i for i in np.linspace(0.0, 0.5, steps+1)])
        if (num == 0):
          AUC = np.reshape(100*nu[1:], (-1,1))
          print(AUC)
        for n in nu:
          if n> 0:
              lof = Investigate_lof(train_z, test_z, actual, norm, n)
              auc = np.append(auc, [lof]) 
              
        auc = np.reshape(auc, (-1,1))  
        AUC  = np.insert(AUC , [num+1], auc, axis=1)
        num = num+1
        print(AUC)
        
    else:
      AUC = np.genfromtxt(path + "Parameter_k/" + data + "_auc_k.csv", delimiter=",")
                
    plt.figure(figsize=(6,3)) 
#    plt.title(data , fontsize=16)
    plt.xlim([0.0, max(AUC[:,0]) + 1])
    plt.ylim([0.25,1.0])
    plt.yticks(fontsize=12)  
    
    plt.plot(AUC[:,0], AUC[:,1],  'r-o', ms=6, mec="r", label = 'DAE-LOF', markevery = 5)
    plt.plot(AUC[:,0], AUC[:,2],  'g-^', ms=6, mec="g", label = 'SAE-LOF', markevery = 5)
    plt.plot(AUC[:,0], AUC[:,3],  'b-x', ms=6, mec="b", label = 'DVAE-LOF', markevery = 5)
    np.savetxt(path + "Parameter_k/" + data + "_auc_k.csv", AUC, delimiter=",", fmt='%f' )
    
    plt.legend(bbox_to_anchor=(1.0, 0.35), ncol = 2, fontsize = 'large')         
    plt.xlabel(r"$k$", fontsize=20)
    plt.ylabel('AUC' , fontsize=16)
      
    plt.tight_layout()
    plt.savefig(path + "Parameter_k/" + data + "_k.pdf")
    plt.show()     
    
    
"************************** Investigate Bandwidth ****************************"
path    = "D:/Python_code/SDA-02/Results/Exp_Hidden/"
method = ["DAE", "SHRINK", "VAE"]
label  = ["DAE", "SAE", "DVAE"]

datasets = ["PageBlocks", "WPBC", "PenDigits", "GLASS", "Shuttle", "Arrhythmia",\
                 "CTU13_10", "CTU13_08","CTU13_09","CTU13_13",\
                 "Spambase", "UNSW", "NSLKDD", "InternetAds"]

datasets = ["Spambase", "NSLKDD", "UNSW", "InternetAds", "CTU13_13"]         
datasets = ["CTU13_13"]             
for data in datasets:
  #investigate_bandwidth("maxabs", data, path_method, path)
  parameter_gamma_svm("maxabs", data, label, method, path,1)
#  parameter_nu_svm("maxabs", data, label, method, path,1)
#  parameter_k_lof("maxabs", data, label, method, path,1)
