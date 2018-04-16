# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:45:52 2017

@author: VANLOI
"""

#Visualize hidden data
from sklearn import preprocessing
from ProcessingData import load_data
from High_dimension_visualize import visualize_n_dimension
import matplotlib.pyplot as plt
import numpy as np

datasets = ["PageBlocks", "WPBC", "PenDigits", "GLASS", "Shuttle", "Arrhythmia",\
                 "CTU13_10", "CTU13_08","CTU13_09","CTU13_13",\
                 "Spambase", "UNSW", "NSLKDD", "InternetAds"]  
                 
datasets = ["CTU13_10", "CTU13_09", "Spambase", "UNSW", "NSLKDD", "InternetAds"]

datasets = ["UNSW"]
                                 
path_method = ["NEW_STOPPING_DAE/", "NEW_STOPPING_SHRINK/", "NEW_STOPPING_VAE/"]
method = ["DAE", "SAE", "DVAE"]


"********* nD, no scaler ***********"
def visualize_z():
  path   = "D:/Python_code/SDA-02/Results/Exp_Hidden/"
  for data_name in datasets:
    i = 0
    plt.subplots(ncols=3, nrows = 3, figsize=(6, 6))
    for m in path_method:
      _,_, actual = load_data(data_name)
      train_z = np.genfromtxt(path + m + data_name + "_train_z.csv", delimiter=",")
      test_z  = np.genfromtxt(path + m + data_name + "_test_z.csv", delimiter=",")
    
      test_X0 = test_z[(actual==1)]
      test_X1 = test_z[(actual==0)]
      np.random.shuffle(train_z)
      np.random.shuffle(test_X0)
      np.random.shuffle(test_X1) 
    
      n = 1000  
      train_z = train_z[:n]
      test_X0 = test_X0[:n]
      test_X1 = test_X1[:n]
      dataset = [train_z, test_X0, test_X1]
    
      j = 0
      color = ['b', 'g', 'r']
      label = ["Normal train", "Normal test", "Anomaly test"]
      for data, c, l in zip(dataset, color, label):
        num = i*3 + j + 1
        fig= plt.subplot(3, 3, num)
        if (data_name == "CTU13_10"):
          if (i==0):
            plt.ylim((-1.1, 1.1))
            plt.xlim((-1.1, 1.1))
          elif (i==1):
            plt.ylim((-0.11, 0.11))
            plt.xlim((-0.11, 0.11))     
          elif (i==2):
            plt.ylim((-0.011, 0.011))
            plt.xlim((-0.011, 0.011)) 

        if (data_name == "InternetAds" ):
          if (i==0):
            plt.ylim((-1.1, 1.1))
            plt.xlim((-1.1, 1.1))
          elif (i==1):
            plt.ylim((-0.21, 0.21))
            plt.xlim((-0.21, 0.21))     
          elif (i==2):
            plt.ylim((-0.051, 0.051))
            plt.xlim((-0.051, 0.051))
            
        if (data_name == "Spambase"):
          if (i==0):
            plt.ylim((-1.1, 1.1))
            plt.xlim((-1.1, 1.1))
          elif (i==1):
            plt.ylim((-0.051, 0.051))
            plt.xlim((-0.051, 0.051))     
          elif (i==2):
            plt.ylim((-0.031, 0.031))
            plt.xlim((-0.031, 0.031))
            
        if (data_name == "CTU13_08"):        
          if (i==0):
            plt.ylim((-1.1, 1.1))
            plt.xlim((-1.1, 1.1))
          elif (i==1):
            plt.ylim((-0.051, 0.051))
            plt.xlim((-0.051, 0.051))     
          elif (i==2):
            plt.ylim((-0.031, 0.031))
            plt.xlim((-0.031, 0.031))
            
        if (data_name == "CTU13_09"):        
          if (i==0):
            plt.ylim((-1.1, 1.1))
            plt.xlim((-1.1, 1.1))
          elif (i==1):
            plt.ylim((-0.051, 0.051))
            plt.xlim((-0.051, 0.051))     
          elif (i==2):
            plt.ylim((-0.051, 0.051))
            plt.xlim((-0.051, 0.051))
            
        if (data_name == "CTU13_13"):        
          if (i==0):
            plt.ylim((-1.1, 1.1))
            plt.xlim((-1.1, 1.1))
          elif (i==1):
            plt.ylim((-0.021, 0.021))
            plt.xlim((-0.021, 0.021))     
          elif (i==2):
            plt.ylim((-0.021, 0.021))
            plt.xlim((-0.021, 0.021))
            
        if (data_name =="NSLKDD"):
          if (i==0):
            plt.ylim((-1.1, 1.1))
            plt.xlim((-1.1, 1.1))
          elif (i==1):
            plt.ylim((-0.051, 0.051))
            plt.xlim((-0.051, 0.051))             
          elif (i==2):
            plt.ylim((-0.061, 0.061))
            plt.xlim((-0.061, 0.061))    
            
        elif (data_name == "UNSW"):
          if (i==0):
            plt.ylim((-1.1, 1.1))
            plt.xlim((-1.1, 1.1))
          elif (i==1):
            plt.ylim((-0.051, 0.051))
            plt.xlim((-0.051, 0.051))             
          elif (i==2):
            plt.ylim((-0.061, 0.061))
            plt.xlim((-0.061, 0.061)) 
        #plt.title(data_name)    
        plt.plot(data[:,0], data[:,1], c+'o', ms=2, mec= c , label= l) 
        plt.xticks(fontsize=9, rotation=90)
        plt.yticks(fontsize=9, rotation=0)       
        


        if i == 0:
          plt.legend(bbox_to_anchor=(1.02, 1.25), ncol = 1, fontsize = 'medium')
        fig.xaxis.set_ticks_position('bottom')    #Disable bottom and left ticks
        fig.yaxis.set_ticks_position('left')  
        
        fig.axes.get_xaxis().set_visible(False) 
        fig.axes.get_yaxis().set_visible(False)  
   
        if (i == 2 or i == 0 or  i == 1):
          fig.axes.get_xaxis().set_visible(True)
#          xticks = fig.xaxis.get_major_ticks()  
#          xticks[0].label1.set_visible(False)   #Disable the last yticks           
#          xticks[-1].label1.set_visible(False)    #Disable the first yticks           
#          if (j == 1):
#              plt.xlabel(data_name, fontsize=16)
                   
        if (j == 0):
          fig.axes.get_yaxis().set_visible(True)  
          plt.ylabel(method[i], fontsize=12)  
          
#          yticks = fig.yaxis.get_major_ticks() 
#          yticks[0].label1.set_visible(False)    #Disable the first yticks 
#          yticks[-1].label1.set_visible(False)   #Disable the last yticks
         
        j = j + 1
      i = i + 1
    
#    plt.tight_layout()    
    plt.subplots_adjust(wspace=0.08, hspace=0.35) 
    plt.savefig(path + "Visualize_z/" + data_name + "_Visualize_nD_1.pdf")
    plt.show() 


"********** scaler and nD ************"
def visualize_nD_scale():
  path     = "D:/Python_code/SDA-02/Results/Exp_Hidden/"
  for data_name in datasets:
    i = 0
    plt.subplots(ncols=3, nrows = 3, figsize=(6, 6))
    for m in path_method:
      _,_, actual = load_data(data_name)
      train_z = np.genfromtxt(path + m + data_name + "_train_z.csv", delimiter=",")
      test_z  = np.genfromtxt(path + m + data_name + "_test_z.csv", delimiter=",")
      
      scaler = preprocessing.StandardScaler()                
      scaler.fit(train_z)
      train_z = scaler.transform(train_z)
      test_z  = scaler.transform(test_z)
    
      test_X0 = test_z[(actual==1)]
      test_X1 = test_z[(actual==0)]
    
      np.random.shuffle(train_z)
      np.random.shuffle(test_X0)
      np.random.shuffle(test_X1) 
    
      n = 1000  
      train_z = train_z[:n]
      test_X0 = test_X0[:n]
      test_X1 = test_X1[:n]
      dataset = [train_z, test_X0, test_X1]
    
      j = 0
      color = ['b', 'g', 'r']
      label = ["Normal train", "Normal test", "Anomaly test"]
      for data, c, l in zip(dataset, color, label):
        num = i*3 + j + 1
        fig= plt.subplot(3, 3, num)
        if (i==0):
          plt.ylim((-5.1, 5.1))
          plt.xlim((-5.1, 5.1))
        else:
          plt.ylim((-40.1, 40.1))
          plt.xlim((-40.1, 40.1))
          
        plt.plot(data[:,0], data[:,1], c+'o', ms=2, mec= c , label= l) 
        plt.xticks(fontsize=10, rotation=90)
        plt.yticks(fontsize=10, rotation=0)       
        


        if i == 0:
          plt.legend(bbox_to_anchor=(1.02, 1.25), ncol = 1, fontsize = 'medium')
        fig.xaxis.set_ticks_position('bottom')    #Disable bottom and left ticks
        fig.yaxis.set_ticks_position('left')  
        
        fig.axes.get_xaxis().set_visible(False) 
        fig.axes.get_yaxis().set_visible(False)  
   
#        if (i == 2 or i == 0):
        fig.axes.get_xaxis().set_visible(True)
        xticks = fig.xaxis.get_major_ticks()  
        xticks[0].label1.set_visible(False)   #Disable the last yticks           
        xticks[-1].label1.set_visible(False)    #Disable the first yticks           
#          if (j == 1):
#              plt.xlabel(data_name, fontsize=16)
                   
        if (j == 0):
          fig.axes.get_yaxis().set_visible(True)  
          plt.ylabel(method[i], fontsize=14)  
          
          yticks = fig.yaxis.get_major_ticks() 
          yticks[0].label1.set_visible(False)    #Disable the first yticks 
          yticks[-1].label1.set_visible(False)   #Disable the last yticks
         
        j = j + 1
      i = i + 1
    
#    plt.tight_layout()    
    plt.subplots_adjust(wspace=0.08, hspace=0.35) 
    plt.savefig(path + "Visualize_z/" + data_name + "_Visualize_nD.pdf")
    plt.show()  


"********** scaler and 2D ************"
def visualize_2D_scale():
  path     = "D:/Python_code/SDA-02/Results/Exp_Hidden/Visualize_z/"
  for data_name in datasets:
    i = 0
    plt.subplots(ncols=3, nrows = 3, figsize=(6, 6))
    for m in path_method:
      _,_, actual = load_data(data_name)
      train_z = np.genfromtxt(path + m + data_name + "_train_z.csv", delimiter=",")
      test_z  = np.genfromtxt(path + m + data_name + "_test_z.csv", delimiter=",")
      
      scaler = preprocessing.StandardScaler()                
      scaler.fit(train_z)
      train_z = scaler.transform(train_z)
      test_z  = scaler.transform(test_z)
    
      test_X0 = test_z[(actual==1)]
      test_X1 = test_z[(actual==0)]
    
      np.random.shuffle(train_z)
      np.random.shuffle(test_X0)
      np.random.shuffle(test_X1) 
    
      n = 1000  
      train_z = train_z[:n]
      test_X0 = test_X0[:n]
      test_X1 = test_X1[:n]
      dataset = [train_z, test_X0, test_X1]
    
      j = 0
      color = ['b', 'g', 'r']
      label = ["Normal train", "Normal test", "Anomaly test"]
      for data, c, l in zip(dataset, color, label):
        num = i*3 + j + 1
        fig= plt.subplot(3, 3, num)
        
        ["CTU13_10", "UNSW", "NSLKDD"]
        if(data_name == "NSLKDD"):
          if (i==0):
            plt.ylim((-5.1, 5.1))
            plt.xlim((-5.1, 5.1))
          else:
            plt.ylim((-40.1, 40.1))
            plt.xlim((-40.1, 40.1))
        else:
          if (i==0):
            plt.ylim((-5.1, 5.1))
            plt.xlim((-5.1, 5.1))
          else:
            plt.ylim((-30.1, 30.1))
            plt.xlim((-30.1, 30.1))
          
        plt.plot(data[:,0], data[:,1], c+'o', ms=2, mec= c , label= l) 
        plt.xticks(fontsize=10, rotation=90)
        plt.yticks(fontsize=10, rotation=0)       
        


        if i == 0:
          plt.legend(bbox_to_anchor=(1.02, 1.25), ncol = 1, fontsize = 'medium')
        fig.xaxis.set_ticks_position('bottom')    #Disable bottom and left ticks
        fig.yaxis.set_ticks_position('left')  
        
        fig.axes.get_xaxis().set_visible(False) 
        fig.axes.get_yaxis().set_visible(False)  
   
#        if (i == 2 or i == 0):
        fig.axes.get_xaxis().set_visible(True)
        xticks = fig.xaxis.get_major_ticks()  
        xticks[0].label1.set_visible(False)   #Disable the last yticks           
        xticks[-1].label1.set_visible(False)    #Disable the first yticks           
#          if (j == 1):
#              plt.xlabel(data_name, fontsize=16)
                   
        if (j == 0):
          fig.axes.get_yaxis().set_visible(True)  
          plt.ylabel(method[i], fontsize=14)  
          
          yticks = fig.yaxis.get_major_ticks() 
          yticks[0].label1.set_visible(False)    #Disable the first yticks 
          yticks[-1].label1.set_visible(False)   #Disable the last yticks
         
        j = j + 1
      i = i + 1
    
#    plt.tight_layout()    
    plt.subplots_adjust(wspace=0.08, hspace=0.35) 
    plt.savefig(path  + "2D/" +  data_name + "_Visualize_2D.pdf")
    plt.show()  

visualize_z()
#visualize_nD_scale()
#visualize_2D_scale()