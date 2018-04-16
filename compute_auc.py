# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 22:20:49 2017

@author: VANLOI
This program is to compute AUC for LOF, IOF and Mixture Gaussian on hidden data
read from file. Hidden data producing from DAE (greedy layer-wise), Shrink AE and
Dirac Delta VAE.
"""
from sklearn.metrics import roc_curve, auc
from LOF import LocalOutlierFactor
from sklearn import mixture
from sklearn.ensemble import IsolationForest

from BaseOneClass import CentroidBasedOneClassClassifier

from ProcessingData import normalize_data, load_data
import numpy as np

"LOF"
def auc_LOF(training_set, testing_set, actual):
  
    training_set, testing_set =  normalize_data(training_set, testing_set, "maxabs")
    neighbors = (int)(len(training_set)*0.1)
    clf_lof = LocalOutlierFactor(n_neighbors=neighbors)
    clf_lof.fit(training_set)
    predict = clf_lof._decision_function(testing_set)
    FPR, TPR, thresholds = roc_curve(actual, predict)
    lof = auc(FPR, TPR)
    
    return lof

"Mixture Gaussian"
def auc_MX(training_set, testing_set, actual):
  
    training_set, testing_set =  normalize_data(training_set, testing_set, "maxabs")

    clf_lof = mixture.GaussianMixture(n_components=1, covariance_type='full')
    clf_lof.fit(training_set)
    predict = clf_lof.score_samples(testing_set)
    FPR, TPR, thresholds = roc_curve(actual, predict)
    lof = auc(FPR, TPR)
    
    return lof

"Isolated Outlier Forest"
def auc_IOF(training_set, testing_set, actual):    
    # fit the model
    rng = np.random.RandomState(42)
    training_set, testing_set =  normalize_data(training_set, testing_set, "maxabs")
  
    clf_iof = IsolationForest(random_state=rng)
    clf_iof.fit(training_set)
    score_iof = clf_iof.predict(testing_set)
    
    FPR, TPR, thresholds = roc_curve(actual, score_iof)
    iof= auc(FPR, TPR)
    return iof

"*************** Centroid AE - Hidden layer **************"
def auc_CEN(training_set, testing_set, actual): 

    
    CEN = CentroidBasedOneClassClassifier()    
    CEN.fit(training_set)  
    predictions_cen = -CEN.get_density(testing_set)
    
    FPR_cen, TPR_cen, thresholds_cen = roc_curve(actual, predictions_cen)
    cen = auc(FPR_cen, TPR_cen) 
    return cen


"************************ Compute AUC for each data set **********************"
"""This function is used to compute AUC from latent data when we want to reproduce
experimental results."""
def compute_auc_data():
  list_data =  ["PageBlocks", "WPBC", "PenDigits", "GLASS", "Shuttle", "Arrhythmia",\
              "CTU13_10", "CTU13_08","CTU13_09","CTU13_13",\
              "Spambase", "UNSW", "NSLKDD", "InternetAds"]                 
  methods = ["input", "DAE", "Shrink", "VAE"]        
  path   = "D:/Python_code/SDA-02/Results/Exp_Hidden/"

  lof_auc   = np.empty([0,14])      
  for method in methods:
    auc1   = np.empty([0,1]) 
    for data in list_data:
      _,_, actual = load_data(data)      #load original data  
      if method == "input": 
        train_X, test_X, actual = load_data(data) 
      elif method == "DAE":
        train_X = np.genfromtxt(path + "NEW_STOPPING_DAE/" + data + "_train_z.csv", delimiter=",")
        test_X  = np.genfromtxt(path + "NEW_STOPPING_DAE/" + data + "_test_z.csv", delimiter=",")
      elif method == "Shrink":
        train_X = np.genfromtxt(path + "NEW_STOPPING_SHRINK/" + data + "_train_z.csv", delimiter=",")
        test_X  = np.genfromtxt(path + "NEW_STOPPING_SHRINK/" + data + "_test_z.csv", delimiter=",")      
      elif method == "VAE":  
        train_X = np.genfromtxt(path + "NEW_STOPPING_VAE/" + data + "_train_z.csv", delimiter=",")
        test_X  = np.genfromtxt(path + "NEW_STOPPING_VAE/" + data + "_test_z.csv", delimiter=",")

      lof = auc_LOF(train_X, test_X, actual)
      #lof = auc_IOF(train_X, test_X, actual)
      #lof = auc_CEN(train_X, test_X, actual)
    
      auc1 = np.append(auc1, lof)
      print(auc1)

    lof_auc = np.append(lof_auc, auc1)  
    np.set_printoptions(precision=3, suppress=True)
    print(auc1)
  lof_auc = np.reshape(lof_auc, (-1,14))  
  np.savetxt(path +  "LOF_11_per.csv", lof_auc, delimiter=",", fmt='%f' )  




"****** For compute AUC for each attack group in NSL-KDD and UNSW-NB15 ********"   
def compute_auc_group_attacks():
  list_data = ["Probe", "DoS", "R2L", "U2R",\
             "Fuzzers", "Analysis", "Backdoor", "DoS_UNSW", "Exploits",\
             "Generic", "Reconnaissance", "Shellcode", "Worms"]
              
  methods = ["input", "DAE", "Shrink", "VAE"]        

  path   = "D:/Python_code/SDA-02/Results/Exp_Hidden/NEW_GROUP/"

  lof_auc   = np.empty([0,13])      
  for method in methods:
    auc1   = np.empty([0,1]) 
    for data in list_data:
      _,_, actual = load_data(data)      #load original data  
      if method == "input": 
        train_X, test_X, actual = load_data(data) 
      elif method == "DAE":
        train_X = np.genfromtxt(path + "DAE/" + data + "_train_z.csv", delimiter=",")
        test_X  = np.genfromtxt(path + "DAE/" + data + "_test_z.csv", delimiter=",")
      elif method == "Shrink":
        train_X = np.genfromtxt(path + "SHRINK/" + data + "_train_z.csv", delimiter=",")
        test_X  = np.genfromtxt(path + "SHRINK/" + data + "_test_z.csv", delimiter=",")      
      elif method == "VAE":  
        train_X = np.genfromtxt(path + "VAE/" + data + "_train_z.csv", delimiter=",")
        test_X  = np.genfromtxt(path + "VAE/" + data + "_test_z.csv", delimiter=",")

      lof = auc_LOF(train_X, test_X, actual)  
      auc1 = np.append(auc1, lof)
      print(auc1)

    lof_auc = np.append(lof_auc, auc1)  
    np.set_printoptions(precision=3, suppress=True)
    print(auc1)
  lof_auc = np.reshape(lof_auc, (-1,13))  
  np.savetxt(path +  "LOF_10_UNSW.csv", lof_auc, delimiter=",", fmt='%f' )  
"******************************************************************************"



def compute_confusion_matrix():
  list_data = ["Probe", "DoS", "R2L", "U2R"]
  path   = "D:/Python_code/SDA-02/Results/Exp_Hidden/NEW_GROUP/"
  
  train_X = np.genfromtxt(path + "ShrinkAE/NSLKDD_train_z.csv", delimiter=",")
  test_X = np.genfromtxt(path + "ShrinkAE/NSLKDD_test_z.csv", delimiter=",")
  _,_, actual = load_data("NSLKDD")      #load original data
  
#  auc_cen = auc_CEN(train_X,test_X, actual)
#  print(auc_cen)
  
  test_normal = test_X[(actual == 1)]
  y_normal = ~(actual[(actual==1)]).astype(np.bool) # False
  
  
  CEN = CentroidBasedOneClassClassifier(threshold=0.92)    
  CEN.fit(train_X)  

  pre_label0 =  CEN.predict(test_normal) 
 
  Test_P   = sum(pre_label0 == y_normal)
  Actual_P = len(test_normal)
  FN       =  Actual_P - Test_P
  print ("\nActual Normal: %d,\nTest normal:%d, \nFP:%d" %(Actual_P, Test_P, FN))
  
  Test_N   = 0
  Actual_N = 0
  FP       = 0 
  for data in list_data:
    _,_, actual = load_data(data)      #load original data  
    test_X  = np.genfromtxt(path + "ShrinkAE/" + data + "_test_z.csv", delimiter=",")
    test_anomaly = test_X[(actual == 0)]
    y_attack = ~(actual[(actual==0)]).astype(np.bool) #True
    
    Actual_N = len(test_anomaly)
    pre_label1 =  CEN.predict(test_anomaly)  
    

    Test_N = sum(pre_label1 == y_attack)
    FP  = Actual_N - Test_N
    
    print ("\nActual " + data + ": %d,\nTest anomaly: %d, \nFP:%d" % (Actual_N, Test_N, FP))
    
    #matrix = np.append(lof_auc, auc1)  
    #np.set_printoptions(precision=3, suppress=True)
  #lof_auc = np.reshape(lof_auc, (-1,4))  
  #np.savetxt(path +  "confusion_matrix.csv", lof_auc, delimiter=",", fmt='%f' )

compute_confusion_matrix()
#compute_auc_data()
#compute_auc_group_attacks()