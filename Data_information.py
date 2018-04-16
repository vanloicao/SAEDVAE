# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:07:52 2017

@author: VANLOI
"""

import numpy as np
from ProcessingData import load_data
import csv

#%%
"Get data information"

path = "D:/Python_code/SDA-02/Results/Exp_Hidden/"
list_data = ["PageBlocks", "WPBC", "PenDigits", "GLASS", "Shuttle", "Arrhythmia",\
             "CTU13_10", "CTU13_08","CTU13_09","CTU13_13",\
             "Spambase", "UNSW", "NSLKDD", "InternetAds"]

data_infor = np.empty([0,5])
header     = np.column_stack(["Data", "Dimension", "Training", "Normal", "Anomaly"])
data_infor = np.append(data_infor, header)

for data in list_data:
  train_set, test_set, actual = load_data(data)
  test_normal  = test_set[(actual == 1)]
  test_anomaly = test_set[(actual == 0)]

  d         = train_set.shape[1]
  n_train   = train_set.shape[0]
  n_normal  = test_normal.shape[0]
  n_anomaly = test_anomaly.shape[0]
  infor =  np.column_stack([ data, d, n_train, n_normal, n_anomaly])
  data_infor = np.append(data_infor, infor)

data_infor   = np.reshape(data_infor,(-1,5))
#np.savetxt(path +  "data_information.csv", data_infor, fmt='%.18e', delimiter=' ')
#np.savetxt(path +  "data_information.csv", data_infor, delimiter=",", fmt= ('%s,%f,%f,%f,%f') )

with open(path + "data_information.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(data_infor)

print(data_infor)