# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:47:44 2016

@author: caoloi
"""
import numpy as np
from sklearn import preprocessing

seed = 0

"Normalize training and testing sets"
def normalize_data(train_X, test_X, scale):
    if ((scale == "standard") | (scale == "maxabs") | (scale == "minmax")):
        if (scale == "standard"):
            scaler = preprocessing.StandardScaler()
        elif (scale == "maxabs"):
            scaler = preprocessing.MaxAbsScaler()
        elif (scale == "minmax"):
            scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X  = scaler.transform(test_X)
    else:
        print ("No scaler")
    return train_X, test_X

"***********************Pre-processing NSL-KDD dataset************************"
def process_KDD(group_attack):
    d0 = np.genfromtxt("Data/NSLKDD/NSLKDD_Train.csv", delimiter=",")
    d1 = np.genfromtxt("Data/NSLKDD/NSLKDD_Test.csv", delimiter=",")
    # Train: 67343, Test normal: 9711, Dos: 7458, R2L   2887, U2R: 67, Probe 2422

    n_train = 6734

    np.random.seed(seed)
    np.random.shuffle(d0)
    np.random.shuffle(d1)

    "Pre-processing training set"
    dy = d0[:,-1]                       # put labels(the last column) to dy
    train_X = d0[(dy==0)]               # Normal(class 0) to train_X
    train_X = train_X[:,0:-1]           # discard the last column (labels)
    train_X = train_X[:n_train]         # downsample training set

    "Pre-processing Testing set"
    dy = d1[:,-1]                       # put labels to dy
    dX = d1[:,0:-1]                     # put data to dX without last column (labels)

    dX0 = dX[(dy==0)]                   # Normal, class 0
    if (group_attack == "Probe"):
        dX1 = dX[(dy==1)]
    elif (group_attack == "DoS"):
        dX1 = dX[(dy==2)]
    elif (group_attack == "R2L"):
        dX1 = dX[(dy==3)]
    elif (group_attack == "U2R"):
        dX1 = dX[(dy==4)]
    elif (group_attack == "NSLKDD"):
        dX1 = dX[(dy>0)]
    else:
        print ("No group of attack")

    test_X0 = dX0                       #normal test
    test_X1 = dX1                       #anomaly test
    #normal and anomaly test
    test_X = np.concatenate((test_X0, test_X1))

    #Create label for normal and anomaly test examples, and then combine two sets
    test_y0 = np.full((len(test_X0)), False, dtype=bool)
    test_y1 = np.full((len(test_X1)), True,  dtype=bool)
    test_y =  np.concatenate((test_y0, test_y1))
    #create binary label (1-normal, 0-anomaly) for compute AUC later
    actual = (~test_y).astype(np.int)

    return train_X, test_X, actual

"***************************Processing UNSW Dataset***************************"
def process_UNSW(group_attack):

    """group_attack: {'Fuzzers': 1, 'Exploits': 5, 'Normal': 0, 'Generic': 6, 'Worms': 9,
    'Analysis': 2, 'Backdoor': 3, 'DoS': 4, 'Reconnaissance': 7, 'Shellcode': 8}"""
    #DoS --> DoS_UNSW to avoid the conflict with DoS in NSL-KDD
    d0 = np.genfromtxt("Data/UNSW/UNSW_Train.csv", delimiter=",")
    d1 = np.genfromtxt("Data/UNSW/UNSW_Test.csv", delimiter=",")
    #Training set: 175341 (56000 normal, 119341 anomaly)
    #Testing set:   82332 (37000 normal,  45332 anomaly)

    n_train = 5600         #downsample training set 10%

    np.random.seed(seed)
    np.random.shuffle(d0)
    np.random.shuffle(d1)

    "Pre-processing training set"
    dy = d0[:,-1]                       # put labels(the last column) to dy
    train_X = d0[(dy==0)]               # Normal(class 0) to train_X
    train_X = train_X[:,0:-1]           # discard the last column (labels)
    train_X = train_X[:n_train]         # downsample training set

    "Pre-processing Testing set"
    dy = d1[:,-1]                       # put labels to dy
    dX = d1[:,0:-1]                     # put data to dX without last column (labels)


    """group_attack: {'Fuzzers': 1, 'Exploits': 5, 'Normal': 0, 'Generic': 6, 'Worms': 9,
    'Analysis': 2, 'Backdoor': 3, 'DoS': 4, 'Reconnaissance': 7, 'Shellcode': 8}"""

    dX0 = dX[(dy==0)]                   # Normal, class 0

    if (group_attack == "Fuzzers"):
        dX1 = dX[(dy==1)]
    elif (group_attack == "Analysis"):
        dX1 = dX[(dy==2)]
    elif (group_attack == "Backdoor"):
        dX1 = dX[(dy==3)]
    elif (group_attack == "DoS_UNSW"):
        dX1 = dX[(dy==4)]
    elif (group_attack == "Exploits"):
        dX1 = dX[(dy==5)]
    elif (group_attack == "Generic"):
        dX1 = dX[(dy==6)]
    elif (group_attack == "Reconnaissance"):
        dX1 = dX[(dy==7)]
    elif (group_attack == "Shellcode"):
        dX1 = dX[(dy==8)]
    elif (group_attack == "Worms"):
        dX1 = dX[(dy==9)]
    elif (group_attack == "UNSW"):
        dX1 = dX[(dy>0)]
    else:
        print ("No group of attack")

    test_X0 = dX0                       #normal test
    test_X1 = dX1                       #anomaly test
    #normal and anomaly test
    test_X = np.concatenate((test_X0, test_X1))

    #Create label for normal and anomaly test examples, and then combine two sets
    test_y0 = np.full((len(test_X0)), False, dtype=bool)
    test_y1 = np.full((len(test_X1)), True,  dtype=bool)
    test_y =  np.concatenate((test_y0, test_y1))
    #create binary label (1-normal, 0-anomaly) for compute AUC later
    actual = (~test_y).astype(np.int)

    return train_X, test_X, actual

"*************************Read data from CSV file*****************************"
def process_PenDigits():
    #16 real-values + 1 class (normal: class 0, anomaly: class 2 or 1-9)
    #each class is equaly to each others in train and test set
    d0 = np.genfromtxt("Data/pendigits_train.csv", delimiter=",")
    d1 = np.genfromtxt("Data/pendigits_test.csv", delimiter=",")

    # shuffle
    np.random.seed(seed)
    np.random.shuffle(d0)
    np.random.shuffle(d1)

    "Pre-processing training set"
    dy = d0[:,-1]                       # put labels(the last column) to dy
    train_X = d0[(dy==0)]               # Normal(class 0) to train_X
    train_X = train_X[:,0:-1]           # discard the last column (labels)

    "Pre-processing Testing set"
    dy = d1[:,-1]                       # put labels to dy
    dX = d1[:,0:-1]                     # put data to dX without last column (labels)

    dX0 = dX[(dy == 0)]                  # Normal, class 0
    dX1 = dX[(dy == 2)]                   # Anomaly, class 2 (or 1-9)

    test_X0 = dX0                       #normal test
    test_X1 = dX1                       #anomaly test
    #normal and anomaly test
    test_X = np.concatenate((test_X0, test_X1))

    #Create label for normal and anomaly test examples, and then combine two sets
    test_y0 = np.full((len(test_X0)), False, dtype=bool)
    test_y1 = np.full((len(test_X1)), True,  dtype=bool)
    test_y =  np.concatenate((test_y0, test_y1))
    #create binary label (1-normal, 0-anomaly) for compute AUC later
    actual = (~test_y).astype(np.int)

    return train_X, test_X, actual


"*************************Read Shuttle data*****************************"
def process_Shuttle():
    #9 attributes + 1 class (1-7), train_set: 43500(34108 normal), test_set: 14500
    #Consider normal: class 1, anomaly: classes 2 - 7
    d0 = np.genfromtxt("Data/shuttle_train.csv", delimiter=",")
    d1 = np.genfromtxt("Data/shuttle_test.csv", delimiter=",")
    n_train = 3410  #downsample 10%
    # shuffle
    np.random.seed(seed)
    np.random.shuffle(d0)
    np.random.shuffle(d1)

    "Pre-processing training set"
    dy = d0[:,-1]                       # put labels(the last column) to dy
    train_X = d0[(dy==1)]               # Normal(class 1) to train_X
    train_X = train_X[:,0:-1]           # discard the last column (labels)

    train_X = train_X[:n_train]

    "Pre-processing Testing set"
    dy = d1[:,-1]                       # put labels to dy
    dX = d1[:,0:-1]                     # put data to dX without last column (labels)

    dX0 = dX[(dy==1)]                   # Normal, class 1
    dX1 = dX[(dy>1)]                    # Anomaly, class 2-7

    test_X0 = dX0                       #normal test
    test_X1 = dX1                       #anomaly test
    #normal and anomaly test
    test_X = np.concatenate((test_X0, test_X1))

    #Create label for normal and anomaly test examples, and then combine two sets
    test_y0 = np.full((len(test_X0)), False, dtype=bool)
    test_y1 = np.full((len(test_X1)), True,  dtype=bool)
    test_y =  np.concatenate((test_y0, test_y1))
    actual = (~test_y).astype(np.int)

    return train_X, test_X, actual


"*************************Read Annthyroid data*****************************"
def process_Annthyroid():
    #21 attributes + 1 class (3 - healthy, 1-2 - hypothyroidism)
    #Train_set (3488 1-heathy, 284 hypo), Test_set(3178-heathy, 250-hypo)
    # 3 - normal, 1 - hyperfunction and 2 - subnormal functioning
    d0 = np.genfromtxt("Data/annthyroid_train.csv", delimiter=",")
    d1 = np.genfromtxt("Data/annthyroid_test.csv", delimiter=",")
    n_train = 3488
    # shuffle
    np.random.seed(seed)
    np.random.shuffle(d0)
    np.random.shuffle(d1)

    "Pre-processing training set"
    dy = d0[:,-1]                       # put labels(the last column) to dy
    train_X = d0[(dy==3)]               # Normal(class 3) to train_X
    train_X = train_X[:,0:-1]           # discard the last column (labels)
    train_X = train_X[:n_train]

    "Pre-processing Testing set"
    dy = d1[:,-1]                       # put labels to dy
    dX = d1[:,0:-1]                     # put data to dX without last column (labels)

    dX0 = dX[(dy==3)]                   # Normal, class 1
    dX1 = dX[(dy<3)]                    # Anomaly, class 1, 2. better if choosing only class 1

    test_X0 = dX0                       #normal test
    test_X1 = dX1                       #anomaly test
    #normal and anomaly test
    test_X = np.concatenate((test_X0, test_X1))

    #Create label for normal and anomaly test examples, and then combine two sets
    test_y0 = np.full((len(test_X0)), False, dtype=bool)
    test_y1 = np.full((len(test_X1)), True,  dtype=bool)
    test_y =  np.concatenate((test_y0, test_y1))
    actual = (~test_y).astype(np.int)

    return train_X, test_X, actual

"*************************Read wilt data*****************************"
def process_wilt():
    #5 attributes + 1 class(0,1), Train_set(4265 cover land, 74 diseased tree)
    #Test_set (313 cover land, 187 diseased tree)
    d0 = np.genfromtxt("Data/wilt_train.csv", delimiter=",")
    d1 = np.genfromtxt("Data/wilt_test.csv", delimiter=",")
    n_train = 4265
    # shuffle
    np.random.seed(seed)
    np.random.shuffle(d0)
    np.random.shuffle(d1)

    "Pre-processing training set"
    dy = d0[:,-1]                       # put labels(the last column) to dy
    train_X = d0[(dy==0)]               # Normal(class 1) to train_X
    train_X = train_X[:,0:-1]           # discard the last column (labels)
    train_X = train_X[:n_train]         #downsample\


    "Pre-processing Testing set"
    dX = d1[:,0:-1]                     # put data to dX without last column (labels)
    dy = d1[:,-1]                       # put labels to dy

    dX0 = dX[(dy==0)]                   # Normal, class 0 (cover land)
    dX1 = dX[(dy==1)]                   # Anomaly, class 1 (diseased tree)

    test_X0 = dX0                       #normal test
    test_X1 = dX1                       #anomaly test
    #normal and anomaly test
    test_X = np.concatenate((test_X0, test_X1))

    #Create label for normal and anomaly test examples, and then combine two sets
    test_y0 = np.full((len(test_X0)), False, dtype=bool)
    test_y1 = np.full((len(test_X1)), True,  dtype=bool)
    test_y =  np.concatenate((test_y0, test_y1))
    actual = (~test_y).astype(np.int)

    return train_X, test_X, actual

"*****************************Load dataset*****************************"
def load_data(dataset):
    NSLKDD = ["Probe", "DoS", "R2L", "U2R", "NSLKDD"]
    UNSW   = ["Fuzzers", "Analysis", "Backdoor", "DoS_UNSW", "Exploits", "Generic",\
            "Reconnaissance", "Shellcode", "Worms", "UNSW"]
    CTU13  = ["CTU13_06","CTU13_07","CTU13_08","CTU13_09","CTU13_10","CTU13_12","CTU13_13"]

    if (dataset == "C-heart"):
        d = np.genfromtxt("Data/C-heart.csv", delimiter=",")
        label_threshold = 0
        # 13 attributes + 1 class [0 - Level0(164); level 1,2,3,4 - (139), 6 missing)
        #Some features may be CATEGORICAL, don't need to preprocessing

    elif (dataset == "ACA"):
        d = np.genfromtxt("Data/australian.csv", delimiter=",")
        label_threshold = 0
        # 14 feature + 1 class [ 0 (383 normal), 1 (307 anomaly)]
        # 8 CATEGORICAL FEATURES NEED TO BE PREPROCESSED

    elif (dataset == "WBC"):
        d = np.genfromtxt("Data/wbc.csv", delimiter=",")
        label_threshold = 2
        #9 real-value + 1 class[2-benign(458); 4-malignant(241)], 16 missing

    elif (dataset == "WDBC"):
        d = np.genfromtxt("Data/wdbc.csv", delimiter=",")
        label_threshold = 2
        # 30 real-value + 1 class [2 - benign(357); 4 - malignant(212)]

    elif (dataset in NSLKDD):
        train_X, test_X, actual = process_KDD(dataset)
        return train_X, test_X, actual
        #Tree CATEGORICAL FEATURES NEED TO BE PREPROCESSED

    elif (dataset in UNSW):
        train_X, test_X, actual = process_UNSW(dataset)
        return train_X, test_X, actual

    #**************************** TABLE - 1 **********************************
    elif (dataset == "GLASS"):
        d = np.genfromtxt("Data/glass.csv", delimiter=",")
        label_threshold = 4
        # 9 attributes + 1 class [1-4 - 163 window glass (normal); 5-7 - 51 Non-window glass (anomaly)]

    elif (dataset == "Ionosphere"):
        d = np.genfromtxt("Data/ionosphere.csv", delimiter=",")
        label_threshold = 0
        d = d[:,2:]
        # Ignore the first and second features, using 32 features
        # 34 attributes + 1 class [0 - 225 good (normal); 1 - 126 bad (anomaly)]

    elif (dataset == "PenDigits"):
        train_X, test_X, actual = process_PenDigits()
        #16 real-value + 1 class attribute (0 as Normal - 2 ( or 1,2,3,4,5,6,7,8,9)
        #Training 780 normal, testing 363 normal and 364 anomaly
        return train_X, test_X, actual

    elif (dataset == "Shuttle"):
        train_X, test_X, actual = process_Shuttle()
        #9 real-value + 1 class (1 as Normal -  class 2-7 as anomaly)
        #train_set: 43500, test_set: 14500
        return train_X, test_X, actual

    elif (dataset == "WPBC"):
        d = np.genfromtxt("Data/wpbc.csv", delimiter=",")
        label_threshold = 0
        d = d[:,1:]   #remove ID feature
        # 32 attributes + class [0 - 151 nonrecur (normal); 1 - 47 recur (anomaly)], 4 missing


    #**************************** TABLE - 2 **********************************
    elif (dataset == "Annthyroid"):
        train_X, test_X, actual = process_Annthyroid()
        #21 attributes + 1 class(Class 3 as Normal -  class 1, 2 as anomaly)
        #Training 3488 normal, testing 3178 normal and 250 anomaly
        return train_X, test_X, actual

    elif (dataset =="Arrhythmia"):
        d = np.genfromtxt("Data/arrhythmia.csv", delimiter=",")
        # 452, 245 normal (class 1), 207 anomaly (classes 2 - 16)
        # 279 attributes, 19 attributes have value 0, 1 attribute has many missing data.
        label_threshold = 1

    elif (dataset =="Cardiotocography"):
        d = np.genfromtxt("Data/Cardiotocography.csv", delimiter=",")
        # 21, 1655 normal (class 1), 471 anomaly (classes 2, 3)
        label_threshold = 1

    elif (dataset =="Heartdisease"):
        d = np.genfromtxt("Data/heartdisease.csv", delimiter=",")
        # 270 instances: 150 absence (1-normal) and 120 presence (2-anomaly)
        #13 attributes including some: ORDERED and NOMINAL features
        label_threshold = 1

    elif (dataset =="Hepatitis"):
        d = np.genfromtxt("Data/hepatitis.csv", delimiter=",")
        # 155 instances, 19 attributes + class label( 2- Live 123, 1-die 32)
        #(remove missing: remain class 2: 67, class 1: 13)
        label_threshold = 2

    elif (dataset =="InternetAds"):
        d = np.genfromtxt("Data/internet-ad.csv", delimiter=",")
        # 3264 instances, 1558 attributes + class(0: nonad, 1: Ads), many missing
        label_threshold = 0

    elif (dataset =="PageBlocks"):
        d = np.genfromtxt("Data/page-blocks.csv", delimiter=",")
        # 5473 instances, 10 attributes + class(1 (4913): text, 2-5: hiriz line (329), graphic (28),
        # line (88) or picture (115)
        label_threshold = 1

    elif (dataset =="Parkinson"):
        d = np.genfromtxt("Data/parkinsons.csv", delimiter=",")
        # 195 instances: 48 Healthy (0), 147 Parkinson (1), 22 real-value
        label_threshold = 0

    elif (dataset =="Pima"):
        d = np.genfromtxt("Data/pima.csv", delimiter=",")
        # 768,  500 normal (class 0), 268 diabetes (class 1)
        # 8 real, integer - values attributes
        label_threshold = 0

    elif (dataset =="Spambase"):
        d = np.genfromtxt("Data/spambase.csv", delimiter=",")
        # 4601,  2788 non-spam (normal, 0), 1813 spam (anomaly, 1), 57 real-values
        label_threshold = 0

    elif (dataset == "Wilt"):
        train_X, test_X, actual = process_wilt()
        #5 attributes + 1 class attribute, (0) Non-wilt as Normal - (1) wilt as anomaly
        #Training 4388 ( 4265 - normal, 74 anomaly), testing 500 (313 - normal, 187 - anomaly)
        return train_X, test_X, actual


    elif (dataset =="CTU13_08"):
        d = np.genfromtxt("Data/CTU13/CTU13_08.csv", delimiter=",")
        label_threshold = 1

    elif (dataset =="CTU13_09"):
        d = np.genfromtxt("Data/CTU13/CTU13_09.csv", delimiter=",")
        label_threshold = 1

    elif (dataset =="CTU13_10"):
        d = np.genfromtxt("Data/CTU13/CTU13_10.csv", delimiter=",")
        label_threshold = 1

    elif (dataset =="CTU13_13"):
        d = np.genfromtxt("Data/CTU13/CTU13_13.csv", delimiter=",")
        label_threshold = 1


    elif (dataset =="CTU13_06"):
        d = np.genfromtxt("Data/CTU13/CTU13_06.csv", delimiter=",")
        label_threshold = 1

    elif (dataset =="CTU13_07"):
        d = np.genfromtxt("Data/CTU13/CTU13_07.csv", delimiter=",")
        label_threshold = 1

    elif (dataset =="CTU13_12"):
        d = np.genfromtxt("Data/CTU13/CTU13_12.csv", delimiter=",")
        label_threshold = 1

    else:
        print ("Incorrect data")

    "*************************Chosing dataset*********************************"
    d = d[~np.isnan(d).any(axis=1)]    #discard the '?' values

    np.random.seed(seed)
    np.random.shuffle(d)

    dX = d[:,0:-1]              #put data to dX without the last column (labels)
    dy = d[:,-1]                #put label to dy

    if (dataset =="Hepatitis"):
        dy = dy < label_threshold
    else:
        dy = dy > label_threshold
                                # dy=True with anomaly labels
                                # separate into normal and anomaly
    dX0 = dX[~dy]               # Normal data
    dX1 = dX[dy]                # Anomaly data
    dy0 = dy[~dy]               # Normal label
    dy1 = dy[dy]                # Anomaly label

    #print("Normal: %d Anomaly %d" %(len(dX0), len(dX1)))
    if (dataset in CTU13):
        split = 0.4             #split 40% for training, 60% for testing
    else:
        split = 0.8             #split 80% for training, 20% for testing

    idx0  = int(split * len(dX0))
    idx1  = int(split * len(dX1))

    train_X = dX0[:idx0]        # train_X is 80% of the normal class

    # test set is the other half of the normal class and all of the anomaly class
    test_X = np.concatenate((dX0[idx0:], dX1[idx1:]))  # 30% of normal and 30% of anomaly
    test_y = np.concatenate((dy0[idx0:], dy1[idx1:]))  # 30% of normal and 30% of anomaly label
    #conver test_y into 1 or 0 for computing AUC later
    actual = (~test_y).astype(np.int)

    return train_X, test_X, actual




