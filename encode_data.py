# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:55:12 2017

@author: VANLOI
"""

import codecs
import numpy as np
from sklearn.preprocessing import OneHotEncoder


"Convert string into number"
def string_to_number(uni_values, data):           
    #create a dictionary
    dic = dict()
    idx = 1
    for key in uni_values:
        dic[key]= idx
        idx=idx+1
        
    #replace category values by numbers
    k=0
    for element in data:
        data[k] = dic[element[0]]
        k=k+1
             
    return data
    
 
"Real-value encoder for NSL-KDD" 
def real_value_encode_KDD(train, test, list_features): 
    
    raw_train = train
    raw_test  = test
    for i in list_features:
        #extract category features
        f_train = []
        f_test  = []
        f_train = raw_train[:,i:i+1]
        f_test  = raw_test[:,i:i+1]
        feature = np.concatenate((f_train, f_test))
        uni_values, indices = np.unique(feature, return_index=True)  #find the set of different values
       
        """
        if (i==1):
            uni_values = ['tcp','udp','icmp']     #re-order protocol
        elif (i==2):
            uni_values = ['http', 'domain_u','smtp','ftp_data'	,'other','private','ftp','telnet',\
                        'urp_i','finger','eco_i','auth','ecr_i','IRC','pop_3','ntp_u','time','X11',\
                        'domain','urh_i','red_i','ssh', 'tim_i','shell','imap4','tftp_u','Z39_50','aol',\
                        'bgp','courier','csnet_ns','ctf','daytime','discard','echo','efs','exec','gopher',\
                        'harvest','hostnames','http_2784','http_443','http_8001','iso_tsap','klogin','kshell',\
                        'ldap','link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat',\
                        'nnsp','nntp','pm_dump','pop_2','printer','remote_job','rje','sql_net','sunrpc','supdup',\
                        'systat','uucp','uucp_path','vmnet','whois']
        else:
            uni_values =['SF','REJ','S1','S0','RSTO','RSTR','S2','S3','OTH','SH','RSTOS0']
        """        
        
        print ("\nFeature_" + str(i) + ":")
        print (uni_values)
        np.savetxt("Data/NSLKDD/feature_" + str(i)+ "_value.csv",  uni_values  ,delimiter=",", fmt="%s")  
        
        f_train = string_to_number(uni_values, f_train)
        f_test  = string_to_number(uni_values, f_test)
        
        #delete the feature from raw_data
        raw_train = np.delete(raw_train, i, axis = 1)
        raw_test  = np.delete(raw_test,  i, axis = 1)
        #insert new features into array
        raw_train = np.insert(raw_train, [i], f_train, axis=1)
        raw_test  = np.insert(raw_test , [i], f_test, axis=1)
    
    return raw_train, raw_test


"One-Hot-Encoder for NSL-KDD"
def one_hot_encode_KDD(raw_train, raw_test, list_features): 

    for i in list_features:
        #extract category features
        f_train = []
        f_test  = []
        f_train = raw_train[:,i:i+1]
        f_test  = raw_test[:,i:i+1]
        feature = np.concatenate((f_train, f_test))
      
        uni_values, indices = np.unique(feature, return_index=True)  
 
        f_train = string_to_number(uni_values, f_train)
        f_test  = string_to_number(uni_values, f_test)
        
        value = np.copy(uni_values)
        value  = np.reshape(value, (len(value),1))
        f_dic  = string_to_number(uni_values, value)  
        
        print (uni_values)
        print (f_dic)
        
        "One Hot Encoder"
        enc = OneHotEncoder()
        enc.fit(f_dic)  
        new_f_train = enc.transform(f_train).toarray()
        new_f_test  = enc.transform(f_test).toarray()
        
        #delete the feature from raw_data
        raw_train = np.delete(raw_train, i, axis = 1)
        raw_test  = np.delete(raw_test, i, axis = 1)
        #insert new features into array
        raw_train = np.insert(raw_train, [i], new_f_train, axis=1)
        raw_test  = np.insert(raw_test , [i], new_f_test, axis=1)
        
    return raw_train, raw_test
        

"Split data into four groups of attacks and normal"
def split_NSLKDD_group(raw_data): 
    
    probe_label  = ['ipsweep', 'nmap', 'portsweep', 'satan',\
                    'mscan', 'saint'] # add 2 more attacks from test set 
    
    dos_label    = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop',\
                    'apache2', 'mailbomb', 'processtable', 'udpstorm'] #add 4 more attacks from test set
                    
                    
    r2l_label    = ['ftp_write','guess_passwd','imap','multihop','phf','spy',\
                    'warezclient','warezmaster',\
                     'httptunnel','named', 'sendmail',\
                    'snmpgetattack','snmpguess', 'worm', 'xlock', 'xsnoop']#add 8 more attacks from test set
                    
    u2r_label    = ['buffer_overflow','loadmodule','perl','rootkit',\
                    'xterm', 'ps', 'sqlattack']  #add 3 more from test set
    
    dim = raw_data.shape[1]
    n   = raw_data.shape[0]
    
    #extract label column
    label =  raw_data[:,(dim-1):dim]
    
    for i in range(n):
        if  (label[i] in probe_label):
            label[i] = 1                #Probe
        elif (label[i] in dos_label):
            label[i] = 2                #DoS
        elif (label[i] in r2l_label):  
            label[i] = 3                #R2L   
        elif (label[i] in u2r_label):  
            label[i] = 4                #U2R
        elif (label[i] == "normal"):    #Normal 
            label[i] = 0
        else:
            print("No group of attacks chosen")
            
     #delete the feature from raw_data
    raw_data = np.delete(raw_data, [dim-1], axis = 1)
    #insert new features into array
    raw_data = np.insert(raw_data, [dim-1], label, axis=1) 
    
    return raw_data
    
    
def encode_NSLKDD(encoder_type):
    raw_train   = np.genfromtxt("Data/Original_IDSData/KDDTrain+.csv", delimiter="," , dtype = 'str')
    raw_test    = np.genfromtxt("Data/Original_IDSData/KDDTest+.csv",  delimiter="," , dtype = 'str')    
#    a1 =   raw_train[:,-1]
#    a2 =   raw_test[:,-1]
#    raw_train = raw_train[np.float64(a1) > 15]
#    raw_test  = raw_test[np.float64(a2) > 15]

    #remove the difficulty level in the last column
    raw_train = raw_train[:,0:-1]
    raw_test  = raw_test[:,0:-1]
    #list of the categorical features
    "1 - protocol_type","2 - service","3 - flag"
    list_features = [3,2,1] 
    
    "1 - duration: continuous"
    "5 - src_bytes: continuous"
    "6 - dst_bytes: continuous"   
    "13- num_compromised: continuous (may not need to log2)"
    "16- num_root: continuous (may not need to log2)"
    "https://github.com/thinline72/nsl-kdd"
    "https://github.com/jmnwong/NSL-KDD-Dataset"
    
    "convert extremely large values features into small values by log2 function"
    feature_log = [1, 5, 6]     #index: 0, 4, 5
    for i in feature_log:
        raw_train[:,i-1:i] = np.log2((raw_train[:,i-1:i]).astype(np.float64)+1)
        raw_test[:, i-1:i] = np.log2((raw_test[:, i-1:i]).astype(np.float64)+1)    
    
    if (encoder_type == "real_value"):
        pro_train, pro_test = real_value_encode_KDD(raw_train, raw_test, list_features)
    elif (encoder_type == "one_hot"):
        pro_train, pro_test = one_hot_encode_KDD(raw_train, raw_test, list_features)
    else:
        print ("no encoder data is choosen")

    pro_train = split_NSLKDD_group(pro_train)
    pro_test  = split_NSLKDD_group(pro_test)
    
    print ("\n")
    print ("train set: ",len(pro_train) )
    print ("test set: ", len(pro_test) )

    np.savetxt("Data/NSLKDD/NSLKDD_Train.csv", pro_train  ,delimiter=",", fmt="%s")
    np.savetxt("Data/NSLKDD/NSLKDD_Test.csv",  pro_test   ,delimiter=",", fmt="%s")    
    
    
"********************************Encode for UNSW******************************"   
"Real-value encoder for NSL-KDD" 
def real_value_encode_UNSW(train, test, list_features): 
    
    raw_train = train
    raw_test  = test
    for i in list_features:
        #extract category features
        
        protocol = ['tcp', 'udp', 'icmp']
        if (i == 1):
            f1_train = raw_train[:,i:i+1]
            f1_test  = raw_test[:,i:i+1]
            for j in range(len(f1_train)):
                if (f1_train[j] not in protocol):
                    f1_train[j] = 'other'   
                    
            for k in range(len(f1_test)):     
                if (f1_test[k] not in protocol):
                    f1_test[k] = 'other'   
            #delete the feature from raw_data
            raw_train = np.delete(raw_train, i, axis = 1)
            raw_test  = np.delete(raw_test,  i, axis = 1)
            #insert new features into array
            raw_train = np.insert(raw_train, [i], f1_train, axis=1)
            raw_test  = np.insert(raw_test , [i], f1_test, axis=1)
    
    
        f_train = []
        f_test  = []
        f_train = raw_train[:,i:i+1]
        f_test  = raw_test[:,i:i+1]
    
        feature = np.concatenate((f_train, f_test))
        
        #find the set of different values
        uni_values, indices = np.unique(feature, return_index=True) 
                  
        print ("\nFeature_" + str(i) + "_: ")
        print (uni_values)
        np.savetxt("Data/UNSW/feature_" + str(i)+ "_value.csv",  uni_values  ,delimiter=",", fmt="%s")  
        
        f_train = string_to_number(uni_values, f_train)
        f_test  = string_to_number(uni_values, f_test)
        
        #delete the feature from raw_data
        raw_train = np.delete(raw_train, i, axis = 1)
        raw_test  = np.delete(raw_test,  i, axis = 1)
        #insert new features into array
        raw_train = np.insert(raw_train, [i], f_train, axis=1)
        raw_test  = np.insert(raw_test , [i], f_test, axis=1)
    
    return raw_train, raw_test


"One-Hot-Encoder for UNSW"
def one_hot_encode_UNSW(raw_train, raw_test, list_features): 

    for i in list_features:
        
#        protocol = ['tcp', 'udp', 'icmp']
#        if (i == 1):
#            f1_train = raw_train[:,i:i+1]
#            f1_test  = raw_test[:,i:i+1]
#            for j in range(len(f1_train)):
#                if (f1_train[j] not in protocol):
#                    f1_train[j] = 'other'   
#                    
#            for k in range(len(f1_test)):     
#                if (f1_test[k] not in protocol):
#                    f1_test[k] = 'other'   
#            #delete the feature from raw_data
#            raw_train = np.delete(raw_train, i, axis = 1)
#            raw_test  = np.delete(raw_test,  i, axis = 1)
#            #insert new features into array
#            raw_train = np.insert(raw_train, [i], f1_train, axis=1)
#            raw_test  = np.insert(raw_test , [i], f1_test, axis=1)


        #extract category features
        f_train = []
        f_test  = []
        f_train = raw_train[:,i:i+1]
        f_test  = raw_test[:,i:i+1]
        feature = np.concatenate((f_train, f_test))
        
        uni_values, indices = np.unique(feature, return_index=True)  
        
        f_train = string_to_number(uni_values, f_train)
        f_test  = string_to_number(uni_values, f_test)
        
        value = np.copy(uni_values)
        value  = np.reshape(value, (len(value),1))
        f_dic  = string_to_number(uni_values, value)  
        
        print (uni_values)
        print (f_dic)
        
        "One Hot Encoder"
        enc = OneHotEncoder()
        enc.fit(f_dic)  
        new_f_train = enc.transform(f_train).toarray()
        new_f_test  = enc.transform(f_test).toarray()
        
        #delete the feature from raw_data
        raw_train = np.delete(raw_train, i, axis = 1)
        raw_test  = np.delete(raw_test, i, axis = 1)
        #insert new features into array
        raw_train = np.insert(raw_train, [i], new_f_train, axis=1)
        raw_test  = np.insert(raw_test , [i], new_f_test, axis=1)
        
    return raw_train, raw_test   
   
"Split data into nine groups of attacks and normal"

def Label_to_Number_UNSW(raw_data): 
    dim = raw_data.shape[1]
    #extract label column
    label =  raw_data[:,(dim-1):dim]
    
    uni_values, indices = np.unique(label, return_index=True) 
    print ("\nList of labels")    
    print (uni_values)
    
    dic = dict()           #create a dictionary
    idx = 0
    #They are real labels of UNSW dataset
    #'Fuzzers': 1, 'Analysis': 2, 'Backdoor': 3, 'DoS': 4, 'Exploits': 5,
    #'Generic': 6, 'Reconnaissance': 7, 'Shellcode': 8, 'Worms': 9
    for key in uni_values:
        if   (key == 'Normal'):
            idx = 0
        elif (key == 'Fuzzers'):
            idx=1
        elif (key == 'Analysis'):
            idx=2        
        elif (key == 'Backdoor'):
            idx=3        
        elif (key == 'DoS'):
            idx=4        
        elif (key == 'Exploits'):
            idx=5
        elif (key == 'Generic'):
            idx=6
        elif (key == 'Reconnaissance'):
            idx=7
        elif (key == 'Shellcode'):
            idx=8
        elif (key == 'Worms'):
            idx=9  
        else:
            print("No attack group match")               
        dic[key]= idx
        
    print ("\nDictionalry for labels") 
    print (dic) 
    
    #convert label into real-number               
    k=0
    for element in label:
        label[k] = dic[element[0]]
        k=k+1  
            
    print ("\nClass number")
    print (label)
                           
    #Replace the label feature with categorical values by label class with numbers
    raw_data = np.delete(raw_data, [dim-1], axis = 1)
    raw_data = np.insert(raw_data, [dim-1], label, axis=1) 
    
    return raw_data
    
    

def encode_UNSW(encoder_type):
    """The information of UNSW15 in document is different from dataset. 
    They put a wrong names for train and test sets, these names need to be swapped"""

    with codecs.open("Data/Original_IDSData/UNSW_NB15_test.csv", encoding='utf-8-sig') as f_train:
        raw_train = np.genfromtxt(f_train, delimiter="," , dtype = 'str' )
    with codecs.open("Data/Original_IDSData/UNSW_NB15_train.csv", encoding='utf-8-sig') as f_test:
        raw_test = np.genfromtxt(f_test, delimiter="," , dtype = 'str' )
        
    #remove the class labels (0, 1) in the last column, and keep the specific 
    #labels (name of attacks or normal) in the sencond last column.    
    raw_train = raw_train[:,0:-1]
    raw_test  = raw_test[:,0:-1]
    list_features = [3,2,1]                        #list of categorical features
    
    "convert extremely large values features into small values by log2 function"
    #7-sbytes, 8-dbytes, 12-Sload, 13-Dload, 21-stcpb, 22-dtcpb  
    #http://networksorcery.com/enp/protocol/tcp.htm 
    feature_log = [7, 8, 12, 13, 21, 22]  #index: 6,7,11,12,20,21
    for i in feature_log:
        raw_train[:,i-1:i]  = np.log2((raw_train[:,i-1:i]).astype(np.float64)+1)
        raw_test[:, i-1:i]  = np.log2((raw_test[:, i-1:i]).astype(np.float64)+1)
    
    "Convert categorical values into number"
    if (encoder_type == "real_value"):
        train_data, test_data = real_value_encode_UNSW(raw_train, raw_test, list_features)
    elif (encoder_type == "one_hot"):
        train_data, test_data = one_hot_encode_UNSW(raw_train, raw_test, list_features)
    else:
        print ("no encoder data is choosen")
    
    "Convert labels (name of attacks) to class number"
    pro_train = Label_to_Number_UNSW(train_data)
    pro_test  = Label_to_Number_UNSW(test_data)   
    
    print ("train set: ",len(pro_train) )
    print ("test set: ", len(pro_test) )

    np.savetxt("Data/UNSW/UNSW_Train.csv", pro_train  ,delimiter=",", fmt="%s")
    np.savetxt("Data/UNSW/UNSW_Test.csv",  pro_test ,delimiter=",", fmt="%s")





"****************************** Processing CTU 13 *****************************"
def find_uni_lable(label):
    uni_label, indices = np.unique(label, return_index=True)
    uni_label =np.reshape(uni_label,(-1,1))
    print(uni_label)

"One-Hot-Encoder for UNSW"
def one_hot_encode_CTU13(data, list_features): 
    data1 = data
    for i in list_features:
        feature = []
        feature = data1[:,i:i+1]
        
        uni_values, indices = np.unique(feature, return_index=True)  
        feature = string_to_number(uni_values, feature)
        
        value = np.copy(uni_values)
        value  = np.reshape(value, (-1,1))
        f_dic  = string_to_number(uni_values, value)  
        
        print (uni_values)
        print (f_dic)
        
        "One Hot Encoder"
        enc = OneHotEncoder()
        enc.fit(f_dic)  
        new_f = enc.transform(feature).toarray()
        print(new_f)
        #delete the feature from raw_data
        data1 = np.delete(data1, i, axis = 1)
        #insert new features into array
        data1 = np.insert(data1, [i], new_f, axis=1)
    return data1     

"Real-value encoder for NSL-KDD" 
def real_value_encode_CTU13(data, list_features): 
    raw_train = data
    for i in list_features:
        feature = []
        feature = raw_train[:,i:i+1]
        
        uni_values, indices = np.unique(feature, return_index=True)  #find the set of different values
        print ("\nFeature_" + str(i) + ":", len(uni_values))
        print (uni_values)
        np.savetxt("Data/CTU13/feature_" + str(i)+ "_value.csv",  uni_values  ,delimiter=",", fmt="%s")  

        feature = string_to_number(uni_values, feature)
        #delete the feature from raw_data
        raw_train = np.delete(raw_train, i, axis = 1)
        #insert new features into array
        raw_train = np.insert(raw_train, [i], feature, axis=1)
    return raw_train

def Label_to_Number_CTU13(raw_data):
    dim = raw_data.shape[1]
    labels =  raw_data[:,(dim-1):dim]
    
    print(labels)
    for j in range(len(labels)):
        l = labels[j]
        if ('Normal' in l[0]):
            labels[j] = 1    
        elif ('Botnet' in l[0]):
            labels[j] = 2
        elif ('Background' in l[0]):
            labels[j] = 3
            
    print ("\nClass number")
    print (labels)
    #Replace the label feature with categorical values by label class with numbers
    raw_data = np.delete(raw_data, [dim-1], axis = 1)
    raw_data = np.insert(raw_data, [dim-1], labels, axis=1) 
    return raw_data

    
def Preprocess_CTU():
    raw_data = np.genfromtxt("Data/Original_CTU13/12.capture20110819.binetflow.txt",
                             delimiter=",", dtype = 'str', skip_header=1)      
                             
    "Select set of features, remove Start time (0), remove IP address (3, 6)"
    selected_features = [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14] 
    data = raw_data[:, selected_features]
    #print(data[:5])
    
    
    print("\nReal-Value Enconder, port number (2,4) and state (5), index start from 0")
    real_value_features=[5,4,2]
    data1 = real_value_encode_CTU13(data, real_value_features)    
    #print(data1[:5])     
    
    
    "Log scale some large value features, dur(0), port(2,4), state(5), 8,9,10"
    print("\nLog")
    feature_log = [0, 2, 4, 5, 8, 9, 10]
    for i in feature_log:
        new  = np.log2((data1[:,i:i+1]).astype(np.float64)+1)
        data1[:,i:i+1] = new
        #print( new)
        
        
    print("\nOne-hot-encode, protocol(1), dir(3), sTos(6), dTos(7)")
    one_hot_features=[7,6,3,1]
    data1 = one_hot_encode_CTU13(data1, one_hot_features)    
    #print(data1)     
    
    
    data1 = Label_to_Number_CTU13(data1)
    #print(data1)
    label = data1[:,-1] 
        
    "Split data into three Background, Botnet and Noraml data"    
    i_data = np.asanyarray((label == '1') | (label == '2'))
    
    Normal_Botnet = data1[i_data]
    Background    = data1[~i_data]
    print("\nBackground: %d, \nNormal and Botnet: %d" %(len(Background), len(Normal_Botnet)))
    
    np.savetxt("Data/CTU13/CTU13_12.csv", Normal_Botnet ,delimiter=",", fmt="%s")                         



"****************************** Processing CTU 13 *****************************"
#Process four datasets together, so processed datasets have the dimension.
"""
def find_uni_lable(label):
    uni_label, indices = np.unique(label, return_index=True)
    uni_label =np.reshape(uni_label,(-1,1))
    print(uni_label)

"One-Hot-Encoder for CTU-13"
def one_hot_encode_CTU13(data, list_features, one_hot_features): 
    data1 = data
    for i, uni_values in zip(list_features, one_hot_features):
        feature = []
        feature = data1[:,i:i+1]
    
        feature = string_to_number(uni_values, feature)
        
        value = np.copy(uni_values)
        value  = np.reshape(value, (-1,1))
        f_dic  = string_to_number(uni_values, value)  
        
        print("\n")
        print (uni_values)
        print (f_dic)
        
        "One Hot Encoder"
        enc = OneHotEncoder()
        enc.fit(f_dic)  
        new_f = enc.transform(feature).toarray()
        print(new_f)
        #delete the feature from raw_data
        data1 = np.delete(data1, i, axis = 1)
        #insert new features into array
        data1 = np.insert(data1, [i], new_f, axis=1)
    return data1     

"Real-value encoder for CTU-13" 
def real_value_encode_CTU13(data, list_features, uni_feature_value): 
    raw_train = data
    for i, uni_values in zip(list_features, uni_feature_value):
        feature = []
        feature = raw_train[:,i:i+1]
        
#        print ("\nFeature_" + str(i) + ":", len(uni_values))
#        print (uni_values)
#        np.savetxt("Data/CTU13/feature_" + str(i)+ "_value.csv",  uni_values  ,delimiter=",", fmt="%s")  

        feature = string_to_number(uni_values, feature)
        #delete the feature from raw_data
        raw_train = np.delete(raw_train, i, axis = 1)
        #insert new features into array
        raw_train = np.insert(raw_train, [i], feature, axis=1)
    return raw_train

def Label_to_Number_CTU13(raw_data):
    dim = raw_data.shape[1]
    labels =  raw_data[:,(dim-1):dim]
    
    print(labels)
    for j in range(len(labels)):
        l = labels[j]
        if ('Normal' in l[0]):
            labels[j] = 1    
        elif ('Botnet' in l[0]):
            labels[j] = 2
        elif ('Background' in l[0]):
            labels[j] = 3
            
    print ("\nClass number")
    print (labels)
    #Replace the label feature with categorical values by label class with numbers
    raw_data = np.delete(raw_data, [dim-1], axis = 1)
    raw_data = np.insert(raw_data, [dim-1], labels, axis=1) 
    return raw_data

    
def Preprocess_CTU():  
    data_08 = np.genfromtxt("Data/Original_CTU13/8.capture20110816-3.binetflow.txt",
                             delimiter=",", dtype = 'str', skip_header=1) 
    data_09 = np.genfromtxt("Data/Original_CTU13/9.capture20110817.binetflow.txt",
                             delimiter=",", dtype = 'str', skip_header=1)
    data_10 = np.genfromtxt("Data/Original_CTU13/10.capture20110818.binetflow.txt",
                             delimiter=",", dtype = 'str', skip_header=1)
    data_13 = np.genfromtxt("Data/Original_CTU13/13.capture20110815-3.binetflow.txt",
                             delimiter=",", dtype = 'str', skip_header=1)
    #Select set of features, remove Start time (0), remove IP address (3, 6)   
    selected_features = [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]
    data_08 =  data_08[:, selected_features]                         
    data_09 =  data_09[:, selected_features]     
    data_10 =  data_10[:, selected_features] 
    data_13 =  data_13[:, selected_features] 
    
    CTU_data = [data_08, data_09, data_10, data_13]    
    #List of features need to be preprocessed
    list_feature  = [7, 6, 5, 4, 3, 2, 1]
    
    f7 = np.empty([0,1]) 
    f6 = np.empty([0,1]) 
    f5 = np.empty([0,1]) 
    f4 = np.empty([0,1]) 
    f3 = np.empty([0,1]) 
    f2 = np.empty([0,1])  
    f1 = np.empty([0,1]) 
    for data in CTU_data:
        for i in list_feature:
            feature = data[:,i:i+1]
            uni_values, indices = np.unique(feature, return_index=True)  #find the set of different values
             
            if (i == 1):
                f1 = np.append(f1, uni_values)
                uni_values1, indices = np.unique(f1, return_index=True)
                f1 = uni_values1
                
            elif (i == 2):
                f2 = np.append(f2, uni_values)
                uni_values2, indices = np.unique(f2, return_index=True)
                f2 = uni_values2
                
            elif (i == 3):
                f3 = np.append(f3, uni_values)
                uni_values3, indices = np.unique(f3, return_index=True)
                f3 = uni_values3      
                
            elif (i == 4):
                f4 = np.append(f4, uni_values)
                uni_values4, indices = np.unique(f4, return_index=True)
                f4 = uni_values4 
                
            elif (i == 5):
                f5 = np.append(f5, uni_values)
                uni_values5, indices = np.unique(f5, return_index=True)
                f5 = uni_values5
                
            elif (i == 6):
                f6 = np.append(f6, uni_values)
                uni_values6, indices = np.unique(f6, return_index=True)
                f6 = uni_values6
    
            elif (i == 7):
                f7 = np.append(f7, uni_values)
                uni_values7, indices = np.unique(f7, return_index=True)
                f7 = uni_values7      
        data = None
                         
    print("\nUni feature values")
    uni_feature_value = [f5, f4, f2]
    f_name_value = ["feature_5", "feature_4", "feature_2" ]
    for fv, f_name1 in zip(uni_feature_value, f_name_value):
        print("\n" + f_name1 +  ": %d" %len(fv))
        print(fv)
        np.savetxt("Data/CTU13/" + f_name1 + "_value1.csv",  fv  ,delimiter=",", fmt="%s") 
        
        
    print("\nUni feture on hot")    
    uni_feature_onehot = [f7, f6, f3, f1] 
    f_name_onhot = ["feature_7", "feature_6", "feature_3", "feature_1"]    
    for fo, f_name2 in zip(uni_feature_onehot, f_name_onhot) :
        print("\n" + f_name2 +": %d" %len(fo))
        print(fo)
        np.savetxt("Data/CTU13/" + f_name2 + "_onhot.csv",  fo  ,delimiter=",", fmt="%s")         
        
        
    real_value_features=[5,4,2]
    one_hot_features=[7,6,3,1]
    log_feature = [0, 2, 4, 5, 8, 9, 10]

    name = ["CTU13_08", "CTU13_09", "CTU13_10", "CTU13_13"]
    for data, n in zip(CTU_data, name):
        print("\nReal-Value Enconder, port number (2, 4) and state (5), index start from 0")
        data = real_value_encode_CTU13(data, real_value_features, uni_feature_value)        
    
        print("\nLog scale some large value features, dur(0), port(2,4), state(5), 8,9,10")
        for i in log_feature:
            new  = np.log2((data[:,i:i+1]).astype(np.float64)+1)
            data[:,i:i+1] = new 
            new = None
        
        print("\nOne-hot-encode, protocol(1), dir(3), sTos(6), dTos(7)")
        data = one_hot_encode_CTU13(data, one_hot_features, uni_feature_onehot)    
    
        data = Label_to_Number_CTU13(data)
        label = data[:,-1] 
        
        "Split data into three Background, Botnet and Noraml data"    
        i_data = np.asanyarray((label == '1') | (label == '2'))
        i_normal = np.asanyarray(label == '1')  
        i_botnet = np.asanyarray(label == '2')
        
    
        Normal_Botnet = data[i_data]
        Normal        = data[i_normal]
        Botnet        = data[i_botnet]
        Background    = data[~i_data]
        print("\nNormal and Botnet: %d, \nNormal: %d, \nBotnet: %d, \nBackground: %d, "\
                         %(len(Normal_Botnet), len(Normal), len(Botnet), len(Background)))
    
        np.savetxt("Data/CTU13/" + n + ".csv", Normal_Botnet ,delimiter=",", fmt="%s")      
        
        Normal_Botnet = None
        Normal        = None
        Botnet        = None
        Background    = None
        data          = None 
"""

   
def check_data_information():
    raw_data = np.genfromtxt("Data/CTU13/13.capture20110815-3.binetflow.txt",
                             delimiter=",", dtype = 'str', skip_header=1) 
    "Found unique labels"
    labels = raw_data[:,-1]
    
    uni_label, indices = np.unique(labels, return_index=True)
    uni_label =np.reshape(uni_label,(-1,1))
    print(uni_label)    
    
    print(len(raw_data))

    i_botnet      = np.asanyarray([('Botnet'     in s) for s in labels])
    i_cc          = np.asanyarray([('CC'         in s) for s in labels])
    i_normal      = np.asanyarray([('Normal'     in s) for s in labels])
    i_background  = np.asanyarray([('Background' in s) for s in labels])
    
    label_botnet = labels[i_botnet]
    uni_botnet, indices = np.unique(label_botnet, return_index=True)
    uni_botnet = np.reshape(uni_botnet,(-1,1))
    print("\n Label ALL Botnet")
    print(uni_botnet)    
    
    label_CC = labels[i_cc]
    uni_cc, indices = np.unique(label_CC, return_index=True)
    uni_cc = np.reshape(uni_cc,(-1,1))
    print("\n Label CC")
    print(uni_cc)  

    label_botnet1 = labels[(i_botnet & (~i_cc))]
    uni_botnet1, indices = np.unique(label_botnet1, return_index=True)
    uni_botnet1 = np.reshape(uni_botnet1,(-1,1))
    print("\n Label Botnet")
    print(uni_botnet1)

    label_background = labels[i_background]
    uni_background, indices = np.unique(label_background, return_index=True)
    uni_background = np.reshape(uni_background,(-1,1))
    print("\n Label Background")
    print(uni_background)

    label_normal = labels[i_normal]
    uni_normal, indices = np.unique(label_normal, return_index=True)
    uni_normal = np.reshape(uni_normal,(-1,1))
    print("\n Label Background")
    print(uni_normal)

    CC         = raw_data[i_cc]           #botnet master CC 
    Botnet     = raw_data[(i_botnet & (~i_cc))]         #botnet

    Normal     = raw_data[i_normal]
    Background = raw_data[i_background]
    
    print("\nTotal: %d \nBotnet: %d, \nCC: %d, \nBotnet_all %d , \nNormal: %d, \nBackground: %d"\
        %(len(raw_data), len(Botnet), len(CC), len(Botnet) + len(CC) , len(Normal), len(Background)))    
   
# one_hot, real_value    
encode_UNSW("one_hot")
encode_NSLKDD("one_hot")
#check_data_information()   
#Preprocess_CTU()

    
