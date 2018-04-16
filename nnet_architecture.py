# -*- coding: utf-8 -*-
"""
Created on Thu Apr 06 12:59:05 2017

@author: VANLOI
"""
NSLKDD = ["Probe", "DoS", "R2L", "U2R", "NSLKDD"]
UNSW   = ["Fuzzers", "Analysis", "Backdoor", "DoS_UNSW", "Exploits", "Generic",\
            "Reconnaissance", "Shellcode", "Worms", "UNSW"]
CTU13  = ["CTU13_08", "CTU13_13", "CTU13_10", "CTU13_09"]
        
def hyper_parameters(data):            
    if (data == "C-heart"):          #13   
        h_sizes = [10, 7, 4]     #PPSN, manually

    elif (data == "ACA"):            #14 
        h_sizes = [11, 7, 4] 
        
    elif (data == "WBC"):            #9
        h_sizes = [7, 6, 4]          #tot   
        
    elif (data == "WDBC"):           #30
        h_sizes = [22, 14, 6] 
        
    elif (data in NSLKDD):           #41, 122 
        h_sizes = [85, 49, 12]    
        
    elif (data in UNSW):            #42, 196
        h_sizes = [136, 75, 15]
        
    #Table - 1  
    elif (data == "GLASS"):          #9
        h_sizes = [7, 6, 4] 

    elif (data == "Ionosphere"):     
        h_sizes = [23, 15, 6]        #32 
        #34 features, but using only 32 features. 
        
    elif (data == "PenDigits"):
        h_sizes = [12, 9, 5]        #16 cung tot

    elif (data == "Shuttle"):
        h_sizes = [7, 6, 4]         #9
        
    elif (data == "WPBC"):
        h_sizes = [23, 15, 6]       #32
        
    #Table - 2  
    elif (data == "Annthyroid"):
        h_sizes = [16, 10, 5]      #21 
        
    elif (data == "Arrhythmia"):
        h_sizes = [178, 98, 17]   #259 
        #h_sizes = [138, 17]       #140
        #h_sizes = [50, 10, 2]
        
    elif (data == "Cardiotocography"):
        h_sizes = [16, 10, 5]     #21
        
    elif (data == "Heartdisease"):
        h_sizes = [10, 7, 4]     #13

    elif (data == "Hepatitis"):
        h_sizes = [14, 10, 5]     #19 features  - Ket qua tot  nhat
        
    elif (data == "InternetAds"):
        h_sizes = [1052, 546, 40] #1558 features 
        
    elif (data == "PageBlocks"):
        h_sizes = [8, 6, 4]    #10 features
        
    elif (data == "Parkinson"):
        h_sizes = [16, 11, 5]   #22 
        
    elif (data == "Pima"):
        h_sizes = [6, 5, 3]     #8  
        
    elif (data == "Spambase"):
        h_sizes = [41, 24, 8]  #57  

    elif (data == "Wilt"):
        h_sizes = [4, 4, 3]   #5
    
    
    elif (data == "waveform"):
        h_sizes = [14, 7, 5]
        
    elif (data == "CTU13_08"):
        h_sizes = [29, 18, 7]   #5
        #h_sizes = [27, 15, 2]

    elif (data == "CTU13_09"):
        h_sizes = [30, 18, 7]   #5      
        #h_sizes = [28, 15, 2]

    elif (data == "CTU13_10"):
        h_sizes = [28, 17, 7]   #5      
        #h_sizes = [26, 14, 2]

    elif (data == "CTU13_13"):
        h_sizes = [29, 18, 7]   #5   
        #h_sizes = [27, 15, 2]

    elif (data == "CTU13_06"):
        h_sizes = [28, 17, 7]   #5      
        
    elif (data == "CTU13_07"):
        h_sizes = [25, 15, 6]   #5      
        
    elif (data == "CTU13_12"):
        h_sizes = [26, 17, 7]   #5 
        
    return h_sizes    
    
      