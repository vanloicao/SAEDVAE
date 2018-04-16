# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:48:37 2016

@author: VANLOI
"""
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np



from ProcessingData import load_data


def Plotting_AUC(name_dataset, path_result, training_size,
                 FPR_auto, TPR_auto, auc_auto,
                 FPR_cen, TPR_cen, auc_cen,
                 FPR_kde, TPR_kde, auc_kde):
    print ("\n*********************** Plot AUC *************************")
    plt.figure(figsize=(6,6))
    plt.title('The ROC curves - '+ name_dataset, fontsize=16)
    plt.plot(FPR_auto, TPR_auto, 'g-^'  , label='OCAE      (AUC = %0.3f)'% auc_auto, markevery = 150 , markersize = 6)
    plt.plot(FPR_cen, TPR_cen,   'b-o' ,  label='OCCEN    (AUC = %0.3f)'% auc_cen, markevery = 150 , markersize = 6)
    plt.plot(FPR_kde, TPR_kde, 'r-x' , label='OCKDE    (AUC = %0.3f)'% auc_kde, markevery = 150 , markersize = 6)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.savefig(path_result + "fig_" + name_dataset +"_" + training_size + "_Auc.pdf")
    plt.show()

def Plotting_End2End_RE(RE, epoch, ymin, ymax, data_name, path):
    """Plotting RE on train_set and validation_set of the End-to-End traing
    process"""

    plt.figure(figsize=(6,3))
    #plt.title('End-to-End training RE on ' + data_name, fontsize=16)
    plt.xlim([0.0, epoch + 1.0])
    plt.ylim([ymin,ymax])

    x  = RE[:,0]
    y1 = RE[:,1]
    y2 = RE[:,2]
    plt.plot(x, y1,  'b', label = 'Validation set')
    plt.plot(x, y2,  'r', label = 'Training set')
    plt.legend(loc='upper right')
    plt.ylabel('Error', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.savefig(path + data_name +"_End2End_loss.pdf")
    plt.show()

def Plotting_Loss_Component(LOSS, RE, ymin, ymax, data_name, path):
    """Plotting RE on train_set and validation_set of the End-to-End traing
    process"""

    plt.figure(figsize=(6,3))
    #plt.title('End-to-End training RE on ' + data_name, fontsize=16)

    x  = LOSS[:,0]
    y1 = LOSS[:,1]
    y2 = LOSS[:,2]
    y3 = RE[:,2]

    plt.xlim([0.0, max(x) + 1.0])
    plt.ylim([ymin,ymax])

    plt.plot(x, y1,  'b', label = 'Recon error')
    plt.plot(x, y2,  'g', label = 'KL-divergence')
    plt.plot(x, y3,  'r', label = 'Training error')
    plt.legend(loc='upper right')
    plt.ylabel('Errors', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.savefig(path + data_name +"_traing_errors.pdf")
    plt.show()


"*****************************************************************************"
def Plotting_Monitor(RE, ymin, ymax, data_name, path):
    """Plotting RE on train_set and validation_set of the End-to-End traing
    process"""

    plt.figure(figsize=(6,3))
    #plt.title('Monitoring AUC every 5 epoch on ' + data_name, fontsize=16)
    ax = plt.subplot(111)

    x   = RE[:,0]
    lof = RE[:,1]
    cen = RE[:,2]
    dis = RE[:,3]
    kde = RE[:,4]
    svm5 = RE[:,5]
    svm1 = RE[:,6]
    ae  = RE[:,7]

    plt.xlim([0.0, max(x) + 1.0])
    plt.ylim([ymin,ymax])
    ax = plt.subplot(111)

    plt.plot(x, lof,  'r-o', label = 'LOF', markevery = 20 , markersize = 6)
    plt.plot(x, cen,  'b-x', label = 'CEN', markevery = 20 , markersize = 6)
    plt.plot(x, dis,  'g-^', label = 'MDIS', markevery = 20 , markersize = 6)
    plt.plot(x, kde,  'y-x', label = 'KDE', markevery = 20 , markersize = 6)
    plt.plot(x, svm5,  'r-^', label = 'SVM05', markevery = 20 , markersize = 6)
    plt.plot(x, svm1,  'g-o', label = 'SVM01', markevery = 20 , markersize = 6)
    plt.plot(x, ae,   'b-^', label = 'AE' , markevery = 20 , markersize = 6)

    ax.legend(bbox_to_anchor=(0.99, 0.28), ncol= 3, fontsize = 'medium')

    #plt.legend(loc='upper right')
    plt.ylabel('AUC', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.savefig(path + data_name + "_Monitor_AUCs.pdf")
    plt.show()


#def Plotting_Monitor(AUC_hidden, epoch, ymin, ymax, data_name, path):
#    """Plotting RE on train_set and validation_set of the End-to-End traing
#    process"""
#
#    plt.figure(figsize=(8,4))
#
#    x   = AUC_hidden[:,0]
#    lof = AUC_hidden[:,1]
#    cen = AUC_hidden[:,2]
#    dis = AUC_hidden[:,3]
#    kde = AUC_hidden[:,4]
#    svm5 = AUC_hidden[:,5]
#    svm1 = AUC_hidden[:,6]
#    ae  = AUC_hidden[:,7]
#
#    ax0 = plt.subplot2grid((1, 8), (0, 0), colspan=7)
#    plt.xlim([0.0, epoch + 1.0])
#    plt.ylim([ymin,ymax])
#    ax0.plot(x, lof,  'r-o', label = 'LOF', markevery = 20 , markersize = 6)
#    ax0.plot(x, cen,  'b-x', label = 'CEN', markevery = 20 , markersize = 6)
#    ax0.plot(x, dis,  'g-^', label = 'DIS', markevery = 20 , markersize = 6)
#    ax0.plot(x, kde,  'y-x', label = 'KDE', markevery = 20 , markersize = 6)
#    ax0.plot(x, svm5, 'r-^', label = 'SVM05', markevery = 20 , markersize = 6)
#    ax0.plot(x, svm1, 'g-o', label = 'SVM01', markevery = 20 , markersize = 6)
#    ax0.plot(x, ae,   'b-^', label = 'AE' , markevery = 20 , markersize = 6)
#
#    ax0.legend(bbox_to_anchor=(0.99, 0.32), ncol= 3)
#    #plt.legend(loc='upper right')
#    plt.ylabel('AUC', fontsize=14)
#    plt.xlabel('Epochs', fontsize=14)
#
#    ax1 = plt.subplot2grid((1, 8), (0, 7))
#    plt.ylim([ymin,ymax])
#    ax1.plot(x, lof,  'r-o', label = 'LOF', markevery = 20 , markersize = 6)
#    ax1.plot(x, cen,  'b-x', label = 'CEN', markevery = 20 , markersize = 6)
#    ax1.plot(x, dis,  'g-^', label = 'DIS', markevery = 20 , markersize = 6)
#    ax1.plot(x, kde,  'y-x', label = 'KDE', markevery = 20 , markersize = 6)
#    ax1.plot(x, svm5, 'r-^', label = 'SVM05', markevery = 20 , markersize = 6)
#    ax1.plot(x, svm1, 'g-o', label = 'SVM01', markevery = 20 , markersize = 6)
#    ax1.plot(x, ae,   'b-^', label = 'AE' , markevery = 20 , markersize = 6)
#    ax1.axes.get_yaxis().set_visible(False)
#    ax1.axes.get_xaxis().set_visible(False)
#
#    plt.tight_layout()
#    plt.savefig(path + data_name + "_Monitor_AUCs.pdf")
#    plt.show()

def Plotting_Pre_RE(RE, n_layers, epoch, ymin, ymax, batch_size, data_name):
    """Plotting REs of each dAE in the pre-training process"""
    plt.figure(figsize=(8,4))
    plt.title('Pre-training RE on' + data_name + '- Batch size = ' + str(batch_size), fontsize=16)
    plt.xlim([0.0, epoch + 1.0])
    plt.ylim([ymin,ymax])

    color = ['b', 'g', 'r', 'y']
    label = ["layer 1", "layer 2", "layer 3", "layer 4"]

    ax = plt.subplot(111)
    x  = RE[:,0]
    for i in range(n_layers):
        y = RE[:,i+1]
        plt.plot(x, y,  color[i], label = label[i])

    ax.legend(bbox_to_anchor=(0.99, 0.99), ncol=n_layers)
    #plt.legend(loc='upper right')
    plt.ylabel('Reconstruction errors', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.show()

"Plotting recontruction error from three autoencoders together"
def Plotting_Pre_RE1(re, stop_epoch, n_layers, ymin, ymax, batch_size, data_name, path):
    """Plotting REs of each dAE in the pre-training process"""
    plt.figure(figsize=(8,4))
    #plt.title('Pre-training RE on ' + data_name + ' - Batch size = ' + str(batch_size), fontsize=16)

    max_epoch = np.max(stop_epoch)

    plt.xlim([0.0, max_epoch + 1.0])
    plt.ylim([ymin,ymax])

    color = ['b', 'g', 'r', 'y']
    label = ["layer 1", "layer 2", "layer 3", "layer 4"]

    ax = plt.subplot(111)

    for i in range(n_layers):
        x = None
        y = None
        x = np.array(range(int(stop_epoch[i])))   #stop epoches of layer i
        y = re[:,i]                               #pre-train errors of layer i
        y = y[:len(x)]
        plt.plot(x, y,  color[i], label = label[i]) #plot pre-train errors of each layer

    ax.legend(bbox_to_anchor=(0.99, 0.99), ncol=n_layers)
    #plt.legend(loc='upper right')
    plt.ylabel('Reconstruction Error', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.savefig(path + data_name + "_Pre_train.pdf")
    plt.show()

#"Each subplot for ploting reconstruction error of each autoencoder"
#def Plotting_Pre_RE1(re, stop_epoch, n_layers, ymin, ymax, batch_size, data_name):
#    """Plotting REs of each dAE in the pre-training process"""
#
#    max_epoch = np.max(stop_epoch)
#    color = ['b-', 'g-', 'r-']
#    plt.subplots(ncols=3, nrows = 1, figsize=(8, 3))
#
#    for i in range(n_layers):
#        x = np.array(range(int(stop_epoch[i])))   #stop epoches of layer i
#        y = re[:,i]                               #pre-train errors of layer i
#        y = y[:len(x)]
#        fig = plt.subplot(1, 3, i+1)
#
#        plt.xlim([0.0, max_epoch + max_epoch/20])
#        plt.ylim([ymin,ymax])
#
#        plt.plot(x, y, color[i])
#        plt.legend(['Layer ' + str(i+1)])
#        if (i==0):
#            plt.ylabel('Reconstruction Error', fontsize=14)
#        else:
#            fig.axes.get_yaxis().set_visible(False)
#
#        plt.xlabel('Epochs', fontsize=14)
#
#        plt.yticks(fontsize=10)
#        plt.xticks(rotation = 30, fontsize=10)
#
#        #hide the first zero in axes matplotlib
#        ax = plt.gca()
#        xticks = ax.xaxis.get_major_ticks()
#        xticks[0].label1.set_visible(False)
#
#    plt.subplots_adjust(wspace=0.005, hspace=0)
#    plt.savefig(path + data_name + "_Pre_train.pdf")
#    plt.show()


def Plotting_AUC_RE(AUC_RE, dataset, ymin, ymax, path):
    """Plotting AUC against training-RE when evaluting the model. This is aim to
    do gridsearch over batch_sizes to choose the best performanced model.
    Hopfully, the smaller training-RE the model produces, the higher accuracy
    when evaluting the model on testing set"""
    plt.figure(figsize=(8,4))
    plt.title('AUC against RE - '+ dataset, fontsize=16)

    #Sorted AUC_RE by reconstruction error
    AUC_RE = AUC_RE[np.argsort(AUC_RE[:,9])]

    x = AUC_RE[:,9]
    plt.xlim( x[0] - (x[-1]-x[0])/20 , x[-1] + (x[-1]-x[0])/20)
    plt.ylim([ymin, ymax])

    y01 = AUC_RE[:,2]  #AUC of LOF
    y11 = AUC_RE[:,3]  #AUC of CEN
    y21 = AUC_RE[:,4]  #AUC of NDIS
    y31 = AUC_RE[:,5]  #AUC of KDE
    y41 = AUC_RE[:,6]  #AUC of KDE
    y51 = AUC_RE[:,8]  #AUC of AE


    ax = plt.subplot(111)

    plt.plot(x, y01,  'b-p', label = 'LOF', markersize = 6)
    plt.plot(x, y11,  'r-p', label = 'CEN', markersize = 6)
    plt.plot(x, y21,  'g-^', label = 'NDIS',markersize = 6)
    plt.plot(x, y31,  'y-d', label = 'KDE',markersize = 6)
    plt.plot(x, y41,  'r-s', label = 'SVM05',markersize = 6)
    plt.plot(x, y51,  'b-s', label = 'AE',markersize = 6)
    #b: blue | g: green | r: red | c: cyan | m: magenta | y: yellow | k: black | w: white

    ax.legend(bbox_to_anchor=(0.99, 0.25), ncol=3)
    plt.ylabel('AUC Value', fontsize=14)
    plt.xlabel('Reconstruction Error x 100', fontsize=14)
    plt.savefig(path + "AUC_RE_" + dataset +".pdf")
    plt.show()

def Plotting_AUC_Batch_Size(AUC_RE, dataset, ymin, ymax, path):
    """Plotting AUC against training-RE when evaluting the model. This is aim to
    do gridsearch over batch_sizes to choose the best performanced model.
    Hopfully, the smaller training-RE the model produces, the higher accuracy
    when evaluting the model on testing set"""
    plt.figure(figsize=(8,4))
    plt.title('AUC against RE - '+ dataset, fontsize=16)

    #Sorted AUC_RE by reconstruction error
    #AUC_RE = AUC_RE[np.argsort(AUC_RE[:,9])]

    x = AUC_RE[:,0]
    plt.xlim( x[0] - 1 , x[-1] + 1)
    plt.ylim([ymin, ymax])

    y01 = AUC_RE[:,2]  #AUC of LOF
    y11 = AUC_RE[:,3]  #AUC of CEN
    y21 = AUC_RE[:,4]  #AUC of NDIS
    y31 = AUC_RE[:,5]  #AUC of KDE
    y41 = AUC_RE[:,6]  #AUC of KDE
    y51 = AUC_RE[:,8]  #AUC of AE


    ax = plt.subplot(111)

    plt.plot(x, y01,  'b-p', label = 'LOF', markersize = 6)
    plt.plot(x, y11,  'r-p', label = 'CEN', markersize = 6)
    plt.plot(x, y21,  'g-^', label = 'NDIS',markersize = 6)
    plt.plot(x, y31,  'y-d', label = 'KDE',markersize = 6)
    plt.plot(x, y41,  'r-s', label = 'SVM05',markersize = 6)
    plt.plot(x, y51,  'b-s', label = 'AE',markersize = 6)
    #b: blue | g: green | r: red | c: cyan | m: magenta | y: yellow | k: black | w: white

    ax.legend(bbox_to_anchor=(0.99, 0.25), ncol=3)
    plt.ylabel('AUC Value', fontsize=14)
    plt.xlabel('Reconstruction Error x 100', fontsize=14)
    plt.savefig(path + "AUC_RE_" + dataset +".pdf")
    plt.show()

def Plotting_AUC_BW(AUC_Hidden, dataset, xmax, ymin, ymax, training_size, path ):
    plt.figure(figsize=(10,6))
    plt.title('AUC against BW - '+ dataset, fontsize=16)
    plt.xlim([0.0, xmax])
    plt.ylim([ymin, ymax])

    x   = AUC_Hidden[:,0]
    y11 = AUC_Hidden[:,1]
    y21 = AUC_Hidden[:,2]
    y31 = AUC_Hidden[:,3]
    y41 = AUC_Hidden[:,4]
    y51 = AUC_Hidden[:,5]

    plt.plot(x, y11,  'b-s', label = 'KDE      - Hidden',markersize = 6)
    plt.plot(x, y21,  'r-p', label = 'Negative Distance', markersize = 6)
    plt.plot(x, y31,  'g-^', label = 'SVM(0.5) - Hidden',markersize = 6)
    plt.plot(x, y41,  'y-d', label = 'SVM(0.2) - Hidden',markersize = 6)
    plt.plot(x, y51,  'm-s', label = 'SVM(0.1) - Hidden',markersize = 6)
    #b: blue | g: green | r: red | c: cyan | m: magenta | y: yellow | k: black | w: white
    plt.legend(loc='lower right')
    plt.ylabel('AUC Value', fontsize=14)
    plt.xlabel('Bandwidth', fontsize=14)
    plt.savefig(path + "AUC_BW_" + dataset + "_"+ training_size +".pdf")
    plt.show()


def plot_auc_size_input(data, data_name, sizes, path):

    # data to plot
    n_groups = 5
    LOF     = data[:,0:1]
    CEN     = data[:,1:2]
    DIS     = data[:,2:3]
    KDE     = data[:,3:4]
    SVM05   = data[:,4:5]
    SVM01   = data[:,-1]

    plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)

#    plt.title(data_name + ' Attack Group', fontsize=16)
    plt.ylim([0.0, 1.0])
#    plt.ylim([0.0, m + m/5])

    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 1.0

    plt.bar(index + bar_width, LOF, bar_width,
                 alpha=opacity, color='b', label='LOF')

    plt.bar(index + 2*bar_width, CEN, bar_width,
                 alpha=opacity,color='g',label='CEN')

    plt.bar(index + 3*bar_width , DIS, bar_width,
                 alpha=opacity,color='r',label='DIS')

    plt.bar(index + 4*bar_width, KDE, bar_width,
                 alpha=opacity,color='y',label='KDE')

    plt.bar(index + 5*bar_width, SVM05, bar_width,
                 alpha=opacity,color='c',label='SVM05')

    plt.bar(index + 6*bar_width, SVM01, bar_width,
                 alpha=opacity,color='maroon',label='SVM01')

    ax.legend(bbox_to_anchor=(1.04, 0.42), ncol=1, fontsize = 'small')

    plt.xlabel('Size of training set', fontsize=16)
    plt.ylabel('AUC', fontsize=16)
    plt.yticks(fontsize=12)

    ax.yaxis.grid(True)

    plt.xticks(index + 2*bar_width, ('0.5%('+str(sizes[0])+')', '1%('+str(sizes[1])+')', '5%('+str(sizes[2])+')', '10%('+str(sizes[3])+')', '20%('+str(sizes[4])+')'),rotation=15,fontsize=12)

    plt.tight_layout()
    plt.savefig(path + data_name + "_auc_size.pdf")
    plt.show()




"Plot AUC vs Size of training data - 1"
def plot_auc_size_1(data, data_name, sizes, path):
    # data to plot
    n_groups = 5
    LOF     = data[:,0:1]
    CEN     = data[:,1:2]
    DIS     = data[:,2:3]
    KDE     = data[:,3:4]
    SVM05   = data[:,4:5]
    SVM01   = data[:,5:6]
#    RE      = data[:,-1]

    plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)

#    plt.title(data_name + ' Attack Group', fontsize=16)
    plt.ylim([0.0, 1.0])
#    plt.ylim([0.0, m + m/5])

    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 1.0

    plt.bar(index + bar_width, LOF, bar_width,
                 alpha=opacity, color='cyan', label='LOF')

    plt.bar(index + 2*bar_width, CEN, bar_width,
                 alpha=opacity,color='yellow',label='CEN')

    plt.bar(index + 3*bar_width , DIS, bar_width,
                 alpha=opacity,color='magenta',label='NDIS')

    plt.bar(index + 4*bar_width, KDE, bar_width,
                 alpha=opacity,color='blue',label='KDE')

    plt.bar(index + 5*bar_width, SVM05, bar_width,
                 alpha=opacity,color='lightblue',label=r'SVM$_{\nu = 0.5}$')

    plt.bar(index + 6*bar_width, SVM01, bar_width,
                 alpha=opacity,color='plum',label=r'SVM$_{\nu = 0.1}$')

#    plt.bar(index + 7*bar_width, RE, bar_width,
#                 alpha=opacity,color='springgreen',label='RE-Based')
    #xx-small, x-small, small, medium, large, x-large, xx-large
    ax.legend(bbox_to_anchor=(1.0, 0.30), ncol=2, fontsize = 'large')

    plt.xlabel('Size of training set', fontsize=16)
    plt.ylabel('AUC', fontsize=16)
    plt.yticks(fontsize=12)

    ax.yaxis.grid(True)

    plt.xticks(index + 2*bar_width, (str(sizes[0]),\
                                     str(sizes[1]),\
                                     str(sizes[2]),\
                                     str(sizes[3]),\
                                     str(sizes[4])),\
                                     rotation=0,fontsize=12)

    plt.tight_layout()
    plt.savefig(path + data_name + "_auc_size.pdf")
    plt.show()

#from matplotlib import rcParams
##rcParams['mathtext.default'] = 'regular'
#rcParams['text.usetex']=True
# Mathtext font size
# \tiny, \small, \normalsize, \large, \Large, \LARGE, \huge and \Huge
"Plot AUC vs Size of training data - 2"
def plot_auc_size_2(data, data_name, name, sizes, method, path):
    cl = ["LOF", "CEN", "MDIS", "KDE", r'OCSVM$_{\nu=0.5}$', r'OCSVM$_{\nu=0.1}$']
    # data to plot
    n_groups = 6
    Z500    = data[:,0:1]
    Z1000     = data[:,1:2]
    Z2000     = data[:,2:3]
    Z5000     = data[:,3:4]
    Z10000      = data[:,-1]
#    RE      = data[:,-1]

    plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)

    plt.title(name , fontsize=16)
    plt.ylim([0.5, 1.0])
#    plt.ylim([0.0, m + m/5])

    index = np.arange(n_groups)
    bar_width = 0.125
    space     = 0.0
    opacity = 1.0

    plt.bar(index + bar_width+space, Z500, bar_width,
                 alpha=opacity, color='cyan', label='500')

    plt.bar(index + 2*(bar_width+space), Z1000, bar_width,
                 alpha=opacity,color='yellow',label='1000')

    plt.bar(index + 3*(bar_width+space) , Z2000, bar_width,
                 alpha=opacity,color='magenta',label='2000')

    plt.bar(index + 4*(bar_width+space), Z5000, bar_width,
                 alpha=opacity,color='blue',label='5000')

    plt.bar(index + 5*(bar_width+space), Z10000, bar_width,
                 alpha=opacity,color='lightblue',label='10000')


#    plt.bar(index + 7*bar_width, RE, bar_width,
#                 alpha=opacity,color='springgreen',label='RE-Based')
    #xx-small, x-small, small, medium, large, x-large, xx-large
    ax.legend(bbox_to_anchor=(1.01, 0.61), ncol=1, fontsize = 'x-large')

#    plt.xlabel('Size of training set', fontsize=16)
    plt.ylabel('AUC', fontsize=16)
    plt.yticks(fontsize=14)

    ax.yaxis.grid(True)

    plt.xticks(index + 3*bar_width, (method + cl[0],\
                                     method + cl[1],\
                                     method + cl[2],\
                                     method + cl[3],\
                                     method + cl[4],\
                                     method + cl[5]),\
                                     rotation=20,fontsize=14)

    plt.tight_layout()
    plt.savefig(path + data_name + "_auc_size.pdf")
    plt.show()



"Visualize the hidden data in two dimension"
def visualize_hidden(train_set, test_set, actual, data_name, data, path):

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_set)

    train_set = scaler.transform(train_set)
    test_set  = scaler.transform(test_set)

    test_X0 = test_set[(actual==1)]
    test_X1 = test_set[(actual==0)]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    if data == "train":
        plt.plot(train_set[:,0], train_set[:,1], 'bo', ms=5, mec="b", label="Normal Train")
    elif data == "normal":
        plt.plot(test_X0[:,0],   test_X0[:,1],   'go', ms=5, mec="g", label="Normal Test")
    else:
        plt.plot(test_X1[:,0],   test_X1[:,1],   'r^', ms=5, mec="r", label= "Anomaly Test")

    ax.legend(bbox_to_anchor=(1.0, 1.0), ncol=3 )

    plt.axis('equal')
    plt.ylim((-10.0, 10.0))
    plt.xlim((-10.0, 10.0))
    plt.tight_layout()
    #plt.savefig(path + data_name + "_v_hid_train_" + dataset + ".pdf")
    plt.show()
    plt.close



"Each subplot for ploting reconstruction error of each autoencoder"
def visualize_hidden1(train_set, test_set, actual, data_name, path):
    """Plotting REs of each dAE in the pre-training process"""

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_set)

    train_set = scaler.transform(train_set)
    test_set  = scaler.transform(test_set)

    test_X0 = test_set[(actual==1)]
    test_X1 = test_set[(actual==0)]

    plt.subplots(ncols=3, nrows = 1, figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.axis('equal')
    plt.ylim((-10.0, 10.0))
    plt.xlim((-10.0, 10.0))
    plt.plot(train_set[:,0], train_set[:,1], 'bo', ms=5, mec="b", label="Normal Train")
    plt.legend(["Normal Train"])

    fig = plt.subplot(1, 3, 2)
    plt.axis('equal')
    plt.ylim((-10.0, 10.0))
    plt.xlim((-10.0, 10.0))
    plt.plot(test_X0[:,0],   test_X0[:,1],   'go', ms=5, mec="g", label="Normal Test")
    fig.axes.get_yaxis().set_visible(False)
    plt.legend(["Normal Test"])

    fig = plt.subplot(1, 3, 3)
    plt.axis('equal')
    plt.ylim((-10.0, 10.0))
    plt.xlim((-10.0, 10.0))
    plt.plot(test_X1[:,0],   test_X1[:,1],   'r^', ms=5, mec="r", label= "Anomaly Test")
    fig.axes.get_yaxis().set_visible(False)
    plt.legend(["Anomaly Test"])

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig(data_name + "_Visualize.pdf")
    plt.show()


"Investigate Bandwidth/gramma parameters of SVM and KDE"
def Plot_AUC_Bandwidth(auc, data_name, X_max, n_features ,path):

    bw    = auc[:,0]
    kde   = auc[:,1]
    svm05 = auc[:,2]
    svm01 = auc[:,3]
    default_bw = (n_features/2.0)**0.5

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(111)

    plt.xlim([0.0, X_max])
    plt.ylim([0.40,1.0])

    plt.plot(bw, svm05,  'r-o', ms=6, mec="r", label =r'$\mathrm{SVM}_{\nu = 0.5}$', markevery = 3)
    plt.plot(bw, svm01,  'g-^', ms=6, mec="g", label =r'$\mathrm{SVM}_{\nu = 0.1}$', markevery = 3)
    plt.plot(bw, kde,    'b-x', ms=6, mec="b", label= 'KDE', markevery = 3)
    plt.legend(bbox_to_anchor=(1.01, 0.15), ncol=3)
    ax1.set_ylabel('AUC', fontsize=14)
    ax1.set_xlabel('Bandwidth', fontsize=14)


    ax2 = ax1.twiny()
    new_tick_locations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    def tick_function(bw):
        gamma = 1.0/(2.0*bw*bw)
        return ["%.3f" % z for z in gamma]

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r"Gamma($\gamma$) =  $1/(2*bandwidth^{2})$", fontsize=14 )


    sparse_data = ["Arrhythmia", "Spambase", "UNSW", "NSLKDD", "InternetAds"]
    if (data_name in sparse_data):
        x_text = 4.0
    else:
        x_text = 0.5
    ax1.annotate('default value',
            xy=(default_bw, 1.0), xytext=(x_text, 1.06),
            arrowprops=dict(facecolor='green', arrowstyle="->"))

    plt.tight_layout()
    plt.savefig(path + "Bandwith_auc/" + data_name + "_BW.pdf")
    plt.show()



def plot_sparsity_auc_bar(data, improve_auc, spa_score, method, path):

    #Using CTU13-13 to demonstrate for four CTU13 datasets, [6,7,8,9]
    id_data = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13]
    improve_auc = improve_auc[id_data]
    spa_score   = spa_score[id_data]        #sparsity score
    labels      = data[id_data]

    n_groups = len(id_data)
    LOF     = improve_auc[:,2:3]
    CEN     = improve_auc[:,3:4]
    DIS     = improve_auc[:,4:5]
    KDE     = improve_auc[:,5:6]
    SVM05   = improve_auc[:,6:7]
    SVM01   = improve_auc[:,7:8]

    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)

    #plt.title(data_name + ' Attack Group', fontsize=16)
    plt.ylim([-0.45, 0.45])

    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 1.0

    plt.bar(index + 1*bar_width,   LOF,   bar_width, alpha=opacity, color='b', label='LOF')
    plt.bar(index + 2*bar_width, CEN,   bar_width, alpha=opacity, color='g',label='CEN')
    plt.bar(index + 3*bar_width, DIS,   bar_width, alpha=opacity, color='r',label='NDIS')
    plt.bar(index + 4*bar_width, KDE,   bar_width, alpha=opacity, color='y',label='KDE')
    plt.bar(index + 5*bar_width, SVM05, bar_width, alpha=opacity, color='c',label='SVM05')
    plt.bar(index + 6*bar_width, SVM01, bar_width, alpha=opacity, color='maroon',label='SVM01')
    #xx-small, x-small, small, medium, large, x-large, xx-large
    ax.legend(bbox_to_anchor=(0.44, 1.0), ncol=3, fontsize = 'medium')

    plt.xlabel('Sparsity of data', fontsize=14)
    plt.ylabel('($\mathrm{AUC}_{\mathrm{hidden}}$' + '-' + '$\mathrm{AUC}_{\mathrm{input}}$)', fontsize=14)
    plt.yticks(fontsize=12)

    ax.yaxis.grid(True)
    plt.xticks(index + 3*bar_width,(str(spa_score[i][1]) + '-' + str(labels[i]) for i in range(n_groups)),rotation=60,fontsize=11)

    plt.tight_layout()
    plt.savefig(path + "auc_sparsity_" + method + "_bar.pdf")
    plt.show()


def plot_sparsity_auc(data, improve_auc, spa_score, method, path):

    #Using CTU13-13 to demonstrate for four CTU13 datasets, [6,7,8,9]
    id_data = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13]
    improve_auc = improve_auc[id_data]
    spa_score   = spa_score[id_data]        #sparsity score
    labels = data[id_data]                  #name of datasets

    LOF     = improve_auc[:,2:3]
    CEN     = improve_auc[:,3:4]
    DIS     = improve_auc[:,4:5]
    KDE     = improve_auc[:,5:6]
    SVM05   = improve_auc[:,6:7]
    SVM01   = improve_auc[:,7:8]

    plt.figure(figsize=(8, 4.5))
    ax = plt.subplot(111)


    plt.ylim([-0.4, 0.4])
    plt.xlim([-0.02, max(spa_score[:,1])+0.01])
    plt.xticks(spa_score[:,1],spa_score[:,1], rotation=90)

    #'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
    plt.plot(spa_score[:,1], LOF,   'b-s', ms=4, mec="b", label ='LOF', markevery = 1)
    plt.plot(spa_score[:,1], CEN,   'r-p', ms=4, mec="r", label ='CEN', markevery = 1)
    plt.plot(spa_score[:,1], DIS,   'g-^', ms=4, mec="g", label= 'NDIS', markevery = 1)
    plt.plot(spa_score[:,1], KDE,   'y-d', ms=4, mec="y", label ='KDE', markevery = 1)
    plt.plot(spa_score[:,1], SVM05, 'm-o', ms=4, mec="m", label =r'$\mathrm{SVM}_{\nu = 0.5}$', markevery = 1)
    plt.plot(spa_score[:,1], SVM01, 'c-x', ms=4, mec="c", label= r'$\mathrm{SVM}_{\nu = 0.1}$', markevery = 1)

    #xx-small, x-small, small, medium, large, x-large, xx-large
    ax.legend(bbox_to_anchor=(0.47, 1.0), ncol=3, fontsize = 'medium')

    plt.xlabel('Sparsity of data', fontsize=14)
    plt.ylabel('($\mathrm{AUC}_{\mathrm{hidden}}$' + '-' + '$\mathrm{AUC}_{\mathrm{input}}$)', fontsize=14)
    plt.yticks(fontsize=12)
    ax.yaxis.grid(True)

    ax.twiny()
    plt.xlim([-0.02, max(spa_score[:,1])+0.01])
    plt.xticks(spa_score[:,1], labels, rotation=90)

    plt.tight_layout()
    plt.savefig(path + "auc_sparsity_" + method + "_line.pdf")
    plt.show()



def plot_dimension_auc(data, improve_auc, spa_dim, method, path):

    #Using CTU13-13 to demonstrate for four CTU13 datasets, [6,7,8,9]
    id_data = [0, 1, 2, 4, 5, 9, 10, 11, 12, 13]
    improve_auc = improve_auc[id_data]
    spa_dim     = spa_dim[id_data]        #sparsity score

    improve_auc = np.insert(improve_auc, [0], spa_dim, axis=1)

    #improve_auc = sorted(improve_auc, key=lambda a_entry: a_entry[2])
    improve_auc = improve_auc[improve_auc[:,2].argsort()]
    dim     = np.asanyarray(improve_auc[:,2], dtype = int)
    idx     = np.asanyarray(improve_auc[:,0], dtype = int)
    labels1 = []
    for d in idx:
      labels1 = np.append(labels1, data[d])

                     #name of datasets

    LOF     = improve_auc[:,5:6]
    CEN     = improve_auc[:,6:7]
    DIS     = improve_auc[:,7:8]
    KDE     = improve_auc[:,8:9]
    SVM05   = improve_auc[:,9:10]
    SVM01   = improve_auc[:,10:11]

    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)

    log_dim = np.round(np.log(dim+1-9), 2)

    plt.ylim([-0.45, 0.45])
    plt.xlim([-0.1, max(log_dim)+0.1])
    plt.xticks(log_dim, dim, rotation=90)

    #'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
    plt.plot(log_dim, LOF,   'b-s', ms=4, mec="b", label ='LOF', markevery = 1)
    plt.plot(log_dim, CEN,   'r-p', ms=4, mec="r", label ='CEN', markevery = 1)
    plt.plot(log_dim, DIS,   'g-^', ms=4, mec="g", label= 'NDIS', markevery = 1)
    plt.plot(log_dim, KDE,   'y-d', ms=4, mec="y", label ='KDE', markevery = 1)
    plt.plot(log_dim, SVM05, 'm-o', ms=4, mec="m", label =r'$\mathrm{SVM}_{\nu = 0.5}$', markevery = 1)
    plt.plot(log_dim, SVM01, 'c-x', ms=4, mec="c", label= r'$\mathrm{SVM}_{\nu = 0.1}$', markevery = 1)

    #xx-small, x-small, small, medium, large, x-large, xx-large
    ax.legend(bbox_to_anchor=(0.47, 1.0), ncol=3, fontsize = 'medium')

    plt.xlabel('Dimension in log scale', fontsize=14)
    plt.ylabel('($\mathrm{AUC}_{\mathrm{hidden}}$' + '-' + '$\mathrm{AUC}_{\mathrm{input}}$)', fontsize=14)
    plt.yticks(fontsize=12)
    ax.yaxis.grid(True)

    ax.twiny()
    plt.xlim([-0.1, max(log_dim)+0.1])
    plt.xticks(log_dim, labels1, rotation=90)

    plt.tight_layout()
    plt.savefig(path + "auc_dimension_" + method + ".pdf")
    plt.show()



"****************** Plot histogram of z_mu, z_var and z **********************"
def histogram_z(x, name, alpha, epoch, path):

    mu    = np.mean(x)
    sigma = np.std(x)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 20, normed=1, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    y = mlab.normpdf( bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=1)

    if (name == 'mu'):
        title = r'$\mathrm{Histogram\ of\ \mu}_{\mathrm{z}}\ (\mathrm{\alpha\ = ' + str(alpha) + ',}\ \mathrm{epoch\ = }'+ str(epoch) + ')$'
        xlabel = r'$\mathrm{\mu}_{\mathrm{z}}$'
    elif (name == 'var'):
        title = r'$\mathrm{Histogram\ of\ \sigma}_{\mathrm{z}}\ (\mathrm{\alpha\ = ' + str(alpha) + ',}\ \mathrm{epoch\ = }'+ str(epoch) + ')$'
        xlabel = r'$\mathrm{\sigma}_{\mathrm{z}}$'
    else:
        title = r'$\mathrm{Histogram\ of\ z}\ (\mathrm{\alpha\ = ' + str(alpha) + ',}\ \mathrm{epoch\ = }'+ str(epoch) + ')$'
        xlabel = r'$\mathrm{z}$'

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel('Probability', fontsize=14)

    plt.title(title, fontsize=18)
    plt.axis([-3, 3, 0, max(y)+ 0.1*max(y)])
    plt.grid(True)
    plt.savefig(path + "Visualize_histogram/" + "his_" + name + "_" + str(alpha) + "_" +  str(epoch) + ".pdf")
    plt.show()
