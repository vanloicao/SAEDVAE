# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 23:26:14 2017

@author: VANLOI
This module contains a number of functions for visulizing training losses
and AUCs during the training process of SAE and DVAE
"""
from Plot_Curves import Plotting_Loss_Component, Plotting_Monitor
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

#list_data = ["CTU13_10", "Spambase", "UNSW", "NSLKDD", "InternetAds",\
#            "PageBlocks", "PenDigits", "GLASS"]
#data_name = ["CTU13-10", "Spambase", "UNSW-NB15", "NSL-KDD", "InternetAds",\
#           "PageBlocks", "PenDigits", "GLASS"]

#list_data = ["PenDigits", "CTU13_10", "NSLKDD", "UNSW"]
#data_name = ["PenDigits", "CTU13-10", "NSL-KDD","UNSW"]
#stop_point= [305, 115, 265, 240]

list_data = ["NSLKDD"]
data_name = ["NSL-KDD"]
stop_point = 265

#%%
"plot one dataset - lambda = 0.1, 1.0, 5.0, 10 and 50"
def plot_loss_auc_shrink5():
    n_data = len(list_data)
    n_lambda = 5
    path = "D:/Python_code/SDA-02/Results/Exp_Hidden/NEW_ERROR_SHRINK10/"
    lamda = ["lamda_01/","lamda_1/","lamda_5/","lamda_10/","lamda_50/"]
    legend = [r'$\mathrm{\lambda_{SAE}= ' + str(0.1) + '}$',
              r'$\mathrm{\lambda_{SAE}= ' + str(1.0) + '}$',
              r'$\mathrm{\lambda_{SAE}= ' + str(5.0) + '}$',
              r'$\mathrm{\lambda_{SAE}= ' + str(10) + '}$',
              r'$\mathrm{\lambda_{SAE}= ' + str(50) + '}$']

    plt.subplots(ncols=n_lambda, nrows = n_data, figsize=(10, 20))
    for i in range(n_lambda):
      j = 0
      for j in range(len(list_data)):
        num = i*n_data + j + 1
        loss = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_loss_component.csv", delimiter=",")
        x  = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_monitor_auc.csv", delimiter=",")

        y = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
        _, idx = np.unique(y, return_index=True)
        auc = x[idx]
        auc = auc[auc[:,0].argsort()]
        #[stop_ep, lof, cen, dis, kde, svm05, svm01, ae]

        x  = loss[:,0]
        y1 = loss[:,1]
        y2 = loss[:,2]
        y3 = y1 + y2

        ax1 = plt.subplot(n_lambda, n_data, num)
        plt.ylim((-0.002, 0.061))
        #plt.xticks(rotation='vertical')
        lns1 = ax1.plot(x, y1,  'b',      label = 'RE loss')
        lns2 = ax1.plot(x, y2,  'olive',  label = 'Shrink loss')
        lns3 = ax1.plot(x, y3,  'm',      label = 'SAE loss')

        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        if (j == 0):
          ax1.axes.get_yaxis().set_visible(True)
          plt.ylabel('Error ('+ legend[i]+')' , fontsize=18)
          plt.yticks(fontsize=13)
#          yticks =  ax1.yaxis.get_major_ticks()
#          yticks[5].label1.set_visible(False)


        ax2 = ax1.twinx()
        plt.ylim((0.4, 1.02))
        plt.xlim((-8,1005))

        lns4 = ax2.plot(auc[:,0], auc[:,1], 'g-o', label = 'SAE-LOF', markevery = 10 )
        lns5 = ax2.plot(auc[:,0], auc[:,2], 'c-x', label = 'SAE-CEN', markevery = 10 )
        lns6 = ax2.plot(auc[:,0], auc[:,5], 'r-^', label = r'SAE-OCSVM$_{\nu = 0.5}$', markevery = 10)
        #lns7 = ax2.plot([0.0, 0.0], [0.0, 0.0], 'g--', label = 'Stopping point', markevery = 10 )
        if (i==3):
          lns7 = ax2.plot([stop_point, stop_point], [0.0, 1.02], 'g--', label = 'Early stopping point', markevery = 10 )
          labs = [l.get_label() for l in lns7]
          ax1.legend(lns7, labs, bbox_to_anchor=(0.65, 0.85), ncol = 1, fontsize = 'x-large')
          #"PenDigits" (0.7, 0.85) , "CTU13-10" (0.5 0.85), "NSL-KDD" 0.65, 0.85,"UNSW" 0.62, 0.99
        #xx-small, x-small, small, medium, large, x-large, xx-large
        "Legend setting"
        if (i == 0):
          #plt.title(data_name[j], fontsize=18)
          if (j == 0):
            lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, bbox_to_anchor=(1.0, 0.62), ncol = 2, fontsize = 'x-large')
            #"PenDigits" 1.0, 0.9 , "CTU13-10" 1.0, 0.65, "NSL-KDD" 1.0, 0.62,"UNSW" 1.0, 0.99

        ax2.axes.get_yaxis().set_visible(False)
        if (j == 0):
          ax2.axes.get_yaxis().set_visible(True)
          plt.ylabel('AUC', fontsize=18)
          plt.yticks(fontsize=13)
#          yticks =  ax2.yaxis.get_major_ticks()
#          yticks[5].label1.set_visible(False)

        ax2.axes.get_xaxis().set_visible(False)
        if (i == (n_lambda-1)):
          plt.xlabel('Epoch', fontsize=18)
          plt.xticks(fontsize=13)
          ax2.axes.get_xaxis().set_visible(True)
#          xticks = ax2.xaxis.get_major_ticks()
#          if (j<2):
#            xticks[6].label1.set_visible(False)        #Disable the first xticks

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.00, hspace=0.04)
    plt.savefig(path + "loss_auc_SAE2_"+ list_data[j-1] + "_5.pdf")
    plt.show()



#%%
"plot one dataset (CTU13_10) - lambda = 5.0 and 10"
def plot_loss_auc_shrink2():

    list_data = ["CTU13_10"]
    data_name = ["CTU13-10"]

    n_data = len(list_data)
    n_lambda = 2
    path = "D:/Python_code/SDA-02/Results/Exp_Hidden/NEW_ERROR_SHRINK10/"
    lamda = ["lamda_5/","lamda_10/"]
    legend = [r'$\mathrm{\lambda_{SAE}= ' + str(5.0) + '}$',
              r'$\mathrm{\lambda_{SAE}= ' + str(10) + '}$']

#    lamda = ["lamda_01/","lamda_1/","lamda_5/","lamda_10/","lamda_50/"]
#    legend = [r'$\mathrm{\lambda_{SAE}= ' + str(0.1) + '}$',
#              r'$\mathrm{\lambda_{SAE}= ' + str(1.0) + '}$',
#              r'$\mathrm{\lambda_{SAE}= ' + str(5.0) + '}$',
#              r'$\mathrm{\lambda_{SAE}= ' + str(10) + '}$',
#              r'$\mathrm{\lambda_{SAE}= ' + str(50) + '}$']
    plt.subplots(ncols=n_lambda, nrows = n_data, figsize=(10, 8))
    for i in range(n_lambda):
      j = 0
      for j in range(len(list_data)):
        num = i*n_data + j + 1
        loss = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_loss_component.csv", delimiter=",")
        re   = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_training_error.csv", delimiter=",")

        x  = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_monitor_auc.csv", delimiter=",")

        y = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
        _, idx = np.unique(y, return_index=True)
        auc = x[idx]
        auc = auc[auc[:,0].argsort()]

        x  = loss[:,0]
        y1 = loss[:,1]
        y2 = loss[:,2]
        y3 = y1 + y2
        #y3 = re[:,2]

        ax1 = plt.subplot(n_lambda, n_data, num)
        plt.ylim((-0.002, 0.061))
        #plt.xticks(rotation='vertical')
        lns1 = ax1.plot(x, y1,  'b',      label = 'RE loss')
        lns2 = ax1.plot(x, y2,  'olive',  label = 'Shrink loss')
        lns3 = ax1.plot(x, y3,  'm',      label = 'SAE loss')

        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        if (j == 0):
          ax1.axes.get_yaxis().set_visible(True)
          plt.ylabel('Error ('+ legend[i]+')' , fontsize=18)
          plt.yticks(fontsize=13)
#          yticks =  ax1.yaxis.get_major_ticks()
#          yticks[5].label1.set_visible(False)


        ax2 = ax1.twinx()
        plt.ylim((0.4, 1.02))
        plt.xlim((-8,1005))

        lns4 = ax2.plot(auc[:,0], auc[:,1], 'g-o', label = 'SAE-LOF', markevery = 10 )
        lns5 = ax2.plot(auc[:,0], auc[:,2], 'c-x', label = 'SAE-CEN', markevery = 10 )
        lns6 = ax2.plot(auc[:,0], auc[:,5], 'r-^', label = r'SAE-OCSVM$_{\nu = 0.5}$', markevery = 10)
        #lns7 = ax2.plot([0.0, 0.0], [0.0, 0.0], 'g--', label = 'Stopping point', markevery = 10 )
        if (i==1):
          lns7 = ax2.plot([115, 115], [0.0, 1.02], 'g--', label = 'Early stopping point', markevery = 10 )
          labs = [l.get_label() for l in lns7]
          ax1.legend(lns7, labs, bbox_to_anchor=(0.5, 0.85), ncol = 1, fontsize = 'x-large')

        #xx-small, x-small, small, medium, large, x-large, xx-large
        "Legend setting"
        if (i == 0):
          #plt.title(data_name[j], fontsize=18)
          if (j == 0):
            lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, bbox_to_anchor=(1.0, 0.85), ncol = 2, fontsize = 'x-large')

        ax2.axes.get_yaxis().set_visible(False)
        if (j == 0):
          ax2.axes.get_yaxis().set_visible(True)
          plt.ylabel('AUC', fontsize=18)
          plt.yticks(fontsize=13)
#          yticks =  ax2.yaxis.get_major_ticks()
#          yticks[5].label1.set_visible(False)

        ax2.axes.get_xaxis().set_visible(False)
        if (i == (n_lambda-1)):
          plt.xlabel('Epoch', fontsize=18)
          plt.xticks(fontsize=13)
          ax2.axes.get_xaxis().set_visible(True)
#          xticks = ax2.xaxis.get_major_ticks()
#          if (j<2):
#            xticks[6].label1.set_visible(False)        #Disable the first xticks

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.00, hspace=0.04)
    plt.savefig(path + "loss_auc_SAE2_"+ list_data[j-1] + "_2.pdf")
    plt.show()

#%%
"plot one dataset (CTU13_10) - Lambda = 10"
def plot_loss_auc_shrink1():

    list_data = ["CTU13_10"]
    data_name = ["CTU13-10"]
    n_data = len(list_data)
    n_lambda = 1
    path = "D:/Python_code/SDA-02/Results/Exp_Hidden/NEW_ERROR_SHRINK10/"
    lamda = ["lamda_10/"]
    legend = [r'$\mathrm{\lambda_{SAE}= ' + str(10) + '}$']

    plt.subplots(ncols=n_lambda, nrows = n_data, figsize=(10, 4.5))
    for i in range(n_lambda):
      j = 0
      for j in range(len(list_data)):
        num = i*n_data + j + 1
        loss = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_loss_component.csv", delimiter=",")
        re   = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_training_error.csv", delimiter=",")

        x  = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_monitor_auc.csv", delimiter=",")

        y = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
        _, idx = np.unique(y, return_index=True)
        auc = x[idx]
        auc = auc[auc[:,0].argsort()]

        x  = loss[:,0]
        y1 = loss[:,1]
        y2 = loss[:,2]
        y3 = y1 + y2
        #y3 = re[:,2]

        ax1 = plt.subplot(n_lambda, n_data, num)
        plt.title('\n \n', fontsize=12)
        plt.ylim((-0.002, 0.061))
        #plt.xticks(rotation='vertical')
        lns1 = ax1.plot(x, y1,  'b',      label = 'RE loss')
        lns3 = ax1.plot(x, y2,  'olive',  label = 'Shrink loss')
        lns5 = ax1.plot(x, y3,  'm',      label = 'SAE loss')

        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        if (j == 0):
          ax1.axes.get_yaxis().set_visible(True)
          plt.ylabel('Error ('+ legend[i]+')' , fontsize=18)
          plt.yticks(fontsize=13)
#          yticks =  ax1.yaxis.get_major_ticks()
#          yticks[5].label1.set_visible(False)


        ax2 = ax1.twinx()
        plt.ylim((0.4, 1.02))
        plt.xlim((-8,1005))

        lns2 = ax2.plot(auc[:,0], auc[:,1], 'g-o', label = 'SAE-LOF', markevery = 10 )
        lns4 = ax2.plot(auc[:,0], auc[:,2], 'c-x', label = 'SAE-CEN', markevery = 10 )
        lns6 = ax2.plot(auc[:,0], auc[:,5], 'r-^', label = r'SAE-OCSVM$_{\nu = 0.5}$', markevery = 10)
        #lns7 = ax2.plot([0.0, 0.0], [0.0, 0.0], 'g--', label = 'Stopping point', markevery = 10 )
        if (i==0):
          lns7 = ax2.plot([115, 115], [0.0, 1.02], 'g--', label = 'Early stopping point', markevery = 10 )
          labs = [l.get_label() for l in lns7]
          lg1=ax1.legend(lns7, labs, bbox_to_anchor=(0.15, 0.6), ncol = 1, fontsize = 'x-large')

        #xx-small, x-small, small, medium, large, x-large, xx-large
          "Legend setting"

          #plt.title(data_name[j], fontsize=18)
          if (j == 0):
            lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, bbox_to_anchor=(0.05, 1.28), ncol = 3, fontsize = 'x-large')
            ax1.axes.add_artist(lg1)

        ax2.axes.get_yaxis().set_visible(False)
        if (j == 0):
          ax2.axes.get_yaxis().set_visible(True)
          plt.ylabel('AUC', fontsize=18)
          plt.yticks(fontsize=13)
#          yticks =  ax2.yaxis.get_major_ticks()
#          yticks[5].label1.set_visible(False)

        ax2.axes.get_xaxis().set_visible(False)
        if (i == (n_lambda-1)):
          plt.xlabel('Epoch', fontsize=18)
          plt.xticks(fontsize=13)
          ax2.axes.get_xaxis().set_visible(True)
#          xticks = ax2.xaxis.get_major_ticks()
#          if (j<2):
#            xticks[6].label1.set_visible(False)        #Disable the first xticks

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.00, hspace=0.04)
    plt.savefig(path + "loss_auc_SAE2_"+ list_data[j-1] + "_1.pdf")
    plt.show()



#%%
"Plotting the loss function and its components on Shrink AE for n_data"
def plot_loss_component_shrink(n_data):
    path = "D:/Python_code/SDA-02/Results/Exp_Hidden/NEW_ERROR_SHRINK/"
    lamda = ["lamda_01/","lamda_1/","lamda_5/","lamda_10/","lamda_50/"]
    legend = [r'$\mathrm{\lambda\ = ' + str(0.1) + '}$',
              r'$\mathrm{\lambda\ = ' + str(1.0) + '}$',
              r'$\mathrm{\lambda\ = ' + str(5.0) + '}$',
              r'$\mathrm{\lambda\ = ' + str(10) + '}$',
              r'$\mathrm{\lambda\ = ' + str(50) + '}$']

    plt.subplots(ncols=5, nrows = n_data, figsize=(15, 10))
    for i in range(5):
      j = 0
      for j in range(len(list_data)):
        num = i*n_data + j + 1
        loss = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_loss_component.csv", delimiter=",")
        re  = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_training_error.csv", delimiter=",")

        x  = loss[:,0]
        y1 = loss[:,1]
        y2 = loss[:,2]
        y3 = re[:,2]

        fig = plt.subplot(5, n_data, num)
        plt.ylim((0.0, 0.0495))
        plt.ylim((0.0, 1.0))
        #plt.xticks(rotation='vertical')

        plt.plot(x, y1,  'b', label = 'Recon loss')
        plt.plot(x, y2,  'y', label = 'Shrink loss')
        plt.plot(x, y3,  'r', label = 'Loss function')

        if i == 0:
          #xx-small, x-small, small, medium, large, x-large, xx-large
          if (j == 0):
            plt.legend(loc='upper right', ncol = 1, fontsize = 'x-large')
          plt.title(data_name[j], fontsize=20)

        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        if (i == 4):
          fig.axes.get_xaxis().set_visible(True)
          plt.xlabel('Epoch', fontsize=18)

          for xtick in fig.xaxis.get_major_ticks():
            xtick.label.set_fontsize(14)
          xtick.label1.set_visible(False)     #Disable the first xticks

        if (j == 0):
          fig.axes.get_yaxis().set_visible(True)
          if (i == 2):
            plt.ylabel('Error\n' +  legend[i] , fontsize=20)
          else:
            plt.ylabel( legend[i] , fontsize=20)

          for ytick in fig.yaxis.get_major_ticks():
            ytick.label.set_fontsize(14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.02)
    plt.savefig(path + "loss_component_shrink.pdf")
    plt.show()


#%%
"Plotting the loss function and its components on DVAE for n_data"
def plot_loss_component_vae(n_data):
    path = "D:/Python_code/SDA-02/Results/Exp_Hidden/NEW_ERROR_VAE/"
    lamda = ["lamda_0001/","lamda_001/","lamda_005/","lamda_01/","lamda_05/"]
    legend = [r'$\mathrm{\lambda\ = ' + str(0.001) + '}$',
              r'$\mathrm{\lambda\ = ' + str(0.01) + '}$',
              r'$\mathrm{\lambda\ = ' + str(0.05) + '}$',
              r'$\mathrm{\lambda\ = ' + str(0.1) + '}$',
              r'$\mathrm{\lambda\ = ' + str(0.5) + '}$']

    plt.subplots(ncols=5, nrows = n_data, figsize=(15, 9))
    for i in range(5):
      j = 0
      for j in range(len(list_data)):
        num = i*n_data + j + 1
        loss = np.genfromtxt(path + lamda[i] +  list_data[j] +  "_loss_component.csv", delimiter=",")
        re   = np.genfromtxt(path + lamda[i] + list_data[j] +  "_training_error.csv", delimiter=",")

        x  = loss[:,0]
        y1 = loss[:,1]
        y2 = loss[:,2]
        y3 = re[:,2]

        fig = plt.subplot(5, n_data, num)
        plt.ylim((0.0, 0.41))

        plt.plot(x, y1,  'b', label = 'Recon loss')
        plt.plot(x, y2,  'g', label = 'Shrink loss')
        plt.plot(x, y3,  'r', label = 'Loss function')

        if (i == 0) and (j==0):
          plt.legend(loc='upper right', ncol = 1, fontsize = 'x-large')

        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        if (i == 0):
          plt.title(data_name[j], fontsize=20)
        if (i == 4):
          fig.axes.get_xaxis().set_visible(True)
          plt.xlabel('Epoch', fontsize=18)

          for xtick in fig.xaxis.get_major_ticks():
            xtick.label.set_fontsize(14)
          xtick.label1.set_visible(False)     #Disable the first xticks

        if (j == 0):
          fig.axes.get_yaxis().set_visible(True)
          if (i == 2):
            plt.ylabel('Error\n' +  legend[i] , fontsize=20)
          else:
            plt.ylabel( legend[i] , fontsize=20)

          plt.yticks(np.arange(0.0, 0.35, 0.1))

          for ytick in fig.yaxis.get_major_ticks():
            ytick.label.set_fontsize(14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(path + "loss_component_vae.pdf")
    plt.show()

#%%
if __name__=="__main__":
	plot_loss_auc_shrink1()
	plot_loss_auc_shrink2()
	plot_loss_auc_shrink5()
