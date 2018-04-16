# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 01:56:46 2017

@author: VANLOI
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


path = "D:/Python_code/SDA-02/Results/Exp_Hidden/"

def plot_boxplot_svm_nu():
  df = pd.read_csv(path + "Parameter_nu/" + "Boxplot_nu.csv")
                  
  df = df[['Dataset','DAE','SAE', 'DVAE']]
  dd=pd.melt(df,id_vars=['Dataset'],value_vars=['DAE','SAE', 'DVAE'],var_name='Representation by')

  flatui = ["#2ecc71", "#9b59b6","#e74c3c"]
  color = sns.color_palette(flatui)
  fig, ax = plt.subplots()
  fig.set_size_inches(8, 4)

  flierprops = dict(marker='x', markerfacecolor='r', markersize=4,
                  linestyle='none', markeredgecolor='r')
  ax = sns.boxplot(x='Dataset',y='value',data=dd, hue='Representation by', palette=color, width=0.6,flierprops=flierprops)
  #plt.legend(bbox_to_anchor=(1.005, 1), loc=1, borderaxespad=0.)
  #plt.legend(bbox_to_anchor=(1.05, 0.4), ncol = 1, fontsize = 'medium')

  #ax.axes.set_title("Title",fontsize=30)
  ax.set_xlabel("Datasets",fontsize=15)
  ax.set_ylabel("AUC",fontsize=15)
  plt.xticks(fontsize=12, rotation=10)
  plt.yticks(fontsize=12, rotation=0)
  plt.ylim([0.2,1.0])
  #sns.set_style("whitegrid")
  plt.tight_layout()
  plt.savefig(path+ "Parameter_nu/" + "boxplot_SVM.pdf")


def plot_boxplot_lof_k():
  path = "D:/Python_code/SDA-02/Results/Exp_Hidden/"
  df = pd.read_csv(path + "Parameter_k/" + "Boxplot_k.csv") 
                  
  df = df[['Dataset','DAE','SAE', 'DVAE']]
  dd=pd.melt(df,id_vars=['Dataset'],value_vars=['DAE','SAE', 'DVAE'],var_name='Representation by')

  flatui = ["#2ecc71", "#9b59b6","#e74c3c"]
  color = sns.color_palette(flatui)
  fig, ax = plt.subplots()
  fig.set_size_inches(8, 4)

  flierprops = dict(marker='x', markerfacecolor='r', markersize=4,
                  linestyle='none', markeredgecolor='r')
  ax = sns.boxplot(x='Dataset',y='value',data=dd, hue='Representation by', palette=color, width=0.6,flierprops=flierprops)
  #plt.legend(bbox_to_anchor=(1.005, 1), loc=1, borderaxespad=0.)
  #plt.legend(bbox_to_anchor=(1.05, 0.4), ncol = 1, fontsize = 'medium')

  #ax.axes.set_title("Title",fontsize=30)
  ax.set_xlabel("Datasets",fontsize=15)
  ax.set_ylabel("AUC",fontsize=15)
  plt.xticks(fontsize=12, rotation=10)
  plt.yticks(fontsize=12, rotation=0)
  plt.ylim([0.2,1.0])
  #sns.set_style("whitegrid")
  plt.tight_layout()
  plt.savefig(path+ "Parameter_k/" + "boxplot_LOF.pdf")



def plot_boxplot_svm_gamma():
  path = "D:/Python_code/SDA-02/Results/Exp_Hidden/"
  df = pd.read_csv(path + "Parameter_gamma/" + "Boxplot_gamma.csv") 
                  
  df = df[['Dataset','DAE','SAE', 'DVAE']]
  dd=pd.melt(df,id_vars=['Dataset'],value_vars=['DAE','SAE', 'DVAE'],var_name='Representation by')

  flatui = ["#2ecc71", "#9b59b6","#e74c3c"]
  color = sns.color_palette(flatui)
  fig, ax = plt.subplots()
  fig.set_size_inches(8, 4)
  
  flierprops = dict(marker='x', markerfacecolor='r', markersize=4,
                  linestyle='none', markeredgecolor='r')
  ax = sns.boxplot(x='Dataset',y='value',data=dd, hue='Representation by', palette=color, width=0.6,flierprops=flierprops)
  #plt.legend(bbox_to_anchor=(1.005, 1), loc=1, borderaxespad=0.)
  #plt.legend(bbox_to_anchor=(1.05, 0.4), ncol = 1, fontsize = 'medium')

  #ax.axes.set_title("Title",fontsize=30)
  ax.set_xlabel("Datasets",fontsize=15)
  ax.set_ylabel("AUC",fontsize=15)
  plt.xticks(fontsize=12, rotation=10)
  plt.yticks(fontsize=12, rotation=0)
  plt.ylim([0.2,1.0])
#  # Select which box you want to change    
#  mybox = ax.artists[0]
#  # Change the appearance of that box
#  mybox.set_facecolor('red')
#  mybox.set_edgecolor("#9b59b6")
#  mybox.set_linewidth(1.0)
  #sns.set_style("whitegrid")
  plt.tight_layout()
  plt.savefig(path+ "Parameter_gamma/" + "boxplot_SVM_gamma.pdf")

plot_boxplot_svm_nu()
plot_boxplot_lof_k()
plot_boxplot_svm_gamma()