# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 18:12:38 2017

@author: VANLOI
"""
from __future__ import print_function, division

#import sys
import numpy
import numpy as np
import downhill
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from hiddenLayer import HiddenLayer
from hiddenLayer import dA

from ProcessingData import load_data, normalize_data
from Methods import auc_density, auc_AEbased
from Plot_Curves import Plotting_End2End_RE, Plotting_Pre_RE, Plotting_AUC_RE
from Plot_Curves import Plotting_AUC_Batch_Size, Plotting_Monitor, plot_auc_size_input, visualize_hidden1
from Plot_Curves import Plotting_Loss_Component, plot_auc_size_2
from nnet_architecture import hyper_parameters
from stopping_para import stopping_para_shrink, stopping_para_shrink_same_batch
#import timeit as tm

path = "D:/Python_code/SDA-02/Results/"

#Check whether weights matrix is updated or not
def check_weight_update(sda):
    np.set_printoptions(precision=4, suppress=True)
    "Check whether weights matrix is updated or not"
    for i in range(sda.n_layers):
        print("\n %d" %i)
        print (sda.Autoencoder_layers[i].W.get_value(borrow=True))

    for j in range(sda.n_layers, 2*sda.n_layers):
        print("\n %d" % j)
        print (sda.Autoencoder_layers[j].W.eval())



class SdA(object):
    def __init__(self, numpy_rng, theano_rng=None, n_ins=100,
                 hidden_layers_sizes=[50, 30], corruption_levels=[0.1, 0.1]):

        self.encoder = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.decoder = []

        assert self.n_layers > 0
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images

        "*************** Encoder **************************"
        for i in range(self.n_layers):
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i - 1]
                layer_input = self.encoder[-1].output
            act_function = T.tanh
            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first layer
            encoder_layer = HiddenLayer(rng=numpy_rng,
                                    input       = layer_input,
                                    n_in        = input_size,
                                    n_out       = hidden_layers_sizes[i],
                                    activation  = act_function)
            self.encoder.append(encoder_layer)
            self.params.extend(encoder_layer.params)

            "**************************************************"
            """Construct a dAE that shared weights with this layer"""
            dA_layer = dA(numpy_rng = numpy_rng,
                          theano_rng = theano_rng,
                          #input = layer_input,                #if use early-stopping for pre-train, it is disable
                          n_visible = input_size,
                          n_hidden  = hidden_layers_sizes[i],
                          W         = encoder_layer.W,
                          bhid      = encoder_layer.b)             #bvis: will be create dA itself

            self.dA_layers.append(dA_layer)
            #dA will not initialize wieghts and bias

        "*************** Decoder *****************************"
        i = self.n_layers-1
        while (i >=0):
            input_size = hidden_layers_sizes[i]
            if ( i > 0):
                output_size = hidden_layers_sizes[i-1]
            else:
                output_size =  n_ins

            if (i==self.n_layers-1):
                layer_input = self.encoder[-1].output
            else:
                layer_input = self.decoder[-1].output

            decoder_layer = HiddenLayer(rng     = numpy_rng,
                                        input   = layer_input,
                                        n_in    = input_size,
                                        n_out   = output_size,
                                        activation = T.tanh,
                                        W = self.encoder[i].W.T,
                                        b = self.dA_layers[i].b_prime)    #this is bvis of dA
            self.decoder.append(decoder_layer)
            self.params.append(decoder_layer.b)
            i = i - 1

        "******************* End To End Cost function ************************"
        y = self.encoder[-1].output
        z = self.decoder[-1].output

        lamda = 10.0
        self.shrink = lamda*(((y)**2).mean(1)).mean()
        self.recon = (((self.x - z)**2).mean(1)).mean()
        self.end2end_cost = self.recon + self.shrink

        #mean(1) is within example, mean(0) is within each feature

    "****** Error on train_x and valid_x before optimization process **********"
    def Loss_train_valid(self, train_x, valid_x):
        index = T.lscalar('index')

        train_size = train_x.get_value().shape[0]
        tm = theano.function([index],
                             outputs = self.end2end_cost,
                             givens={self.x: train_x[index : train_size]})

        valid_size = valid_x.get_value().shape[0]
        vm = theano.function([index],
                             outputs = self.end2end_cost,
                             givens={self.x: valid_x[index : valid_size]})

        return tm(0), vm(0)

    "****************** Compute recon and shrink component *******************"
#    def Loss_recon_shrink(self, train_x):
#        index = T.lscalar('index')
#        train_size = train_x.get_value().shape[0]
#        loss_com = theano.function([index],
#                             outputs = [self.recon, self.shrink],
#                             givens={self.x: train_x[index : train_size]})
#        return loss_com(0)



    "Compute loss for each batch"
    def Loss_recon_shrink_batch(self, train_x, batch_size):

        index = T.lscalar('index')
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        loss_com = theano.function([index],
                             outputs = [self.recon, self.shrink],
                             givens={self.x: train_x[batch_begin : batch_end]})
        return loss_com

    "Compute loss for n_batch from train_x"
    def Loss_recon_shrink(self, train_x, batch_size):
        n_train = train_x.get_value().shape[0]
        n_batches = (int)(n_train/batch_size)
        loss_com = self.Loss_recon_shrink_batch(train_x, batch_size)
        loss = np.empty([0,2])
        for batch_index in range(n_batches):
          l = loss_com(index = batch_index)
          loss = np.append(loss, [l[0], l[1]])
        loss = np.reshape(loss, (-1,2))

        return (loss.mean(0))


    "Get data from the middle hidden layer Deep Autoencoder"
    def get_hidden_data(self,data_set):

        data_size = data_set.get_value().shape[0]
        index = T.lscalar('index')
        hidden_data = theano.function([index],
                                      outputs = self.encoder[-1].output,
                                      givens={self.x: data_set[index : data_size]})
        #Get hidden_data from Autoencoder is the same getting data from the last
        return hidden_data(0)

    #Get hidden data from hidden layer i-th for pre-training
    def get_hidden_i(self,data_set, i):

        data_size = data_set.get_value().shape[0]
        index = T.lscalar('index')
        hidden_data = theano.function([index],
                                      outputs = self.encoder[i].output,
                                      givens={self.x: data_set[index : data_size]})
        #Get hidden_data from Autoencoder is the same getting data from the last
        return hidden_data(0)

    "Get data from the output of Autoencoder"
    def get_output_data(self,data_set):

        data_size = data_set.get_value().shape[0]
        index = T.lscalar('index')
        output_data = theano.function([index],
                                      outputs = self.decoder[-1].output,
                                      givens={self.x: data_set[index : data_size]})
        return output_data(0)



    def Compute_AUC_Hidden(self, train_set, test_set, actual, norm, data_name):

        output_test  = self.get_output_data(test_set)       #get prediction values
        train_hidden = self.get_hidden_data(train_set)      #get hidden values
        test_hidden  = self.get_hidden_data(test_set)       #get hidden values

        "Compute performance of classifiers on latent data"
        lof, cen, dis, kde, svm05, svm01 = auc_density(train_hidden, test_hidden, actual, norm)
        ae                               = auc_AEbased(test_set.get_value(), output_test, actual)
        return lof, cen, dis, kde, svm05, svm01, ae

    "**************************************************************************"
    def Save_Hidden_Data(self, train_set, test_set, data_name, path):

        train_hidden = self.get_hidden_data(train_set)      #get hidden values
        test_hidden  = self.get_hidden_data(test_set)       #get hidden values
        np.savetxt(path + data_name + "_train_z.csv", train_hidden, delimiter=",", fmt='%f' )
        np.savetxt(path + data_name + "_test_z.csv", test_hidden, delimiter=",", fmt='%f' )

    "**************************************************************************"
    def Save_Hidden_Data_Size(self, train_set, test_set, data_name, size, path):

        train_hidden = self.get_hidden_data(train_set)      #get hidden values
        test_hidden  = self.get_hidden_data(test_set)       #get hidden values
        np.savetxt(path + "data/" + data_name + "_train_z_" + str(size) + ".csv", train_hidden, delimiter=",", fmt='%f' )
        np.savetxt(path + "data/"+ data_name + "_test_z_" + str(size) +".csv", test_hidden, delimiter=",", fmt='%f' )

    "******** Training End-to-End Early-stopping by Downhill Package *********"
    def End2end_Early_stopping(self, numpy_rng, dataset, n_validate, data_name,
                               batch_size, end2end_lr, algo, norm, patience, validation):

        train_X, test_X, actual = dataset
        valid_x = train_X.get_value()[:n_validate]
        train_x = train_X.get_value()[n_validate:]

        "for compute tm and vm before optimization process"
        t = theano.shared(numpy.asarray(train_x, dtype=theano.config.floatX), borrow=True)
        v = theano.shared(numpy.asarray(valid_x, dtype=theano.config.floatX), borrow=True)

        "Use downhill for training network"
        #'adadelta' 'adagrad (default 0.01)' 'adam''esgd' 'nag''rmsprop' 'rprop' 'sgd'
        opt = downhill.build(algo = algo, params= self.params,
                             loss = self.end2end_cost, inputs = [self.x])

        train = downhill.Dataset(train_x, batch_size = batch_size, rng = numpy_rng)
        valid = downhill.Dataset(valid_x, batch_size = len(valid_x), rng = numpy_rng)

        "for monitoring before optimization process"
        stop_ep = 0

#        LOSS = np.empty([0,4])
#        monitor = np.empty([0,8])
                                       #performance before fine-tuning
#        lof,cen,dis,kde,svm05,svm01,ae = self.Compute_AUC_Hidden(train_X, test_X, actual, norm, data_name)
#        a = [stop_ep, lof, cen, dis, kde, svm05, svm01, ae]
#        monitor = np.append(monitor, a )
                                        #Loss component
#        loss = self.Loss_recon_shrink(t, batch_size)
#        LOSS = np.append(LOSS,[0.0, loss[0], loss[1], loss[0]+loss[1]])
#                                        #error before training

        for tm1, vm1 in opt.iterate(train,                        # 10, 5, 1e-2, 0.0
                                  valid,
                                  patience = patience,                # 10
                                  validate_every= validation,            # 5
                                  min_improvement = 1e-3,       # 1e-3
                                  #learning_rate =  end2end_lr, # 1e-4
                                  momentum = 0.0,
                                  nesterov = False):


            stop_ep = stop_ep + 1
#            loss = self.Loss_recon_shrink(t, batch_size)
#            LOSS = np.append(LOSS,[stop_ep, loss[0], loss[1], loss[0]+loss[1]])
#
##            "******* Classification Results after End to End training ******"
#            if ((stop_ep%1 == 0) and (stop_ep > 0)):
#                lof,cen,dis,kde,svm05,svm01,ae = self.Compute_AUC_Hidden(train_X, test_X, actual, norm, data_name)
#                a = [stop_ep, lof, cen, dis, kde, svm05, svm01, ae]
#            monitor = np.append(monitor, a)

            if (stop_ep >= 1000):
                break

        #Plotting AUC and save to csv file
#        monitor = np.reshape(monitor, (-1,8))
#        Plotting_Monitor(monitor, 0.4, 1.0, data_name, path)
#        np.savetxt(path + data_name + "_monitor_auc.csv", monitor, delimiter=",", fmt='%f' )


#        LOSS = np.reshape(LOSS, (-1,4))
#        Plotting_Loss_Component(LOSS, RE, 0.0, 0.1, data_name, path)
#        np.savetxt(path + data_name + "_loss_component.csv", LOSS, delimiter=",", fmt='%f' )

        return  [stop_ep, vm1['loss'], tm1['loss']]



def test_SdA(pre_lr=0.01, end2end_lr=1e-4, algo = 'sgd',
             dataset=[], data_name = "WBC", n_validate = 0, norm = "maxabs",
             batch_size=10, hidden_sizes = [1,1,1], corruptions = [0.0, 0.0, 0.0],
             patience = 1, validation = 1):

    numpy_rng = numpy.random.RandomState(89677)   # numpy random generator 89677
    train_X, test_X, actual = dataset             # dataset is already normalised

    input_size = train_X.get_value().shape[1]     # input size = dimension
    train_x    = train_X.get_value()[n_validate:]   # 80% for pre-training, 20% for validation
    n_train_batches   = train_x.shape[0]
    n_train_batches //= batch_size                  # number of batches for pre-training

    # construct the stacked denoising autoencoder class
    sda = SdA(numpy_rng = numpy_rng, n_ins = input_size,
              hidden_layers_sizes = hidden_sizes)

    #check_weight_update(sda)    #Check whether weights matrix is updated or not
    RE = sda.End2end_Early_stopping(numpy_rng, dataset, n_validate, data_name,
                               batch_size, end2end_lr, algo, norm, patience, validation)

    return sda, RE

"********************************* Main experiment ***************************"
def Main_Test():

    list_data = ["PageBlocks", "WPBC", "PenDigits", "GLASS", "Shuttle", "Arrhythmia",\
                 "CTU13_10", "CTU13_08","CTU13_09","CTU13_13",\
                 "Spambase", "UNSW", "NSLKDD", "InternetAds"]

    norm         = "maxabs"                 #standard, maxabs[-1,1] or minmax[0,1]
    corruptions  = [0.1, 0.1, 0.1]

    print ("Shrink - 10")
    print ("+ Data: ", list_data)
    print ("+ Scaler: ", norm)
    print ("+ Corruptions: ", corruptions)

    AUC_Hidden = np.empty([0,10])     #store auc of all hidden data
    num = 0
    for data in list_data:
        num = num + 1
        h_sizes = hyper_parameters(data)                   #Load hyper-parameters

        train_set, test_set, actual = load_data(data)      #load original data
        train_X, test_X = normalize_data(train_set, test_set, norm)  #Normalize data

        train_X = theano.shared(numpy.asarray(train_X, dtype=theano.config.floatX), borrow=True)
        test_X  = theano.shared(numpy.asarray(test_X,  dtype=theano.config.floatX), borrow=True)

        datasets = [(train_X), (test_X), (actual)]          #Pack data for training AE

        in_dim   = train_set.shape[1]                       #dimension of input data
        n_vali   = (int)(train_set.shape[0]/5)              #size of validation set
        n_train  = len(train_set) - n_vali                  #size of training set
        #batch     = int(n_train/20)                          #Training set will be split training set into 20 batches

        pat, val, batch, n_batch = stopping_para_shrink(n_train)


        print ("\n" + str(num) + ".", data, "..." )
        print (" + Hidden Sizes: ",in_dim, h_sizes, "- Batch_sizes:", batch)
        print (" + Data: %d (%d train, %d vali) - %d normal, %d anomaly"\
            %(len(train_set), n_train, n_vali, \
            len(test_set[(actual == 1)]), len(test_set[(actual == 0)])))
        print(" + Patience: %5.0d, Validate: %5.0d,  \n + Batch size: %5.0d, n batch:%5.0d"\
             %(pat, val, batch, n_batch))

#        AUC_RE   = np.empty([0,10])

                               #adadelta, 'adagrad' 'adam''esgd' 'nag''rmsprop' 'rprop' 'sgd'
#        if (num==1):
        sda, re = test_SdA(pre_lr       = 1e-2,              #re = [stop_ep, vm, tm]
                            end2end_lr   = 1e-4,
                            algo         = 'adadelta',
                            dataset      = datasets,
                            data_name    = data,
                            n_validate   = n_vali,
                            norm         = norm,
                            batch_size   = batch,
                            hidden_sizes = h_sizes,
                            corruptions  = corruptions,
                            patience     = pat,
                            validation   = val)


        #*******Computer AUC on hidden data*************
        lof,cen,dis,kde,svm05,svm01,ae  = sda.Compute_AUC_Hidden(train_X, test_X, actual, norm, data)
        auc_hidden = np.column_stack([batch, re[0], lof, cen, dis, kde, svm05, svm01, ae , 100*re[2]])
        AUC_Hidden = np.append(AUC_Hidden, auc_hidden)

        #save hidden data to files
#        sda.Save_Hidden_Data(train_X, test_X, data, path)

        #Display for saving to document
#        AUC_RE   = np.append(AUC_RE, auc_hidden)
#        AUC_RE   = np.reshape(AUC_RE,(-1,10))
#
#        np.set_printoptions(precision=3, suppress=True)
#        column_list = [2,3,4,5,6,7,8,9]
#        print (AUC_RE[:,column_list])

    AUC_Hidden  =  np.reshape(AUC_Hidden, (-1, 10))
    np.set_printoptions(precision=3, suppress=True)
    column_list = [2,3,4,5,6,7,8,9]
    print("    LOF    CEN    MDIS   KDE   SVM5    SVM1    AE    RE*100")
    print (AUC_Hidden[:,column_list])

#    AUC_Hidden  =  np.reshape(AUC_Hidden, (-1, 10))
#    np.savetxt(path +  "AUC_Hidden.csv", AUC_Hidden, delimiter=",", fmt='%f' )


if __name__ == '__main__':
    Main_Test()
