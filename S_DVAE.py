# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 19:05:51 2017

@author: VANLOI
"""
from __future__ import print_function, division

import sys
import numpy as np
import downhill
import theano
import theano.tensor as T

from mlp import HiddenLayer
from dA import dA
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from ProcessingData import load_data, normalize_data
from Methods import auc_density, auc_AEbased
from Plot_Curves import Plotting_End2End_RE, Plotting_Pre_RE, Plotting_AUC_RE, Plotting_Pre_RE1
from Plot_Curves import Plotting_AUC_Batch_Size, Plotting_Monitor, plot_auc_size_input, visualize_hidden
from Plot_Curves import visualize_hidden1, histogram_z, Plotting_Loss_Component, plot_auc_size_2
from nnet_architecture import hyper_parameters
from stopping_para import stopping_para_vae, stopping_para_vae_same_batch

path = "D:/Python_code/SDA-02/Results/"

"Check whether weights matrix is updated or not"
def check_weight_update(sda):
    np.set_printoptions(precision=4, suppress=True)
    "Check whether weights matrix is updated or not"
    for i in range(sda.n_layers):
        print("\n %d" %i)
        print (sda.Autoencoder_layers[i].W.get_value(borrow=True))

    for j in range(sda.n_layers, 2*sda.n_layers):
        print("\n %d" % j)
        print (sda.Autoencoder_layers[j].W.eval())

    print ("\n ************************************ ")



class SdA(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=100, hidden_layers_sizes=[50, 30]):

        self.encoder = []
        self.decoder = []
        self.params  = []
        self.n_layers = len(hidden_layers_sizes)

        """set seed one time per dataset, rng will randomly generate different
        number in each mini_batch_size and, different in each epoch. This is
        repetive the same number when we create new SdA object. See theano_rondom.py"""
        self.rng = theano.tensor.shared_randomstreams.RandomStreams(numpy_rng.randint(2 ** 30))

        assert self.n_layers > 0
#        if not theano_rng:
#            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        # the data is presented as rasterized images
        self.x = T.matrix('x')

        for i in range(self.n_layers):
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if (i == 0):
                input_size  = n_ins
                layer_input = self.x
            else:
                input_size  = hidden_layers_sizes[i - 1]
                layer_input = self.encoder[-1].output

            "Not the middle hidden layer"
            if (i < self.n_layers-1):
                encoder_layer = HiddenLayer(rng = numpy_rng,
                                    input       = layer_input,
                                    n_in        = input_size,
                                    n_out       = hidden_layers_sizes[i],
                                    activation  = T.tanh)
                self.encoder.append(encoder_layer)
                self.params.extend(encoder_layer.params)

            else:
                "The middle hidden layer, linear"
                encoder_mu = HiddenLayer(rng = numpy_rng,
                                    input    = self.encoder[i-1].output,
                                    n_in     = input_size,
                                    n_out    = hidden_layers_sizes[i])

                encoder_var = HiddenLayer(rng= numpy_rng,
                                    input    = self.encoder[i-1].output,
                                    n_in     = input_size,
                                    n_out    = hidden_layers_sizes[i])

                self.encoder.append(encoder_mu)
                self.params.extend(encoder_mu.params)

                self.encoder.append(encoder_var)
                self.params.extend(encoder_var.params)


        "*************** Sample z **************"
        mu       = self.encoder[-2].output
        log_var  = self.encoder[-1].output
        sample_z = self.sample_z(mu, log_var)

        "*************** Decoder ***************"
        i = self.n_layers-1
        while (i >=0):
            input_size = hidden_layers_sizes[i]
            if ( i > 0):
                output_size = hidden_layers_sizes[i-1]
            else:
                output_size =  n_ins

            if (i==self.n_layers-1):  #the first layer in decoder
                layer_input = sample_z
                decoder_layer = HiddenLayer(rng = numpy_rng,
                                        input   = layer_input,
                                        n_in    = input_size,
                                        n_out   = output_size,
                                        activation = T.tanh)   #may be linear
                self.decoder.append(decoder_layer)
                self.params.extend(decoder_layer.params)
            else:
                layer_input = self.decoder[-1].output
                decoder_layer = HiddenLayer(rng = numpy_rng,
                                        input   = layer_input,
                                        n_in    = input_size,
                                        n_out   = output_size,
                                        activation = T.tanh,
                                        W = self.encoder[i].W.T)

                self.decoder.append(decoder_layer)
                self.params.append(decoder_layer.b)
            i = i - 1

        "******************* End To End Cost function ************************"
        z_mu  = self.encoder[-2].output
        z_var = self.encoder[-1].output
        y     = self.decoder[-1].output

        self.alpha = 1e-8
        self.lamda = 0.05
        
        self.recon = (((self.x - y)**2).mean(1)).mean()
        """When compute a constant together with theano variable, it will be converted
        into the same shape as the theano variable. We may compute mean over features
        of each example instead of sum. This is to avoid the difference in dimension of
        each data. Default lamda = 0.05, alpha = 1e-8"""

        alpha = self.alpha
        self.KL    = T.mean((0.5/alpha)*T.mean(T.exp(z_var) + z_mu**2 - alpha - alpha * z_var + alpha*T.log(alpha), 1))
        self.end2end_cost = self.recon + self.lamda*T.log10(self.KL+1)
        
        #Experiment: lamda = 0.05; alpha = 1e-8    1
        #mean(1) is within example, mean(0) is within each feature
        
    "**************************** Sample z ***********************************"
    def sample_z(self, mu, log_var):
        eps = self.rng.normal(mu.shape, 0.0, 1.0 , dtype = theano.config.floatX)
        sample_z = mu + T.exp(log_var / 2) * eps
        return sample_z

    "********************** Compute KL and Recon Loss *************************"
#    def Recon_KL_Loss(self, data_set):
#        data_size = data_set.get_value().shape[0]
#        index = T.lscalar('index')
#        KL1 = self.lamda*T.log10(self.KL+1)
#        Loss = theano.function([index],
#                               outputs = [self.recon, KL1],
#                               givens={self.x: data_set[index : data_size]})
#        return Loss(0)



    def Recon_KL_loss_batch(self, train_x, batch_size):

        index = T.lscalar('index')
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size
        KL1 = self.lamda*T.log10(self.KL+1)
        loss_com = theano.function([index],
                             outputs = [self.recon, KL1],
                             givens={self.x: train_x[batch_begin : batch_end]})
        return loss_com

    def Recon_KL_Loss(self, train_x, batch_size):
        n_train = train_x.get_value().shape[0]
        n_batches = (int)(n_train/batch_size)
        loss_com = self.Recon_KL_loss_batch(train_x, batch_size)
        loss = np.empty([0,2])
        for batch_index in range(n_batches):
          l = loss_com(index = batch_index)
          loss = np.append(loss, [l[0], l[1]])
        loss = np.reshape(loss, (-1,2))

        return (loss.mean(0))

    "************************** Get Mu and Log_var ****************************"
    def get_mu_logvar(self,data_set):
        data_size = data_set.get_value().shape[0]
        index   = T.lscalar('index')
        mu      =  self.encoder[-2].output
        log_var =  self.encoder[-1].output
        mu_logvar = theano.function([index],
                                    outputs = [mu, log_var],
                                    givens={self.x: data_set[index : data_size]})
        return mu_logvar(0)


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

    "**************************** Get hidden data z **************************"
    def get_hidden_data(self,data_set):
        data_size = data_set.get_value().shape[0]
        index = T.lscalar('index')
        mu      =  self.encoder[-2].output
        log_var =  self.encoder[-1].output
        z = self.sample_z(mu, log_var)
        hidden_data = theano.function([index],
                                      outputs = z,
                                      givens={self.x: data_set[index : data_size]})
        return hidden_data(0)


    "************get data from the output of Autoencoder**********************"
    def get_output_data(self,data_set):
        data_size = data_set.get_value().shape[0]
        index  = T.lscalar('index')
        y_data = theano.function([index],
                                      outputs = self.decoder[-1].output,
                                      givens={self.x: data_set[index : data_size]})
        return y_data(0)

    "******************** Histogram z, z_mu and z_var ************************"
    def Plot_histogram_z(self, train_set, test_set, actual, epoch, path):
        z_train    = self.get_hidden_data(train_set)
        mu, logvar = self.get_mu_logvar(train_set)

        np.savetxt(path + "Visualize_histogram/" + "z_train_" + str(epoch) + "_" + str(self.alpha)+".csv", z_train, delimiter=",", fmt='%f' )
#        np.savetxt(path + "Visualize_histogram/" + "z_mu" + str(epoch) + ".csv",  mu, delimiter=",", fmt='%f' )
#        np.savetxt(path + "Visualize_histogram/" + "z_var" + str(epoch) + ".csv", np.exp(logvar), delimiter=",", fmt='%f' )

#        histogram_z(mu[:,0],             'mu' , self.alpha, epoch, path)
#        histogram_z(np.exp(logvar[:,0]), 'var', self.alpha, epoch, path)
        histogram_z(z_train[:,0],        'z'  , self.alpha, epoch, path)

    "********************** Standard deviation of z ***************************"
    def Compute_Std(self, train_set, test_set, actual, data_name, path):
        z_train    = self.get_hidden_data(train_set)
        z_test     = self.get_hidden_data(test_set)

        visualize_hidden1(z_train, z_test, actual, data_name, path)
        #std on each feature over data
        std = np.std(z_train, axis = 0)
        np.set_printoptions(precision=6, suppress=True)
        print("\n+ Standard Deviation of Hidden data:")
        print(std)

    "********************* Compute AUC on hidden data *************************"
    def Compute_AUC_Hidden(self, train_set, test_set, actual, norm, data_name):
        z_train = self.get_hidden_data(train_set)           #get hidden values
        z_test  = self.get_hidden_data(test_set)            #get hidden values
        y_test  = self.get_output_data(test_set)            #get prediction values
        "Compute performance of classifiers on latent data"
        lof, cen, dis, kde, svm05, svm01 = auc_density(z_train, z_test, actual, norm)
        ae                               = auc_AEbased(test_set.get_value(), y_test, actual)
        return lof, cen, dis, kde, svm05, svm01, ae

    "**************************************************************************"
    def Save_Hidden_Data(self, train_set, test_set, data_name, path):
        z_train = self.get_hidden_data(train_set)      #get hidden values
        z_test  = self.get_hidden_data(test_set)       #get hidden values
        np.savetxt(path + data_name + "_train_z.csv", z_train, delimiter=",", fmt='%f' )
        np.savetxt(path + data_name + "_test_z.csv" ,  z_test, delimiter=",", fmt='%f' )

    "**************************************************************************"
    def Save_Hidden_Data_Size(self, train_set, test_set, data_name, size, path):
        z_train = self.get_hidden_data(train_set)      #get hidden values
        z_test  = self.get_hidden_data(test_set)       #get hidden values
        np.savetxt(path + "data/"+ data_name + "_train_z_" + str(size)+ ".csv", z_train, delimiter=",", fmt='%f' )
        np.savetxt(path + "data/"+ data_name + "_test_z_" + str(size)+ ".csv" ,  z_test, delimiter=",", fmt='%f' )

    "******** Training End-to-End Early-stopping by Downhill Package *********"
    def End2end_Early_stopping(self, numpy_rng, dataset, n_validate, data_name,
                               batch_size, end2end_lr, algo, norm, patience, validation):

        train_X, test_X, actual = dataset
        valid_x = train_X.get_value()[:n_validate]
        train_x = train_X.get_value()[n_validate:]
        "for compute tm and vm before optimization process"
        t = theano.shared(numpy.asarray(train_x, dtype=theano.config.floatX), borrow=True)
        v = theano.shared(numpy.asarray(valid_x, dtype=theano.config.floatX), borrow=True)

        "Training network by downhill"
        #'adadelta' 'adagrad (default 0.01)' 'adam''esgd' 'nag''rmsprop' 'rprop' 'sgd'
        opt = downhill.build(algo = algo, params= self.params, loss = self.end2end_cost, inputs = [self.x])
        train = downhill.Dataset(train_x, batch_size = batch_size, rng = numpy_rng)
        valid = downhill.Dataset(valid_x, batch_size = len(valid_x), rng = numpy_rng)

        "***** Monitor before optimization *****"
        stop_ep = 0
        RE = np.empty([0,3])
        monitor = np.empty([0,8])
#        LOSS = np.empty([0,3])
#        self.Plot_histogram_z(train_X, test_X, actual, 0, path)
                                    #AUC before optimization
        lof,cen,dis,kde,svm05,svm01,ae = self.Compute_AUC_Hidden(train_X, test_X, actual, norm, data_name)
        a = np.column_stack([0, lof, cen, dis, kde, svm05, svm01, ae])
        monitor = np.append(monitor, a)
                                    #Loss components before optimization

#        loss = self.Recon_KL_Loss(t, batch_size)
#        LOSS = np.append(LOSS,[stop_ep, loss[0], loss[1]])
                                    #Error before optimization
        tm1, vm1 = self.Loss_train_valid(t, v)
        RE = np.append(RE, np.column_stack([stop_ep, vm1, tm1]))

        for tm1, vm1 in opt.iterate(train,                     # 5, 5, 1e-2, 0.9
                                  valid,
                                  patience = patience,               # 10
                                  validate_every = validation,            # 5
                                  min_improvement = 1e-3,       # 1e-3
                                  #learning_rate =  end2end_lr,  # 1e-4
                                  momentum = 0.0,
                                  nesterov = False):
            stop_ep = stop_ep+1
#            loss = self.Recon_KL_Loss(t, batch_size)
#            LOSS = np.append(LOSS,[stop_ep, loss[0], loss[1]])
#            "******* Monitor optimization ******"
            if ((stop_ep%200 == 0 ) and (stop_ep > 0)):
                #self.Plot_histogram_z(train_X, test_X, actual, stop_ep, path)
                lof,cen,dis,kde,svm05,svm01,ae = self.Compute_AUC_Hidden(train_X, test_X, actual, norm, data_name)
                a = np.column_stack([stop_ep, lof, cen, dis, kde, svm05, svm01, ae])

            monitor = np.append(monitor, a)
            re = np.column_stack([stop_ep, vm1['loss'], tm1['loss']])
            RE = np.append(RE, re)

            if (stop_ep >= 1000):
                break

        #Plotting AUC and save to csv file
        monitor = np.reshape(monitor, (-1,8))
#        Plotting_Monitor(monitor, 0.4, 1.0, data_name, path)
#        np.savetxt(path + data_name + "_monitor_auc1.csv", monitor, delimiter=",", fmt='%f' )


#        LOSS = np.reshape(LOSS, (-1,3))
#        Plotting_Loss_Component(LOSS, RE, 0.0, 0.5, data_name, path)
#        np.savetxt(path + data_name + "_loss_component.csv", LOSS, delimiter=",", fmt='%f' )

        RE = np.reshape(RE, (-1,3))
#        Plotting_End2End_RE(RE, stop_ep, 0.0, 0.4, data_name, path)
#        np.savetxt(path +  data_name + "_training_error1.csv", RE, delimiter=",", fmt='%f' )

        np.set_printoptions(precision=6, suppress=True)
        print ("\n ",RE[stop_ep])

        return RE[stop_ep]



def test_SdA(pre_lr=0.01, end2end_lr=1e-4, algo = 'sgd',
             dataset=[], data_name = "WBC", n_validate = 0, norm = "maxabs",
             batch_size=10, hidden_sizes = [1,1,1], corruptions = [0.0, 0.0, 0.0],
             patience = 1, validation = 1):

    numpy_rng = numpy.random.RandomState(89677)     # numpy random generator 89677
    train_X, test_X, actual = dataset               # dataset is already normalised

    input_size = train_X.get_value().shape[1]       # input size = dimension
    train_x    = train_X.get_value()[n_validate:]   # 80% for pre-training, 20% for validation
    n_train_batches   = train_x.shape[0]
    n_train_batches //= batch_size                  # number of batches for pre-training

    # construct the stacked denoising autoencoder class
    sda = SdA(numpy_rng = numpy_rng, n_ins = input_size,
              hidden_layers_sizes = hidden_sizes)


    RE = sda.End2end_Early_stopping(numpy_rng, dataset, n_validate, data_name,
                               batch_size, end2end_lr, algo, norm, patience, validation)
    return sda, RE


def Main_Test():
	
    list_data =  ["PageBlocks", "WPBC", "PenDigits", "GLASS", "Shuttle", "Arrhythmia",\
                 "CTU13_10", "CTU13_08","CTU13_09","CTU13_13",\
                 "Spambase", "UNSW", "NSLKDD", "InternetAds"]

    norm         = "maxabs"            # standard, maxabs[-1,1] or minmax[0,1]
    corruptions  = [0.1, 0.1, 0.1]

    print ("+ VAE: 0.05, Group")
    print ("+ Data: ", list_data)
    print ("+ Scaler: ", norm)

    AUC_Hidden = np.empty([0,10])     #store auc of all hidden data
    num = 0                           #a counter
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
        #batch    = int(n_train/20)                           #Training set will be split training set into 20 batches
                                                    #print data information
        pat, val, batch, n_batch = stopping_para_vae(n_train)

        print ("\n" + str(num) + ".", data, "..." )
        print (" + Hidden Sizes: ",in_dim, h_sizes, "- Batch_sizes:", batch)
        print (" + Data: %d (%d train, %d vali) - %d normal, %d anomaly"\
            %(len(train_set), n_train, n_vali, \
            len(test_set[(actual == 1)]), len(test_set[(actual == 0)])))

        print(" + Patience: %5.0d, Validate: %5.0d,  \n + Batch size: %5.0d, n batch:%5.0d"\
             %(pat, val, batch, n_batch))

        AUC_RE   = np.empty([0,10])
                               #adadelta, 'adagrad' 'adam''esgd' 'nag''rmsprop' 'rprop' 'sgd'
        #if (num==1):
        sda, re = test_SdA(pre_lr       = 1e-2,            #re = [stop_ep, vm, tm]
                                   end2end_lr   = 1e-4,
                                   algo         = 'adadelta',
                                   dataset      = datasets,
                                   data_name    = data,
                                   n_validate   = n_vali,
                                   norm         = norm,
                                   batch_size   = batch,
                                   hidden_sizes = h_sizes,
                                   corruptions = corruptions,
                                   patience     = pat,
                                   validation   = val)

        #Computer AUC on hidden data
        lof,cen,dis,kde,svm05,svm01,ae  = sda.Compute_AUC_Hidden(train_X, test_X, actual, norm, data)
        auc_hidden = np.column_stack([batch, re[0], lof, cen, dis, kde, svm05, svm01, ae , 100*re[2]])
        AUC_Hidden = np.append(AUC_Hidden, auc_hidden)

        #compute standard deviation of z
        #sda.Compute_Std(train_X, test_X, actual, data, path)
        #save hidden data to files
#        sda.Save_Hidden_Data(train_X, test_X, data, path)

        #store AUC_input AUC_hidden and RE to AUC_RE for each data
#        AUC_RE   = np.append(AUC_RE, auc_hidden)
#        AUC_RE   = np.reshape(AUC_RE,(-1,10))
#
#        print("\n+ AUC input, AUC hidden:")
#        np.set_printoptions(precision=3, suppress=True)
#        column_list = [2,3,4,5,6,7,8,9]
#        print (AUC_RE[:,column_list])
#
#        AUC_Hidden = np.append(AUC_Hidden, auc_hidden)
        AUC_Hidden  =  np.reshape(AUC_Hidden, (-1, 10))
        np.set_printoptions(precision=3, suppress=True)
        column_list = [2,3,4,5,6,7,8,9]
        print("    LOF    CEN    MDIS   KDE   SVM5    SVM1    AE    RE*100")
        print (AUC_Hidden[:,column_list])
        print("\n")
        
    AUC_Hidden  =  np.reshape(AUC_Hidden, (-1, 10))
    np.set_printoptions(precision=3, suppress=True)
    column_list = [2,3,4,5,6,7,8,9]
    print("    LOF    CEN    MDIS   KDE   SVM5    SVM1    AE    RE*100")
    print (AUC_Hidden[:,column_list])

#    #store AUC_input and AUC_hidden to AUC_Input, AUC_Hidden
#    AUC_Hidden  =  np.reshape(AUC_Hidden, (-1, 10))
#    np.savetxt(path +  "AUC_Hidden.csv", AUC_Hidden, delimiter=",", fmt='%f' )


if __name__ == '__main__':
    Main_Test()
