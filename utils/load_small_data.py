#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loading the small base data for N = 30 

"""
import pickle
import numpy as np

# -------------------------------------------------------------------------------
# load data
# -------------------------------------------------------------------------------

f = open('data/data_HST_1220_5000_2A.pickle', 'rb') 
data, data_ivar = pickle.load(f)
f.close()

# -------------------------------------------------------------------------------
# initialize parameters
# -------------------------------------------------------------------------------

def save_joint_spectra_labels_small():

    for qq in range(data.shape[0]): 
     
        # remove one object for test set, rest as training set
        ind_test = np.zeros(data.shape[0], dtype = bool)
        ind_test[qq] = True
        ind_train = ~ind_test 
    
        # -------------------------------------------------------------------------------
        # re-scale all input data to be Gaussian with zero mean and unit variance
        # -------------------------------------------------------------------------------
        
        qs = np.nanpercentile(data[ind_train, :], (2.5, 50, 97.5), axis=0)
        pivots = qs[1]
        scales = (qs[2] - qs[0]) / 4.
        scales[scales == 0] = 1.
        data_scaled = (data - pivots) / scales
        data_ivar_scaled = data_ivar * scales**2 
        
        # -------------------------------------------------------------------------------
        # prepare rectangular training data
        # -------------------------------------------------------------------------------
        
        # spectra
        X = data_scaled[:, 7:] 
        X_var = 1 / data_ivar_scaled[:, 7:]
        
        # labels
        inds_label = np.zeros(data_scaled.shape[1], dtype = bool)
        inds_label[0] = True # black hole mass
        inds_label[1] = True # redshift
        inds_label[6] = True # Lbol
        Y = data_scaled[:, inds_label] 
        Y_var = (1 / data_ivar_scaled[:, inds_label])
        
        data_scaled = np.hstack((X,Y))
        return data_scaled
        
def load_joint_spectra_labels_small():
    
       data = np.loadtxt(fname='data/small_quasar.csv', dtype=float, delimiter=',')
       return data


