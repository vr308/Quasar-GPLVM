#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loading the small base data for N = 30 

"""
import pickle
import numpy as np
from astropy.io import fits
# -------------------------------------------------------------------------------
# load data
# -------------------------------------------------------------------------------

f = open('data/data_HST_1220_5000_2A.pickle', 'rb') 
data, data_ivar = pickle.load(f)
f.close()

hdu = fits.open('data/data_norm_sdss16_SNR10_random_1.fits')  

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

def load_spectra_labels_large():
    
    issues = hdu[4].data
    wave = hdu[0].data  
    X = hdu[1].data[issues == 0.]
    X_ivar = hdu[2].data[issues == 0.]
    masks = hdu[3].data[issues == 0.]
    Y = hdu[5].data[issues == 0.]
    Y_ivar = hdu[6].data[issues == 0.]
    snr = hdu[7].data[issues == 0.]
    
    # set missing values to NaN
    X[masks == 0.] = np.nan
    
    # full data set will have 23085 quasars (or ~80000), only 1000 now
    X = X[:1000, :]
    Y = Y[:1000, :]
    
    means_X = np.nanmean(X, axis = 0)
    means_Y = np.nanmean(Y, axis = 0)
    std_X = np.nanstd(X, axis = 0)
    std_Y = np.nanstd(Y, axis = 0)
    
    X = (X - means_X) / std_X
    Y = (Y - means_Y) / std_Y
    
    return X, Y, means_X, std_X, means_Y, std_Y, snr
    
  


