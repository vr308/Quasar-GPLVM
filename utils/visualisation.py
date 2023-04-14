#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation code for quasar spectra reconstruction plots, label reproduction and test predictions

"""

import matplotlib.pylab as plt
import numpy as np

def plot_spectra_reconstructions(X_train_recon_orig, X_train_orig, X_pred_var, obj_id):
    
    plt.figure(figsize=(9,5))
    plt.plot(X_train_recon_orig[obj_id], c='b', alpha=0.6, label='Predicted mean')
    plt.plot(X_train_orig[obj_id], c='r', alpha=0.7, label='Ground truth')
    
    lower = X_train_recon_orig[obj_id] - 1.96*X_pred_var[obj_id]
    upper = X_train_recon_orig[obj_id] + 1.96*X_pred_var[obj_id]
    
    plt.fill_between(np.arange(len(X_train_recon_orig.T)), lower, upper, color='b', alpha=0.5, label=r'$\pm2\sigma$')
    plt.title('Spectra for obj id ' + str(obj_id))
    plt.legend(fontsize='small')
    plt.tight_layout()


def plot_y_label_comparison(Y_train_recon_orig, Y_train_orig, Y_pred_var, snr, col_id, title):
    
    plt.figure()
    plt.scatter(Y_train_orig[:,col_id], Y_train_recon_orig[:,col_id], c=snr,cmap='jet')
    plt.errorbar(Y_train_orig[:,col_id], Y_train_recon_orig[:,col_id], yerr=1*Y_pred_var[:,col_id], fmt='None')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k',alpha=0.7, scalex=False, scaley=False)
    #plt.plot(np.arange(5), np.arange(5),'--',c='k', alpha=0.7)
    plt.ylabel('Predicted')
    plt.xlabel('Measured')
    plt.title(title)
    plt.colorbar()
    plt.xlim(xpoints[0], xpoints[-1])
    plt.ylim(xpoints[0], xpoints[-1])
    plt.tight_layout()
    
#def plot_latent_variable_corner_plots()
    
    


