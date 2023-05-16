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
    

def plot_partial_spectra_reconstruction_report(X_partial_recon_orig, X_train_orig, X_partial_pred_sigma):
    
    plt.figure(figsize=(9,15))
    
    obs_regions = [(0,280), (280,590), (140,350), (310,400)]
    
    for i in np.arange(4):
    
        plt.subplot(4,1,i+1)
        plt.plot(X_partial_recon_orig[i], c='b', alpha=0.5, label='Predicted mean')
        plt.plot(X_train_orig, c='r', linestyle='--', alpha=0.8, label='Ground truth')
        lower = X_partial_recon_orig[i] - 3*X_partial_pred_sigma[i]
        upper = X_partial_recon_orig[i] + 3*X_partial_pred_sigma[i]
        plt.axvspan(xmin=obs_regions[i][0], xmax=obs_regions[i][1], color='orange', alpha=0.3, label='Observed Region')
        plt.fill_between(np.arange(len(X_partial_recon_orig.T)), lower, upper, color='b', alpha=0.3, label=r'$\pm2\sigma$')
        plt.tight_layout()
        plt.suptitle('Quasar Spectra Reconstruction [from partial observations]')
        if i == 3:
            plt.legend(fontsize='small')


def plot_y_label_comparison(Y_train_recon_orig, Y_train_orig, Y_pred_var, snr, col_id, title):
    
    plt.figure()
    plt.scatter(Y_train_orig[:,col_id], Y_train_recon_orig[:,col_id], c=snr,cmap='jet')
    plt.errorbar(Y_train_orig[:,col_id], Y_train_recon_orig[:,col_id], yerr=1*Y_pred_var[:,col_id], fmt='None')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k',alpha=0.7, scalex=False, scaley=False)
    #plt.plot(np.arange(5), np.arange(5),'--',c='k', alpha=0.7)
    plt.ylabel(r'Predicted ', fontsize='large')
    plt.xlabel(r'Measured ', fontsize='large')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('SNR')
    plt.xlim(xpoints[0], xpoints[-1])
    plt.ylim(xpoints[0], xpoints[-1])
    plt.tight_layout()
    
def spectra_reconstruction_report(X_train_recon_orig, X_train_orig, X_pred_sigma, samples):
    
    
    plt.figure(figsize=(9,15))
    
    obj_ids = [1,4,78,101,145,178]
    
    for i in np.arange(6):
    
        obj_id = obj_ids[i]
        plt.subplot(6,1,i+1)
        plt.plot(X_train_recon_orig[obj_id], c='b', alpha=0.5, label='Predicted mean')
        plt.plot(X_train_orig[obj_id], c='r', linestyle='--', alpha=0.8, label='Ground truth')
        lower = X_train_recon_orig[obj_id] - 2*X_pred_sigma[obj_id]
        upper = X_train_recon_orig[obj_id] + 2*X_pred_sigma[obj_id]
        plt.fill_between(np.arange(len(X_train_recon_orig.T)), lower, upper, color='b', alpha=0.3, label=r'$\pm2\sigma$')
        if i == 0:
            plt.legend(fontsize='small')
        if i == 2:
            plt.ylabel('Normalised Flux')
        if i == 5:
            plt.xlabel('rest-frame wavelength')
    #plt.suplabel('normalised flux')
    plt.tight_layout()
    plt.suptitle('Quasar Spectra Reconstruction')


    # for i, j  in zip(np.arange(6),[7,8,9,10,11,12]):
    
    #     obj_id = obj_ids[i]
    #     plt.subplot(6,2,j)
    #     plt.plot((samples[:,:,obj_id]).T, color='b', alpha=0.3, label='Posterior Samples')
    #     #plt.plot((samples[:,:,obj_id]*std_X + means_X).T, color='b', alpha=0.3, label='Posterior Samples')
    #     plt.plot(X_train[obj_id], c='r', linestyle='--', alpha=0.7, label='Ground truth')
    #     #lower = X_train_recon_orig[obj_id] - 2*X_pred_sigma[obj_id]
    #     #upper = X_train_recon_orig[obj_id] + 2*X_pred_sigma[obj_id]
    #     #plt.fill_between(np.arange(len(X_train_recon_orig.T)), lower, upper, color='b', alpha=0.5, label=r'$\pm2\sigma$')
    # plt.legend(fontsize='small')
    # plt.tight_layout()
    
    
#def plot_latent_variable_plots(shared_model, dim1, dim2)
    
    


