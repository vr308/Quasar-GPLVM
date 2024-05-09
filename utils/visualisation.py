#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation code for quasar spectra reconstruction plots, label reproduction and test predictions

"""

import matplotlib.pylab as plt
import numpy as np

def plot_spectra_reconstructions(wave, X_test_recon_orig, X_test_orig, X_pred_var, obj_id):
    
    plt.figure(figsize=(9,5))
    plt.plot(wave, X_test_recon_orig[obj_id], c='b', alpha=0.6, label='predicted spectrum')
    plt.plot(wave, X_test_orig[obj_id], c='r', alpha=0.7, label='data')
    
    lower = X_test_recon_orig[obj_id] - 1.96*X_pred_var[obj_id]
    upper = X_test_recon_orig[obj_id] + 1.96*X_pred_var[obj_id]
    
    plt.fill_between(wave, lower, upper, color='b', alpha=0.5, label=r'$\pm2\sigma$')
    plt.title('Spectrum for obj id ' + str(obj_id))
    plt.legend(fontsize='small')
    plt.ylabel('normalised flux')
    plt.xlabel(r'rest-frame wavelength [$\AA$]')
    plt.tight_layout()
    

def plot_partial_spectra_reconstruction_report(wave, X_partial_recon_orig, X_test_orig, X_partial_pred_sigma, bhm_value):
    
    plt.figure(figsize=(9,15))
    
    obs_regions = [(0,280), (280,585), (140,350), (310,400)]
    
    for i in np.arange(4):
    
        plt.subplot(4,1,i+1)
        plt.plot(wave, X_partial_recon_orig[i], c='b', alpha=0.5, label='predicted spectrum')
        plt.plot(wave, X_test_orig, c='r', linestyle='dotted', alpha=0.8, label=r'data: $\log_{10}(M_{\bullet}/M_{\odot})$ ' + str(np.round(bhm_value.item(),4)))
        lower = X_partial_recon_orig[i] - 3*X_partial_pred_sigma[i]
        upper = X_partial_recon_orig[i] + 3*X_partial_pred_sigma[i]
        plt.axvspan(xmin=wave[obs_regions[i][0]], xmax=wave[obs_regions[i][1]], color='orange', alpha=0.3, label='Observed Region')
        plt.fill_between(wave, lower, upper, color='b', alpha=0.3, label=r'$\pm2\sigma$')
        plt.tight_layout()
        plt.ylabel('normalised flux')
        plt.ylim(ymin=0)
        if i == 0:
            plt.legend(fontsize='medium')
        if i == 3:
            plt.xlabel(r'rest-frame wavelength [$\AA$]')
        plt.suptitle('Quasar Spectra Reconstruction [from partial observations]')

def plot_y_label_comparison(Y_test_recon_orig, Y_test_orig, Y_test_pred_var, col_id, title, colors, clabel, xlabel, ylabel, cmap):
    
    plt.figure()
    plt.scatter(Y_test_orig[:,col_id].cpu().detach(), Y_test_recon_orig[:,col_id].cpu().detach(), c=colors, cmap=cmap)
    plt.errorbar(Y_test_orig[:,col_id].cpu().detach(), Y_test_recon_orig[:,col_id].cpu().detach(), c='peru', alpha=0.5, yerr=3*Y_test_pred_var[:,col_id].cpu().detach(), fmt='None')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k',alpha=0.7, scalex=False, scaley=False)
    #plt.plot(np.arange(5), np.arange(5),'--',c='k', alpha=0.7)
    plt.ylabel(ylabel, fontsize='large')
    plt.xlabel(xlabel, fontsize='large')
    plt.title(title)
    cbar = plt.colorbar()
    #cbar.set_label()
    plt.xlim(xpoints[0], xpoints[-1])
    plt.ylim(xpoints[0], xpoints[-1])
    plt.tight_layout()
    
def spectra_reconstruction_report(wave, X_test_recon_orig, X_test_orig, X_pred_sigma, bhms):
    
    plt.figure(figsize=(7,15))
    
    obj_ids = [1,4,13,78]
    
    for i in np.arange(4):
    
        obj_id = obj_ids[i]
        plt.subplot(4,1,i+1)
        plt.plot(wave, X_test_recon_orig[obj_id], c='b', alpha=0.5, label='predicted spectrum')
        plt.plot(wave, X_test_orig[obj_id], c='r', linestyle='--', alpha=0.8, label=r'data; $\log_{10}(M_{\bullet}/M_{\odot})$ ' + str(np.round(bhms[obj_id].item(),4)))
        lower = X_test_recon_orig[obj_id] - 2*X_pred_sigma[obj_id]
        upper = X_test_recon_orig[obj_id] + 2*X_pred_sigma[obj_id]
        plt.fill_between(wave, lower, upper, color='b', alpha=0.3, label=r'$\pm2\sigma$')
        plt.legend(fontsize='medium')
        plt.ylabel('normalised flux')
        if i == 3:
            plt.xlabel(r'rest-frame wavelength [$\AA$]')
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
    
    


