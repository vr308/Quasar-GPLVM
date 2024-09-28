#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared GPLVM with NN encoder

@author: vr308

"""

from models.shared_gplvm import SharedGPLVM, predict_joint_latent
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import os 
import pickle as pkl
import numpy as np
import gc
from tqdm import trange
from prettytable import PrettyTable
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from models.likelihood import GaussianLikelihoodWithMissingObs, GaussianLikelihood
from utils.load_data import load_spectra_labels, load_synthetic_labels_no_redshift
from utils.visualisation import plot_spectra_reconstructions, plot_y_label_comparison, plot_partial_spectra_reconstruction_report, spectra_reconstruction_report
from models.likelihood import GaussianLikelihoodWithMissingObs
from utils.load_data import load_spectra_labels
from utils.visualisation import plot_spectra_reconstructions, plot_y_label_comparison, spectra_reconstruction_report, plot_partial_spectra_reconstruction_report
from utils.metrics import rmse_lum_bhm_edd, nll_lum_bhm_edd, rmse
from models.likelihood import GaussianLikelihoodWithMissingObs
from utils.load_data import load_spectra_labels
from utils.visualisation import plot_spectra_reconstructions, plot_y_label_comparison, spectra_reconstruction_report, plot_partial_spectra_reconstruction_report
from utils.metrics import rmse_lum_bhm_edd, nll_lum_bhm_edd, rmse

## Import class and experiment configuration here

from utils.config import hdu, BASE_SEED, latent_dim, test_size, num_inducing

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setting torch and numpy seed for reproducibility
    
    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    
    # Load joint spectra and label data 
    
    X, Y, means_X, std_X, means_Y, std_Y, X_ivar, Y_ivar, snr, wave  = load_spectra_labels(hdu)

    data = np.hstack((X,Y))[0:15000]
    
    XY_train, XY_test, train_idx, test_idx = train_test_split(data, np.arange(len(data)), test_size=test_size, random_state=BASE_SEED)
    snr_test = snr[test_idx]

    XY_train = torch.Tensor(XY_train).to(device)
    XY_test = torch.Tensor(XY_test).to(device)
    std_X = torch.Tensor(std_X).to(device)
    std_Y = torch.Tensor(std_Y).to(device)
    means_X = torch.Tensor(means_X).to(device)
    means_Y = torch.Tensor(means_Y).to(device)
    
    # Experiment config
      
    N = len(XY_train)
    data_dim = XY_train.shape[1]
    
    spectra_dim = X.shape[1]
    label_dim = Y.shape[1]
      
    # Shared Model 
    
    shared_model = SharedGPLVM(N, spectra_dim, label_dim, latent_dim, num_inducing, latent_config='nn_encoder').to(device)
    
    # Missing data Likelihood
    
    likelihood_spectra = GaussianLikelihoodWithMissingObs(batch_shape = shared_model.model_spectra.batch_shape).to(device)
    likelihood_labels = GaussianLikelihoodWithMissingObs(batch_shape = shared_model.model_labels.batch_shape).to(device)
 
    # Declaring objective to be optimised along with optimiser
    
    mll_spectra = VariationalELBO(likelihood_spectra, shared_model.model_spectra, num_data=len(XY_train)).to(device)
    mll_labels = VariationalELBO(likelihood_labels, shared_model.model_labels, num_data=len(XY_train)).to(device)

    optimizer = torch.optim.Adam([
        dict(params=shared_model.parameters(), lr=0.01),
        dict(params=likelihood_spectra.parameters(), lr=0.01),
        dict(params=likelihood_labels.parameters(), lr=0.01)
    ])
      
    ############## Training loop - optimises the objective wrt kernel hypers, ######
    ################  variational params and inducing inputs using the optimizer provided. ########
    
    loss_list = []
    iterator = trange(5000, leave=True)
    batch_size = 128

    for i in iterator: 
        
        mask = torch.isnan(XY_train)  # Create mask indicating where the NaNs are
        batch_index = shared_model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample_batch = shared_model.Z.forward(XY_train, mask, batch_idx =batch_index)
        
        ### Getting the output of the two groups of GPs
        
        output_spectra = shared_model.model_spectra(sample_batch)
        output_labels = shared_model.model_labels(sample_batch)
        
        ### Adding together the ELBO losses 
        
        joint_loss = -mll_spectra(output_spectra, XY_train[batch_index].T[0:spectra_dim]).sum() -mll_labels(output_labels, XY_train[batch_index].T[spectra_dim:]).sum()
        loss_list.append(joint_loss.item())
        
        iterator.set_description('Loss: ' + str(float(np.round(joint_loss.item(),2))) + ", iter no: " + str(i))
        joint_loss.backward()
        shared_model.inducing_inputs.grad = shared_model.inducing_inputs.grad.to(device)
        optimizer.step()
        
    Z_train = shared_model.Z.Z
    
    ids = np.arange(200)
    
    X_train = XY_train[::,0:-4][ids]
    Y_train = XY_train[:,-4::][ids]
    
    X_train_orig = X_train*std_X + means_X
    Y_train_orig = Y_train*std_Y + means_Y
  
    X_train_recon, X_train_pred_covar = shared_model.model_spectra.reconstruct_y(Z_train[ids], XY_recon_pass[0:200], ae=True)
    Y_train_recon, Y_train_pred_covar = shared_model.model_labels.reconstruct_y(Z_train[ids], XY_recon_pass[0:200], ae=True)
        
    #X_train_recon =  X_train_recon.T
    Y_train_recon =  Y_train_recon.T

    # X_train_recon =  X_train_recon.T.detach().numpy()
    # Y_train_recon =  Y_train_recon.T.detach().numpy()
    
    #X_train_recon_orig = X_train_recon*std_X + means_X
    Y_train_recon_orig = Y_train_recon*std_Y + means_Y
    
    #vars_X_noiseless = np.array([(m.diag()).cpu().detach().numpy() for m in X_train_pred_covar]).T ## extracting diagonals per dimensions
    #vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().cpu().detach().numpy() for m in vars_X_noiseless])
    
    # X_train_recon_orig = X_train_recon*std_X + means_X
    # Y_train_recon_orig = Y_train_recon*std_Y + means_Y
    
    vars_X_noiseless = np.array([(m.diag()).cpu().detach().numpy() for m in X_train_pred_covar]).T ## extracting diagonals per dimensions
    vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().cpu().detach().numpy() for m in vars_X_noiseless])
    
    diags_Y_list = [m.diag().sqrt() for m in Y_train_pred_covar]
    diags_Y = torch.cat(diags_Y_list).reshape(len(ids),4)
    
    #X_train_pred_sigma = np.sqrt(vars_X_noisy)*std_X.cpu().numpy()
    Y_train_pred_sigma = diags_Y*std_Y

    # vars_X_noiseless = np.array([(m.diag()).detach().numpy() for m in X_train_pred_covar]).T ## extracting diagonals per dimensions
    # vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().detach().numpy() for m in vars_X_noiseless])
    
    # diags_Y = np.array([m.diag().sqrt().detach().numpy() for m in Y_train_pred_covar]).T #