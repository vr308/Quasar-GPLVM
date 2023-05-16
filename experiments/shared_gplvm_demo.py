#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for Shared GPLVM with split GPs

TODO: 
    
    Test with MAP inference
    Implement GPU deployment
    Clean-up experiment scripts / model classes
    
"""
from models.shared_gplvm import SharedGPLVM
from models.latent_variable import PointLatentVariable, MAPLatentVariable
from sklearn.model_selection import train_test_split
import torch
import os 
import pickle as pkl
import numpy as np
from tqdm import trange
from prettytable import PrettyTable
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from models.likelihood import GaussianLikelihoodWithMissingObs
from utils.load_data import load_spectra_labels
from utils.visualisation import plot_spectra_reconstructions, plot_y_label_comparison, plot_partial_spectra_reconstruction_report
from utils.metrics import rmse_missing, nll

## Import class and experiment configuration here

from utils.config import *

save_model = True

if __name__ == '__main__':
    
    # Setting torch and numpy seed for reproducibility
    
    SEED = BASE_SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load joint spectra and label data 
    
    X, Y, means_X, std_X, means_Y, std_Y, snr = load_spectra_labels(hdu)
    
    data = np.hstack((X,Y))
    
    XY_train, XY_test = train_test_split(data, test_size=test_size, random_state=SEED)
    
    XY_train = torch.Tensor(XY_train)
    XY_test = torch.Tensor(XY_test)
    
    if torch.cuda.is_available():
        
        XY_train = XY_train.cuda()
        XY_test = XY_test.cuda()
    
    # Experiment config
    
    N = len(XY_train)
    data_dim = XY_train.shape[1]
    latent_dim = latent_dim_q
    n_inducing = num_inducing
    
    spectra_dim = X.shape[1]
    label_dim = Y.shape[1]
      
    # Shared Model
    
    shared_model = SharedGPLVM(N, spectra_dim, label_dim, latent_dim, n_inducing)
    
    # Likelihood
    
    likelihood_spectra = GaussianLikelihoodWithMissingObs(batch_shape = shared_model.model_spectra.batch_shape)
    likelihood_labels = GaussianLikelihoodWithMissingObs(batch_shape = shared_model.model_labels.batch_shape)
    
    # Deploy model and likelihoods on cuda
    
    if torch.cuda.is_available():
        
        shared_model = shared_model.cuda()
        likelihood_spectra = likelihood_spectra.cuda()
        llikelihood_labels = likelihood_labels.cuda()


    # Declaring objective to be optimised along with optimiser
    
    mll_spectra = VariationalELBO(likelihood_spectra, shared_model.model_spectra, num_data=len(XY_train)).cuda()
    mll_labels = VariationalELBO(likelihood_labels, shared_model.model_labels, num_data=len(XY_train)).cuda()

    optimizer = torch.optim.Adam([
        dict(params=shared_model.parameters(), lr=0.005),
        dict(params=likelihood_spectra.parameters(), lr=0.005),
        dict(params=likelihood_labels.parameters(), lr=0.005)
    ])
    
    shared_model.get_trainable_param_names()
    
    ############## Training loop - optimises the objective wrt kernel hypers, ######
    ################  variational params and inducing inputs using the optimizer provided. ########
    
    loss_list = []
    iterator = trange(50, leave=True)
    batch_size = 100
    
    for i in iterator: 
        batch_index = shared_model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample = shared_model.Z.Z  # a full sample returns latent Z across all N
        sample_batch = sample[batch_index]
        
        ### Getting the output of the two groups of GPs
        
        output_spectra = shared_model.model_spectra(sample_batch)
        output_labels = shared_model.model_labels(sample_batch)
        
        ### Adding together the ELBO losses 
        
        joint_loss = -mll_spectra(output_spectra, XY_train[batch_index].T[0:spectra_dim]).sum() -mll_labels(output_labels, XY_train[batch_index].T[spectra_dim:]).sum()
        loss_list.append(joint_loss.item())
        
        iterator.set_description('Loss: ' + str(float(np.round(joint_loss.item(),2))) + ", iter no: " + str(i))
        joint_loss.backward()
        optimizer.step()
        
    Z_train = shared_model.Z.Z
        
    ####################### Save / Load trained model ##########################################
    
    if os.path.isfile('trained_models/gplvm_1000_shared.pkl'):
          with open('trained_models/gplvm_1000_shared.pkl', 'rb') as file:
              model_sd, likl_sd = pkl.load(file)
              model.load_state_dict(model_sd)
              likelihood_sp.load_state_dict(likl_sd)
              likelihood_lb.load_state_dict(likl_sd)


    with open('trained_models/gplvm_1000_shared.pkl', 'wb') as file:
        pkl.dump((shared_model.cpu().state_dict(), likelihood_spectra.cpu().state_dict(), likelihood_labels.cpu()), file)
        
    ####################### Split Reconstruction Framework (Training and Test) ##############
    
    
    ids = np.arange(200)
    
    X_train = XY_train[::,0:-4]
    Y_train = XY_train[:,-4::]
    
    X_train_orig = X_train*std_X + means_X
    Y_train_orig = Y_train*std_Y + means_Y
    
    X_train_recon, X_train_pred_covar = shared_model.model_spectra.reconstruct_y(torch.Tensor(shared_model.Z.Z)[0:200], X_train[0:200], ae=False)
    Y_train_recon, Y_train_pred_covar = shared_model.model_labels.reconstruct_y(torch.Tensor(shared_model.Z.Z)[0:200], Y_train[0:200], ae=False)
        
    X_train_recon =  X_train_recon.T.detach().numpy()
    Y_train_recon =  Y_train_recon.T.detach().numpy()
    
    X_train_recon_orig = X_train_recon*std_X + means_X
    Y_train_recon_orig = Y_train_recon*std_Y + means_Y
    
    vars_X_noiseless = np.array([(m.diag()).detach().numpy() for m in X_train_pred_covar]).T ## extracting diagonals per dimensions
    vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().detach().numpy() for m in vars_X_noiseless])
    
    diags_Y = np.array([m.diag().sqrt().detach().numpy() for m in Y_train_pred_covar]).T #
    
    X_train_pred_sigma = np.sqrt(vars_X_noisy)*std_X 
    Y_train_pred_sigma = diags_Y*std_Y
    
    ################ Testing #######################################################################
    
    # Initialise test model at training params
   
    TEST = True

    if TEST:
        
        X_test = XY_test[::,0:-4]
        Y_test = XY_test[:,-4::]
        
        X_test_orig = X_test*std_X + means_X
        Y_test_orig = Y_test*std_Y + means_Y
        
        test_model = shared_model.initialise_model_test(len(Y_test))

        test_loss, test_model, Z_test = predict_joint_latent(test_model, X_test, None, likelihood_spectra, likelihood_labels, lr=0.001, prior_z = None, steps = 2000)

        X_test_recon, X_test_pred_covar = test_model.model_spectra.reconstruct_y(Z_test.Z, X_test, ae=False)
        Y_test_recon, Y_test_pred_covar = test_model.model_labels.reconstruct_y(Z_test.Z, Y_test, ae=False)
        
        X_test_recon =  X_test_recon.T.detach().numpy()
        Y_test_recon =  Y_test_recon.T.detach().numpy()
        
        X_test_recon_orig = X_test_recon*std_X + means_X
        Y_test_recon_orig = Y_test_recon*std_Y + means_Y
        
        vars_X_noiseless = np.array([(m.diag()).detach().numpy() for m in X_test_pred_covar]).T ## extracting diagonals per dimensions
        vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().detach().numpy() for m in vars_X_noiseless])
    
        diags_Y = np.array([m.diag().sqrt().detach().numpy() for m in Y_train_pred_covar]).T #
        
        X_test_pred_sigma = np.sqrt(vars_X_noisy)*std_X 
        Y_test_pred_sigma = diags_Y*std_Y
    
    ################## Partial observation region (Spectra reconstruction) ####################################################
    
    idx = 138
    
    test_point = torch.full(torch.Size([4,590]), torch.nan)
    
    obs_region_1 = np.arange(0,280)    ## first half of spectra 
    obs_region_2 = np.arange(280,590)  ## last half of spectra 
    obs_region_3 = np.arange(140,350)  ## obs middle part 
    obs_region_4 = np.arange(310,400)  ## obs a small internal section 
    
    test_point[0,obs_region_1] = X_test[idx][obs_region_1]
    test_point[1,obs_region_2] = X_test[idx][obs_region_2]
    test_point[2,obs_region_3] = X_test[idx][obs_region_3]
    test_point[3,obs_region_4] = X_test[idx][obs_region_4]
    
    #Z_latent = torch.Tensor(test_model.Z.Z)[idx].repeat(4).reshape(4,10)
    test_model = shared_model.initialise_model_test(4)
    test_point_Y = Y_test[idx].repeat(4).reshape(4,4)

    test_loss, test_model, Z_partial = predict_joint_latent(test_model, test_point, test_point_Y, likelihood_spectra, likelihood_labels, lr=0.005, prior_z = None, steps = 10000)
    
    X_partial_recon, X_partial_pred_covar = shared_model.model_spectra.reconstruct_y(Z_partial.Z, test_point, ae=False)

    X_partial_recon_orig = X_partial_recon.T.detach().numpy()*std_X + means_X
    
    vars_X_noiseless = np.array([(m.diag()).detach().numpy() for m in X_partial_pred_covar]).T ## extracting diagonals per dimensions
    vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().detach().numpy() for m in vars_X_noiseless])
    X_partial_pred_sigma = np.sqrt(vars_X_noisy)*std_X 
    
    plot_partial_spectra_reconstruction_report(X_partial_recon_orig, X_test_orig[idx], X_partial_pred_sigma)
    
    ####################### Visualisation ################################
    
    # plt.plot(np.isnan(XY_train).sum(axis=0)) ## check the presence of the data 
    
    # ids = np.arange(200)
    # col_range = np.arange(68,1000)
    
    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma[ids], obj_id=24)
    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma[ids], obj_id=13)
    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma[ids], obj_id=78)
    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma[ids], obj_id=151)
  
    plot_y_label_comparison(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], snr[ids], col_id = 1, title='Luminosity')
    plot_y_label_comparison(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], snr[ids], col_id = 2, title='Black hole mass')
    plot_y_label_comparison(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], snr[ids], col_id = 3, title='Eddington luminosity')
        
    # # # ################################
    # Compute the metrics:
                
    # 1) Reconstruction error - Train & Test
    
    mse_test = rmse_missing(Y_test, Y_test_recon.T)
    nll_test = nll(Y_test_pred, Y_test, Y_std)
    
    print(f'Test Reconstruction error {model_name} = ' + str(mse_test))
    print(f'Neg. test log likelihood {model_name} = ' + str(nll))
    
    mse_test = rmse_missing(Y_test, Y_test_recon.T)
    nll_test = nll(Y_test_pred, Y_test, Y_std)

    print(f'Test Reconstruction error {model_name} = ' + str(mse_test))
    print(f'Neg. test log likelihood {model_name} = ' + str(nll))
