#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for Shared GPLVM with split GPs
    
"""
from models.shared_gplvm import SharedGPLVM, predict_joint_latent
from models.latent_variable import PointLatentVariable, MAPLatentVariable
from sklearn.model_selection import train_test_split
import torch
import os 
import pickle as pkl
import numpy as np
import gc
from tqdm import trange
from prettytable import PrettyTable
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from models.likelihood import GaussianLikelihoodWithMissingObs, FixedNoiseGaussianLikelihood
from utils.load_data import load_spectra_labels
from utils.visualisation import plot_spectra_reconstructions, plot_y_label_comparison, spectra_reconstruction_report, plot_partial_spectra_reconstruction_report
from utils.metrics import rmse_lum_bhm_edd, nll_lum_bhm_edd, rmse

## Import class and experiment configuration here

from utils.config import *

save_model = True

if __name__ == '__main__':
    
    # Setting torch and numpy seed for reproducibility
    
    SEED = BASE_SEED
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load joint spectra and label data 
    
    X, Y, means_X, std_X, means_Y, std_Y, X_sigma, Y_sigma, snr = load_spectra_labels(hdu)
    
    data = np.hstack((X,Y))
    
    XY_train, XY_test, train_idx, test_idx = train_test_split(data, np.arange(len(Y)), test_size=test_size, random_state=SEED)
    snr_test = snr[test_idx]
    
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
    
    # Missing data Likelihood
    
    likelihood_spectra = GaussianLikelihoodWithMissingObs(batch_shape = shared_model.model_spectra.batch_shape)
    likelihood_labels = GaussianLikelihoodWithMissingObs(batch_shape = shared_model.model_labels.batch_shape)
    
    # Fixed Noise Gaussian Likelihood 
    
    likelihood_spectra = FixedNoiseGaussianLikelihood(noise=torch.Tensor(X_sigma), learn_additional_noise=False, batch_shape=shared_model.model_spectra.batch_shape)
    likelihood_labels = FixedNoiseGaussianLikelihood(noise=torch.Tensor(Y_sigma), learn_additional_noise=False, batch_shape=shared_model.model_labels.batch_shape)
    
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
    iterator = trange(10000, leave=True)
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
    
    model_name = 'shared_gplvm_' + 'n_' + size + '_latent_dim_' + str(latent_dim) + '_n_inducing_' + str(n_inducing) + '_random_seed_' + str(BASE_SEED)
    
    if save_model:
   
        with open('trained_models/' + model_name + '.pkl', 'wb') as file:
            pkl.dump((shared_model.state_dict(), likelihood_spectra.state_dict(),       likelihood_labels.state_dict()), file)
            
            
    ## Loading pre-saved model
    
    # if os.path.isfile('trained_models/' + model_name + '.pkl'):
    #       with open('trained_models/'+ model_name + '.pkl', 'rb') as file:
    #           model_sd, likl_spectra_sd, likl_lb_sd = pkl.load(file)
    #           shared_model.load_state_dict(model_sd)
    #           likelihood_spectra.load_state_dict(likl_spectra_sd)
    #           likelihood_labels.load_state_dict(likl_lb_sd)

    ####################### Split Reconstruction Framework (Training and Test) ##############
    
    # ids = np.arange(200)
    
    # X_train = XY_train[::,0:-4]
    # Y_train = XY_train[:,-4::]
    
    # X_train_orig = X_train*std_X + means_X
    # Y_train_orig = Y_train*std_Y + means_Y
    
    # X_train_recon, X_train_pred_covar = shared_model.model_spectra.reconstruct_y(torch.Tensor(shared_model.Z.Z)[0:200], X_train[0:200], ae=False)
    # Y_train_recon, Y_train_pred_covar = shared_model.model_labels.reconstruct_y(torch.Tensor(shared_model.Z.Z)[0:200], Y_train[0:200], ae=False)
        
    # X_train_recon =  X_train_recon.T.detach().numpy()
    # Y_train_recon =  Y_train_recon.T.detach().numpy()
    
    # X_train_recon_orig = X_train_recon*std_X + means_X
    # Y_train_recon_orig = Y_train_recon*std_Y + means_Y
    
    # vars_X_noiseless = np.array([(m.diag()).detach().numpy() for m in X_train_pred_covar]).T ## extracting diagonals per dimensions
    # vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().detach().numpy() for m in vars_X_noiseless])
    
    # diags_Y = np.array([m.diag().sqrt().detach().numpy() for m in Y_train_pred_covar]).T #
    
    # X_train_pred_sigma = np.sqrt(vars_X_noisy)*std_X 
    # Y_train_pred_sigma = diags_Y*std_Y
    
    ################ Testing #######################################################################
    
    # Initialise test model at training params
   
    TEST = True

    if TEST:
        
        X_test = XY_test[::,0:-4]
        Y_test = XY_test[:,-4::]
        
        X_test_orig = X_test.cpu()*std_X + means_X
        Y_test_orig = Y_test.cpu()*std_Y + means_Y
        
        test_model = shared_model.initialise_model_test(len(Y_test), latent_dim)
        
        if torch.cuda.is_available():
            test_model = test_model.cuda()

        test_loss, test_model, Z_test = predict_joint_latent(test_model, X_test, Y_test, likelihood_spectra, likelihood_labels, lr=0.005, prior_z = None, steps = 10000)

        #X_test_pred, X_test_recon, X_test_pred_covar = test_model.model_spectra.reconstruct_y(Z_test.Z, X_test, ae=False)
        #Y_test_pred, Y_test_recon, Y_test_pred_covar = test_model.model_labels.reconstruct_y(Z_test.Z, Y_test, ae=False)
        
        X_test_pred = likelihood_spectra(test_model.model_spectra(Z_test.Z))
        Y_test_pred = likelihood_labels(test_model.model_labels(Z_test.Z))
        
        torch.cuda.empty_cache()
        gc.collect()
        
        X_test_recon, X_test_pred_covar = X_test_pred.loc , X_test_pred.covariance_matrix
        Y_test_recon, Y_test_pred_covar = Y_test_pred.loc , Y_test_pred.covariance_matrix
        
        X_test_recon = X_test_pred.loc 

        X_test_recon =  X_test_recon.T.cpu().detach().numpy()
        Y_test_recon =  Y_test_recon.T.cpu().detach().numpy()
        
        X_test_recon_orig = X_test_recon*std_X + means_X
        Y_test_recon_orig = Y_test_recon*std_Y + means_Y
        
        spec_noise = likelihood_spectra.noise_covar.noise.sqrt().cpu().detach().flatten()
        sigma_X = np.array([m.diag().sqrt().cpu().detach().numpy() for m in X_test_pred_covar]).T #
        sigma_Y = np.array([m.diag().sqrt().cpu().detach().numpy() for m in Y_test_pred_covar]).T #

        X_test_pred_sigma = (sigma_X + np.tile(spec_noise, reps=20).reshape(20,657))*std_X
        Y_test_pred_sigma = sigma_Y*std_Y
        
        torch.cuda.empty_cache()
        gc.collect()
        
    ############### Compute and save the metrics for X and Y ##########################
                    
    
    # X_rmse_test = rmse(X_test_orig.cpu(), X_test_recon_orig)
    # Y_rmse_test = rmse_lum_bhm_edd(Y_test_orig.cpu(), Y_test_recon_orig)
    # X_nll_test = nll(X_test_pred, X_test, std_X)
    # Y_nll_test = nll_lum_bhm_edd(Y_test_pred_filter, Y_test_filter, std_Y)

    # print('X, Y -> Test Reconstruction error  = ' + str(X_rmse_test) +  '   ' + str(Y_rmse_test))
    # #print('X, Y -> Neg. test log likelihood  = ' + str(X_nll_test) +   '   ' + str(Y_nll_test))
    
    # metrics = {
    #         'model_name': model_name,
    #         'X_test_rmse': X_rmse_test.item(),
    #         #'X_test_nlpd': X_nll_test.item(),
    #         'Y_test_rmse': str(np.array(Y_rmse_test)),
    #         'Y_test_nlpd': str(np.array(Y_nll_test)),
    #         'Y_rmse_all_in': np.mean(np.array(Y_rmse_test)).item(),
    #         'Y_nll_all_in': np.mean(np.array(Y_nll_test)).item()
    #          }
    
    # results_filename = f"results/{model_name}__.json"
    # with open(results_filename, "w") as fp:
    #        json.dump(metrics, fp, indent=4)
     
 #    ################## Partial observation region (Spectra reconstruction) ####################################################
    
 #    idx = 135
    
 #    test_point = torch.full(torch.Size([4,590]), np.nan).cuda()
    
 #    obs_region_1 = np.arange(0,280)    ## first half of spectra 
 #    obs_region_2 = np.arange(280,590)  ## last half of spectra 
 #    obs_region_3 = np.arange(140,350)  ## obs middle part 
 #    obs_region_4 = np.arange(310,400)  ## obs a small internal section 
    
 #    test_point[0,obs_region_1] = X_test[idx][obs_region_1]
 #    test_point[1,obs_region_2] = X_test[idx][obs_region_2]
 #    test_point[2,obs_region_3] = X_test[idx][obs_region_3]
 #    test_point[3,obs_region_4] = X_test[idx][obs_region_4]
    
 #    #Z_latent = torch.Tensor(test_model.Z.Z)[idx].repeat(4).reshape(4,10)
 #    test_model = shared_model.initialise_model_test(4, latent_dim)
    
 #    if torch.cuda.is_available():
 #        test_model = test_model.cuda()
        
 #    test_point_Y = Y_test[idx].repeat(4).reshape(4,4)

 #    test_loss, test_model, Z_partial = predict_joint_latent(test_model, test_point, test_point_Y, likelihood_spectra, likelihood_labels, lr=0.003, prior_z = None, steps = 5000, batch_size=4)
    
 #    X_partial_pred, X_partial_recon, X_partial_pred_covar = shared_model.model_spectra.reconstruct_y(Z_partial.Z, test_point, ae=False)

 #    X_partial_recon_orig = X_partial_recon.T.cpu().detach().numpy()*std_X + means_X
    
 #    spec_noise = likelihood_spectra.noise_covar.noise.sqrt().cpu().detach().flatten()
 #    #sigma_X = np.array([m.diag().sqrt().cpu().detach().numpy() for m in X_partial_pred_covar]).T #
    
 #    vars_X_noiseless = np.array([(m.diag()).cpu().detach().numpy() for m in X_partial_pred_covar]).T ## extracting diagonals per dimensions
 #    vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().cpu().detach().numpy() for m in vars_X_noiseless])
 #    X_partial_pred_sigma = np.sqrt(vars_X_noisy)*std_X 
    
 #    plot_partial_spectra_reconstruction_report(X_partial_recon_orig, X_test_orig[idx], X_partial_pred_sigma)
    
 #    # ####################### Visualisation: Reconstructing spectra and labels ################################
    
 #    # # plt.plot(np.isnan(XY_train).sum(axis=0)) ## check the presence of the data 
    
 #    ids = np.arange(200)
 #    # # col_range = np.arange(68,1000)
    
 #    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma, obj_id=24)
 #    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma, obj_id=13)
 #    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma, obj_id=7)
 #    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma, obj_id=151)
    
 #    spectra_reconstruction_report(X_test_recon_orig, X_test_orig[0:20], X_test_pred_sigma)
  
 #    # plot_y_label_comparison(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], Y_test_sigma[ids],  snr_test[ids], col_id = 1, title='Luminosity')
 #    # plot_y_label_comparison(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], Y_test_sigma[ids], snr_test[ids], col_id = 2, title='Black hole mass')
 #    # plot_y_label_comparison(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], snr_test[ids], col_id = 3, title='Eddington luminosity')
    
 #    plot_y_label_report(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], Y_test_sigma[ids], snr_test[ids])
        
 #    # # # # ################################
 #    ## Extra plots
    
 # #plt.scatter(Y_test_recon_orig[:,2][ids], Y_test_recon_orig[:,1][ids], c=Y_test_orig[:,0][ids].cpu().detach().numpy())
 
 #     plt.figure()
 #     plt.plot(loss_100_2, label='Q=2')
 #     plt.plot(loss_100_5, label='Q=5',alpha=0.8)
 #     plt.plot(loss_100_10, label='Q=10', alpha=0.7)
 #     plt.plot(loss_100_15, label='Q=15', alpha=0.7)
 #     plt.xticks(fontsize='small')
 #     plt.yticks(fontsize='small')
 #     plt.legend(fontsize='small')
 #     plt.title('Joint ELBOs for varying latent dim. Q', fontsize='small')
     

