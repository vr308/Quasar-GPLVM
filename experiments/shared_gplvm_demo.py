#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for Shared GPLVM with split GPs

TODO: 
    
    Test with MAP inference
    Clean-up experiment scripts / model classes
    
"""
from models.shared_gplvm import SharedGPLVM, predict_joint_latent
from models.latent_variable import PointLatentVariable, MAPLatentVariable
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
from utils.metrics import rmse_missing, nll
from models.likelihood import GaussianLikelihoodWithMissingObs, FixedNoiseGaussianLikelihood
from utils.load_data import load_spectra_labels
from utils.visualisation import plot_spectra_reconstructions, plot_y_label_comparison, spectra_reconstruction_report, plot_partial_spectra_reconstruction_report
from utils.metrics import rmse_lum_bhm_edd, nll_lum_bhm_edd, rmse


## Import class and experiment configuration here

from utils.config import hdu, BASE_SEED, latent_dim, test_size, num_inducing

save_model = True

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setting torch and numpy seed for reproducibility
    
    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    
    # Load joint spectra and label data 
    
<<<<<<< HEAD
    X, Y, means_X, std_X, means_Y, std_Y, snr, wave = load_spectra_labels(hdu)
<<<<<<< HEAD

    data = np.hstack((X,Y))[0:15000]
    
    XY_train, XY_test, train_idx, test_idx = train_test_split(data, np.arange(len(data)), test_size=test_size, random_state=BASE_SEED)
=======
=======
    X, Y, means_X, std_X, means_Y, std_Y, X_sigma, Y_sigma, snr = load_spectra_labels(hdu)
>>>>>>> c0280532bebcb054486a8c1b0d6e1e595e1373f5
    
    data = np.hstack((X,Y))[0:15000]
    
<<<<<<< HEAD
    XY_train, XY_test = train_test_split(data, test_size=test_size, random_state=BASE_SEED)
=======
    XY_train, XY_test, train_idx, test_idx = train_test_split(data, np.arange(len(Y)), test_size=test_size, random_state=SEED)
    snr_test = snr[test_idx]
>>>>>>> c0280532bebcb054486a8c1b0d6e1e595e1373f5
>>>>>>> 9780a4ba4a3b2abd9eef1347850076dbb0bb6666
    
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
    
    shared_model = SharedGPLVM(N, spectra_dim, label_dim, latent_dim, num_inducing, latent_config='point').to(device)
    
    # Missing data Likelihood
    
    likelihood_spectra = GaussianLikelihoodWithMissingObs(batch_shape = shared_model.model_spectra.batch_shape).to(device)
    likelihood_labels = GaussianLikelihoodWithMissingObs(batch_shape = shared_model.model_labels.batch_shape).to(device)

    # Fixed Noise Gaussian Likelihood 
    
    likelihood_spectra = FixedNoiseGaussianLikelihood(noise=torch.Tensor(X_sigma), learn_additional_noise=False, batch_shape=shared_model.model_spectra.batch_shape)
    likelihood_labels = FixedNoiseGaussianLikelihood(noise=torch.Tensor(Y_sigma), learn_additional_noise=False, batch_shape=shared_model.model_labels.batch_shape)
    
    # Deploy model and likelihoods on cuda
    
    if torch.cuda.is_available():
        
        shared_model = shared_model.cuda()
        likelihood_spectra = likelihood_spectra.cuda()
        llikelihood_labels = likelihood_labels.cuda()

    # Declaring objective to be optimised along with optimiser
    
    mll_spectra = VariationalELBO(likelihood_spectra, shared_model.model_spectra, num_data=len(XY_train)).to(device)
    mll_labels = VariationalELBO(likelihood_labels, shared_model.model_labels, num_data=len(XY_train)).to(device)

    optimizer = torch.optim.Adam([
        dict(params=shared_model.parameters(), lr=0.001),
        dict(params=likelihood_spectra.parameters(), lr=0.001),
        dict(params=likelihood_labels.parameters(), lr=0.001)
    ])
      
    ############## Training loop - optimises the objective wrt kernel hypers, ######
    ################  variational params and inducing inputs using the optimizer provided. ########
    
    loss_list = []
    iterator = trange(5000, leave=True)
    batch_size = 128

    for i in iterator: 
        
        batch_index = shared_model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample_batch = shared_model.Z.Z[batch_index]
        #sample_batch= shared_model.Z.sample(batch_index)  # a full sample returns latent Z across all N
        #sample_batch = sample[batch_index]
        
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
    
    ########### Plot latents ##########
    
    
    shared_model.model_spectra.covar_module.base_kernel.lengthscale
    shared_model.model_labels.covar_module.base_kernel.lengthscale
    
    X_train = XY_train[::,0:-4]
    Y_train = XY_train[:,-4::]
    
    X_train_orig = X_train*std_X + means_X
    Y_train_orig = Y_train*std_Y + means_Y
    
    Z_df = pd.DataFrame(Z_train.cpu().detach())
    
    Z_df['bhm'] = Y_train_orig[:,1].cpu().detach()
    Z_df['lumin'] = Y_train_orig[:,0].cpu().detach()
    Z_df['edd'] = Y_train_orig[:,2].cpu().detach()

    sns.pairplot(Z_df, vars = Z_df.columns[0:10], corner=True, plot_kws={"s": 3}, hue='bhm')

    ####################### Save / Load trained model ##########################################
    
    # if os.path.isfile('trained_models/gplvm_1000_shared.pkl'):
    #       with open('trained_models/gplvm_1000_shared.pkl', 'rb') as file:
    #           model_sd, likl_sd = pkl.load(file)
    #           model.load_state_dict(model_sd)
    #           likelihood_sp.load_state_dict(likl_sd)
    #           likelihood_lb.load_state_dict(likl_sd)

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
    ids = torch.randperm(300)

    # X_train = XY_train[::,0:-4]
    # Y_train = XY_train[:,-4::]
    
    #X_train = XY_train[::,0:-4][200:500]
    Y_train = XY_train[:,-4::][ids]
    
    #X_train_orig = X_train*std_X + means_X
    with open('trained_models/gplvm_11000_shared.pkl', 'wb') as file:
         pkl.dump((shared_model.cpu().state_dict(), likelihood_spectra.cpu().state_dict(), likelihood_labels.cpu()), file)
        
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
    
    X_train = XY_train[::,0:-3][0:500]
    Y_train = XY_train[:,-3::][0:500]
    
    X_train_orig = X_train*std_X + means_X
    Y_train_orig = Y_train*std_Y + means_Y
  
    #a, X_train_recon, X_train_pred_covar = shared_model.model_spectra.reconstruct_y(torch.Tensor(shared_model.Z.Z)[0:200], X_train[0:200], ae=False)
    a, Y_train_recon, Y_train_pred_covar = shared_model.model_labels.reconstruct_y(torch.Tensor(Z_train)[ids], Y_train, ae=False)
        
    #X_train_recon =  X_train_recon.T
    Y_train_recon =  Y_train_recon.T

    # X_train_orig = X_train*std_X + means_X
    # Y_train_orig = Y_train*std_Y + means_Y
    
    # X_train_recon, X_train_pred_covar = shared_model.model_spectra.reconstruct_y(torch.Tensor(shared_model.Z.Z)[0:200], X_train[0:200], ae=False)
    # Y_train_recon, Y_train_pred_covar = shared_model.model_labels.reconstruct_y(torch.Tensor(shared_model.Z.Z)[0:200], Y_train[0:200], ae=False)
        
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
    
    # X_train_pred_sigma = np.sqrt(vars_X_noisy)*std_X 
    # Y_train_pred_sigma = diags_Y*std_Y

    ################ Testing ###################################################################################
    
    # Initialise test model at training params
   
    TEST = True

    if TEST:
        
        X_test = XY_test[::,0:-4]
        Y_test = XY_test[:,-4::]
        
        X_test_orig = X_test.cpu()*std_X + means_X
        Y_test_orig = Y_test.cpu()*std_Y + means_Y
        
        test_model = shared_model.initialise_model_test(len(Y_test), latent_dim).to(device)

        test_loss, test_model, Z_test = predict_joint_latent(test_model, X_test, None, likelihood_spectra, likelihood_labels, lr=0.001, prior_z = None, steps = 2000)

        X_test_recon, X_test_pred_covar = test_model.model_spectra.reconstruct_y(Z_test.Z, X_test, ae=False)
        a, Y_test_recon, Y_test_pred_covar = test_model.model_labels.reconstruct_y(Z_test.Z, Y_test, ae=False)
        
        #X_test_recon =  X_test_recon.T.detach().numpy()
        #Y_test_recon =  Y_test_recon.T.detach().numpy()

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
        
        X_test_recon_orig = X_test_recon.T*std_X + means_X
        Y_test_recon_orig = Y_test_recon.T*std_Y + means_Y
        
        vars_X_noiseless = np.array([(m.diag()).cpu().detach().numpy() for m in X_test_pred_covar]).T ## extracting diagonals per dimensions
        vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().cpu().detach().numpy() for m in vars_X_noiseless])
    
        diags_Y_list = [m.diag().sqrt() for m in Y_test_pred_covar]
        diags_Y = torch.cat(diags_Y_list).reshape(len(Y_test),4)
        
        X_test_pred_sigma = np.sqrt(vars_X_noisy)*std_X.cpu().numpy()
        Y_test_pred_sigma = diags_Y*std_Y
        
        X_test_recon_orig = X_test_recon_orig.cpu().detach().numpy()
        X_test_orig = X_test_orig.cpu().detach().numpy()

        plot_spectra_reconstructions(wave, X_test_recon_orig, X_test_orig, X_test_pred_sigma, obj_id=13)
        plot_spectra_reconstructions(wave, X_test_recon_orig, X_test_orig, X_test_pred_sigma, obj_id=78)
        plot_spectra_reconstructions(wave, X_test_recon_orig, X_test_orig, X_test_pred_sigma, obj_id=93)
        
        spectra_reconstruction_report(wave, X_test_recon_orig, X_test_orig, X_test_pred_sigma, Y_test_orig[:,1])

        #plot_spectra_samples(X_test_recon_orig, X_test_orig, X_test)
        
    ###### Simulate spectra for varying properties #####################################
    
    Y_synthetic_lumin, Y_synthetic_bhm, Y_synthetic_edd = load_synthetic_labels_no_redshift(Y_test, Y_test_orig, means_Y, std_Y, device)
    
    ## bhm
    
    test_model_bhm = shared_model.initialise_model_test(len(Y_synthetic_bhm), latent_dim).to(device)
    test_loss_bhm, test_model_bhm, Z_test_bhm = predict_joint_latent(test_model_bhm, None, Y_synthetic_bhm, likelihood_spectra, likelihood_labels, lr=0.001, prior_z = None, steps = 5000)
    
    X_test_recon_bhm, X_test_pred_covar_bhm = test_model_bhm.model_spectra.reconstruct_y(Z_test_bhm.Z[0:100], X_test[0:100], ae=False)
    X_test_recon_orig_bhm = X_test_recon_bhm.T*std_X + means_X
    Y_test_recon_orig_bhm = Y_synthetic_bhm*std_Y + means_Y
    
    vars_X_noiseless = np.array([(m.diag()).cpu().detach().numpy() for m in X_test_pred_covar_bhm]).T ## extracting diagonals per dimensions
    vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().cpu().detach().numpy() for m in vars_X_noiseless])
 
    
    ## Lumin
    
    test_model_lumin = shared_model.initialise_model_test(len(Y_synthetic_lumin), latent_dim).to(device)
    test_loss_lumin, test_model_lumin, Z_test_lumin = predict_joint_latent(test_model_lumin, None, Y_synthetic_lumin, likelihood_spectra, likelihood_labels, lr=0.0001, prior_z = None, steps = 200)
    
    X_test_recon_lumin, X_test_pred_covar_lumin = test_model_lumin.model_spectra.reconstruct_y(Z_test_lumin.Z[0:100], X_test[0:100], ae=False)
    X_test_recon_orig_lumin = X_test_recon_lumin.T*std_X + means_X
    Y_test_recon_orig_lumin = Y_synthetic_lumin*std_Y + means_Y
    
    vars_X_noiseless = np.array([(m.diag()).cpu().detach().numpy() for m in X_test_pred_covar_lumin]).T ## extracting diagonals per dimensions
    vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().cpu().detach().numpy() for m in vars_X_noiseless])
 
    
    ## Edd

    test_model_edd = shared_model.initialise_model_test(len(Y_synthetic_edd), latent_dim).to(device)
    test_loss_edd, test_model_edd, Z_test_edd = predict_joint_latent(test_model_edd, None, Y_synthetic_edd, likelihood_spectra, likelihood_labels, lr=0.01, prior_z = None, steps = 5000)

    X_test_recon_edd, X_test_pred_covar_edd = test_model_edd.model_spectra.reconstruct_y(Z_test_edd.Z[0:100], X_test[0:100], ae=False)
    X_test_recon_orig_edd = X_test_recon_edd.T*std_X + means_X
    Y_test_recon_orig_edd = Y_synthetic_edd*std_Y + means_Y
    
    # ## red
    
    # test_model_red = shared_model.initialise_model_test(len(Y_synthetic_red), latent_dim).to(device)
    # test_loss_red, test_model_red, Z_test_red = predict_joint_latent(test_model_red, None, Y_synthetic_red, likelihood_spectra, likelihood_labels, lr=0.01, prior_z = None, steps = 5000)

    # X_test_recon_red, X_test_pred_covar_red = test_model_red.model_spectra.reconstruct_y(Z_test_red.Z[0:100], X_test[0:100], ae=False)
    # X_test_recon_orig_red = X_test_recon_red.T*std_X + means_X
    # Y_test_recon_orig_red = Y_synthetic_red*std_Y + means_Y
     
    # np.savetxt(fname='X_recon_synthetic_red.txt', X=X_test_recon_orig_red.cpu().detach(), delimiter=',')
    # np.savetxt(fname='Y_orig_synthetic_red.txt', X=Y_test_recon_orig_red.cpu().detach(), delimiter=',')

    # vars_X_noiseless = np.array([(m.diag()).cpu().detach().numpy() for m in X_test_pred_covar_red]).T ## extracting diagonals per dimensions
    # vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().cpu().detach().numpy() for m in vars_X_noiseless])
 
    # np.savetxt(fname='X_test_pred_covar_red.txt', X=vars_X_noisy, delimiter=',')

    # X_test_recon_orig_ = np.loadtxt(fname = 'X_recon_synthetic_red.txt', delimiter=',')
    # Y_test_recon_orig_ = np.loadtxt(fname = 'Y_orig_synthetic_red.txt', delimiter=',')
    
    
    #### plotting
    
    import matplotlib
    import matplotlib.pylab as plt
    
    parameters = np.array(Y_test_recon_orig_lumin[:,0].cpu().detach())
    
    norm = matplotlib.colors.Normalize(
    vmin=np.min(parameters),
    vmax=np.max(parameters))

    # choose a colormap
    c_m = matplotlib.cm.spring
    
    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])
    
    for i in np.arange(100):
        
        plt.plot(wave, X_test_recon_orig_lumin[i].cpu().detach(), color=s_m.to_rgba(parameters[i]))
    
    plt.errorbar(wave, X_test_recon_orig_lumin.cpu().detach().mean(dim=0), yerr=np.sqrt(vars_X_noisy.mean(axis=0))*2, alpha=0.3)
    
    # having plotted the 11 curves we plot the colorbar, using again our
    # ScalarMappable
    cbar = plt.colorbar(s_m)
    cbar.set_label('Luminosity')
    plt.ylabel('Normalised Flux')
    plt.xlabel('Restframe wavelength')
    plt.title('Spectra  generated for simulated Luminosity')


    ################## Partial observation region (Spectra reconstruction) ####################################################
    
    idx = 125
    
    test_point_X = torch.full(torch.Size([4,586]), torch.nan).to(device)
    
    obs_region_1 = np.arange(0,280)    ## first half of spectra 
    obs_region_2 = np.arange(280,586)  ## last half of spectra 
    obs_region_3 = np.arange(140,350)  ## obs middle part 
    obs_region_4 = np.arange(310,400)  ## obs a small internal section 
    
    test_point_X[0,obs_region_1] = X_test[idx][obs_region_1]
    test_point_X[1,obs_region_2] = X_test[idx][obs_region_2]
    test_point_X[2,obs_region_3] = X_test[idx][obs_region_3]
    test_point_X[3,obs_region_4] = X_test[idx][obs_region_4]
    
    #Z_latent = torch.Tensor(test_model.Z.Z)[idx].repeat(4).reshape(4,10)
    test_model = shared_model.initialise_model_test(4, latent_dim).to(device)
    test_point_Y = Y_test[idx].repeat(4).reshape(4,3).to(device)

    test_loss, test_model, Z_partial = predict_joint_latent(test_model, test_point_X, test_point_Y, likelihood_spectra, likelihood_labels, lr=0.001, prior_z = None, steps = 200, batch_size=4)
    X_partial_recon, X_partial_pred_covar = shared_model.model_spectra.reconstruct_y(Z_partial.Z, test_point_X, ae=False)

    X_partial_recon_orig = X_partial_recon.T*std_X + means_X
    
    vars_X_noiseless = np.array([m.diag().cpu().detach().numpy() for m in X_partial_pred_covar]).T ## extracting diagonals per dimensions
    vars_X_noisy = np.array([m + likelihood_spectra.noise_covar.noise.flatten().cpu().detach().numpy() for m in vars_X_noiseless])
    std_X_cpu = std_X.cpu().detach().numpy()
    X_partial_pred_sigma = np.sqrt(vars_X_noisy)*std_X_cpu
    
    bhm_value = Y_test_orig[:,1][idx]
    
    plot_partial_spectra_reconstruction_report(wave, X_partial_recon_orig.cpu().detach(), X_test_orig[idx].cpu().detach(), X_partial_pred_sigma, bhm_value)
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
    
    ids = np.arange(200)
    # col_range = np.arange(68,1000)
    
    plot_spectra_reconstructions(X_train_recon_orig, X_train_orig, X_train_pred_sigma[ids], obj_id=24)
    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma[ids], obj_id=13)
    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma[ids], obj_id=78)
    plot_spectra_reconstructions(X_test_recon_orig, X_test_orig, X_test_pred_sigma[ids], obj_id=151)
    
    #### Plotting label reconstruction 
    
    xlabel_lumin = r'Measured $\log_{10}(L_{\mathrm{lbol}}/\mathrm{ergs}^{-1})$'
    ylabel_lumin = r'Predicted $\log_{10}(L_{\mathrm{lbol}}/\mathrm{ergs}^{-1})$'
    lumin_title = 'Bolometric luminosity'
     
    xlabel_bhm = r'Measured $\log_{10}(M_{\bullet}/M_{\odot})$'
    ylabel_bhm = r'Predicted $\log_{10}(M_{\bullet}/M_{\odot})$'
    bhm_title = 'Black hole mass'
    
    xlabel_edd = r'Measured $\log_{10}\lambda_{\mathrm{Edd}}$'
    ylabel_edd = r'Predicted $\log_{10}\lambda_{\mathrm{Edd}}$'
    edd_title = 'Eddington Ratio'
     
  
    plot_y_label_comparison(Y_test_recon_orig, Y_test_orig, Y_test_pred_sigma,  \
                            col_id = 0, title=lumin_title, 
                            colors=Y_test_orig[:,0].cpu().detach(), \
                            clabel = None,
                            xlabel = xlabel_lumin, \
                            ylabel = ylabel_lumin, \
                            cmap= 'spring')
        
    plot_y_label_comparison(Y_test_recon_orig, Y_test_orig, Y_test_pred_sigma, \
                            col_id = 1, title=bhm_title, colors=Y_test_orig[:,1].cpu().detach(), \
                            clabel = None,
                            xlabel = xlabel_bhm, \
                            ylabel = ylabel_bhm, \
                            cmap= 'jet')
        
    plot_y_label_comparison(Y_test_recon_orig, Y_test_orig, Y_test_pred_sigma, \
                            col_id = 2, title=edd_title,  colors=Y_test_orig[:,2].cpu().detach(), 
                            clabel = None,
                            xlabel = xlabel_edd, \
                            ylabel = ylabel_edd, \
                            cmap= 'summer')
        
    c = snr[ids]
        
    plot_y_label_comparison(Y_train_recon_orig, Y_train_orig, Y_train_pred_sigma, \
                            col_id = 1, title=lumin_title,  colors=c, 
                            clabel=None,
                            xlabel = xlabel_lumin, \
                            ylabel = ylabel_lumin, \
                            cmap= 'jet')
        
    plot_y_label_comparison(Y_train_recon_orig, Y_train_orig, Y_train_pred_sigma, \
                            col_id = 2, title=bhm_title,  colors=c, 
                            clabel=None,
                            xlabel = xlabel_bhm, \
                            ylabel = ylabel_bhm, \
                            cmap= 'jet')
         
    plot_y_label_comparison(Y_train_recon_orig, Y_train_orig, Y_train_pred_sigma, \
                            col_id = 3, title=edd_title,  colors=c, 
                            clabel=None,
                            xlabel = xlabel_edd, \
                            ylabel = ylabel_edd, \
                            cmap= 'jet')    
 
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
     

