#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for Shared GPLVM with split GPs

TODO: 
    
    Test with MAP inference
    Test with Gaussian latent vars
    Implement GPU deployment

"""

from models.gplvm import BayesianGPLVM
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
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel, RQKernel
from gpytorch.distributions import MultivariateNormal
from models.likelihood import GaussianLikelihoodWithMissingObs
from utils.load_data import load_joint_spectra_labels_small, load_spectra_labels_large
from utils.visualisation import plot_spectra_reconstructions, plot_y_label_comparison, plot_partial_spectra_reconstruction_report

class QuasarDemoModel(BayesianGPLVM):
     def __init__(self, Z, n, data_dim, latent_dim, n_inducing, inducing_inputs):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
    
        # Sparse Variational Formulation
        
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, inducing_inputs, q_u, learn_inducing_locations=False)
    
        super(QuasarDemoModel, self).__init__(Z, q_f)
        
        # Kernel 
        self.shared_inducing_inputs = inducing_inputs

        self.mean_module = ConstantMean(batch_shape=self.batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
                   
     def forward(self, Z):
         
        self.variational_strategy.inducing_points = self.shared_inducing_inputs
        
        mean_z = self.mean_module(Z)
        covar_z = self.covar_module(Z)
        dist = MultivariateNormal(mean_z, covar_z)
        return dist

class SharedGPLVM(gpytorch.Module):
    
     def __init__(self, n, spectra_dim, label_dim, latent_dim, n_inducing, latent_config='point', kernel_config='standard'):
         
        super(SharedGPLVM, self).__init__()

        self.n = n
        self.latent_dim = latent_dim
        
        # Define prior for Z

        Z_prior_mean = torch.zeros(self.n, latent_dim)  # shape: N x Q
        Z_init = torch.nn.Parameter(torch.zeros(n, latent_dim))
          
        # LatentVariable configuration
        
        if latent_config == 'map':
            
            prior_z = NormalPrior(Z_prior_mean, torch.ones_like(Z_prior_mean))
            Z = MAPLatentVariable(n, latent_dim, Z_init, prior_z)
        
        elif latent_config == 'point':
        
            Z = PointLatentVariable(Z_init)
            
        self.Z = Z
        self.inducing_inputs = torch.nn.Parameter(torch.randn(n_inducing, latent_dim))
        
        self.model_spectra = QuasarDemoModel(self.Z, n, spectra_dim, latent_dim, n_inducing, self.inducing_inputs)
        self.model_labels = QuasarDemoModel(self.Z, n, label_dim, latent_dim, n_inducing, self.inducing_inputs)
        

     def _get_batch_idx(self, batch_size):
            
         valid_indices = np.arange(self.n)
         batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
         return np.sort(batch_indices)
     
     def get_trainable_param_names(self):
        
        ''' Prints a list of parameters (model + variational) which will be 
        learnt in the process of optimising the objective '''
        
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        
     def initialise_model_test(self, n_test):
        
        ## Initialise test models for both the internal models: spectra and labels
        self.n = n_test
        
        self.model_spectra.n = n_test
        self.model_labels.n = n_test
        
        Z_init_test = torch.nn.Parameter(torch.randn(self.n, latent_dim))

        self.model_spectra.Z.reset(Z_init_test)
        self.model_labels.Z.reset(Z_init_test)
        
        return self
    
def predict_joint_latent(test_model, X_test, Y_test, likelihood_spectra, likelihood_labels, lr=0.001, prior_z = None, steps = 2000):
   
    # Initialise a new test optimizer with just the test model latents
    
    test_optimizer = torch.optim.Adam(test_model.Z.parameters(), lr=lr)
    
    mll_spectra = VariationalELBO(likelihood_spectra, test_model.model_spectra, num_data=test_model.n)
    mll_labels = VariationalELBO(likelihood_labels, test_model.model_labels, num_data=test_model.n)
    
    print('---------------Learning variational parameters for test ------------------')
    
    for name, param in test_model.Z.named_parameters():
        print(name)
        
    loss_list = []
    iterator = trange(steps, leave=True)
    batch_size = 100
    
    for i in iterator: 
        
           joint_loss = 0.0
           batch_index = test_model._get_batch_idx(batch_size)
           test_optimizer.zero_grad()
           sample = test_model.Z.Z  # a full sample returns latent Z across all N
           sample_batch = sample[batch_index]
           
           ### Getting the output of the two groups of GPs
           
           output_spectra = test_model.model_spectra(sample_batch)
           output_labels = test_model.model_labels(sample_batch)
           
           ### Adding together the ELBO losses 
           
           if X_test is not None:
               joint_loss += -mll_spectra(output_spectra, X_test[batch_index].T).sum() 
               
           if Y_test is not None:
               joint_loss += -mll_labels(output_labels, Y_test[batch_index].T).sum()
               
           loss_list.append(joint_loss.item())
           iterator.set_description('Loss: ' + str(float(np.round(joint_loss.item(),2))) + ", iter no: " + str(i))
           joint_loss.backward()
           test_optimizer.step()
        
    return loss_list, test_model, test_model.Z

if __name__ == '__main__':
    
    # Setting seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)

    # Load joint spectra and label data 
    
    #data = load_joint_spectra_labels_small()
    X, Y, means_X, std_X, means_Y, std_Y, snr = load_spectra_labels_large()
    
    data = np.hstack((X,Y))
    
    XY_train, XY_test = train_test_split(data, test_size=200, random_state=SEED)
    
    XY_train = torch.Tensor(XY_train)
    XY_test = torch.Tensor(XY_test)
    
    # Experiment config
    
    N = len(XY_train)
    data_dim = XY_train.shape[1]
    latent_dim = 10
    n_inducing = 90
    
    spectra_dim = X.shape[1]
    label_dim = Y.shape[1]
      
    # Shared Model
    
    shared_model = SharedGPLVM(N, spectra_dim, label_dim, latent_dim, n_inducing)
    
    # Likelihood
    
    likelihood_spectra = GaussianLikelihoodWithMissingObs(batch_shape = shared_model.model_spectra.batch_shape)
    likelihood_labels = GaussianLikelihoodWithMissingObs(batch_shape = shared_model.model_labels.batch_shape)

    # Declaring objective to be optimised along with optimiser
    
    mll_spectra = VariationalELBO(likelihood_spectra, shared_model.model_spectra, num_data=len(XY_train))
    mll_labels = VariationalELBO(likelihood_labels, shared_model.model_labels, num_data=len(XY_train))

    optimizer = torch.optim.Adam([
        dict(params=shared_model.parameters(), lr=0.005),
        dict(params=likelihood_spectra.parameters(), lr=0.005),
        dict(params=likelihood_labels.parameters(), lr=0.005)
    ])
    
    shared_model.get_trainable_param_names()
    
    ############## Training loop - optimises the objective wrt kernel hypers, ######################################
    ################  variational params and inducing inputs using the optimizer provided. ###########################
    
    loss_list = []
    iterator = trange(5000, leave=True)
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
        
    ####################### Save trained model ##########################################
    
    # if os.path.isfile('trained_models/gplvm_1000_shared.pkl'):
    #       with open('trained_models/gplvm_1000_shared.pkl', 'rb') as file:
    #           model_sd, likl_sd = pkl.load(file)
    #           model.load_state_dict(model_sd)
    #           likelihood_sp.load_state_dict(likl_sd)
    #           likelihood_lb.load_state_dict(likl_sd)


    # with open('trained_models/gplvm_1000_shared.pkl', 'wb') as file:
    #     pkl.dump((shared_model.cpu().state_dict(), likelihood_spectra.cpu().state_dict(), likelihood_labels.cpu()), file)
        
    ####################### Split Reconstruction Framework (Training and Test) ################################################
    
    
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

        # with torch.no_grad():
                
        #         pca = False
        #         prior_x_test = None
                
        #         shared_model.model_spectra.eval()
        #         shared_model.model_labels.eval()
        #         likelihood_spectra.eval()
        #         likelihood_labels.eval()
                
        #         losses_test,  Z_test = shared_model.model_labels.predict_joint_latent(Y_train, Y_test, optimizer.defaults['lr'], likelihood_labels, SEED, prior_x=prior_x_test, ae=False, model_name='pca', pca=pca)
        #         losses_test,  Z_test = shared_model.model_labels.predict_latent(Y_train, Y_test, optimizer.defaults['lr'], likelihood_labels, SEED, prior_x=prior_x_test, ae=False, model_name='pca', pca=pca)

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
    
    Z_latent = torch.Tensor(test_model.Z.Z)[idx].repeat(4).reshape(4,10)

    X_partial_recon, X_partial_pred_covar = shared_model.model_spectra.reconstruct_y(Z_latent, test_point, ae=False)

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
  
    plot_y_label_comparison(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], snr[ids], col_id = 0, title='Redshift z')
    plot_y_label_comparison(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], snr[ids], col_id = 1, title='Luminosity')
    plot_y_label_comparison(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], snr[ids], col_id = 2, title='Black hole mass')
    plot_y_label_comparison(Y_test_recon_orig[ids], Y_test_orig[ids], Y_test_pred_sigma[ids], snr[ids], col_id = 3, title='Eddington luminosity')
  
    # plot_spectra_reconstructions(X_train_recon_orig, X_train_orig[ids], X_train_pred_sigma[ids], obj_id=13)
    # plot_spectra_reconstructions(X_train_recon_orig, X_train_orig[ids], X_train_pred_sigma[ids], obj_id=78)
    # plot_spectra_reconstructions(X_train_recon_orig, X_train_orig[ids], X_train_pred_sigma[ids], obj_id=151)
    
    # plot_y_label_comparison(Y_train_recon_orig[ids], Y_train_orig[ids], Y_train_pred_sigma[ids], snr[ids], col_id = 0, title='Redshift z')
    # plot_y_label_comparison(Y_train_recon_orig[ids], Y_train_orig[ids], Y_train_pred_sigma[ids], snr[ids], col_id = 1, title='Luminosity')
    # plot_y_label_comparison(Y_train_recon_orig[ids], Y_train_orig[ids], Y_train_pred_sigma[ids], snr[ids], col_id = 2, title='Black hole mass')
    # plot_y_label_comparison(Y_train_recon_orig[ids], Y_train_orig[ids], Y_train_pred_sigma[ids], snr[ids], col_id = 3, title='Eddington luminosity')
  
    # # # ################################
    # # Compute the metrics:
            
    # from utils.metrics import *
    
    # # 1) Reconstruction error - Train & Test
    
    # mse_train = rmse(Y_train, Y_train_recon.T)
    # mse_test = rmse(Y_test, Y_test_recon.T)
    
    # print(f'Train Reconstruction error {model_name} = ' + str(mse_train))
    # print(f'Test Reconstruction error {model_name} = ' + str(mse_test))
