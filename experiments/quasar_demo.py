#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for bGPLVM Gaussian with different inference modes. 

TODO: 
    
    Test with MAP inference
    Test with Gaussian latent vars

"""

from models.gplvm import BayesianGPLVM
from models.latent_variable import PointLatentVariable, MAPLatentVariable
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import os 
import pickle as pkl
import numpy as np
from tqdm import trange
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
from utils.visualisation import plot_spectra_reconstructions, plot_y_label_comparison

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class QuasarMiniDemoModel(BayesianGPLVM):
     def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False, latent_config='point', kernel_config='standard', spectra_dims=7):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations \tilde{Z_{d}} corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(n_inducing, latent_dim)
    
        # Sparse Variational Formulation
        
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
    
        # Define prior for X
        Z_prior_mean = torch.zeros(self.n, latent_dim)  # shape: N x Q
    
        # Initialise X with PCA or 0s.
        
        if pca == True:
             Z_init = _init_pca(XY_train, latent_dim) # Initialise X to PCA 
        else:
             Z_init = torch.nn.Parameter(torch.zeros(n, latent_dim))
          
        # LatentVariable configuration
        if latent_config == 'map':
            
            prior_z = NormalPrior(Z_prior_mean, torch.ones_like(Z_prior_mean))
            Z = MAPLatentVariable(n, latent_dim, Z_init, prior_z)
            #X = VariationalLatentVariable(self.n, data_dim, latent_dim, X_init, prior_x)
        
        elif latent_config == 'point':
        
            Z = PointLatentVariable(Z_init)

        super(QuasarMiniDemoModel, self).__init__(Z, q_f)
        
        # Kernel 
        
        self.mean_module = ConstantMean(batch_shape=self.batch_shape)
        
        if kernel_config == 'standard':
            
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
            
        elif kernel_config == 'partition':
            
            spectra_dims_list = np.arange(latent_dim)[0:spectra_dims]
            label_dims_list = np.arange(latent_dim)[spectra_dims:]
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=len(spectra_dims_list), active_dims = torch.tensor(spectra_dims_list))*RQKernel(ard_num_dims=len(label_dims_list), active_dims=torch.tensor(label_dims_list)))
            
            #self.covar_module = ScaleKernel(RBFKernel(active_dims=torch.tensor(np.arange(spectra_dims_list)))) + ScaleKernel(RQ(active_dims=torch.tensor(label_dims_list)))

     def forward(self, Z):
        mean_z = self.mean_module(Z)
        covar_z = self.covar_module(Z)
        dist = MultivariateNormal(mean_z, covar_z)
        return dist
    
     def _get_batch_idx(self, batch_size):
            
         valid_indices = np.arange(self.n)
         batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
         return np.sort(batch_indices)
     
if __name__ == '__main__':
    
    # Setting seed for reproducibility
    SEED = 7
    torch.manual_seed(SEED)

    # Load joint spectra and label data 
    
    #data = load_joint_spectra_labels_small()
    X, Y, means_X, std_X, means_Y, std_Y, snr = load_spectra_labels_large()
    
    data = np.hstack((X,Y))
    
    XY_train, XY_test = train_test_split(data, test_size=200, random_state=SEED)
    #lb_train, lb_test = train_test_split(labels, test_size=100, random_state=SEED)
    
    XY_train = torch.Tensor(XY_train)
    XY_test = torch.Tensor(XY_test)
    
    # Setting shapes
    N = len(XY_train)
    data_dim = XY_train.shape[1]
    latent_dim = 10
    n_inducing = 70
    pca = False
      
    # Model
    model = QuasarMiniDemoModel(N, data_dim, latent_dim, n_inducing, pca=pca, kernel_config='partition')
    
    # Likelihood
    likelihood = GaussianLikelihoodWithMissingObs(batch_shape=model.batch_shape)

    # Declaring objective to be optimised along with optimiser
    mll = VariationalELBO(likelihood, model, num_data=len(XY_train))
    
    optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
    ], lr=0.001)
    
    model.get_trainable_param_names()
    
    # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
    # using the optimizer provided.
    
    loss_list = []
    iterator = trange(20000, leave=True)
    batch_size = 100
    for i in iterator: 
        batch_index = model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample = model.sample_latent_variable()  # a full sample returns latent x across all N
        sample_batch = sample[batch_index]
        output_batch = model(sample_batch)
        loss = -mll(output_batch, XY_train[batch_index].T).sum()
        loss_list.append(loss.item())
        iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
        loss.backward()
        optimizer.step()
        
    ####################### Save trained model ##########################################
    
    if os.path.isfile('trained_models/gplvm_1000_clean.pkl'):
          with open('trained_models/gplvm_1000_clean.pkl', 'rb') as file:
              model_sd, likl_sd = pkl.load(file)
              model.load_state_dict(model_sd)
              likelihood.load_state_dict(likl_sd)

    # with open('trained_models/gplvm_1000_clean.pkl', 'wb') as file:
    #     pkl.dump((model.cpu().state_dict(), likelihood.cpu().state_dict()), file)
        
    ####################### Reconstruction Framework ################################################
    
    XY_train_recon, XY_train_pred_covar = model.reconstruct_y(torch.Tensor(model.X.X)[0:300], XY_train[0:300], ae=False, model_name='point')
    
    X_train_recon =  XY_train_recon.T[::,0:-4].detach().numpy()
    Y_train_recon =  XY_train_recon.T[:,-4::].detach().numpy()
    
    X_train_orig = XY_train[::,0:-4]*std_X + means_X
    Y_train_orig = XY_train[:,-4::]*std_Y + means_Y
    
    X_train_recon_orig = X_train_recon*std_X + means_X
    Y_train_recon_orig = Y_train_recon*std_Y + means_Y
    
    diags = np.array([m.diag().sqrt().detach().numpy() for m in XY_train_pred_covar]).T ## extracting diagonals per dimensions
    
    X_pred_var = diags[::,0:-4]*std_X
    Y_pred_var = diags[:,-4::]*std_Y
    
    
    # X_train_orig = XY_train[::,0:-3]*scales[7:] + pivots[7:]
    # Y_train_orig = XY_train[:,-3::]*slabs + plabs
    
    # X_train_recon_orig = X_train_recon*scales[7:] + pivots[7:]
    # Y_train_recon_orig = Y_train_recon*slabs + plabs
    
    # diags = np.array([m.diag().detach().numpy() for m in XY_train_pred_covar]).T ## extracting diagonals per dimensions
    
    # X_pred_var = diags[::,0:-3]*scales[7:]
    # Y_pred_var = diags[:,-3::]*slabs
    
    ########################### Testing Framework #################################
    
    
    ###################### Visualisation ################################
    plt.plot(np.isnan(XY_train).sum(axis=0)) ## check the presence of the data 
    
    ids = np.arange(300)
    #col_range = np.arange(68,1000)

    plot_spectra_reconstructions(X_train_recon_orig, X_train_orig[ids], X_pred_var[ids], obj_id=24)
    plot_spectra_reconstructions(X_train_recon_orig, X_train_orig[ids], X_pred_var[ids], obj_id=13)
    plot_spectra_reconstructions(X_train_recon_orig, X_train_orig[ids], X_pred_var[ids], obj_id=211)
    plot_spectra_reconstructions(X_train_recon_orig, X_train_orig[ids], X_pred_var[ids], obj_id=151)

    
    plot_y_label_comparison(Y_train_recon_orig[ids], Y_train_orig[0:300], Y_pred_var[ids], snr[0:300], col_id = 0, title='Redshift z [test]')
    plot_y_label_comparison(Y_train_recon_orig[ids], Y_train_orig[0:300], Y_pred_var[ids], snr[0:300], col_id = 1, title='Logbol (luminosity) [test]')
    plot_y_label_comparison(Y_train_recon_orig[ids], Y_train_orig[0:300], Y_pred_var[ids], snr[0:300], col_id = 2, title='MBH (log10) [test]')
    plot_y_label_comparison(Y_train_recon_orig[ids], Y_train_orig[0:300], Y_pred_var[ids], snr[0:300], col_id = 3, title='lam_edd [test]')

    # # ################################
    # # Compute the metrics:
            
    from utils.metrics import *
    
    # # 1) Reconstruction error - Train & Test
    
    # mse_train = rmse(Y_train, Y_train_recon.T)
    # mse_test = rmse(Y_test, Y_test_recon.T)
    
    # print(f'Train Reconstruction error {model_name} = ' + str(mse_train))
    # print(f'Test Reconstruction error {model_name} = ' + str(mse_test))
