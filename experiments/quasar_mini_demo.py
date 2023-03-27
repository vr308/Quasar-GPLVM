#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for bGPLVM Gaussian with different inference modes. 

"""

from utils.load_small_data import load_joint_spectra_labels_small
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
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from models.likelihood import GaussianLikelihoodWithMissingObs

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class QuasarMiniDemoModel(BayesianGPLVM):
     def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False):
         
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(n_inducing, latent_dim)
    
        # Sparse Variational Formulation
        
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
    
        # Define prior for X
        X_prior_mean = torch.zeros(self.n, latent_dim)  # shape: N x Q
    
        # Initialise X with PCA or 0s.
        
        if pca == True:
             X_init = _init_pca(Y_train, latent_dim) # Initialise X to PCA 
        else:
             X_init = torch.nn.Parameter(torch.zeros(n, latent_dim))
          
        # LatentVariable configuration
        #prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
        #X = MAPLatentVariable(n, latent_dim, X_init, prior_x)
        #X = VariationalLatentVariable(self.n, data_dim, latent_dim, X_init, prior_x)
        X = PointLatentVariable(X_init)

        super(QuasarMiniDemoModel, self).__init__(X, q_f)
        
        # Kernel 
        
        self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))


     def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
     def _get_batch_idx(self, batch_size):
            
         valid_indices = np.arange(self.n)
         batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
         return np.sort(batch_indices)
     
if __name__ == '__main__':
    
    # Setting seed for reproducibility
    SEED = 7
    torch.manual_seed(SEED)

    # Load some data
    
    #N, d, q, X, Y, labels = load_real_data('oilflow')
    data = load_joint_spectra_labels_small()
    
    #Y_train, Y_test = train_test_split(data, test_size=0, random_state=SEED)
    #lb_train, lb_test = train_test_split(labels, test_size=100, random_state=SEED)
    
    Y_train = torch.Tensor(data)
    #Y_test = torch.Tensor(Y_test)
    
    # Setting shapes
    N = len(Y_train)
    data_dim = Y_train.shape[1]
    latent_dim = 10
    n_inducing = 25
    pca = False
      
    # Model
    model = QuasarMiniDemoModel(N, data_dim, latent_dim, n_inducing, pca=pca)
    
    # Likelihood
    likelihood = GaussianLikelihoodWithMissingObs()

    # Declaring objective to be optimised along with optimiser
    mll = VariationalELBO(likelihood, model, num_data=len(Y_train))
    
    optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
    ], lr=0.001)
    
    model.get_trainable_param_names()
    
    # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
    # using the optimizer provided.
    
    loss_list = []
    iterator = trange(5000, leave=True)
    batch_size = 31
    for i in iterator: 
        batch_index = model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample = model.sample_latent_variable()  # a full sample returns latent x across all N
        sample_batch = sample[batch_index]
        output_batch = model(sample_batch)
        loss = -mll(output_batch, Y_train[batch_index].T).sum()
        loss_list.append(loss.item())
        iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
        loss.backward()
        optimizer.step()