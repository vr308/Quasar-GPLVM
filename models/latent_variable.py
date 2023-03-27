
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Variable class with sub-classes that determine type of inference for the latent variable

"""
import torch
import gpytorch
import numpy as np
from torch.distributions import kl_divergence
from gpytorch.mlls.added_loss_term import AddedLossTerm

class LatentVariable(gpytorch.Module):
    
    """
    :param n (int): Size of the latent space.
    :param latent_dim (int): Dimensionality of latent space.

    """

    def __init__(self, n, dim):
        super().__init__()
        self.n = n
        self.latent_dim = dim
        
    def forward(self, x):
        raise NotImplementedError
        
    def reset(self):
         raise NotImplementedError
        
class PointLatentVariable(LatentVariable):
    def __init__(self, X_init):
        n, latent_dim = X_init.shape
        super().__init__(n, latent_dim)
        self.register_parameter('X', X_init)

    def forward(self):
        return self.X
    
    def reset(self, X_init_test):
        self.__init__(X_init_test)
        
class MAPLatentVariable(LatentVariable):
    
    def __init__(self, X_init, prior_x):
        n, latent_dim = X_init.shape
        super().__init__(n, latent_dim)
        self.prior_x = prior_x
        self.register_parameter('X', X_init)
        self.register_prior('prior_x', prior_x, 'X')

    def forward(self):
        return self.X
    
    def reset(self, X_init_test, prior_x_test):
        self.__init__(X_init_test, prior_x_test)

class kl_gaussian_loss_term(AddedLossTerm):
    
    def __init__(self, q_x, p_x, n, data_dim):
        self.q_x = q_x
        self.p_x = p_x
        self.n = n
        self.data_dim = data_dim
        
    def loss(self): 
        # G 
        kl_per_latent_dim = kl_divergence(self.q_x, self.p_x).sum(axis=0) # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum()/self.n # scalar
        # inside the forward method of variational ELBO, 
        # the added loss terms are expanded (using add_) to take the same 
        # shape as the log_lik term (has shape data_dim)
        # so they can be added together. Hence, we divide by data_dim to avoid 
        # overcounting the kl term
        return (kl_per_point/self.data_dim)
    

class VariationalLatentVariable(LatentVariable):
    
    def __init__(self, X_init, prior_x, data_dim):
        n, latent_dim = X_init.shape
        super().__init__(n, latent_dim)
        
        self.data_dim = data_dim
        self.prior_x = prior_x
        # G: there might be some issues here if someone calls .cuda() on their BayesianGPLVM
        # after initializing on the CPU

        # Local variational params per latent point with dimensionality latent_dim
        self.q_mu = torch.nn.Parameter(X_init)
        self.q_log_sigma = torch.nn.Parameter(torch.randn(n, latent_dim))     
        # This will add the KL divergence KL(q(X) || p(X)) to the loss
        self.register_added_loss_term("x_kl")

    def forward(self, batch_idx=None):
        
        if batch_idx is None:
            batch_idx = np.arange(self.n) 
        
        q_mu_batch = self.q_mu[batch_idx, ...]
        q_log_sigma_batch = self.q_log_sigma[batch_idx, ...]

        q_x = torch.distributions.Normal(q_mu_batch, q_log_sigma_batch.exp())

        self.prior_x.loc = self.prior_x.loc[:len(batch_idx), ...]
        self.prior_x.scale = self.prior_x.scale[:len(batch_idx), ...]
        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, len(batch_idx), self.data_dim)        
        self.update_added_loss_term('x_kl', x_kl)
        return q_x.rsample()
    
    def reset(self, X_init_test, prior_x_test, data_dim):
        self.__init__(X_init_test, prior_x_test, data_dim)