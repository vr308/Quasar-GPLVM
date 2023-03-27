#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point Net Encoder class for amortised inference 

"""
import gpytorch
import torch
from torch import nn
from torch.distributions import kl_divergence
from gpytorch.mlls.added_loss_term import AddedLossTerm
import torch.nn.functional as F
import numpy as np
from models.partial_gaussian import PointNet
from models.latent_variable import LatentVariable, kl_gaussian_loss_term

class PointNetEncoder(LatentVariable):
    def __init__(self, n, data_dim, latent_dim, prior_x, inter_dim=5, h_dims=(5, 5), rho_dims=(5, 5)):
        super().__init__(n, latent_dim)
        
        self.data_dim = data_dim
        self.prior_x = prior_x
        self.pointnet = PointNet(latent_dim, inter_dim, h_dims=h_dims, rho_dims=rho_dims,
                 min_sigma=1e-6, init_sigma=None, nonlinearity=torch.tanh)
        self.register_added_loss_term("x_kl")

    def forward(self, Y):
        q_x = self.pointnet(Y)
        x_kl = kl_gaussian_loss_term(q_x, self.prior_x, self.n, self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)  # Update the KL term
        return q_x.rsample()
    