#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gpytorch.models import ApproximateGP
from gpytorch.mlls import VariationalELBO
from prettytable import PrettyTable
from tqdm import trange
import pickle as pkl
import numpy as np
from copy import deepcopy
import torch

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class BayesianGPLVM(ApproximateGP):
    def __init__(self, Z, variational_strategy):
        
        """The GPLVM model class for unsupervised learning. The current class supports
        
        (a) Point estimates for latent Z when prior_z = None 
        (b) MAP Inference for Z when prior_z is not None and inference == 'map'
        (c) Gaussian variational distribution q(Z) when prior_z is not None and inference == 'variational'

        :param Z (LatentVariable): An instance of a sub-class of the LatentVariable class.
                                    One of,
                                    PointLatentVariable / 
                                    MAPLatentVariable / 
                                    GaussianLatentVariable to
                                    facilitate inference with (a), (b) or (c) respectively.
       
        """
     
        super(BayesianGPLVM, self).__init__(variational_strategy)
        
        # Assigning Latent Variable 
        self.Z = Z
    
    def forward(self):
        raise NotImplementedError
          
    def sample_latent_variable(self, *args, **kwargs):
        sample = self.Z(*args, **kwargs)
        return sample
    
    def initialise_model_test(self, Y_test, model_name, seed, prior_z=None):
        
        if model_name == 'gauss': # deepcopy wont work
            test_model = pkl.loads(pkl.dumps(self)) 
            assert id(test_model) != id(self)
            assert test_model.covar_module.base_kernel.lengthscale[0][0] == self.covar_module.base_kernel.lengthscale[0][0]
        
        else:
            test_model = deepcopy(self) # A says better to re-initialise
        # self is in eval mode - but test_model needs to be in train_mode
        #if torch.cuda.is_available():
        #    test_model = test_model.cuda()
        #    test_model.train()
        #else:
            test_model.train()
        test_model.n = len(Y_test)
        latent_dim = self.Z.latent_dim #q
        
        # reset the latent variable - A says be careful about random seed
        # Initialise Z with PCA or 0s.
  
        Z_init = torch.nn.Parameter(torch.randn(test_model.n, latent_dim))
        
        kwargs = {'Z_init_test': Z_init}
        
        if prior_z is not None:
            kwargs['prior_z_test'] = prior_z
            
        if hasattr(test_model.Z, 'data_dim'):
            kwargs['data_dim'] = Y_test.shape[1]

        test_model.Z.reset(**kwargs)
        
        # freeze gradients for everything except Z
        #for name, parameter in test_model.named_parameters():
        #   if (name.startswith('Z') is False):
        #       parameter.requires_grad = False
        
        return test_model
    
    def predict_latent(self, Y_train, Y_test, lr, likelihood, seed, prior_z=None, ae=True, model_name='nn_gauss', steps=5000):
        
        if ae is True: # A says make into a LatentVariable attribute

            # nn-gauss
            mu_star = self.Z.mu(Y_test)
            sigma_star = self.Z.sigma(Y_test)
            return mu_star, sigma_star 
            
        else:
            # Train for test Z variational params
            
            # The idea here is to initialise a new test model but import all the trained 
            # params from the training model. The variational params of the training data 
            # do not affect the test data.
            
            # Initialise test model at training params
            
            test_model = self.initialise_model_test(Y_test, model_name, seed, prior_z=prior_z)
            
            # Initialise fresh test optimizer 
            optimizer = torch.optim.Adam(test_model.Z.parameters(), lr=lr)
            elbo = VariationalELBO(likelihood, test_model, num_data=len(Y_test))
            
            print('---------------Learning variational parameters for test ------------------')
            for name, param in test_model.Z.named_parameters():
                print(name)
                
            loss_list = []
            iterator = trange(steps, leave=True)
            batch_size = len(Y_test) if len(Y_test) < 100 else 100
            for i in iterator: 
                batch_index = test_model._get_batch_idx(batch_size)
                optimizer.zero_grad()
                sample = test_model.sample_latent_variable()  # a full sample returns latent x across all N
                sample_batch = sample[batch_index]
                sample_batch = sample_batch   #.cuda() #if torch.cuda.is_available() else sample_batch
                sample_batch.requires_grad_(True)
                
                output_batch = test_model(sample_batch)
                loss = -elbo(output_batch, Y_test[batch_index].T).sum()
                loss.requires_grad_(True)
                loss_list.append(loss.item())
                iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
                loss.backward()
                optimizer.step()
                
            return loss_list, test_model.Z
        
    def reconstruct_y(self, Z, Y, ae=False):
        
        if ae is True:
            # encoder based model -> to reconstruct encode first, then decode
            y_pred = self(self.Z.mu(Y))
            y_pred_mean = y_pred.loc.detach()
            y_pred_covar = y_pred.covariance_matrix.detach()
            return y_pred_mean, y_pred_covar
    
        else:
            # Just decode from the Z that is passed
            # Returns a batch of multivariate-normals 
            y_pred = self(Z)
            return y_pred, y_pred.loc, y_pred.covariance_matrix

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
        
    def get_Z_mean(self, Y):
         
         if self.Z.__class__.__name__ in ('PointLatentVariable', 'MAPLatentVariable'):
             
             return self.Z.Z.detach()
         
         elif self.Z.__class__.__name__ == 'GaussianLatentVariable':
             
             return self.Z.q_mu.detach()
         
         elif self.Z.__class__.__name__ == 'NNEncoder':
             
             return self.Z.mu(Y).detach()
    
    def get_Z_scales(self, Y):
     
         if self.Z.__class__.__name__ in ('PointLatentVariable', 'MAPLatentVariable'):
             
             return None
         
         elif self.Z.__class__.__name__ == 'VariationalLatentVariable':
             
             return torch.nn.functional.softplus(self.Z.q_log_sigma).detach()
         
         elif self.Z.__class__.__name__ == 'NNEncoder':
             
             return np.array([torch.sqrt(x.diag()) for x in self.Z.sigma(Y).detach()])
         
    def store(self, losses, likelihood):
        
         self.losses = losses 
         self.likelihood = likelihood

       