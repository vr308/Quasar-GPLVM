#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Negative test log-likelihood and sq. root mean squared error

"""

import torch
import numpy as np
from scipy.stats import stats 

# def nll(X_test_pred, X_test, X_std):

#       try:
          
#           chol = torch.linalg.cholesky(pred.covariance_matrix + torch.eye(len(test_x))*1e-4)
          
#       except RuntimeError:
          
#           print('Not psd for sample ' + str(i))
                 
#       lpd = X_test_pred.log_prob(X_test.T)
#       # return the average
#       avg_lpd_rescaled = lpd.cpu().detach()/len(X_test) - torch.log(torch.Tensor(X_std))
#       return -avg_lpd_rescaled.mean()

def nll_lum_bhm_edd(Y_test_pred, Y_test, Y_std):
    
      ## clear out the Nan
      nan_ids = torch.where(Y_test.isnan())
      lpd = Y_test_pred.log_prob(Y_test.T)
      # return the average
      avg_lpd_rescaled = lpd.cpu().detach()/len(Y_test) - torch.log(torch.Tensor(Y_std))
      return -avg_lpd_rescaled[1:]
  
def rmse(X_test_orig, X_test_recon_orig):
    
    return torch.sqrt(torch.mean(torch.Tensor([np.nanmean(np.square(X_test_orig - X_test_recon_orig))])))

def rmse_lum_bhm_edd(Y_test_orig, Y_test_recon_orig):
    
    rmse_labels = []
    for i in [1,2,3]:
        rmse_labels.append(torch.sqrt(torch.mean(torch.Tensor([np.nanmean(np.square(Y_test_orig[:,i] - Y_test_recon_orig[:,i]))]))))
    return rmse_labels