# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:29:07 2021

@author: USUARIO
"""

import numpy as np


#%%featureNormalize
def featureNormalize(X):
#Esta función permite normalizar las variables de entrada
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    Mu=np.zeros((X.shape[0],X.shape[1]))
    Sigma=np.zeros((X.shape[0],X.shape[1]))
    for i in np.arange(0,X.shape[0]):
        Mu[i,:]=mu
        Sigma[i,:]=sigma
    X_norm=(X-Mu)/Sigma;  
    return(X_norm, mu, sigma)