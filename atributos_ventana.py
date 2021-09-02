# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:10:07 2021

@author: HOLGER
"""
import numpy as np
from scipy import signal,linalg
import centroidef as cen

#%% Atributos Ventana
def atributos_ventana(Data,dt,win,fs):
    Nw=win/dt+1
    Nt=Data.shape[0]
    Z=Data[np.around(np.concatenate((np.concatenate((np.arange((Nw-1)/2+1,0,-1),np.arange(0,Nt)),axis=0),np.arange(Nt-2,Nt-((Nw-1)/2+3),-1)),axis=0)).astype(int),:,0]
    X=Data[np.around(np.concatenate((np.concatenate((np.arange((Nw-1)/2+1,0,-1),np.arange(0,Nt)),axis=0),np.arange(Nt-2,Nt-((Nw-1)/2+3),-1)),axis=0)).astype(int),:,1]
    L=Z.shape[0]
    
    incidencia=np.zeros((int(L-Nw-1),Data.shape[1]))
    dip=np.zeros((int(L-Nw-1),Data.shape[1]))
    strength=np.zeros((int(L-Nw-1),Data.shape[1]))
    e_modw=np.zeros((int(L-Nw-1),Data.shape[1]))
    for iStation in np.arange(0,Data.shape[1]):
        w=np.abs(cen.centroidef(Z[:,iStation],Nw,fs))
        for iwindow in np.arange(0,int(L-Nw-1)):
            M=np.concatenate((X[iwindow:iwindow+int(Nw)+1,iStation].reshape(np.size(X[0:1+int(Nw),0]),1),Z[iwindow:iwindow+int(Nw)+1,iStation].reshape(np.size(Z[0:1+int(Nw),0]),1)),axis=1)
            # Covariance Analysis
            S=(1/Nw)*np.matmul(np.transpose(M),M)
            [Lambdas,V]=linalg.eig(S)#Tambien se considera la parte imaginaria
            Lambda=np.sort(np.abs(Lambdas))[::-1]
            maxv=np.asarray(np.where(Lambdas==Lambda[0]))
            #minv=np.asarray(np.where(Lambdas==Lambda[1]))
            maximo=maxv[0,0]
            incidencia[iwindow,iStation]=np.arccos(np.abs(V[0,maximo]))
            # Complex Covariance Analysis
            C=np.matmul(np.transpose((signal.hilbert2(M))),(signal.hilbert2(M))) #OJO, en matlab solo se multiplica la parte real
            [Lambdas,V]=linalg.eig(C)#Tambien se considera la parte imaginaria
            Lambda=np.sort(np.abs(Lambdas))[::-1]
            maxv=np.asarray(np.where(np.abs(Lambdas)==Lambda[0]))
            #minv=np.asarray(np.where(Lambdas==Lambda[1]))
            maximo=maxv[0,0]
            x0=V[0,maximo]; z0=V[1,maximo]
            dip[iwindow,iStation]=np.arctan(np.real(x0)/np.real(z0))
            strength[iwindow,iStation]=1-Lambda[1]/Lambda[0]
            #SVD Analysis
            u,s,v= linalg.svd(M)
            e_mod=np.sqrt((s[0]*s[0])*(s[1]*s[1]))
            e_modw[iwindow,iStation]=e_mod/w[iwindow]
        
    attribute_2=np.zeros((int(L-Nw-1),Data.shape[1],4))
    attribute_2[:,:,0]=incidencia;
    attribute_2[:,:,1]=dip;
    attribute_2[:,:,2]=strength;
    attribute_2[:,:,3]=20*np.log10(e_modw);
        
    
    return(attribute_2)