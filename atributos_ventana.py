"""
Created on Tue Aug 24 10:10:07 2021

@author: HOLGER
"""
import numpy as np
from scipy import signal,linalg
import centroidef as cen

from scipy.fftpack import fft,ifft

def hilbert_from_scratch(u,Nt,Nx):
    # N : fft length
    # M : number of elements to zero out
    # U : DFT of u
    # v : IDFT of H(U)
    v=np.zeros((Nt,Nx),dtype = "complex_")
    for i in np.arange(0,Nx):
     N = len(u[:,i])
     # take forward Fourier transform
     U = fft(u[:,i])
     M = N - N//2 - 1
     # zero out negative frequency components
     U[N//2+1:] = [0] * M
     # double fft energy except @ DC0
     U[1:N//2] = 2 * U[1:N//2]
     # take inverse Fourier transform
     v[:,i] = ifft(U)
    return v

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
            [Lambdas,V]=np.linalg.eigh(S)#Tambien se considera la parte imaginaria
            Lambda=np.sort(np.abs(Lambdas))[::-1]
            maxv=np.asarray(np.where(np.abs(Lambdas)==Lambda[0]))
            #minv=np.asarray(np.where(Lambdas==Lambda[1]))
            maximo=maxv[0,0]
            incidencia[iwindow,iStation]=np.arccos(np.abs(V[maximo,0]))
            # Complex Covariance Analysis
            C=np.matmul(np.transpose((hilbert_from_scratch(M,M.shape[0],M.shape[1]))),(hilbert_from_scratch(M,M.shape[0],M.shape[1]))) #OJO, en matlab solo se multiplica la parte real
            [Lambdas,V]=np.linalg.eigh(C)#Tambien se considera la parte imaginaria
            Lambda=np.sort(np.abs(Lambdas))[::-1]
            maxv=np.asarray(np.where(np.abs(Lambdas)==Lambda[0]))
            #minv=np.asarray(np.where(Lambdas==Lambda[1]))
            maximo=maxv[0,0]
            x0=V[maximo,0]; z0=V[maximo,1]#cambie el orden
            dip[iwindow,iStation]=np.arctan(np.real(x0)/(np.real(z0)+3e-300))
            strength[iwindow,iStation]=1-Lambda[1]/(Lambda[0]+3e-300)
            #SVD Analysis
            u,s,v= linalg.svd(M)
            e_mod=np.sqrt((s[0]*s[0])*(s[1]*s[1]))
            e_modw[iwindow,iStation]=e_mod/w[iwindow]
        
    attribute_2=np.zeros((int(L-Nw-1),Data.shape[1],4))
    attribute_2[:,:,0]=incidencia;
    attribute_2[:,:,1]=dip*-1;
    attribute_2[:,:,2]=strength;
    attribute_2[:,:,3]=20*np.log10(e_modw+3e-300);
        
    
    return(attribute_2)
