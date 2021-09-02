# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:05:06 2021

@author: HOLGER
"""
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


#%%ATRIBUTOS INSTANTANEOS.
def atributos_instantaneos(Data,dt,dx,Lx,Lt):
    fs=1/dt
    Nt=Data.shape[0];Nx=Data.shape[1]
    #Componente vertical
    Z=Data[:,:,0]
    dv=signal.hilbert2(Z)#Otra vez transformada Hilbert
    for i in np.arange(0,Nt):
        for j in np.arange(0,Nx):
            if Z[i,j]<1e-5:
                dv[i,j]=np.complex(0,np.imag(dv[i,j]))
            else:
                dv[i,j]=np.complex(Z[i,j],np.imag(dv[i,j]))
            
    Av=np.abs(dv)
    phiv=np.angle(dv)
    dphiv=np.arcsin(np.sin(np.diff(phiv,1,1)))
    temp=dphiv[:,-1]
    
    dphiv=np.concatenate((dphiv,temp.reshape(Nt,1)),axis=1)
    fv=abs(fs/(2*np.pi)*np.diff(np.unwrap(np.angle(dv))))
    temp=fv[:,-1]
    fv=np.concatenate((fv,temp.reshape(Nt,1)),axis=1)
    
    #Componente radial    
    X=Data[:,:,1]
    dr=signal.hilbert2(X)
    for i in np.arange(0,Nt):
        for j in np.arange(0,Nx):
            if X[i,j]<1e-5:
                dr[i,j]=np.complex(0,np.imag(dr[i,j]))
            else:
                dr[i,j]=np.complex(X[i,j],np.imag(dr[i,j]))
    Ah=np.abs(dr)
    phir=np.angle(dr)
    dphir=np.arcsin(np.sin(np.diff(phir,1,1)))
    temp=dphir[:,-1]
    dphir=np.concatenate((dphir,temp.reshape(Nt,1)),axis=1)
    fr=abs(fs/(2*np.pi)*np.diff(np.unwrap(np.unwrap(np.angle(dr)))))
    temp=fr[:,-1]
    fr=np.concatenate((fr,temp.reshape(Nt,1)),axis=1)
    fh=fr 

    #Polarization attributes
    rv=np.real(dv); rh=np.real(dr)
    qv=np.imag(dv); qh=np.imag(dr)
    phi=np.arctan2(rh*qv-rv*qh,rv*rh+qv*qh)
    S0=Av*Av+Ah*Ah
    S1=Av*Av-Ah*Ah
    S2=2*Av*Ah*np.cos(phi)
    a=np.sqrt((1/2)*(S0+np.sqrt(S1*S1+S2*S2)))
    b=np.real(np.sqrt(np.abs((1/2)*(S0-np.sqrt(S1*S1+S2*S2)))))
    rho=b/a
    sigma=np.sign(phi)*rho;
    tau=(1/2)*np.arctan(S2/S1)
    temp1=tau>=0
    temp2=tau<=np.pi/2
    temp3=tau>=-np.pi/2
    temp4=tau<0
    np.where(temp1 == True, 1, 0)
    np.where(temp2 == True, 1, 0)
    np.where(temp3 == True, 1, 0)
    np.where(temp4 == True, 1, 0)
    gamma=(np.pi/2-tau)*temp1*temp2+(-np.pi/2-tau)*temp3*temp4
    
    Lx=2
    Av=gaussian_filter1d(Av,sigma=Lx)
    Ah=gaussian_filter1d(Ah,sigma=Lx)
    fv=gaussian_filter1d(fv,sigma=Lx)
    fh=gaussian_filter1d(fh,sigma=Lx)
    dphiv=gaussian_filter1d(dphiv,sigma=Lx)
    dphir=gaussian_filter1d(dphir,sigma=Lx)
    phi=gaussian_filter1d(phi,sigma=Lx)
    a=gaussian_filter1d(a,sigma=Lx)
    b=gaussian_filter1d(b,sigma=Lx)
    rho=gaussian_filter1d(rho,sigma=Lx)
    sigma=gaussian_filter1d(sigma,sigma=Lx)
    tau=gaussian_filter1d(tau,sigma=Lx)
    gamma=gaussian_filter1d(gamma,sigma=Lx)
    
    G=np.sqrt(Av*Ah)
    L=signal.gaussian(Lt,Lt/5)
    
    attribute=np.zeros([Nt,Nx,13])
    for i in np.arange(Z.shape[1]):
        attribute[:,i,0]=20*np.log10(Av[:,i])
        attribute[:,i,1]=20*np.log10(Ah[:,i])
        attribute[:,i,2]=np.convolve(fv[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same')
        attribute[:,i,3]=np.convolve(fh[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same')
        attribute[:,i,4]=np.convolve(dphiv[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same')
        attribute[:,i,5]=np.convolve(dphir[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same')
        attribute[:,i,6]=np.convolve(phi[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same')
        attribute[:,i,7]=20*np.log10(np.convolve(a[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same'))
        attribute[:,i,8]=20*np.log10(np.convolve(b[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same'))
        attribute[:,i,9]=np.convolve(rho[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same')
        attribute[:,i,10]=np.convolve(sigma[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same')
        attribute[:,i,11]=np.convolve(tau[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same')
        attribute[:,i,12]=np.convolve(gamma[:,i]*G[:,i],L,'same')/np.convolve(G[:,i],L,'same')
        
    return(attribute)

