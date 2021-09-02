# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:26:23 2021

@author: USUARIO
"""

import numpy as np
from scipy import signal


#%% Atributos_Velocidad
def atributos_velocidad(Data,dt,dx,si,sox,soy):
    I=Data[:,:,0]
    (m,n)=np.shape(I)
    Sxx_v=np.zeros((m,n))
    Sxy_v=np.zeros((m,n))
    Syy_v=np.zeros((m,n))
#Robust differentiation by convolution with derivative of Gaussian:   
    x=np.arange(-2*si,2*si+1)
    g=np.exp(-0.5*(x/si)*(x/si))
    g=g/np.sum(g)
    gd=-(x*g)/si #is this normalized?
    Ix=signal.convolve2d( signal.convolve2d(I,gd.reshape(1,gd.shape[0]),'same'),g.reshape(g.shape[0],1),'same')
    Iy=signal.convolve2d( signal.convolve2d(I,gd.reshape(gd.shape[0],1),'same'),g.reshape(1,g.shape[0]),'same')
    
    Ixx = Ix*Ix
    Ixy = Ix*Iy
    Iyy = Iy*Iy
 
#Smoothing:
    x=np.arange(-2*sox,2*sox+1)
    gx=np.exp(-0.5*(x/sox)*(x/sox))
    y=np.arange(-2*soy,2*soy+1)
    gy=np.exp(-0.5*(y/soy)*(y/soy))
    Sxx_v=signal.convolve2d( signal.convolve2d(Ixx,gx.reshape(1,gx.shape[0]),'same'),gy.reshape(gy.shape[0],1),'same')
    Sxy_v=signal.convolve2d( signal.convolve2d(Ixy,gx.reshape(1,gx.shape[0]),'same'),gy.reshape(gy.shape[0],1),'same')
    Syy_v=signal.convolve2d( signal.convolve2d(Iyy,gx.reshape(1,gx.shape[0]),'same'),gy.reshape(gy.shape[0],1),'same')
    
    
    I=Data[:,:,1]
    (m,n)=np.shape(I)
    Sxx_h=np.zeros((m,n))
    Sxy_h=np.zeros((m,n))
    Syy_h=np.zeros((m,n))    
#Robust differentiation by convolution with derivative of Gaussian:
    x=np.arange(-2*si,2*si+1)
    g=np.exp(-0.5*(x/si)*(x/si))
    g=g/np.sum(g)
    gd=-x*g/si
    Ix=signal.convolve2d( signal.convolve2d(I,gd.reshape(1,gd.shape[0]),'same'),g.reshape(g.shape[0],1),'same')
    Iy=signal.convolve2d( signal.convolve2d(I,gd.reshape(gd.shape[0],1),'same'),g.reshape(1,g.shape[0]),'same')

    Ixx = Ix*Ix
    Ixy = Ix*Iy
    Iyy = Iy*Iy

#Smoothing:
    x=np.arange(-2*sox,2*sox+1)
    gx=np.exp(-0.5*(x/sox)*(x/sox))
    y=np.arange(-2*soy,2*soy+1)
    gy=np.exp(-0.5*(y/soy)*(y/soy))
    Sxx_h=signal.convolve2d( signal.convolve2d(Ixx,gx.reshape(1,gx.shape[0]),'same'),gy.reshape(gy.shape[0],1),'same')
    Sxy_h=signal.convolve2d( signal.convolve2d(Ixy,gx.reshape(1,gx.shape[0]),'same'),gy.reshape(gy.shape[0],1),'same')
    Syy_h=signal.convolve2d( signal.convolve2d(Iyy,gx.reshape(1,gx.shape[0]),'same'),gy.reshape(gy.shape[0],1),'same')

    vh=0*I
    vv=0*I
    
    for ix in np.arange(n):
        for it in np.arange(m):
            S_v=np.array([[Sxx_v[it,ix],Sxy_v[it,ix]],[Sxy_v[it,ix],Syy_v[it,ix]]])
            (lambda_,vel)=np.linalg.eig(S_v)
            vv[it,ix]=(vel[0,1]*dx)/(vel[1,1]*dt)
            S_h=np.array([[Sxx_h[it,ix],Sxy_h[it,ix]],[Sxy_h[it,ix],Syy_h[it,ix]]])
            (lambda_,vel)=np.linalg.eig(S_h)
            vh[it,ix]=(vel[0,1]*dx)/(vel[1,1]*dt)
      
    attribute_velocity=np.zeros((m,n,2))
    attribute_velocity[:,:,0]=vv
    attribute_velocity[:,:,1]=vh
    return(attribute_velocity)