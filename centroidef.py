# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:11:18 2021

@author: HOLGER
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


#%% Centoride
def centroidef(Dataz,Nw,fs):
    L=np.size(Dataz)
    s=Dataz
    w=np.linspace(int(-fs/2),int(fs/2),int(Nw))
    w=w.reshape(np.size(w),1)
    c=np.zeros(np.size(np.arange(0,int(L-Nw-1))));c=c.reshape(c.shape[0],1)
    for n in np.arange(0,int(L-Nw-1)):
        S=np.fft.fft(s[n:n+int(Nw)])
        S=np.concatenate((S[np.int(np.round(Nw/2)):int(Nw)],S[0:np.int(np.round(Nw/2))]),axis=0)
        S=S.reshape(np.size(S),1)
        num=np.matmul(np.abs(np.transpose(w)),(np.abs(S)*np.abs(S)))
        den=np.matmul(np.abs(np.transpose(S)),np.abs(S))
        
        c[n,0]=num/den
    
    for i in np.arange(0,c.shape[0]):
    #    if c[i,0]>25:
     #       c[i,0]=(c[i-3,0]+c[i-2,0]+c[i-1,0]+c[i-4,0]+c[i,0])/5
        if c[i,0]>Nw/2:
            c[i,0]=Nw/2
        if c[i,0]==Nw/2:
            c[i,0]=(c[i-3,0]+c[i-2,0]+c[i-1,0]+c[i-4,0]+c[i,0])/5
            
    c=gaussian_filter1d(c,sigma=np.round(Nw/2))
    
    return(c)
