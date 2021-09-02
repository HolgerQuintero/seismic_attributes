# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:07:09 2021

@author: HOLGER
"""

import numpy as np
import math
from sklearn.cluster import KMeans 

#%%kmeans_opt
def kmeans_opt(*args):
    m=len(args[0]) #getting the number of samples
    if len(args)>2:
        ToTest=np.array([args[1]])
    else:
        ToTest=math.ceil(np.sqrt(m))
    if len(args)>3:
        Cutoff=np.array([args[2]])
    else:
        Cutoff=0.95
    if len(args)>4:
        Repeats=np.array([args[3]])
    else:
        Repeats=3
    
    D=np.zeros((int(ToTest[0]),1))#initialize the results matrix    
    for c in np.arange(0,int(ToTest[0])):
        tmp=np.zeros((1,c+1))
        clustering=KMeans(n_clusters=c+1,n_init=1,init='random')
        clustering.fit(args[0])
        dist=clustering.transform(args[0])
        for i in np.arange(0,c+1):
            tmp[0,i]=np.sum(dist[:,i])
        tmp_sum=np.sum(tmp)
            
        for cc in np.arange(1,Repeats):#repeat the algo
            clustering=KMeans(n_clusters=c+1,n_init=1,init='random')
            clustering.fit(args[0])
            dist=clustering.transform(args[0])
            for i in np.arange(0,c+1):
                  tmp[0,i]=np.sum(dist[:,i])
            tmp_sum=min(tmp_sum,np.sum(tmp))
            
        D[c,0]=tmp_sum #collect the best so far in the results vecor
    
    Var=D[0:-1]-D[1:] #calculate #variance explained
    PC=np.cumsum(Var)/(D[1]-D[-1])
    
    r=np.argwhere(PC>Cutoff)#find the best index
    K=1+r[0,0]; #get the optimal number of clusters
    
    clustering=KMeans(n_clusters=K+1,n_init=1,init='random') #now rerun one last time with the optimal number of clusters
    clustering.fit(args[0])
    IDX=clustering.fit_predict(args[0]);IDX=IDX.reshape(IDX.shape[0],1)
    C=clustering.cluster_centers_
    SUMD=clustering.transform(args[0])
    SUMD_=np.zeros((1,SUMD.shape[1]))
    for i in np.arange(0,SUMD.shape[1]):
        SUMD_[0,i]=np.sum(SUMD[:,i])
        
            
    return(IDX,C,SUMD_,K)