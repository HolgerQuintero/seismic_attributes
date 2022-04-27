# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:42:38 2022

@author: Holger
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm 
#import dbscan_opt as db


import atributos_instantaneos as ai
import atributos_ventana as av
import atributos_velocidad as avel
import featureNormalize as fn
import kmeans_opt as km
import histograma as hist
from funciones import normalize_2d, graph, interpol, delete_outliers,damaged_geophone,arreglo_mascara

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

#%%Cargar los mismos datos de matlab
#import scipy.io as sio
#mat = sio.loadmat('horizontal.mat')
#Horizontal=mat['horizontal']
#mat = sio.loadmat('vertical.mat')
#Vertical=mat['vertical']

#Salvar como .mat
#sio.savemat('v.mat',{'vertical':Vertical})
#sio.savemat('h.mat',{'horizontal':Horizontal})
#%%Cargar datos de python 
Horizontal=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelo_tenerife\aleatorio\dataX.npy')
Vertical=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelo_tenerife\aleatorio\dataZ.npy')


#%%Si hay alguno geofono dañada en datos reales              
Vertical=damaged_geophone(Vertical)
Horizontal=damaged_geophone(Horizontal)


#%%
(Nt,Nx)=Vertical.shape
dt=2e-3#time sampling
fs=1/dt
dx=10
t=np.arange(0,Nt*dt,dt); 
x=np.arange(0,Nx*dx,dx); 


#Correcion de ganancia de datos(La amplitud decae muy rapido)
Vertical1=Vertical
Horizontal1=Horizontal
for i in np.arange(Nx):
    for j in np.arange(Nt):
        Vertical1[j,i]=Vertical[j,i]*(t[j]*t[j])
        Horizontal1[j,i]=Horizontal[j,i]*(t[j]*t[j])
Vertical=Vertical1
Horizontal=Horizontal1


plt.figure(1)
plt.subplot(121)
plt.imshow(Horizontal[:,:],aspect='auto',vmin=-1e-6,vmax=1e-6, extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
#plt.axvline(x=2000, color="black", linestyle="--")
plt.title('Horizontal shot gather')
plt.ylabel(r'Time (s)')
plt.xlabel(r'Distance (m)')
plt.colorbar(orientation='horizontal')

plt.subplot(122)
plt.imshow(Vertical[:,:],aspect='auto',vmin=-1e-6,vmax=1e-6,extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]], cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
#plt.axvline(x=2000, color="black", linestyle="--")
plt.title('Vertical shot gather')
plt.ylabel(r'Time (s)')
plt.xlabel(r'Distance (m)')
plt.colorbar(orientation='horizontal')

plt.show()

#%% Mascaras (Bien)
H=np.zeros(Vertical.shape)
V=np.zeros(Vertical.shape)

for i in np.arange(0,10):
    ans1=np.concatenate((np.arange(i,Horizontal.shape[0]-1),np.arange(Horizontal.shape[0]-(i+1),Horizontal.shape[0])),axis=0)
    ans2=np.concatenate((np.arange(i,Horizontal.shape[1]-1),np.arange(Horizontal.shape[1]-1-i,Horizontal.shape[1])),axis=0)
    H=abs(Horizontal[ans1[:,None],ans2])+H
    V=abs(Vertical[ans1[:,None],ans2])+V
    
mascara1=((H+V)>(1e-1)*np.mean(H+V))# es por defecto 1e-3
#Buscar funcion que haga la misma transformada Hilbert
#V=np.sqrt(Vertical*Vertical+np.imag(signal.hilbert2(Vertical))*np.imag(signal.hilbert2(Vertical)))
V=np.abs(hilbert_from_scratch(Vertical,Nt,Nx))
#H=np.sqrt(Horizontal*Horizontal+np.imag(signal.hilbert2(Horizontal))*np.imag(signal.hilbert2(Horizontal)))
H=np.abs(hilbert_from_scratch(Horizontal,Nt,Nx))
mascara2=((H*H+V*V)>(1e-6)*np.mean(H*H+V*V))#por defecto 1e-6
mascara=mascara1*mascara2



#Para arreglar la mascara del dato real de tenerife
for i in np.arange(884,Nt):
    mascara[i,:]=True
mascara=arreglo_mascara(mascara,Nt,Nx)



mascara_1D=np.transpose(mascara).reshape(-1)
id_mascara=np.where(mascara_1D==True)
(idy,idx)=np.where(np.transpose(mascara)==True)
ind=np.where(mascara_1D==True)
ind=np.asarray(ind)


#%% Feature Generation
Data=np.zeros([Nt,Nx,2])
Data[:,:,0]=Vertical
Data[:,:,1]=Horizontal
win=100e-3; si=1; sox=8; soy=100; Lx=5; Lt=100;

attribute_1=ai.atributos_instantaneos(Data,dt,dx,Lx,Lt)
attribute_2=av.atributos_ventana(Data,dt,win,fs)   
attribute_3=avel.atributos_velocidad(Data,dt,dx,si,sox,soy)
attribute=np.concatenate((attribute_1,attribute_2,attribute_3),axis=2)

NameAttr=['Ampl. V','Ampl. H','Freq. V','Freq. H',
          'Diff phase V.','Diff phase H.','Phase V-H','Semi-major axis','Semi-minor axis',
          'Ellipticity','Signed Ellip.','Tilt angle','Rise angle','Incidence angle','Dip angle','Strength Pol.','GR detector',
          'Local Vel. V','Local Vel. H.']
Units=['dB','dB','Hz','Hz','rad','rad',
          'rad','dB','dB','Adimensional','Adimensional','rad','rad','rad','rad','Adimensional','dB','m/s','m/s']



#%%Quitar outliers de los atributos de velocidad (k=17 y 18)
k=17
att=attribute[:,:,k]#Número del atributo que se quiera arreglar
attribute[:,:,k]=delete_outliers(att)

k=18
att=attribute[:,:,k]#Número del atributo que se quiera arreglar
attribute[:,:,k]=delete_outliers(att)
#%%Mascara en los atributos
A=attribute
for k in np.arange(0,19):
  for i in np.arange(0,Nt):
    for j in np.arange(0,Nx):
        if mascara[i,j]==False:
            A[i,j,k]=np.nan
       
            

#%%Preparando dato para k-means
X=np.zeros((ind.shape[1],attribute.shape[2]))
for n in np.arange(0,attribute.shape[2]):
    M=attribute[:,:,n]
    MV=np.transpose(M).reshape(-1); #MV=MV.reshape(MV.shape[0],1);
    #temp=MV[ind]
    #temp=temp.reshape(-1);
    X[:,n]=MV[ind]
#%%Normalizacion    (Esta funcionando excelente)

(X_norm, mu, sigma)=fn.featureNormalize(X)
nd=1
X_train=X_norm[::nd,:]
numF=np.size(X_train)
#X_train=np.concatenate((X_train[:,2].reshape(X_train.shape[0],1),X_train[:,16].reshape(X_train.shape[0],1)),axis=1)
#para selecionar solo n atributos
#%% Analisis k-means
MAX=7
(IDX,C,SUMD,K)=km.kmeans_opt(X_train,MAX,0.95)
GruposVector=np.zeros(np.size(mascara))
GruposVector[np.transpose(ind)]=IDX+1
GruposMatrix=GruposVector.reshape(mascara.shape[1],mascara.shape[0]) 
GruposMatrix=np.transpose(GruposMatrix)



masV=Vertical
for i in np.arange(0,Nt):
    for j in np.arange(0,Nx):
        if Vertical[i,j]<-1:
            masV[i,j]=-1
for i in np.arange(0,Nt):
    for j in np.arange(0,Nx):
        if Vertical[i,j]>1:
            masV[i,j]=1
 


#%% GRAFICAS
k=0
fig, axs = plt.subplots(4, 5)
for row in np.arange(0,4):
    for col in np.arange(0,5):
        ax = axs[row, col]
        if k==0:    
            pcm=ax.axis('off')
        else:
            pcm = ax.imshow(attribute[:,:,k-1],cmap=cm.jet,aspect='auto',extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
            ax.set_title(NameAttr[k-1],weight='bold')
            fig.colorbar(pcm, ax=ax,label=Units[k-1])
            ax.set_xlabel('Offset [m]')
            ax.set_ylabel('Time [s]')
        k=k+1
        
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=.9, 
                    wspace=0.85, 
                    hspace=0.95)


plt.show()
    
    
#%% GRafica de clustering
diezmado=120

#Depende de el numero MAX
Temp=GruposVector[ind[0,1::diezmado]]
grupos_numerado=np.asarray(Temp.tolist())
lista_de_grupos=['Grupo 1','Grupo 2','Grupo 3','Grupo 4','Grupo 5','Grupo 6','Grupo 7','Grupo 8','Grupo 9', 'Grupo 10','Grupo 11', 'Grupo 12','Grupo 13', 'Grupo 14','Grupo 15', 'Grupo 16','Grupo 17', 'Grupo 18']



for i in np.arange(0,Nt):
    for j in np.arange(0,Nx):
        if GruposMatrix[i,j]==0:
            GruposMatrix[i,j]=GruposMatrix[i,j]/0   
    


plt.figure(3)
plt.subplot(121)
plt.imshow(masV[:,:],aspect='auto', extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.imshow(GruposMatrix[:,:],aspect='auto', extent=[x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gist_rainbow,alpha = 0.7)
plt.axis([x[0],x[x.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Vertical shot gather')
plt.ylabel(r'Time (s)')
plt.xlabel(r'Distance (m)')
plt.colorbar(orientation='horizontal')


plt.subplot(122)
ax=plt.scatter(X_train[1::diezmado,0],X_train[1::diezmado,2],c=grupos_numerado,alpha=0.6,cmap='gist_rainbow')
plt.title('Vertical shot gather')
plt.ylabel(r'Freq V')
plt.xlabel(r'Ampl V')
plt.grid(True)
plt.legend(handles=ax.legend_elements()[0], labels=lista_de_grupos)
plt.colorbar(orientation='horizontal')
plt.show()

#%% Scatterplot 3D


diezmado=120

#Depende de el numero MAX (Despues puedo hacer una funcion breve)
Temp=GruposVector[ind[0,1::diezmado]]
grupos_numerado=np.asarray(Temp.tolist())
lista_de_grupos=['Grupo 1','Grupo 2','Grupo 3','Grupo 4','Grupo 5','Grupo 6','Grupo 7','Grupo 8','Grupo 9', 'Grupo 10']


# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
pcm=ax.scatter3D(X_train[1::diezmado,16],X_train[1::diezmado,17],X_train[1::diezmado,18],c=grupos_numerado,alpha=0.6,cmap='gist_rainbow',label=lista_de_grupos)
ax.set_title('Vertical shot gather')
ax.set_ylabel(r'Inc. Angle')
ax.set_xlabel(r'Vel. local V')
ax.set_zlabel(r'Vel. local H')
plt.grid(True)
#plt.legend(handles=plt.legend_elements()[0], labels=lista_de_grupos)
legend = ax.legend(*pcm.legend_elements(),title="Grupos:")
ax.add_artist(legend)
#fig.colorbar(pcm, ax=ax,orientation='vertical')
plt.show()

#%%scatterplot_matrix

import seaborn as sns 
import pandas as pd

name=['Ampl_V','Ampl_H','Freq_V','Freq_H',
          'Diff_phase_V.','Diff_phase_H.','Phase_V-H','Semi-major_axis','Semi-minor_axis',
          'Ellipticity','Signed_Ellip','Tilt_angle','Rise_angle','Incidence_angle','Dip_angle','Strength_Pol','GR_detector',
          'Local_Vel_V','Local_Vel_H']
df = pd.DataFrame(X_train[1::diezmado,:])
df.columns = name
df['class'] = grupos_numerado.astype(np.int64)
df1=df.iloc[:,[0,2,16,-1]] #De a 3 atributos(no borrar el -1)

#Escoger a mano los colores del mapa
palette=sns.color_palette(palette='gist_rainbow', n_colors=36)
#sns.palplot(palette) #Para ver el mapa de color
#plt.show()
my_colors = [palette[0], palette[7],  
             palette[14], palette[20],palette[26], palette[-1]]

sns.pairplot(df1, hue='class',palette=my_colors,height=1.5,plot_kws={'alpha':0.6})
plt.show()