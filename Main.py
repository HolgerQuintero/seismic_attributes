"""
Created on Tue Jan 12 10:19:35 2021

@author: HOLGER
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm



import atributos_instantaneos as ai
import atributos_ventana as av
import atributos_velocidad as avel
import featureNormalize as fn
import kmeans_opt as km


#%%Cargar los mismos datos de matlab
import scipy.io as sio
mat = sio.loadmat('horizontal.mat')
Horizontal=mat['horizontal']
mat = sio.loadmat('vertical.mat')
Vertical=mat['vertical']


#%%
Horizontal=np.load('shot_gather_x.npy')
Vertical=np.load('shot_gather_z.npy')
(Nt,Nx)=Vertical.shape
dt=2e-3
fs=1/dt
dx=1
t=np.arange(0,Nt*dt,dt); 
x=np.arange(0,Nx*dx,dx); 


#Problemas con extend en python
plt.figure(1)
plt.subplot(121)
plt.imshow(Horizontal[:,:],vmin=-1e-9,vmax=1e-9, extent=[x[0],x[x.shape[0]-1],100*t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],100*t[t.shape[0]-1],t[0]])
plt.title('Horizontal shot gather')
plt.ylabel(r'Time (ms)')
plt.xlabel(r'Distance (m)')
plt.colorbar(orientation='horizontal')

plt.subplot(122)
plt.imshow(Vertical[:,:],vmin=-1e-9,vmax=1e-9,extent=[x[0],x[x.shape[0]-1],100*t[t.shape[0]-1],t[0]], cmap=cm.gray)
plt.axis([x[0],x[x.shape[0]-1],100*t[t.shape[0]-1],t[0]])
plt.title('Vertical shot gather')
plt.ylabel(r'Time (ms)')
plt.xlabel(r'Distance (m)')
plt.colorbar(orientation='horizontal')

#%% Mask OJO CON LAS MASCARAS
H=np.zeros(Vertical.shape)
V=np.zeros(Vertical.shape)

for i in np.arange(0,10):
    ans1=np.concatenate((np.arange(i,Horizontal.shape[0]-1),np.arange(Horizontal.shape[0]-(i+1),Horizontal.shape[0])),axis=0)
    ans2=np.concatenate((np.arange(i,Horizontal.shape[1]-1),np.arange(Horizontal.shape[1]-1-i,Horizontal.shape[1])),axis=0)
    H=abs(Horizontal[ans1[:,None],ans2])+H
    V=abs(Vertical[ans1[:,None],ans2])+V
mascara1=((H+V)>(1e-3)*np.mean(H+V))
#Buscar funcion que haga la misma transformada Hilbert
V=np.sqrt(Vertical*Vertical+np.imag(signal.hilbert2(Vertical))*np.imag(signal.hilbert2(Vertical)))
#V=abs(signal.hilbert2(Vertical))
H=np.sqrt(Horizontal*Horizontal+np.imag(signal.hilbert2(Horizontal))*np.imag(signal.hilbert2(Horizontal)))
#H=abs(signal.hilbert2(Horizontal))
mascara2=((H*H+V*V)>(1e-6)*np.mean(H*H+V*V))
mascara=mascara1*mascara2
mascara_1D=np.transpose(mascara).reshape(-1)
id_mascara=np.where(mascara_1D==True)#Como son indices entonces llega hasta 301999
(idy,idx)=np.where(np.transpose(mascara)==True)
ind=np.where(mascara_1D==True)
ind=np.asarray(ind)
#%% Feature Generation
Data=np.zeros([Nt,Nx,2])
Data[:,:,0]=Vertical
Data[:,:,1]=Horizontal
win=100e-3; si=1; sox=8; soy=100; Lx=20; Lt=100;

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



            
        



X=np.zeros((ind.shape[1],attribute.shape[2]))
for n in np.arange(0,attribute.shape[2]):
    M=attribute[:,:,n]
    MV=np.transpose(M).reshape(-1); #MV=MV.reshape(MV.shape[0],1);
    #temp=MV[ind]
    #temp=temp.reshape(-1);
    X[:,n]=MV[ind]

#%%PRUEBA
A=attribute
for k in np.arange(0,19):
  for i in np.arange(0,Nt):
    for j in np.arange(0,Nx):
        if mascara[i,j]==False:
            A[i,j,k]=attribute[i,j,k]/0
       
            
    
#%%Normalization    

(X_norm, mu, sigma)=fn.featureNormalize(X)
nd=1
X_train=X_norm[::nd,:]
numF=np.size(X_train)


#%% Analisis k-means
MAX=4
(IDX,C,SUMD,K)=km.kmeans_opt(X_train,MAX,0.95)
GruposVector=np.zeros(np.size(mascara))
GruposVector[ind]=IDX
GruposMatrix=GruposVector.reshape(mascara.shape[0],mascara.shape[1]) 



    


#%% GRAFICAS
def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]
cols = 5
rows = len(NameAttr) // cols + 1
figsize = (10, 10)
axs = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
axs = trim_axs(axs, len(NameAttr))
n=0
for ax, case in zip(axs, NameAttr):
    ax.set_title('%s' % str(case))
    ax.set(xlabel='Time[s]', ylabel='Offset [m]')
    ax.imshow(attribute[:,:,n],cmap=cm.jet,extent=(-0.5, 6.5, -0.5, 5.5))
    n=n+1
    
    
#%% GRafica de uno solo