# -*- coding: utf-8 -*-
"""

@author: TavoGLC
From:

-Parameter estimation of differential equation models-

"""

###############################################################################
#                          Libraries to use  
###############################################################################

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection

from scipy import stats as st
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import curve_fit

###############################################################################
#                    General plot functions 
###############################################################################

#Elimates the left and top lines and ticks in a matplotlib plot  
def PlotStyle(Axes,Title):
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.xaxis.set_tick_params(labelsize=14)
    Axes.yaxis.set_tick_params(labelsize=14)
    Axes.set_title(Title)

#Lollipop plot based on the python graph gallery implementation 
def LollipopPlot(Fig,Time,Data,Regression):
    
    cTime=Time
    cData=Data
    cRegression=Regression
    
    ax=Fig.gca()

    (markers, stemlines, baseline) = ax.stem(cTime, cData,bottom=-0.4,label='Data',basefmt=" ")
    plt.setp(stemlines, linestyle="-", color="red", linewidth=0.5,alpha=0.5 )
    plt.setp(markers, color="red",alpha=0.75 )

    ax.plot(cTime,cRegression,'b-',label='Model',path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
  
    ax.set_ylabel('Temperature',fontsize=16,fontweight='bold')
    ax.set_xlabel('Time',fontsize=16,fontweight='bold')
    ax.legend(loc=0,fontsize=14)
    ax.set_ylim(-0.4,110)
    PlotStyle(ax,'')

#Waterfall plot, based on the matplotlib example 
def WaterfallPlot(Figure,Time,Data):
    
    cFig=Figure
    cTime=Time
    cData=Data

    ax=cFig.gca(projection='3d')

    Container=[]
    Locations=[(L0/Nodes)*k for k in range(Nodes)]
    fColors=plt.cm.viridis(np.linspace(0, 1,Nodes),alpha=0.51)

    for k in range(Nodes):
        
        localSol=np.copy(cData[:,k])
        localSol[0],localSol[-1]=0,0
        
        Container.append(list(zip(cTime, localSol)))

    poly = PolyCollection(Container,edgecolors='black',linewidth=2,facecolors=fColors)

    ax.add_collection3d(poly, zs=Locations, zdir='y')
    ax.view_init(azim=50)

    ax.set_xlabel('Time',fontsize=16,fontweight='bold')
    ax.xaxis.set_tick_params(labelsize=14)
    ax.get_xaxis().labelpad = 15
    ax.set_xlim3d(0, 45)
    ax.set_ylabel('Location',fontsize=16,fontweight='bold')
    ax.yaxis.set_tick_params(labelsize=14)
    ax.get_yaxis().labelpad = 15
    ax.set_ylim3d(0, 1)
    ax.set_zlabel('Temperature',fontsize=16,fontweight='bold')
    ax.zaxis.set_tick_params(labelsize=14)
    ax.set_zlim3d(0, 100)
    ax.grid(False)

###############################################################################
#                    Importando Paquetes a utilizar 
###############################################################################

#PDE model
def MakeModelMatrix(Nodes,k,h,L,Um):
    
    cK=k
    cH=h
    cL=L
    cU=Um
    cN=Nodes
    
    deltaX=cL/cN
    ACoef=cK/deltaX
    
    Matrix=np.zeros((cN,cN))
    
    for k in range(1,cN-1):
        
        Matrix[k,k-1]=ACoef*1
        Matrix[k,k]=-ACoef*2
        Matrix[k,k+1]=ACoef*1
    
    Matrix[-1,-1]=-cH
    
    Vector=np.zeros(cN)
    Vector[-1]=cH*cU
    
    return Matrix,Vector

#Model parameters
k0=0.25
h0=0.4
Nodes=25
L0=1
Um=10
U0=100

#Makes the matrix of coeficients 
ModelData=MakeModelMatrix(Nodes,k0,h0,L0,Um)

#Generates the model to integrate
def PDEModel(Us,t):
    
    model=np.dot(ModelData[0],Us)+ModelData[1]
    
    return model

#Initial conditions and integration time
Init=[U0 for k in range(Nodes)]
SolverTime=np.linspace(0,45,num=250)

#Model integration
Solution=odeint(PDEModel,Init,SolverTime)

###############################################################################
#                           Visualization 
###############################################################################

fig = plt.figure(1,figsize=(13,7))

WaterfallPlot(fig,SolverTime,Solution)

###############################################################################
#                           Visualization 
###############################################################################

fig=plt.figure(2,figsize=(7,7))

im=plt.imshow(Solution,aspect='auto',interpolation='gaussian')

ax=plt.gca()

ax.set_xticks(np.linspace(0,Nodes-1,num=3))
ax.set_xticklabels(['0','0.5','1'])

ax.set_yticks(np.linspace(0,250,num=5))
ax.set_yticklabels([str(k*10) for k in range(5)])

ax.set_ylabel('Time',fontsize=16,fontweight='bold')
ax.set_xlabel('Location',fontsize=16,fontweight='bold')

PlotStyle(ax,'')

cBar=fig.colorbar(im, ax=ax)

cBar.ax.set_ylabel('Temperature',fontsize=16,fontweight='bold', rotation=270)
cBar.ax.get_yaxis().labelpad = 15

###############################################################################
#                           Visualization 
###############################################################################

plt.figure(3,figsize=(10,7))

fColors=plt.cm.viridis(np.linspace(0, 1,Nodes),alpha=0.8)

for k in range(Nodes):
    
    plt.plot(SolverTime,Solution[:,k],color=fColors[k],path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])

ax=plt.gca()
ax.set_ylabel('Temperature',fontsize=16,fontweight='bold')
ax.set_xlabel('Time',fontsize=16,fontweight='bold')
PlotStyle(ax,'')

###############################################################################
#                            Data generation  
###############################################################################

def MakeNoisyData(Data,Noise):
    
    return [val+cal for val,cal in zip(Data,Noise)]

WhiteNoise=[np.random.uniform(low=-1,high=1)*4 for val in Solution[:,1]]
WhiteSignal=MakeNoisyData(Solution[:,Nodes-1],WhiteNoise)

###############################################################################
#                           Visualization 
###############################################################################

fig=plt.figure(4,figsize=(10,7))

LollipopPlot(fig,SolverTime,WhiteSignal,Solution[:,Nodes-1])

###############################################################################
#                    Importando Paquetes a utilizar 
###############################################################################

#Solves the pDE model and return the last node of the model 
def ModelSolver(t,k,h,Us):
    
    cK=k
    cH=h
    cUs=Us
    cTime=t
    
    cModelData=MakeModelMatrix(Nodes,cK,cH,L0,Um)
    
    def LocalModel(cUs,t):
    
        model=np.dot(cModelData[0],cUs)+cModelData[1]
    
        return model
    
    cSol=odeint(LocalModel,cUs,cTime)
    
    return cSol[:,Nodes-1]

#Solution function for curve_fit 
def ModelFit(t,k,h):
    
    return ModelSolver(t,k,h,Init)

#Objetive function for the initial guess optimization 
def SquaredError(ParameterGuess):
    
    try:
        
        cModelParams=curve_fit(ModelFit,SolverTime,WhiteSignal,p0=ParameterGuess)
        cSolution=ModelSolver(SolverTime,cModelParams[0][0],cModelParams[0][1],Init)
    
        error=[(val-sal)**2 for val,sal in zip(cSolution,WhiteSignal)]
        
    except RuntimeError:
        
        error=[1000,1000]
    
    return sum(error)

#Optimization of the initial guess 
g0=[np.random.uniform(low=0,high=0.5) for val in range(2)]
res = minimize(SquaredError, g0, method='nelder-mead',options={'xtol': 1e-6, 'maxiter':1000,'disp': False})

#Paaarameter estimation
ModelParams=curve_fit(ModelFit,SolverTime,WhiteSignal,p0=res.x,bounds=(0, [1., 1.,]))

###############################################################################
#                    Importando Paquetes a utilizar 
###############################################################################

FitData=ModelSolver(SolverTime,ModelParams[0][0],ModelParams[0][1],Init)

fig=plt.figure(5,figsize=(10,7))

LollipopPlot(fig,SolverTime,WhiteSignal,FitData)

###############################################################################
#                    Importando Paquetes a utilizar 
###############################################################################

#Generates a library of similated data 
def MakeDataLibrary(OriginalSignal,LibrarySize):
    
    cOS=OriginalSignal
    cN=LibrarySize
    Container=[]
    
    for k in range(cN):
        
        cNoise=[np.random.uniform(low=-1,high=1)*4 for val in Solution[:,1]]
        Container.append(MakeNoisyData(cOS,cNoise))
        
    return Container

#Data library
Library=MakeDataLibrary(Solution[:,Nodes-1],2000)

#Calculates the parameters for each dataset in the library 
def MakeParameterLibrary(DataLibrary):
    
    cLib=DataLibrary
    nLib=len(DataLibrary)
    Container=[]
    
    for k in range(nLib):
        
        LocalData=cLib[k]
        
        cGuess=[np.random.uniform(low=0,high=0.5) for val in range(2)]
        cOpt=minimize(SquaredError, cGuess, method='nelder-mead',options={'xtol': 1e-6, 'maxiter':1000,'disp': False})

        LocalParams=curve_fit(ModelFit,SolverTime,LocalData,p0=cOpt.x,bounds=(0, [1., 1.,]))
        
        Container.append(list(LocalParams[0]))
        
    return Container

#Parameter library
ParamLib=MakeParameterLibrary(Library)
ParamArray=np.array(ParamLib)

###############################################################################
#                    Importando Paquetes a utilizar 
###############################################################################

#Kernel density estimation of the parameter data 
def DensityEstimation(ParameterLibrary):
    
    cLib=ParameterLibrary
    
    m1=cLib[:,0]
    m2=cLib[:,1]

    cxmin=-0.5
    cxmax=1.1*m1.max()
    cymin=0.95*m2.min()
    cymax=1.05*m2.max()

    X, Y = np.mgrid[cxmin:cxmax:200j, cymin:cymax:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = st.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    return Z,cxmin,cxmax,cymin,cymax

###############################################################################
#                    Importando Paquetes a utilizar 
###############################################################################

Z,xmin,xmax,ymin,ymax=DensityEstimation(ParamArray)

fig = plt.figure(6,figsize=(10,10))

plt.imshow(np.rot90(Z),aspect='auto',cmap=plt.cm.viridis,interpolation='gaussian',extent=[xmin, xmax, ymin, ymax])

ax=plt.gca()

ax.set_ylabel('Parameter -h- ',fontsize=16,fontweight='bold')
ax.set_xlabel('Parameter -k- ',fontsize=16,fontweight='bold')

PlotStyle(ax,'')

cBar=fig.colorbar(im, ax=ax)

cBar.ax.set_ylabel('Density',fontsize=16,fontweight='bold', rotation=270)
cBar.ax.get_yaxis().labelpad = 15
