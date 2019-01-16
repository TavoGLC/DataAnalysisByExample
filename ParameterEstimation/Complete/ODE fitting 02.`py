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

from scipy.integrate import odeint
from scipy.optimize import curve_fit

import scipy.stats as stats

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

###############################################################################
#                    General Model Construction 
###############################################################################

#Performs the dot produt to make the model 
def MakeModel(MatrixCoeficients,InitialConditions):
    
    return np.dot(MatrixCoeficients,InitialConditions)

###############################################################################
#                              ODE system solving  
###############################################################################

SolverTime=np.linspace(0,20,num=150)

#Parameters for Model A
alpha=0.4
beta=1

#Matrix of coeficients for model A
#Model A is refered in this script as model 01
def MakeModelMatrix01(Alpha,Beta):
    
    Matrix=np.zeros((2,2))

    Matrix[0,0]=Alpha
    Matrix[0,1]=-Beta
    Matrix[1,0]=1
    
    return Matrix

#Integrating Model A
Matrix01=MakeModelMatrix01(alpha,beta)
Int=np.array([1,1])

def SODE(InitialConditions,t):
    
    return MakeModel(Matrix01,InitialConditions)

Solution=odeint(SODE,Int,SolverTime)

###############################################################################
#                    Visualisation
###############################################################################

DerivativeLabel=r'$\dfrac{d}{dt} f(t) $'
SolutionLabel=r'$f(t)$'

plt.figure(1,figsize=(9,6))

plt.plot(SolverTime,Solution[:,1],'b-',label=SolutionLabel,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
plt.plot(SolverTime,Solution[:,0],'g-',label=DerivativeLabel,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])

plt.xlabel('Time',fontsize=16,fontweight='bold')
plt.ylabel('Displacement',fontsize=16,fontweight='bold')
plt.legend(loc=0,fontsize=14)

ax=plt.gca()
PlotStyle(ax,'')

###############################################################################
#                        Data Generation
###############################################################################

#Element wise sum of two iterables of the same size, name makes reference to the output rather than the process
def MakeNoisyData(Data,Noise):
    
    return [val+cal for val,cal in zip(Data,Noise)]

WhiteNoise=[np.random.uniform(low=-1,high=1)*3 for val in Solution[:,1]]
WhiteSignal=MakeNoisyData(Solution[:,1],WhiteNoise)

###############################################################################
#                              ODE fitting  
###############################################################################

#Function for parameter estimation
def ModelSolver01(t,Alpha,Beta,InitialConditions):
    
    cAlpha=Alpha
    cBeta=Beta
    cInit=InitialConditions
    
    cMatrix=MakeModelMatrix01(cAlpha,cBeta)
    
    def LocalModel(cInit,t):
        
        return MakeModel(cMatrix,cInit)
    
    Solution=odeint(LocalModel,cInit,t)
    
    return Solution[:,1]

def ModelSolution01(t,Alpha,Beta):
    
    return ModelSolver01(t,Alpha,Beta,Int)
    
Model01Params=curve_fit(ModelSolution01,SolverTime,WhiteSignal)

###############################################################################
#                    Fit solution
###############################################################################

fAlpha=Model01Params[0][0]
fBeta=Model01Params[0][1]

FitSolutionA=ModelSolution01(SolverTime,fAlpha,fBeta)

###############################################################################
#                    Visualization 
###############################################################################

plt.figure(2,figsize=(9,6))

(markers, stemlines, baseline) = plt.stem(SolverTime, WhiteSignal,bottom=-42,label='Data',basefmt=" ")
plt.setp(stemlines, linestyle="-", color="red", linewidth=0.5,alpha=0.5 )
plt.setp(markers, color="red",alpha=0.75 )

plt.plot(SolverTime,FitSolutionA,'b-',label=SolutionLabel,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
    
plt.xlabel('Time',fontsize=16,fontweight='bold')
plt.ylabel('Displacement',fontsize=16,fontweight='bold')
plt.legend(loc=0,fontsize=14)

plt.ylim(-42,75)

ax=plt.gca()
PlotStyle(ax,'')

###############################################################################
#                    Residuals Statistical test  
###############################################################################

ObRes=[signal-model for signal,model in zip(WhiteSignal,FitSolutionA)]

KS=stats.ks_2samp(ObRes,WhiteNoise)

print(KS)

###############################################################################
#                              ODE system  solving  
###############################################################################

SolverTime=np.linspace(0,20,num=120)

#Model B Parameters
k1=0.3
k2=0.25
k3=0.1

#Coeficients matrix for model B
#Model B is refered as model02
def MakeModelMatrix02(K1,K2,K3):
    
    Matrix=np.zeros((3,3))

    Matrix[0,0]=-K1
    Matrix[0,1]=K3

    Matrix[1,0]=K1
    Matrix[1,1]=-(K2+K3)

    Matrix[2,1]=K2
    
    return Matrix

Matrix02=MakeModelMatrix02(k1,k2,k3)
InitialConditions=[5,0,0]

def KineticsSystem(InitialConditions,t):
    
    return MakeModel(Matrix02,InitialConditions)

SystemSolution=odeint(KineticsSystem,InitialConditions,SolverTime)

###############################################################################
#                    Visualization
###############################################################################

plt.figure(3,figsize=(9,6))

plt.plot(SolverTime,SystemSolution[:,0],'b-',label='[A]',path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
plt.plot(SolverTime,SystemSolution[:,1],'g-',label='[B]',path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
plt.plot(SolverTime,SystemSolution[:,2],'m-',label='[C]',path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])

plt.xlabel('Time',fontsize=16,fontweight='bold')
plt.ylabel('Concentration',fontsize=16,fontweight='bold')
plt.legend(loc=0,fontsize=14)

ax=plt.gca()
PlotStyle(ax,'')

###############################################################################
#                            Data Generation
###############################################################################

WhiteNoise=[np.random.uniform(low=-1,high=1)/4 for val in SystemSolution[:,2]]
WhiteSignal=MakeNoisyData(SystemSolution[:,2],WhiteNoise)

###############################################################################
#                              ODE fitting  
###############################################################################

def ModelSolver02(t,K1,K2,K3,InitialConditions):
    
    cK1=K1
    cK2=K2
    cK3=K3
    
    cInit=InitialConditions
    
    cMatrix=MakeModelMatrix02(cK1,cK2,cK3)
    
    def LocalModel(cInit,t):
        
        return MakeModel(cMatrix,cInit)
    
    Solution=odeint(LocalModel,cInit,t)
    
    return Solution[:,2]

def ModelSolution02(t,K1,K2,K3):
    
    return ModelSolver02(t,K1,K2,K3,InitialConditions)
    
    
Model02Params=curve_fit(ModelSolution02,SolverTime,WhiteSignal)

fK1=Model02Params[0][0]
fK2=Model02Params[0][1]
fK3=Model02Params[0][2]

FitSolutionB=ModelSolution02(SolverTime,fK1,fK2,fK3)

###############################################################################
#                        Visualization
###############################################################################

plt.figure(4,figsize=(9,6))

(markers, stemlines, baseline) = plt.stem(SolverTime, WhiteSignal,bottom=0,label='Data',basefmt=" ")
plt.setp(stemlines, linestyle="-", color="red", linewidth=0.5,alpha=0.5 )
plt.setp(markers, color="red",alpha=0.75 )

SolutionLabel='[C]'
plt.plot(SolverTime,FitSolutionB,'m-',label=SolutionLabel,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
    
plt.xlabel('Time',fontsize=16,fontweight='bold')
plt.ylabel('Concentration',fontsize=16,fontweight='bold')
plt.legend(loc=0,fontsize=14)

plt.ylim(0,5.2)

ax=plt.gca()
PlotStyle(ax,'')

###############################################################################
#                    Residuals Statistical test  
###############################################################################

ObRes=[signal-model for signal,model in zip(WhiteSignal,FitSolutionB)]

KS=stats.ks_2samp(ObRes,WhiteNoise)

print(KS)
