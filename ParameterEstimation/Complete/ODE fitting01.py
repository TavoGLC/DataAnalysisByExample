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
from matplotlib.gridspec import GridSpec

###############################################################################
#                    General plot functions 
###############################################################################

#Elimates the left and top lines and ticks in a matplotlib plot 
def PlotStyle(Axes,Title):
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.set_title(Title)

###############################################################################
#                    Global definitions of the model  
###############################################################################

kc=2 #Radioactive decay constant
C0=1 #Initial condition of the model 

###############################################################################
#                                 ODE solver  
###############################################################################

#Numpy array that contains the integration times 
SolverTime=np.linspace(0,2)

def ODE(C,t):
    
    return -kc*C

#Solution of the model 
ModelSolution=odeint(ODE,C0,SolverTime)

plt.figure(1)

plt.plot(SolverTime,ModelSolution,label='ODE Solution',path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
ax=plt.gca()
ax.legend(loc=0)
PlotStyle(ax,'')


###############################################################################
#                          ODE fitting functions
###############################################################################

#General function to solve the ODE model 
def GeneralSolver(t,k,C0):
    
    localK=k
    localC0=C0
    
    def ODEModel(C,t):
    
        return -localK*C

    sol=odeint(ODEModel,localC0,t)
    
    return sol[:,0]

#Solves the ODE model using the initial condition provided above
def ODESolution(t,k):
    
    return GeneralSolver(t,k,C0)

#Element wise sum of two iterables of the same size, name makes reference to the output rather than the process
def MakeNoisyData(Data,Noise):
    
    return [val+cal for val,cal in zip(Data,Noise)]

###############################################################################
#                         ODE fitting visualization  
###############################################################################

#Solving the ODE model 
t_vals=np.linspace(0,2,num=1000)
solution=ODESolution(t_vals,kc)

#Making some simulated data to perform regression 
WhiteNoise=[np.random.uniform(low=-1,high=1)/20 for val in solution]
WhiteSignal=MakeNoisyData(solution,WhiteNoise)
Kp=curve_fit(ODESolution,t_vals,WhiteSignal)[0][0]

#Parameter estimation 
fitSolution=ODESolution(t_vals,Kp)

plt.figure(2)

plt.plot(t_vals,WhiteSignal,'ro',label='Data')
plt.plot(t_vals,fitSolution,label='Regression',path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
ax=plt.gca()
ax.legend(loc=0)
PlotStyle(ax,'')


###############################################################################
#                             Data size impact  
###############################################################################

#Data size lenghts to test 
nums=[1000,500,100,50,25,10]

#Library of ODE solutions 
t_lib=[np.linspace(0,2,num=val) for val in nums]
sol_lib=[ODESolution(clib,kc) for clib in t_lib]

#Library of simulated data 
noises=[[np.random.uniform(low=-1,high=1)/20 for val in sol] for sol in sol_lib]
signal=[MakeNoisyData(sol,nos) for sol,nos in zip(sol_lib,noises)]

#Parameter estimation an performance evaluation 
params=[curve_fit(ODESolution,times,signals)[0][0] for times,signals in zip(t_lib,signal)] 
solutions=[ODESolution(times,kS) for times,kS in zip(t_lib,params)]

paramError=[abs(val-kc)/kc for val in params]

###############################################################################
#                    Data size impact visualization 
###############################################################################

axs=[(i,j) for i in range(3) for j in range(2)]
fig0,axes=plt.subplots(3,2,figsize=(12,8))

for k in range(len(nums)):
    
    axes[axs[k]].plot(t_lib[k],signal[k],'or',label='Data')
    axes[axs[k]].plot(t_lib[k],solutions[k],label='Regression',path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
    axes[axs[k]].legend(loc=0)
    
    title='Data Points = '+str(len(signal[k]))
    
    PlotStyle(axes[axs[k]],title)

plt.tight_layout()

fig1,axs=plt.subplots(1,2,figsize=(10,4))

axs[0].plot(params,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
PlotStyle(axs[0],'Estimated Parameter')

axs[1].plot(paramError,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
PlotStyle(axs[1],'Absolute Error')

plt.tight_layout()

###############################################################################
#                   Data generation for residuals analysis  
###############################################################################

#ODE solution 
t_data=np.linspace(0,2)
sol=ODESolution(t_data,kc)

#Generating noise data with mixed signals 
WhiteNoise=[np.random.uniform(low=-1,high=1)/20 for val in sol]
PeriodicNoise=[np.random.uniform(low=-1,high=1)/30+np.sin(val/np.pi)/30 for val in range(len(t_data))]
LinearNoise=[np.random.uniform(low=-1,high=1)/30-0.04*(val/30) for val in range(len(t_data))]

###############################################################################
#                            Residuals analysis 
###############################################################################

WhiteSignal=MakeNoisyData(sol,WhiteNoise)
PeriodicSignal=MakeNoisyData(sol,PeriodicNoise)
LinearSignal=MakeNoisyData(sol,LinearNoise)

paramWhite=curve_fit(ODESolution,t_data,WhiteSignal)
paramPeriodic=curve_fit(ODESolution,t_data,PeriodicSignal)
paramLinear=curve_fit(ODESolution,t_data,LinearSignal)

fitSolutionWhite=ODESolution(t_data,paramWhite[0][0])
fitSolutionPeriodic=ODESolution(t_data,paramPeriodic[0][0])
fitSolutionLinear=ODESolution(t_data,paramLinear[0][0])

residualsWhite=[val-cal for val,cal in zip(WhiteSignal,fitSolutionWhite)]
residualsPeriodic=[val-cal for val,cal in zip(PeriodicSignal,fitSolutionPeriodic)]
residualsLinear=[val-cal for val,cal in zip(LinearSignal,fitSolutionLinear)]

###############################################################################
#                    Residual analysis visualization 
###############################################################################

def ResidualsPlot(Figure,Time,Signal,FitSolution,Residuals,Noise):
    
    cFig=Figure
    gridSp=GridSpec(2,2)
    
    ax1=cFig.add_subplot(gridSp[:,0])
    ax2=cFig.add_subplot(gridSp[0,1])
    ax3=cFig.add_subplot(gridSp[1,1])
    
    ax1.plot(Time,Signal,'ro',label='Data')
    ax1.plot(Time,FitSolution,label='Regression',path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
    ax1.legend(loc=0)
    PlotStyle(ax1,'Fitted Model')
    
    ax2.plot(Residuals,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
    PlotStyle(ax2,'Residuals')
    
    ax3.plot(Noise,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
    PlotStyle(ax3,'Noise')
    
    plt.tight_layout()


fig2=plt.figure(5,figsize=(13,5))

ResidualsPlot(fig2,t_data,WhiteSignal,fitSolutionWhite,residualsWhite,WhiteNoise)

fig3=plt.figure(6,figsize=(13,5))

ResidualsPlot(fig3,t_data,PeriodicSignal,fitSolutionPeriodic,residualsPeriodic,PeriodicNoise)

fig4=plt.figure(7,figsize=(13,5))

ResidualsPlot(fig4,t_data,LinearSignal,fitSolutionLinear,residualsLinear,LinearNoise)
