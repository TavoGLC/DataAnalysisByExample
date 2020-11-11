#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License
Copyright (c) 2020 Octavio Gonzalez-Lugo 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
@author: Octavio Gonzalez-Lugo

"""

###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import scipy as sc
import pandas as pd 
import itertools as it 
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import preprocessing as pr
from sklearn import decomposition as dec

###############################################################################
# Plotting utility functions
###############################################################################

def PlotStyle(Axes): 
    """
    Parameters
    ----------
    Axes : Matplotlib axes object
        Applies a general style to the matplotlib object

    Returns
    -------
    None.
    """    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=13)
    Axes.yaxis.set_tick_params(labelsize=13)
    
def GetGridShape(TotalNumberOfElements):
    """
    Parameters
    ----------
     TotalNumberOfElements : int
        Total number of elements in the plot.

    Returns
    -------
    nrows : int
        number of rows in the plot.
    ncolumns : int
        number of columns in the plot.

    """
    numberOfUnique=TotalNumberOfElements
    squaredUnique=int(np.sqrt(numberOfUnique))
    
    if squaredUnique*squaredUnique==numberOfUnique:
        nrows,ncolumns=squaredUnique,squaredUnique
    elif squaredUnique*(squaredUnique+1)<numberOfUnique:
        nrows,ncolumns=squaredUnique+1,squaredUnique+1
    else:
        nrows,ncolumns=squaredUnique,squaredUnique+1
    
    return nrows,ncolumns

###############################################################################
# Data Utility Functions
###############################################################################

#Wrapper function to fill missing values 
def ToNumeric(val):
    try:
        return float(val)
    except ValueError:
        return 0

###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "Climate.csv"

###############################################################################
# Loading the data
###############################################################################

Data=pd.read_csv(DataDir)
Data['Date']=pd.to_datetime(Data['  FECHA'],format='%d/%m/%Y')
Data['DayOfYear']=Data['Date'].dt.dayofyear.apply(int)
Data['DayOfWeek']=Data['Date'].dt.dayofweek.apply(int)
Data['Month']=Data['Date'].dt.month.apply(int)
Data[' EVAP']=[ToNumeric(val) for val in Data[' EVAP']]

###############################################################################
# Time series custom features
###############################################################################

plt.figure()
plt.plot(Data.groupby("DayOfYear").max()["PRECIP"],label="Minimum")
plt.plot(Data.groupby("DayOfYear").mean()["PRECIP"],label="Mean")
plt.plot(Data.groupby("DayOfYear").min()["PRECIP"],label="Min")
plt.legend(loc=2)
ax=plt.gca()
ax.set_xlabel("Day Of Year")
ax.set_ylabel("Precipitation (mm)")
PlotStyle(ax)

plt.figure()
plt.plot(Data.groupby("DayOfYear").max()[' EVAP'],label="Minimum")
plt.plot(Data.groupby("DayOfYear").mean()[' EVAP'],label="Mean")
plt.plot(Data.groupby("DayOfYear").min()[' EVAP'],label="Min")
plt.legend(loc=2)
ax=plt.gca()
ax.set_xlabel("Day Of Year")
ax.set_ylabel("Evaporation (mm)")
PlotStyle(ax)

plt.figure()
plt.plot(Data.groupby("DayOfYear").max()["PRECIP"]-Data.groupby("DayOfYear").max()[' EVAP'],label="Minimum")
plt.plot(Data.groupby("DayOfYear").mean()["PRECIP"]-Data.groupby("DayOfYear").mean()[' EVAP'],label="Mean")
plt.plot(Data.groupby("DayOfYear").min()["PRECIP"]-Data.groupby("DayOfYear").min()[' EVAP'],label="Min")
plt.legend(loc=2)
ax=plt.gca()
ax.set_xlabel("Day Of Year")
ax.set_ylabel("Precipitation-Evaporation (mm)")
PlotStyle(ax)

plt.figure()
plt.plot(np.arange(366),np.cumsum(Data.groupby("DayOfYear").max()["PRECIP"]-Data.groupby("DayOfYear").max()[' EVAP']),label="Max")
plt.plot(np.arange(366),np.cumsum(Data.groupby("DayOfYear").mean()["PRECIP"]-Data.groupby("DayOfYear").mean()[' EVAP']),label="Mean")
plt.legend(loc=2)
ax=plt.gca()
ax.set_xlabel("Day Of Year")
ax.set_ylabel("Rainwater Harvest (mm)")
PlotStyle(ax)

###############################################################################
# Fixed Max case
###############################################################################

SurfaceArea=16
Demand=50

def Sizing(Size,Area,Demand,Rainfall,Evaporation):
    """
    Determinates the size of an open body of water that feeds a
    small reforestation operation.
    
    Parameters
    ----------
    Size : float
        Size of the body of water in liters.
    Area : float
        Area of the rainwater harvesting system used to feed the body of water.
    Demand : float
        Amount of water used.
    Rainfall : array-like
        Daily rainfall data.
    Evaporation : array-like
        Daily evaporation data.

    Returns
    -------
    int
        Days with enough water to satisfy the demand.

    """
    
    remaning=Size
    container=[]
    
    for rfall,evap in zip(Rainfall,Evaporation):
        
        remaning=remaning+Area*(rfall-evap)-Demand
        container.append(remaning)
    
    fitness=[1 for val in container if val>Demand]
    
    return sum(fitness)

Sizes=[Sizing(val,SurfaceArea,Demand,Data.groupby("DayOfYear").max()["PRECIP"],Data.groupby("DayOfYear").max()[' EVAP']) for val in range(0,10000,500)]

plt.figure()
plt.plot(np.arange(0,10000,500),Sizes)
ax=plt.gca()
ax.set_xlabel("Reserve Volume (l)")
ax.set_ylabel("Days With Water")
PlotStyle(ax)

###############################################################################
# Stochastic Case 
###############################################################################

PrecipRanges=np.array([list(val) for val in zip(Data.groupby("DayOfYear").min()["PRECIP"],Data.groupby("DayOfYear").max()["PRECIP"])])
EvapRanges=np.array([list(val) for val in zip(Data.groupby("DayOfYear").min()[' EVAP'],Data.groupby("DayOfYear").max()[' EVAP'])])

#Wrapper function to calculate a succesful size
def SuccessfulSize(Size,Area,Demand,Rainfall,Evaporation):
    if Sizing(Size,Area,Demand,Rainfall,Evaporation)==366:
        return 1
    else:
        return 0
    
def StochasticSizing(Size,Area,Demand,RainfallRange,EvaporationRange,iterations=200):
    """
    Stochastic simulation for different rainfall and evaporation profiles.
    Parameters
    ----------
    Size : float
        Size of the body of water in liters.
    Area : float
        Area of the rainwater harvesting system used to feed the body of water.
    Demand : float
        Amount of water used.
    Rainfall : array-like
        Daily rainfall data.
    Evaporation : array-like
        Daily evaporation data.
    iterations : int, optional
        Number of different Rainfall and Evaporation profiles to simulate. 
        The default is 200.

    Returns
    -------
    float
        Fraction of succesful simulations.

    """
    
    counter=0
    for _ in range(iterations):
        rfall=np.random.randint(RainfallRange[:,0],RainfallRange[:,1]+1)
        evap=np.random.randint(EvaporationRange[:,0],EvaporationRange[:,1]+1)
        Success=SuccessfulSize(Size,Area,Demand,rfall,evap)
        counter=counter+Success
    
    return counter/iterations

SSizes=[StochasticSizing(val,SurfaceArea,Demand,PrecipRanges,EvapRanges) for val in range(5000,10000,500)]

plt.figure()
plt.plot(np.arange(5000,10000,500),SSizes)
ax=plt.gca()
ax.set_xlabel("Reserve Volume (l)")
ax.set_ylabel("Succes Fraction")
PlotStyle(ax)
