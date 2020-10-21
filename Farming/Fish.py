#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
tCopyright (c) 2020 Octavio Gonzalez-Lugo 
o use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
import pandas as pd 
import matplotlib.pyplot as plt

###############################################################################
# Plotting functions
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
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "Climate.csv"

Data=pd.read_csv(DataDir)

Data['Date']=pd.to_datetime(Data['  FECHA'],format='%d/%m/%Y')
Data['DayOfYear']=Data['Date'].dt.dayofyear.apply(int)

MeanData=Data.groupby("DayOfYear").mean()
StdData=Data.groupby("DayOfYear").std()

###############################################################################
# Time series custom features
###############################################################################

plt.figure()
plt.plot(MeanData['PRECIP'],label="Mean Scenario")
plt.plot(MeanData['PRECIP']+StdData['PRECIP'],label="Up Scenario")
plt.plot([max([val,0]) for val in MeanData['PRECIP']-StdData['PRECIP']],label="Low Scenario")
plt.legend(loc=0)
ax=plt.gca()
ax.set_xlabel("Time (days)")
ax.set_ylabel("Rain water fall (mm)")
PlotStyle(ax)


###############################################################################
# Single Compartment Model
###############################################################################

def SingleDemandLevel(Size,Demand,Harvested):
    """
    Discrete model of the level evolution trough time of a fish tank 
    used for fertirrigation.
    
    Parameters
    ----------
    Size : float
        Size of the fish tank in liters.
    Demand : float,array-like
        Dayly crop irrigation demand in liters. If float a fixed demand, else an 
        array that contains the daily crop demand.
    Harvested : array-like
        Daily rain water harvested for the fish tank.

    Returns
    -------
    container : list
        Contains the fish tank level evolution trough time.

    """
    container=[Size]
    
    if type(Demand)==int or type(Demand)==float:
        Static=True
    else:
        Static=False
    
    if Static:
        
        lastSize=Size-Demand
        for val in Harvested:
            container.append(lastSize)
            lastSize=min([lastSize-Demand+val,Size])
    else:
        lastSize=Size-Demand[0]
        for val,sal in zip(Harvested,Demand):
            container.append(lastSize)
            lastSize=min([lastSize-sal+val,Size])
        
    return container

###############################################################################
# Single Compartement Parameters
###############################################################################

WaterDemand=650
FishTankSize=30000
TotalArea=380
LinearDemand=np.linspace(100,1.2*WaterDemand,num=len(MeanData))

###############################################################################
# Single Compartment Visualization
###############################################################################

MeanEstimation=SingleDemandLevel(FishTankSize,WaterDemand,TotalArea*MeanData["PRECIP"])
UpperEstimation=SingleDemandLevel(FishTankSize,WaterDemand,TotalArea*(MeanData["PRECIP"]+StdData["PRECIP"]))

plt.figure()
plt.plot(MeanEstimation,label="Fish Tank Level (Mean Scenario)")
plt.plot(UpperEstimation,label="Fish Tank Level (Up Scenario) ")
plt.legend(loc=3)
ax=plt.gca()
ax.set_xlabel("Time (days)")
ax.set_ylabel("Water Level (L)")
PlotStyle(ax)

LinearMeanEstimation=SingleDemandLevel(FishTankSize,LinearDemand,TotalArea*MeanData["PRECIP"])
LinearUpperEstimation=SingleDemandLevel(FishTankSize,LinearDemand,TotalArea*(MeanData["PRECIP"]+StdData["PRECIP"]))

plt.figure()
plt.plot(LinearMeanEstimation,label="Fish Tank Level (Mean Scenario)")
plt.plot(LinearUpperEstimation,label="Fish Tank Level (Up Scenario) ")
plt.legend(loc=3)
ax=plt.gca()
ax.set_xlabel("Time (days)")
ax.set_ylabel("Water Level (L)")
PlotStyle(ax)

###############################################################################
# Dual Compartment Model
###############################################################################

def DualDemandLevel(Sizes,Demand,Areas,RainData,drop=0.1):
    """
    Discrete model of the evolution of two water sources. A fish tank used for fertirrigation 
    and a reserve tank used to replenish the fish tank. 

    Parameters
    ----------
    Sizes : list
        Sizes of the fish tank and the reserve tank in liters .
    Demand : float,array-like
        Dayly crop irrigation demand in liters. If float a fixed demand, else an 
        array that contains the daily crop demand.
    Areas: list
        Cointains the sizes of the rain water harvesting systems that feeds the fish tank and 
        the reserve tank
    RainData : array-like
        Contains the daily rain water fall in the region.
    drop: float (0,1)
        Max level drop at the fish tank before being replenished by the reserve tank

    Returns
    -------
    containerA : list
        Fish tank level evolution.
    containerB : list
        Reserve tank level evolution.

    """
    SizeA,SizeB=Sizes
    AreaA,AreaB=Areas
    
    containerA=[SizeA]
    containerB=[SizeB]
    HarvestedA,HarvestedB=AreaA*RainData,AreaB*RainData
    
    if type(Demand)==int or type(Demand)==float:
        Static=True
    else:
        Static=False
    
    if Static:    
        lastSizeA,lastSizeB=SizeA-Demand,SizeB
        Index=[ val for val in zip(HarvestedA,HarvestedB)]
    else:
        lastSizeA,lastSizeB=SizeA-Demand[0],SizeB
        Index=[val for val in zip(HarvestedA,HarvestedB,Demand)]
        
    for inx in Index:
        
        if Static:
            val,sal=inx
            dal=Demand
        else:
            val,sal,dal=inx
        
        containerA.append(lastSizeA)
        lastSizeA=min([lastSizeA-dal+val,SizeA])
            
        if lastSizeA<SizeA*(1-drop):
            demandB=SizeA-lastSizeA
            if demandB>containerB[-1]:
                lastSizeB=containerB[-1]
                containerB.append(lastSizeB+sal)
            else:
                lastSizeB=min([lastSizeB-demandB+sal,SizeB])
                lastSizeA=SizeA
                containerB.append(lastSizeB)
                
        else:
            lastSizeB=min([containerB[-1]+sal,SizeB])
            containerB.append(lastSizeB)
            
    return containerA,containerB

###############################################################################
# Dual Compartment Parameters
###############################################################################

TanksSize=[30000,45000]
Areas=[80,300]
LinearDemand=np.linspace(100,1.2*WaterDemand,num=len(MeanData))

MeanEstimationA,MeanEstimationB=DualDemandLevel(TanksSize,WaterDemand,Areas,MeanData["PRECIP"])
UpperEstimationA,UpperEstimationB=DualDemandLevel(TanksSize,WaterDemand,Areas,MeanData["PRECIP"]+StdData["PRECIP"])

###############################################################################
# Dual Compartment Visualization
###############################################################################

plt.figure()
plt.plot(MeanEstimationA,label="Fish Tank Level (Mean Scenario) ")
plt.plot(MeanEstimationB,label="Reserve Tank Level (Mean Scenario) ")
plt.legend(loc=3)
ax=plt.gca()
ax.set_xlabel("Time (days)")
ax.set_ylabel("Water Level (L)")
PlotStyle(ax)


plt.figure()
plt.plot(UpperEstimationA,label="Fish Tank Level (Up Scenario) ")
plt.plot(UpperEstimationB,label="Reserve Tank Level (Up Scenario) ")
plt.legend(loc=3)
ax=plt.gca()
ax.set_xlabel("Time (days)")
ax.set_ylabel("Water Level (L)")
ax.set_ylim(20000)
PlotStyle(ax)

###############################################################################
# Stochastic Case 
###############################################################################

Ranges=np.array([list(val) for val in zip(Data.groupby("DayOfYear")["PRECIP"].min(),Data.groupby("DayOfYear")["PRECIP"].max()) ])

StochasticA,StochasticB=DualDemandLevel(TanksSize,WaterDemand,Areas,np.random.randint(Ranges[:,0],Ranges[:,1]+1))

plt.figure()
plt.plot(StochasticA,label="Fish Tank Level (Stochastic Scenario) ")
plt.plot(StochasticB,label="Reserve Tank Level (Stochastic Scenario) ")
plt.legend(loc=3)
ax=plt.gca()
ax.set_xlabel("Time (days)")
ax.set_ylabel("Water Level (L)")
ax.set_ylim(20000)
PlotStyle(ax)

###############################################################################
# Montecarlo Simulation Stochastic Case 
###############################################################################

def DualDemandMontecarlo(Iterations,Drop,TanksSize,WaterDemand,Areas):
    """
    Montecarlo simulation to evaluate how likely is the max drop in the reserve tank  

    Parameters
    ----------
    Iterations: int
        Number of interations in the simulation 
    Drop: float (0,1)
        Max level drop at the reserve tank to be considered succesful
    TanksSizes : list
        Sizes of the fish tank and the reserve tank in liters .
    WaterDemand : float,array-like
        Dayly crop irrigation demand in liters. If float a fixed demand, else an 
        array that contains the daily crop demand.
    Areas: list
        Cointains the sizes of the rain water harvesting systems that feeds the fish tank and 
        the reserve tank


    Returns
    -------
    list
        Success fraction in the simulation
    """
    container=[]
    for k in range(Iterations):
        
        _,localB=DualDemandLevel(TanksSize,WaterDemand,Areas,np.random.randint(Ranges[:,0],Ranges[:,1]+1))
        
        if min(localB)>(1-Drop)*TanksSize[1]:
            
            container.append(1)
        else:
            container.append(0)
            
    return 100*(sum(container)/Iterations)

###############################################################################
# Montecarlo Simulation Stochastic Case Parameters
###############################################################################

MaxIterations=5000
DropValues=np.linspace(0.05,0.5,num=11)
SuccesPercentages=[DualDemandMontecarlo(MaxIterations,val,TanksSize,WaterDemand,Areas) for val in DropValues]
SuccesPercentages15=[DualDemandMontecarlo(MaxIterations,val,TanksSize,1.5*WaterDemand,Areas) for val in DropValues]
SuccesPercentages2=[DualDemandMontecarlo(MaxIterations,val,TanksSize,2*WaterDemand,Areas) for val in DropValues]

plt.figure()
plt.plot(DropValues,SuccesPercentages,label="1 Water Demand")
plt.plot(DropValues,SuccesPercentages15,label="1.5 x Water Demand")
plt.plot(DropValues,SuccesPercentages2,label="2 x Water Demand")
plt.legend(loc=0)
ax=plt.gca()
ax.set_xlabel("Reserve drop(%)")
ax.set_ylabel("Success Fraction")
PlotStyle(ax)

