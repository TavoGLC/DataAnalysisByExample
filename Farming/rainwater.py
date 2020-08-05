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

MonthsNames=['January','February','March','April','May','June','July','August','September','October','November','December']

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


def MakeCorrelationPanel(Data,Headers,PlotSize):
    '''
    Parameters
    ----------
    Data : pandas dataframe
        Contains the data set to be analyzed.
    Headers : list
        list of strings with the data headers inside Data.
    PlotSize : tuple
        contains the size of the generated plot.

    Returns
    -------
    None.

    '''
    nrows,ncolumns=GetGridShape(len(Headers)*len(Headers))
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    
    fig,axes=plt.subplots(nrows,ncolumns,figsize=PlotSize)
    counter=0
    
    for val in Headers:
        for sal in Headers:
            axes[subPlotIndexs[counter]].plot(Data[val],Data[sal],'bo')
            axes[subPlotIndexs[counter]].set_xlabel(val)
            axes[subPlotIndexs[counter]].set_ylabel(sal)
            counter=counter+1
    plt.tight_layout()
        
    [PlotStyle(axes[val]) for val in subPlotIndexs]        

def MakeMeanPlot(Data,figsize):
    '''
    Parameters
    ----------
    Data : pandas dataframe
        Contains the data set to be analyzed.
    figsize : tuple
        contains the size of the generated plot.

    Returns
    -------
    None.

    '''
    fig,axes=plt.subplots(1,3,figsize=figsize,subplot_kw=dict(polar=True))
    dataHeaders=['DayOfWeek','Month','Year']
    
    for ax,name in zip(axes,dataHeaders):
        values=Data.groupby(name).mean()["PRECIP"]
        xticks=values.keys()
        data=values.tolist()
        data+=data[:1]
        angles=np.linspace(0,2*np.pi,len(data))
        ax.plot(angles,data)
        ax.fill(angles,data,'b',alpha=0.1)
        ax.set_xticks(angles)
        if name=='Month':
            ax.set_xticklabels(MonthsNames)
        else:    
            ax.set_xticklabels(xticks)
    plt.tight_layout()

def MakeMonthlyPlot(Data,figsize):
    '''
    Parameters
    ----------
    Data : pandas dataframe
        Contains the data set to be analyzed.
    figsize : tuple
        contains the size of the generated plot.

    Returns
    -------
    None.

    '''
    fig,axes=plt.subplots(4,3,figsize=figsize,subplot_kw=dict(polar=True))
    dataValues=[1,2,3,4,5,6,7,8,9,10,11,12]
    flattenAxes=axes.ravel()
    
    for ax,name in zip(flattenAxes,dataValues):
        values=Data[Data["Month"]==name].groupby('DayOfWeek').mean()["PRECIP"]
        data=values.tolist()
        xticks=values.keys()
        data+=data[:1]
        angles=np.linspace(0,2*np.pi,len(data))
        ax.plot(angles,data)
        ax.fill(angles,data,'b',alpha=0.1)
        ax.set_xticks(angles)
        ax.set_xticklabels(xticks)
        ax.set_title(MonthsNames[name-1],loc='right')
        
    plt.tight_layout()
    
    
###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "Climate.csv"

Data=pd.read_csv(DataDir)

###############################################################################
# Time series custom features
###############################################################################

#Wrapper function for the days of the week
def ToDayOfWeek(DayNumber):
    Days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    return Days[DayNumber]

Data['Date']=pd.to_datetime(Data['  FECHA'],format='%d/%m/%Y')
Data['DayOfWeek']=Data['Date'].dt.dayofweek.apply(ToDayOfWeek)
Data['Month']=Data['Date'].dt.month.apply(int)
Data['Year']=Data['Date'].dt.year.apply(int)

climateHeaders=['PRECIP',' TMAX',' TMIN']

###############################################################################
# Data Visualization 
###############################################################################

MakeCorrelationPanel(Data,climateHeaders,(10,10))

MakeMeanPlot(Data,(10,5))

MakeMonthlyPlot(Data,(12,12))

###############################################################################
# Approximate harvesting
###############################################################################

AvaliableArea=300
MeanHarvestedWater=AvaliableArea*Data.groupby(['Year','Month']).sum().groupby('Month').mean()["PRECIP"]

lowEstimation=Data.groupby(['Year','Month']).sum().groupby('Month').mean()["PRECIP"]-Data.groupby(['Year','Month']).sum().groupby('Month').std()["PRECIP"]
LowEstimationHarvestedWater=[]

for val in lowEstimation:
    if val>0:
        LowEstimationHarvestedWater.append(val*AvaliableArea)
    else:
        LowEstimationHarvestedWater.append(0)
        
HighEstimation=AvaliableArea*(Data.groupby(['Year','Month']).sum().groupby('Month').mean()["PRECIP"]+Data.groupby(['Year','Month']).sum().groupby('Month').std()["PRECIP"])


plt.figure()
plt.plot(MeanHarvestedWater,label="Mean Forecast")
plt.plot(LowEstimationHarvestedWater,label="Low Forecast")
plt.plot(HighEstimation,label="High Forecast")
ax=plt.gca()
ax.set_xlabel("Months")
ax.set_ylabel("Harvested Water (liters)")
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
ax.set_xticklabels(MonthsNames,rotation=45)

ax.legend()
PlotStyle(ax)

###############################################################################
# Water reserve sizing
###############################################################################

MonthlyWaterConsumption=20000

def ReserveWaterState(Reserve,WaterWeights):
    '''
    Parameters
    ----------
    Reserve : int
        size of the water reserve.
    WaterWeights : list
        list with the approximate harvested water.

    Returns
    -------
    container : list
        contains the remaining water during each month of a water reserve of size 
        Reserve.

    '''
    remaning=Reserve
    container=[]

    for val in WaterWeights:
    
        consumption=remaning-MonthlyWaterConsumption
        if consumption+val>=Reserve:
            remaning=Reserve
        else:
            remaning=consumption+val
    
        container.append(remaning)
    
    return container

#Wrapper function to adjust negative values 
def ObjetiveFunction(Reserve,Harvested):
    
    Months=12-np.sum(np.sign(ReserveWaterState(Reserve,Harvested)))
    if Months>=12:
        return 12
    else:
        return Months

###############################################################################
# Water reserve sizing fixed use 
###############################################################################

ReserveSize=[k for k in range(1,100000,1000)]
meanEstimation=[ObjetiveFunction(val,MeanHarvestedWater) for val in ReserveSize]
lowEstimation=[ObjetiveFunction(val,LowEstimationHarvestedWater) for val in ReserveSize]
highEstimation=[ObjetiveFunction(val,HighEstimation) for val in ReserveSize]

plt.figure()
plt.plot(meanEstimation,label="Mean Forecast")
plt.plot(lowEstimation,label="Low Forecast")
plt.plot(highEstimation,label="High Forecast")
ax=plt.gca()
ax.set_xlabel("Reserve Size (m3)")
ax.set_ylabel("Months without water")
ax.legend()
PlotStyle(ax)

###############################################################################
# Water reserve sizing variable use 
###############################################################################

MonthlyUse=[3000,3000,6000,9000,12000,15000,20000,20000,20000,20000,20000,20000]

def VariableReserveWaterState(Reserve,WaterWeights,UsageWeights):
    '''
    Parameters
    ----------
    Reserve : int
        size of the water reserve.
    WaterWeights : list
        list with the approximate harvested water.
    UsageWeights : list
        list with the approximate monthly water usage.

    Returns
    -------
    container : list
        contains the remaining water during each month of a water reserve of size 
        Reserve.

    '''
    remaning=Reserve
    container=[]

    for val,sal in zip(WaterWeights,UsageWeights):
    
        consumption=remaning-sal
        if consumption+val>=Reserve:
            remaning=Reserve
        else:
            remaning=consumption+val
    
        container.append(remaning)
    
    return container

#Wrapper function to adjust for negative values
def ObjetiveFunctionVariable(Reserve,Harvested,Usage):
    
    Months=12-np.sum(np.sign(VariableReserveWaterState(Reserve,Harvested,Usage)))
    if Months>=12:
        return 12
    else:
        return Months
    
meanEstimationV=[ObjetiveFunctionVariable(val,MeanHarvestedWater,MonthlyUse) for val in ReserveSize]
lowEstimationV=[ObjetiveFunctionVariable(val,LowEstimationHarvestedWater,MonthlyUse) for val in ReserveSize]
highEstimationV=[ObjetiveFunctionVariable(val,HighEstimation,MonthlyUse) for val in ReserveSize]

plt.figure()
plt.plot(meanEstimationV,label="Mean Forecast")
plt.plot(lowEstimationV,label="Low Forecast")
plt.plot(highEstimationV,label="High Forecast")
ax=plt.gca()
ax.set_xlabel("Reserve Size (m3)")
ax.set_ylabel("Months without water")
ax.legend()
PlotStyle(ax)
