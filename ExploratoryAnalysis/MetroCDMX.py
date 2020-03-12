#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:23:40 2020

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
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing as pr

from tsfresh.feature_extraction import feature_calculators as fc

###############################################################################
# Data Location
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory+"MetroCDMX.csv"
Data=pd.read_csv(DataDir)
TimeSeries=pd.to_datetime(Data['Fecha'],format='%Y-%m-%d')

###############################################################################
# Plot Functions
###############################################################################

def PlotStyle(Axes): 
    """
    General style used in all the plots 
    
    Axes -> matplotlib axes object
    """ 
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=13)
    Axes.yaxis.set_tick_params(labelsize=13)


def GetGridShape(UniqueVals):
    """
    Calculates the number of rows and columns for a subplot 
    
    UniqueVals -> iterable with the number of unique elements
                  to be in the plot
    """ 
    numberOfUnique=len(UniqueVals)
    squaredUnique=int(np.sqrt(numberOfUnique))
    
    if squaredUnique*squaredUnique==numberOfUnique:
        nrows,ncolumns=squaredUnique,squaredUnique
    elif squaredUnique*(squaredUnique+1)<numberOfUnique:
        nrows,ncolumns=squaredUnique+1,squaredUnique+1
    else:
        nrows,ncolumns=squaredUnique,squaredUnique+1
        
    return nrows,ncolumns

def MakeBarPlot(Data,Ylabels=True):
    """
    Makes a bar plot of the data
    
    Data    -> pandas dataframe to be plotted 
    Ylabels -> Bool optional, used to include the y-axis labels
               default True
    """
    Xaxis=np.arange(len(Data))
    plt.bar(Xaxis,Data)
    
    if Ylabels==True:
        axis=plt.gca()
        axis.set_xticks(Xaxis)
        axis.set_xticklabels(Data.keys(),rotation=70)
        PlotStyle(axis)
    else:
        axis=plt.gca()
        axis.set_xticks([])
        PlotStyle(axis)

def MakeHistogramGrid(Data,Label):
    """
    Makes a subplot of histograms
    
    Data  -> Pandas dataframe
    Label -> Column in the pandas dataframe to be analyzed 
    """
    labelSum=Data.groupby([Label])['Afluencia'].sum()
    dataOrder=labelSum.argsort()
    colors=[plt.cm.viridis(val) for val in np.linspace(0,1,num=dataOrder.size)]

    UniqueVals=Data[Label].unique()
    nrows,ncolumns=GetGridShape(UniqueVals)
    
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(12,12),sharex=True,sharey=True)
    counter=0

    for val in UniqueVals:
        currentColor=colors[dataOrder[val]]
        axes[subPlotIndexs[counter]].hist(Data[Data[Label]==val]['Afluencia'],color=currentColor,label=str(val)+' (Rank -> '+str(dataOrder[val])+')')
        axes[subPlotIndexs[counter]].legend()
        axes[subPlotIndexs[counter]].set_xlabel('Daily Users')
        axes[subPlotIndexs[counter]].set_ylabel('Frequency')
        counter=counter+1
    
    for indx in subPlotIndexs:
        PlotStyle(axes[indx])

def MakeClustersPlot(ClusterData,axes):
    """
    Makes a scatter plot that uses different colors for each 
    cluster
    
    ClusterData -> Data obtained from MakeClusterAnalysis
    axes        -> Matplotlib axes used for the plot 
    """ 
    TransformedData,ClusterLabels,RowNames=ClusterData
    UniqueClusters=np.unique(ClusterLabels)
    colors=[plt.cm.viridis(val,alpha=0.5) for val in np.linspace(0,1,num=UniqueClusters.size)]
    
    for val in UniqueClusters:
    
        Xdata=[TransformedData[k,0] for k in range(len(TransformedData)) if ClusterLabels[k]==val]
        Ydata=[TransformedData[k,1] for k in range(len(TransformedData)) if ClusterLabels[k]==val]
        axes.plot(Xdata,Ydata,'o',color=colors[val],label=str(len(Xdata))+' Elements' )
        axes.legend(loc=4)


###############################################################################
# Analysis Functions
###############################################################################

def ToDayOfWeek(DayNumber):
    """
    Wrapper function to change the day of the week to name
    
    DayNumber -> Pandas day of the week 
    """
    Days=['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo']
    return Days[DayNumber]

def MakeLabelFeatures(Data,Label): 
    """
    Calculates a series of features from the data
    
    Data  -> Pandas dataframe
    Label -> Column in the pandas dataframe to be analyzed 
    """ 
    UniqueVals=Data[Label].unique()
    featuresContainer=[] 
    featuresList=[fc.count_above_mean,fc.count_below_mean,
                  fc.longest_strike_above_mean,fc.longest_strike_below_mean,
                  fc.mean]
    
    for val in UniqueVals:
        
        localData=Data[Data[Label]==val]['Afluencia']
        localContainer=[]
        
        for feature in featuresList:
            localContainer.append(feature(localData))
            
        featuresContainer.append(localContainer)
        
    return np.array(featuresContainer),UniqueVals

def MakeClusterAnalysis(Data,Label): 
    """
    Performs dimensionality reduction and cluster analysis from
    the features matrix 
    
    Data  -> Pandas dataframe
    Label -> Column in the pandas dataframe to be analyzed 
    """ 
    StationsFeatures,RowLabels=MakeLabelFeatures(Data,Label)
    
    Scaler=pr.StandardScaler()
    Scaler.fit(StationsFeatures)
    StationsFeatures=Scaler.transform(StationsFeatures)

    Method=PCA(n_components=2)
    Method.fit(StationsFeatures)
    TransformedData=Method.transform(StationsFeatures)

    ClusterData=KMeans(n_clusters=3,random_state=24).fit(TransformedData)
    ClusterLabels=ClusterData.labels_
    
    return TransformedData,ClusterLabels,RowLabels

def MakeClustersGrid(Data,Label):
    """
    Makes a grid of cluster plots
 
    Data  -> Pandas dataframe
    Label -> Column in the pandas dataframe to be analyzed 
    """ 
    UniqueVals=Data[Label].unique()
    nrows,ncolumns=GetGridShape(UniqueVals)

    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    counter=0
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(12,12))
    
    for val in UniqueVals:
        localData=Data[Data[Label]==val]
        MakeClustersPlot(MakeClusterAnalysis(localData,'Estacion'),axes[subPlotIndexs[counter]])
        axes[subPlotIndexs[counter]].set_title(str(val))
        counter=counter+1
    
    for indx in subPlotIndexs:
        PlotStyle(axes[indx])

###############################################################################
# Data Characteristics
###############################################################################

print(Data.shape)
print(list(Data))

print(Data.isna().sum())

Data['Semana']=TimeSeries.dt.dayofweek.apply(ToDayOfWeek)

plt.figure(1)
linesSize=Data.groupby(['Linea']).size()
MakeBarPlot(linesSize)

plt.figure(2)
yearSize=Data.groupby(['Año']).size()
MakeBarPlot(yearSize)

plt.figure(3)
monthSize=Data.groupby(['Mes']).size()
MakeBarPlot(monthSize)

plt.figure(4)
monthSize=Data.groupby(['Semana']).size()
MakeBarPlot(monthSize)

print('Number of metro stations -> ' + str(Data['Estacion'].unique().size))
print('Number of metro lines -> ' + str(Data['Linea'].unique().size))
print('Years of data -> ' +str(Data['Año'].max()-Data['Año'].min()))
print('Mean Daily afluence -> ' +str(Data['Afluencia'].mean()))

###############################################################################
# Metro afluence
###############################################################################

MakeHistogramGrid(Data,'Año')

MakeHistogramGrid(Data,'Mes')

MakeHistogramGrid(Data,'Semana')

MakeHistogramGrid(Data,'Linea')

###############################################################################
# Data Clustering
###############################################################################

TransformedData,Clabel,Rows=MakeClusterAnalysis(Data,'Estacion')

fig,axes=plt.subplots(2,2,figsize=(12,12))

subPlotIndexs=[(j,k) for j in range(2) for k in range(2)]

MakeClustersPlot([TransformedData,Clabel,Rows],axes[subPlotIndexs[0]])

UniqueClusters=np.unique(Clabel)
for val in UniqueClusters:
    Stations=[Rows[k] for k in range(len(Rows)) if Clabel[k]==val]
    for station in Stations:
        axes[subPlotIndexs[val+1]].hist(Data[Data['Estacion']==station]['Afluencia'],bins=30,alpha=0.5)
        axes[subPlotIndexs[val+1]].set_xlabel('Daily Users')
        axes[subPlotIndexs[val+1]].set_ylabel('Frequency')

for indx in subPlotIndexs:
    PlotStyle(axes[indx])
    
###############################################################################
# Clustering by temporal features
###############################################################################

MakeClustersGrid(Data,'Mes')

MakeClustersGrid(Data,'Semana')
