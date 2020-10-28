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
# Automated Exploratory Analysis Utility Functions
###############################################################################

def FeatureClassification(Data):
    """
    Simple function to classify the kind of feature in the dataset
    Parameters
    ----------
    Data : pandas data frame
        Data to be Analized.

    Returns
    -------
    numerical : list
        List with the names of the numerical features.
    categorical : list
        List with the names of the categorical features.
    timeseries : TYPE
        List with the names of the time series features.

    """
    
    ColumnHeader=list(Data)
    numerical=[]
    categorical=[]
    timeseries=[]
    
    for head in ColumnHeader:
        
        try:
            float(Data[head][0])
            numerical.append(head)
        except ValueError:
            try:
                pd.to_datetime(Data[head])
                timeseries.append(head)
            except Exception:
                categorical.append(head)
        
    return numerical,categorical,timeseries

def UniqueToDictionary(UniqueElements):
    """
    From a list or array of elements returns a dictionary 
    that maps element to index value. 
    Parameters
    ----------
    UniqueElements : list, array
        List with the unique elements in a dataset 

    Returns
    -------
    localDictionary : python dictionary 
        Maps from element to index value
    """
    
    localDictionary={}
    nElements=len(UniqueElements)
    
    for k in range(nElements):
        localDictionary[UniqueElements[k]]=k
        
    return localDictionary

def CountUniqueElements(UniqueElements,TargetArray):
    """
    Counts the frequency of the unique elements in the target array 
    Parameters
    ----------
    UniqueElements : list, array 
        List with the unique elements in a dataset 
    TargetArray : list, array 
        Data set to be analyzed 

    Returns
    -------
    localCounter : list
        List with the frequencies of the unique lements in the target array 

    """
    nUnique=len(UniqueElements)
    localCounter=[0 for k in range(nUnique)]
    UniqueDictionary=UniqueToDictionary(UniqueElements)
    
    for val in TargetArray:
        try:
            localPosition=UniqueDictionary[val]
            localCounter[localPosition]=localCounter[localPosition]+1
        except KeyError:
            pass
    return localCounter

def FormatData(nUniqueA,nUniqueB,Data):
    """
    Helper function to format the co frequency data
    Parameters
    ----------
    nUniqueA : int
        Number of unique labels .
    nUniqueB : int
        Number of unique labels.
    Data : array-like
        Frequencies of each co ocurrence.

    Returns
    -------
    container : array
        Formated Data.

    """
    
    container=np.zeros((nUniqueA,nUniqueB))
    index=[(val,sal) for val in range(nUniqueA) for sal in range(nUniqueB)]
    k=0
    
    for inx in index:
        container[inx]=Data[k]
        k=k+1
    
    return container
    
#Wrapper function to create all the visualizations
def MakeVisualizations(Headers,Data,Functions):
    for val in Functions:
        val(Headers,Data)
        
###############################################################################
# Automated Exploratory Analysis Visualization Functions Numerical Features
###############################################################################

def MakeSimplePlotPanel(Headers,Data):
    """
    Simple visualization with each of the numerical features being plotted
    Parameters
    ----------
    Headers : list
        HEader labels of the numerical features.
    Data : pandas data frame
        Data to be analized.

    Returns
    -------
    None.

    """
    
    nHead=len(Headers)
    nrows,ncolumns=GetGridShape(nHead)
    
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    headColors=[plt.cm.cividis(val) for val in np.linspace(0,1,num=nHead)]
    
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,15))
    
    for val in enumerate(Headers):
        k,hd=val
        axes[subPlotIndexs[k]].plot(Data[hd],'o',color=headColors[k],alpha=0.1,label=hd)
        axes[subPlotIndexs[k]].legend(loc=1)
    
    for inx in subPlotIndexs:
        PlotStyle(axes[inx])

def MakeDistributionPlot(Headers,Data):
    """
    Distribution of each of the numerical features, adds a probability plot 
    to check for normality.
    
    Parameters
    ----------
    Headers : list
        HEader labels of the numerical features.
    Data : pandas data frame
        Data to be analized.

    Returns
    -------
    None.

    """
    
    nHead=len(Headers)
    nrows,ncolumns=GetGridShape(2*nHead)
    
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    headColors=[plt.cm.cividis(val) for val in np.linspace(0,1,num=nHead)]
    
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,15))
    probPlot=False
    k=0
    ck=0
    
    for hd in Headers:
        
        inx=subPlotIndexs[k]
        vals,reg=stats.probplot(Data[hd])
        axes[inx].plot(vals[0],vals[1],'o',color=headColors[ck],label=hd)
        axes[inx].set_xlabel("Probability Plot")
        axes[inx].legend(loc=1)
        k=k+1
        
        inx=subPlotIndexs[k]
        axes[inx].hist(Data[hd],color=headColors[ck],bins=75,label=hd)
        axes[inx].set_xlabel("Distribution")
        axes[inx].legend(loc=1)
        k=k+1
        ck=ck+1
            
    for inx in subPlotIndexs:
        PlotStyle(axes[inx])
    
    plt.tight_layout()

def MakeReducedPlot(Headers,Data):
    """
    Visualization of the PCA analysis over the numerical features of the dataset.
    Parameters
    ----------
    Headers : list
        Header labels of the numerical features.
    Data : pandas data frame
        Data to be analized.
    Returns
    -------
    None.

    """
    
    minmax=pr.MinMaxScaler()
    minmax.fit(Data[Headers])
    standard=pr.StandardScaler()
    standard.fit(Data[Headers])
    
    minData=minmax.transform(Data[Headers])
    stdData=standard.transform(Data[Headers])
    
    PCAmin=dec.PCA(n_components=2)
    PCAmin.fit(minData)
    redMin=PCAmin.transform(minData)
    
    PCAstd=dec.PCA(n_components=2)
    PCAstd.fit(stdData)
    redStd=PCAstd.transform(stdData)
    
    dotColors=[plt.cm.cividis(val) for val in np.linspace(0,1,num=len(redMin[:,0]))]
    
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    
    axes[0].scatter(redMin[:,0],redMin[:,1],marker='o',c=dotColors,alpha=0.5)
    axes[0].set_xlabel("Min Max Scaler")
    PlotStyle(axes[0])
    axes[1].scatter(redStd[:,0],redStd[:,1],marker='o',c=dotColors,alpha=0.5)
    axes[1].set_xlabel("Standard Scaler")
    PlotStyle(axes[1])
    
def MakeCorrelationPlot(Headers,Data):
    """
    Visualization of the correlation between the numerical features
    Parameters
    ----------
    Headers : list
        Header labels of the numerical features.
    Data : pandas data frame
        Data to be analized.

    Returns
    -------
    None.

    """
    correlation=Data[Headers].corr()
    
    fig,axs=plt.subplots(figsize=(10,10))
    cim=axs.imshow(correlation,cmap="cividis")
    fig.colorbar(cim,shrink=0.8)
    
    axs.set_xticks(np.arange(len(Headers)))
    axs.set_yticks(np.arange(len(Headers)))
    axs.set_xticklabels(Headers)
    axs.set_yticklabels(Headers)
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.xaxis.set_tick_params(labelsize=13,rotation=90)
    axs.yaxis.set_tick_params(labelsize=13)

###############################################################################
# Automated Exploratory Analysis Visualization Functions Categorical Features
###############################################################################
    
def MakeFrequencyPlot(Headers,Data):
    """
    Visualization of the frequency of the differnt categorical features

    Parameters
    ----------
    Headers : list
        HEader labels of the categorical features.
    Data : pandas data frame
        Data to be analized.
    Returns
    -------
    None.

    """
    
    nHead=len(Headers)
    nrows,ncolumns=GetGridShape(nHead)
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    k=0
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(10,10))
    for hd in Headers:
        
        localData=Data[hd]
        unique=localData.unique()
        inx=subPlotIndexs[k]
        
        freqs=CountUniqueElements(unique,localData)
        barColors=[plt.cm.cividis(val) for val in np.linspace(0,1,num=len(unique))]
        Xaxis=np.arange(len(unique))
        axes[inx].bar(Xaxis,freqs,color=barColors,edgecolor=barColors)
        axes[inx].set_xticks(Xaxis)
        axes[inx].set_xticklabels(unique,rotation=85)
        axes[inx].set_xlabel(hd)
        k=k+1
    
    for inx in subPlotIndexs:
        PlotStyle(axes[inx])
    
    plt.tight_layout()
    
def MakeCoFrequencyPlot(Headers,Data,MaxPanels=25):
    """
    Visualization of the co ocurrence of the categorical features
    Parameters
    ----------
    Headers : list
        HEader labels of the numerical features.
    Data : pandas data frame
        Data to be analized.
    MaxPanels : int, optional
        Max number of plots in the visualization. The default is 25.

    Returns
    -------
    Indexs : TYPE
        DESCRIPTION.

    """
    
    nHead=len(Headers)
    nCombs=sc.special.binom(nHead,2)
    indexCombs=it.combinations(range(nHead),2)
    
    
    if nCombs<=MaxPanels:
        
        Indexs=list(indexCombs)
        nrows,ncolumns=GetGridShape(nCombs)
        
    else:
        
        Indexs=[]
        nrows,ncolumns=GetGridShape(MaxPanels)
        for k in range(MaxPanels):
            Indexs.append(indexCombs.__next__())
            
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,15))
    k=0
    
    for inx in Indexs:
        
        FeatA,FeatB=Headers[inx[0]],Headers[inx[1]]
        UniqueA=Data[FeatA].unique()
        UniqueB=Data[FeatB].unique()
        pinx=subPlotIndexs[k]
        
        localToken=[val+sal for val in UniqueA for sal in UniqueB]
        localArray=[val+sal for val,sal in zip(Data[FeatA],Data[FeatB])]
        localCounts=CountUniqueElements(localToken,localArray)
        counts=FormatData(UniqueA.size,UniqueB.size,localCounts)
        
        axes[pinx].imshow(counts,cmap="cividis")
        axes[pinx].set_xticks(np.arange(len(UniqueB)))
        axes[pinx].set_yticks(np.arange(len(UniqueA)))
        axes[pinx].set_xticklabels(UniqueB)
        axes[pinx].set_yticklabels(UniqueA)
        axes[pinx].spines['top'].set_visible(False)
        axes[pinx].spines['bottom'].set_visible(False)
        axes[pinx].spines['left'].set_visible(False)
        axes[pinx].spines['right'].set_visible(False)
        axes[pinx].xaxis.set_tick_params(labelsize=13,rotation=90)
        axes[pinx].yaxis.set_tick_params(labelsize=13)
        axes[pinx].set_xlabel(FeatB,fontsize=15)
        axes[pinx].set_ylabel(FeatA,fontsize=15)
        plt.tight_layout()
        
        k=k+1
    
    for inx in subPlotIndexs:
        
        axes[inx].spines['top'].set_visible(False)
        axes[inx].spines['bottom'].set_visible(False)
        axes[inx].spines['left'].set_visible(False)
        axes[inx].spines['right'].set_visible(False)
        
###############################################################################
# Automated Exploratory Analysis Visualization Functions Categorical Features
###############################################################################

NumericalFunctions=[MakeSimplePlotPanel,MakeDistributionPlot,MakeCorrelationPlot,MakeReducedPlot]
CategoricalFunctions=[MakeFrequencyPlot,MakeCoFrequencyPlot]

#Wrapper function for the EDA
def GetEDA(Data):
    
    Numerical,Categorical,TimeSeries=FeatureClassification(Data)
    MakeVisualizations(Numerical,Data,NumericalFunctions)
    MakeVisualizations(Numerical,Data,CategoricalFunctions)

###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "SeoulBikeData.csv"

Data=pd.read_csv(DataDir)
GetEDA(Data)
