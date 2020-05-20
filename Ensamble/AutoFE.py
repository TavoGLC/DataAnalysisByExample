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

import copy
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn import preprocessing as pr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "data_2genre.csv"

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

def MakeCategoricalPlot(Data,CategoricalFeature,PlotSize):
    """
    Parameters
    ----------
    Data: pandas data frame 
        Containes the data for the plot 
    CategoricalFeature : String 
        Categorical value to be analyzed in the dataset 
    PlotSize : tuple
        Size of the plot 

    Returns
    -------
    None.

    """
    localUniqueVals=Data[CategoricalFeature].unique()
    localCounts=CountUniqueElements(localUniqueVals,Data[CategoricalFeature])
    Xaxis=np.arange(len(localUniqueVals))
    localTicksNames=[str(val) for val in localUniqueVals]
    plt.figure(figsize=PlotSize)
    plt.bar(Xaxis,localCounts)
    axis=plt.gca()
    axis.set_xticks(Xaxis)
    axis.set_xticklabels(localTicksNames,rotation=85)
    axis.set_xlabel(CategoricalFeature,fontsize=14)
    PlotStyle(axis)


def GetGridShape(UniqueVals):
    """
    Parameters
    ----------
    UniqueVals : list
        List with the elements of the plot.

    Returns
    -------
    nrows : int
        number of rows in the plot.
    ncolumns : int
        number of columns in the plot.

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

def MakeHistogramGrid(Data,Labels):
    """
    Parameters
    ----------
    Data : pandas dataframe
        data for the histogram plot.
    Labels : list
        list of strings with the headers of Data.

    Returns
    -------
    None.

    """
    
    colors=[plt.cm.viridis(val) for val in np.linspace(0,1,num=len(Labels))]
    nrows,ncolumns=GetGridShape(Labels)
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,15),sharey=True)
    counter=0

    for val in Labels:
        currentColor=colors[counter]
        axes[subPlotIndexs[counter]].hist(Data[val],color=currentColor)
        axes[subPlotIndexs[counter]].set_xlabel(str(val))
        axes[subPlotIndexs[counter]].set_ylabel('Frequency')
        counter=counter+1
    
    for indx in subPlotIndexs:
        PlotStyle(axes[indx])
    plt.tight_layout()
    
def MakeCorrelationPlot(Data,Labels):
    """
    Parameters
    ----------
    Data : Pandas dataframe
        Data to be analized.
    Labels : list
        List with the data headers for correlation analysis.

    Returns
    -------
    None.

    """
    correlation=Data[Labels].corr()
    
    plt.figure(figsize=(10,10))
    plt.imshow(correlation)
    ax=plt.gca()
    ax.set_xticks(np.arange(len(Labels)))
    ax.set_yticks(np.arange(len(Labels)))
    ax.set_xticklabels(Labels)
    ax.set_yticklabels(Labels)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=13,rotation=90)
    ax.yaxis.set_tick_params(labelsize=13)
    
###############################################################################
# Data Analysis Functions
###############################################################################

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


###############################################################################
#Exploratory analysis
###############################################################################

Data=pd.read_csv(DataDir)
Headers=list(Data)

FeatureHeaders=list(set(Headers)-set(['filename','label']))

MakeCategoricalPlot(Data,'label',(10,5))

MakeHistogramGrid(Data,FeatureHeaders)

MakeCorrelationPlot(Data,FeatureHeaders)

###############################################################################
# Feature Functions
###############################################################################

#Feature Skeleton 
def FeatureSkeleton(ValueA,ValueB):
    
    if ValueB==0:
        return 0
    else:
        return ValueA/(ValueB+ValueA)

###############################################################################
# Plotting functions
###############################################################################

def MakeRandomSizePopulation(MaxValue,PopulationSize,LenghtDivider):
    """
    Parameters
    ----------
    MaxValue : int
        Max number of features in the data.
    PopulationSize : int
        Number of indexs to be created.
    LenghtDivider : int
        Regulates the size of the individual in the populations, the greater 
        the value the individual is smaller.

    Returns
    -------
    container : list
        Contains the indexs for each individual in the population.
    """
    container=[]
    localIndex=np.arange(MaxValue)
    
    for k in range(PopulationSize):    
        np.random.shuffle(localIndex)
        currentList=list(localIndex)[0:np.random.randint(2,int(MaxValue/LenghtDivider))]
        container.append(currentList)
        
    return container 

#Wrapper function changes index to pairs of columns
def IndexToPairs(Index):
    return [[Index[k],Index[k+1]] for k in range(len(Index)-1)]


def MakeFeaturesData(Data,Indexs):
    """
    Parameters
    ----------
    Data : array,list
        Contains the data for feature generation.
    Indexs : list
        Location of the features for feature generation.

    Returns
    -------
    array
        feture generated data.
    """
    localData=[]
    localIndexs=IndexToPairs(Indexs)
    
    for pair in localIndexs:
        container=[]
        for val,sal in zip(Data[:,pair[0]],Data[:,pair[1]]):
            container.append(FeatureSkeleton(val,sal))
            
        localData.append(container)
    
    localData=np.array(localData)
    
    return localData.T
        
def MakeDataSets(Data,Target):
    """
    Parameters
    ----------
    Data : array
        X data.
    Target : array
        Y data.

    Returns
    -------
    Scaler : sklearn scaler object
        trained scaler of for the classifier.
    Xtrain : array
        X train data.
    Xtest : array
        X test data.
    Ytrain : array
        Y train data.
    Ytest : array
        Y test data.
    """
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(Data,Target, test_size=0.15,train_size=0.85,random_state=23)
    Scaler=pr.MinMaxScaler()
    Scaler.fit(Xtrain)
    
    return Scaler,Xtrain,Xtest,Ytrain,Ytest
    
def TrainModel(XData,YData,Index):
    """
    Parameters
    ----------
    XData : array
        X data.
    YData : array
        Y data.
    Index : list
        list of index for feature generation.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    XFeatures=MakeFeaturesData(XData,Index)
    scaler,Xtr,Xts,Ytr,Yts=MakeDataSets(XFeatures,YData)
    
    localModel=RandomForestClassifier(n_jobs=-2)
    localModel.fit(Xtr,Ytr)
    Xts=scaler.transform(Xts)
    localY=localModel.predict(Xts)
    
    return roc_auc_score(Yts,localY)

###############################################################################
# Evolutionary strategies
###############################################################################

def TrainOnPopulation(XData,YData,Population):
    """
    Parameters
    ----------
    XData : array
        X data.
    YData : array
        Y data.
    Population : list
        List with the indexs for feature generation.

    Returns
    -------
    fitness : list
        contains the performance of each individual in the population.
    """
    fitness=[]
    for ind in Population:
        fitness.append(TrainModel(XData,YData,ind))
        
    return fitness

def MakeIndexEvolution(IndexPopulation,maxValue,mutProb,recProb):
    """
    Parameters
    ----------
    IndexPopulation : list
        contains the index for feature generation.
    maxValue : int
        max number of features in the data.
    mutProb : float (0,1)
        probability of mutation.
    recProb : float (0,1)
        probability of recombination.

    Returns
    -------
    currentIndexs : list
        contains the modified indexs.
    """
    currentIndexs=copy.deepcopy(IndexPopulation)
    nIndexs=len(IndexPopulation)
    
    for k in range(nIndexs):
        if np.random.random()>mutProb and len(currentIndexs[k])>4:
            randomPosition=np.random.randint(0,len(currentIndexs[k]))
            del currentIndexs[k][randomPosition]
        else:
            randomPosition=np.random.randint(0,len(currentIndexs[k]))
            currentIndexs[k][randomPosition]=np.random.randint(0,maxValue)
            
    for j in range(nIndexs):
        if np.random.random()>recProb:
            rnIndividual=currentIndexs[np.random.randint(0,nIndexs)]
            recIndex=np.random.randint(0,len(rnIndividual))
            recInsertion=np.random.randint(0,len(currentIndexs[j]))
            currentIndexs[j].insert(recInsertion,rnIndividual[recIndex])
            
    return currentIndexs

def TrainOnGenerations(XData,YData,Generations,Population,MaxFeatures):
    """
    Parameters
    ----------
    XData : array
        X data.
    YData : array
        Y data.
    Generations : int
        Number of iterations.
    Population : int
        number of individuals per iteration.
    MaxFeatures : int
        Number of features in the data set.

    Returns
    -------
    fitness : list
        performance of each individual in the population.
    currentPopulation : list
        contains the index of the las population in the iteration.
    """
    currentPopulation=MakeRandomSizePopulation(MaxFeatures,Population,1)
    fitness=TrainOnPopulation(XData,YData,currentPopulation)
    
    for k in range(Generations):
        
        newPopulation=MakeIndexEvolution(currentPopulation,MaxFeatures,0.5,0.5)
        newFitness=TrainOnPopulation(XData,YData,newPopulation)
        
        for k in range(Population):
            if newFitness[k]>fitness[k]:
                currentPopulation[k]=newPopulation[k]
                fitness[k]=newFitness[k]
        
    return fitness,currentPopulation

###############################################################################
# Evolutionary strategies
###############################################################################

XData=np.array(Data[FeatureHeaders])
YData=np.array(Data['label'])

fitn,indxs=TrainOnGenerations(XData,YData,20,25,len(FeatureHeaders))

modelSizes=[len(val) for val in indxs]
minsize=min(modelSizes)
maxsize=max(modelSizes)

widths=[0.5+(val-minsize)/(2*(maxsize-minsize)) for val in modelSizes]

plt.figure(4,figsize=(15,6))
plt.bar(np.arange(len(fitn)),fitn,widths)
plt.xlabel('Models',fontsize=14)
plt.ylabel('ROC AUC',fontsize=14)
ax=plt.gca()
PlotStyle(ax)

