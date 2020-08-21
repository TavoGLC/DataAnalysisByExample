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

import copy

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn import preprocessing as pr
from sklearn.ensemble import RandomForestRegressor

from tsfresh.feature_extraction import feature_calculators as fc

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
    squaredUnique=int(np.sqrt(TotalNumberOfElements))
    
    if squaredUnique*squaredUnique==TotalNumberOfElements:
        nrows,ncolumns=squaredUnique,squaredUnique
    elif squaredUnique*(squaredUnique+1)<TotalNumberOfElements:
        nrows,ncolumns=squaredUnique+1,squaredUnique+1
    else:
        nrows,ncolumns=squaredUnique,squaredUnique+1
    
    return nrows,ncolumns

def MakePlotGrid(Data,Labels,kind="Scatter"):
    """
    Parameters
    ----------
    Data : Pandas dataframe
        Contains the data to be ploted .
    kind : string, optional
        Kind of plot to be created, only takes "Scatter" or "Histogram". 
        The default is "Scatter".

    Returns
    -------
    None.
    """
    
    _,numberOfElements=Data.shape
    
    nrows,ncolumns=GetGridShape(numberOfElements)
    
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,13))
    
    for k in range(numberOfElements):
        
        if kind=="Scatter":
            axes[subPlotIndexs[k]].plot(Data[:,k],'bo')
        elif kind=="Histogram":
            axes[subPlotIndexs[k]].hist(Data[:,k],bins=50)
            axes[subPlotIndexs[k]].set_xlabel(Labels[k])
            
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
    correlation=pd.DataFrame(Data).corr()
    
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
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "daily-min-temperatures.csv"

Data=pd.read_csv(DataDir)

###############################################################################
# Data visualization
###############################################################################

plt.figure(figsize=(10,6))
plt.plot(Data['Temp'])
plt.xlabel('Time',fontsize=14)
plt.ylabel('Daily Min Temperature',fontsize=14)
ax=plt.gca()
PlotStyle(ax)

plt.figure(figsize=(7,7))
plt.hist(Data['Temp'])
plt.xlabel('Daily Min Temperature',fontsize=14)
plt.ylabel('Frequency',fontsize=14)
ax=plt.gca()
PlotStyle(ax)

###############################################################################
# Fragment features
###############################################################################

def MakeSeriesFragments(TimeSeries,FragmentSize,sliding=False):
    """
    Parameters
    ----------
    TimeSeries : list, array 
        1D array or list with the time series values.
    FragmentSize : int
        Size of the fragment taken to split the time series.
    sliding : bool, optional
        Controls the use of the sliding scheme for partition. 
        The default is False.

    Returns
    -------
    array
        2D array with the fragments, 1D array with the forecasted value

    """
    
    XContainer=[]
    YContainer=[]
    nData=len(TimeSeries)
    
    if sliding:
        index=np.arange(0,nData-FragmentSize-1)
    else:
        index=np.arange(0,nData-FragmentSize-1,FragmentSize)
        
    for val in index:
        XContainer.append(TimeSeries[val:val+FragmentSize])
        YContainer.append(TimeSeries[val+FragmentSize]+1)  
    
    return np.array(XContainer),np.array(YContainer)

###############################################################################
# Fragment features
###############################################################################

"""

Contains a series of wrapper functions to calculate several features for the
time series data, most of the features are calculated using the tsfresh library

"""
def C3Lag3(fragment):
    return fc.c3(fragment,3)

def C3Lag5(fragment):
    return fc.c3(fragment,5)

def C3Lag7(fragment):
    return fc.c3(fragment,7)

def C3Lag11(fragment):
    return fc.c3(fragment,11)

def CIDCELag3(fragment):
    return fc.cid_ce(fragment,3)

def CIDCELag5(fragment):
    return fc.cid_ce(fragment,5)

def CIDCELag7(fragment):
    return fc.cid_ce(fragment,7)

def CIDCELag11(fragment):
    return fc.cid_ce(fragment,11)

def ACLag3(fragment):
    return fc.autocorrelation(fragment,3)

def ACLag5(fragment):
    return fc.autocorrelation(fragment,5)

def ACLag7(fragment):
    return fc.autocorrelation(fragment,7)

def ACLag11(fragment):
    return fc.autocorrelation(fragment,11)

def MedianDifference(fragment):
    return fc.median(np.diff(fragment))

def MedianDeviation(fragment):
    return fc.median([val-fc.median(fragment) for val in fragment])

def ReciprocalSum(fragment):
    fragsum=0
    for val in fragment:
        if val!=0:
            fragsum=fragsum+1/val
    return fragsum

def LogSum(fragment):
    fragsum=0
    for val in fragment:
        if val!=0:
            fragsum=fragsum+np.log(1+val)
    return fragsum

def RootSum(fragment):
    fragsum=0
    for val in fragment:
        if val!=0:
            fragsum=fragsum+np.sqrt(val)
    return fragsum   

def NumberOfZeros(fragment):
    fragsum=0
    for val in fragment:
        if val==0:
            fragsum=fragsum+1
    return fragsum

def NonZero(fragment):
    return len(fragment)-NumberOfZeros(fragment)

FeaturesList=[np.min,np.max,np.ptp,fc.mean,fc.median,fc.variance,
              np.argmax,np.argmin,
              fc.standard_deviation,fc.kurtosis,fc.skewness,
              fc.count_above_mean,fc.count_below_mean,
              fc.longest_strike_above_mean,fc.longest_strike_below_mean,
              fc.abs_energy,fc.absolute_sum_of_changes,fc.mean_abs_change,
              fc.mean_change,fc.mean_second_derivative_central,CIDCELag3,
              CIDCELag5,CIDCELag7,CIDCELag11,
              ACLag3,ACLag5,ACLag7,ACLag11,C3Lag3,C3Lag5,C3Lag7,C3Lag11,
              ReciprocalSum,RootSum,NumberOfZeros,NonZero,MedianDifference,
              LogSum,MedianDeviation]

FeaturesNames=["min","max","ptp","mean","median","variance",
              "argmax","argmin",
              "standard_deviation","kurtosis","skewness",
              "count_above_mean","count_below_mean",
              "longest_strike_above_mean","longest_strike_below_mean",
              "abs_energy","absolute_sum_of_changes","mean_abs_change",
              "mean_change","mean_second_derivative_central","CIDCELag3",
              "CIDCELag5","CIDCELag7","CIDCELag11",
              "ACLag3","ACLag5","ACLag7","ACLag11","C3Lag3","C3Lag5","C3Lag7","C3Lag11",
              "ReciprocalSum","RootSum","NumberOfZeros","NonZero","MedianDifference",
              "LogSum","MedianDeviation"]
###############################################################################
# Fragment features
###############################################################################

def MakeFeaturesData(TimeSeries,FragmentSize,sliding,FeaturesList):
    """
    Parameters
    ----------
    TimeSeries : array
        Contains the time series data.
    FragmentSize : int
        Size of the fragment taken to split the time series.
    sliding : bool
        Controls the use of the sliding scheme for partition. 
    FeaturesList : list
        List of feature function names. The function must take only 
        the time series fragment as input

    Returns
    -------
    FeaturesContainer : array
        Contains the calculated features for the time series fragments.
    YData : array
        Contains the forecasted values for each fragment.

    """
    
    XData,YData=MakeSeriesFragments(TimeSeries,FragmentSize,sliding)
    
    FeaturesContainer=[]
    for feature in FeaturesList:
        loopContainer=[]
        for val in XData:
            loopContainer.append(feature(val))
        FeaturesContainer.append(loopContainer)
        
    FeaturesContainer=np.array(FeaturesContainer).T
    FeaturesContainer[np.isnan(FeaturesContainer)]=0
    
    return FeaturesContainer,YData

###############################################################################
# Data Features Visualization 
###############################################################################

XData,YData=MakeFeaturesData(np.array(Data['Temp']),25,True,FeaturesList)
maxFeatures=XData.shape[1]

MakeCorrelationPlot(XData,FeaturesNames)

MakePlotGrid(XData,FeaturesNames,kind="Histogram")

###############################################################################
# Model building
###############################################################################

#Wrapper function to create the datasets
def MakeDataSets(Data,Target,FeaturesIndex):
    
    XData=Data[:,FeaturesIndex]
    cropPoint=int(0.85*len(Target))
    Xtrain=XData[0:cropPoint,:]
    Xtest=XData[cropPoint:len(Target),:]
    Ytrain=Target[0:cropPoint]
    Ytest=Target[cropPoint:len(Target)]
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
    object
        trained RandomForest object.
    float
        Performance of the model
    array
        performance of each feature used
    """
    scaler,Xtr,Xts,Ytr,Yts=MakeDataSets(XData,YData,Index)
    
    localModel=RandomForestRegressor(n_jobs=-2)
    Xtr=scaler.transform(Xtr)
    localModel.fit(Xtr,Ytr)
    Xts=scaler.transform(Xts)
    
    return localModel,localModel.score(Xts,Yts),localModel.feature_importances_

###############################################################################
# Memory heuristics 
###############################################################################
def ForceRange(Index,UpperBound):
    """
    Parameters
    ----------
    Index : list,array
        list of integer values.
    UpperBound : int
        Max integer value valid to be in Index.

    Returns
    -------
    Index : list,array
        list of integer values in the range(0,UpperBound)..

    """
    for k in range(len(Index)):
        if Index[k]>=UpperBound:
            Index[k]=int(np.random.choice(np.arange(0,UpperBound-1),1))
    return Index

def MemoryList(ImportanceList,IndexList,ExcludedList,ToAdd=3,MaxMemory=15):
    """
    Parameters
    ----------
    ImportanceList : list,array
        Contains the importance of each index value.
    IndexList : list,array
        list of integer values.
    ExcludedList : list
        List with the excluded values.
    ToAdd : int, optional
        Max number of worst preforming elements to be added. The default is 3.
    MaxMemory : int, optional
        Max number of elements of the excluded list. The default is 15.

    Returns
    -------
    TYPE
        Updated excluded list.

    """
    
    ImportanceOrder=np.argsort(ImportanceList)
    WorstIndexs=[IndexList[val] for val in ImportanceOrder[0:ToAdd]]
    ExcludedList=list(set(list(ExcludedList)+list(WorstIndexs)))
    
    if len(ExcludedList)>MaxMemory:
        return ExcludedList[0:MaxMemory]
    else:
        return ExcludedList
    
def SingleIndexMemory(Index,ExcludedList,UpperBound):
    """
    Parameters
    ----------
    IndexList : list,array
        list of integer values.
    ExcludedList : list
        List with the excluded values.
    UpperBound : int
        Max integer value valid to be in Index.

    Returns
    -------
    currentIndex : list, array
        Updated index, removes elements from the index found on the 
        ExcludedList.

    """
    
    currentIndex=copy.deepcopy(Index)
    avaliableIndexs=list(set(np.arange(UpperBound))-set(ExcludedList))
    disc=set(currentIndex)-set(ExcludedList)
    
    if len(disc)<len(currentIndex):
        toAppend=len(currentIndex)-len(disc)
        np.random.shuffle(avaliableIndexs)
        currentIndex=list(disc)+avaliableIndexs[0:toAppend]
    
    return currentIndex
    
###############################################################################
# Simulated Annealing
###############################################################################

def AcceptanceProbability(Cost,NewCost,Temperature):
    """
    Parameters
    ----------
    Cost : float
        Cost of the previous state.
    NewCost : float
        cost of the current state .
    Temperature : float
        Current temperature in the optimizer.

    Returns
    -------
    float
        Probability to accept the solution.

    """

    if NewCost>Cost:
        return 1
    else:
        return np.exp(-(NewCost-Cost)/Temperature)

#Temperature relaxation function
def Temperature(Fraction):
    return max(0.01,min(1,1-Fraction))

def RandomNeighbour(Individual,Fraction,UpperBound):
    """
    Parameters
    ----------
    Individual : list
        List of integer values.
    Fraction : float
        Optimizer temperature.
    UpperBound : int
        Max integer value valid to be in Index.
        
    Returns
    -------
    list,array
        Updated individual.

    """
    nToModify=int(len(Individual)/3)
    indexToModify=np.arange(0,len(Individual))
    np.random.shuffle(indexToModify)
    newState=Individual.copy()
    
    for val in indexToModify[0:nToModify]:
        delta=(len(Individual)*(Fraction))*(2*np.random.random())
        newState[val]=int(newState[val]+delta)
    
    return ForceRange(newState,UpperBound)
    
def SimulatedAnnealing(XData,YData,Index,UpperBound,maxSteps=30):
    """
    Parameters
    ----------
    XData : 2D array
        XData.
    YData : array
        YData.
    Index : list
        List of integer values, location of the features to be used.
    UpperBound : int
        Max integer value valid to be in Index.
    maxSteps : int, optional
        Number of steps taken by the optimizer. The default is 30.

    Returns
    -------
    scores : list
        performance of the constructed model.
    states : list
        accepted index values.
    Excluded : list
        list of the excluded features.

    """
    _,score,featureScores=TrainModel(XData,YData,Index)
    Excluded=[]
    Excluded=MemoryList(featureScores,Index,Excluded)
    scores,states=[score],[Index]
    
    for k in range(maxSteps):
        fraction=k/float(maxSteps)
        T=Temperature(fraction)
        newIndex=RandomNeighbour(Index,fraction,UpperBound)
        newIndex=SingleIndexMemory(newIndex,Excluded,UpperBound)
        _,newScore,featureScores=TrainModel(XData,YData,newIndex)
        Excluded=MemoryList(featureScores,newIndex,Excluded)
        
        if AcceptanceProbability(score,newScore,T)>np.random.random():
            score,state=newScore,newIndex
            states.append(state)
            scores.append(score)
            
    return scores,states,Excluded

###############################################################################
# Simulated annealing optimization 
###############################################################################

startIndex=np.arange(0,maxFeatures)
np.random.shuffle(startIndex)

scores,indexs,excluded=SimulatedAnnealing(XData,YData,startIndex[0:10],maxFeatures,maxSteps=100)

plt.figure(figsize=(10,6))
plt.bar(np.arange(len(scores)),scores)
plt.xlabel('Models',fontsize=14)
plt.ylabel('Score',fontsize=14)
ax=plt.gca()
PlotStyle(ax)

ExcludedNames0=[FeaturesNames[val] for val in excluded]

scaler,Xtrain,Xtest,Ytrain,Ytest=MakeDataSets(XData,YData,indexs[np.argmax(scores)])
model,score,importances=TrainModel(XData,YData,indexs[np.argmax(scores)])

Xtest=scaler.transform(Xtest)
Ypreds=model.predict(Xtest)

plt.figure(figsize=(10,6))
plt.plot(Ytest,'b',label="Data")
plt.plot(Ypreds,'r',label="Forecast")
plt.xlabel('Time',fontsize=14)
plt.ylabel('Daily Min Temperature',fontsize=14)
plt.legend()
ax=plt.gca()
PlotStyle(ax)

localLabels=[FeaturesNames[val] for val in indexs[np.argmax(scores)]]

plt.figure(figsize=(10,6))
plt.plot(importances,'b')
plt.xlabel('Features',fontsize=14)
plt.ylabel('Importance',fontsize=14)
ax=plt.gca()
ax.set_xticks(np.arange(len(localLabels)))
ax.set_xticklabels(localLabels)
ax.xaxis.set_tick_params(labelsize=13,rotation=90)
PlotStyle(ax)

###############################################################################
# Genetic algorithm 
###############################################################################

def MakeRandomSizePopulation(MaxValue,PopulationSize,IndexRange):
    """
    Parameters
    ----------
    MaxValue : int
        Max integer value valid for each individual.
    PopulationSize : int
        Number of individuals in the population.
    IndexRange : list
        Upper and lower bounds of the size for the individuals.

    Returns
    -------
    container : list
        Contains the index values for the population.

    """    
    container=[]
    localIndex=np.arange(MaxValue)
    localRange=np.arange(IndexRange[0],IndexRange[1])
    
    for k in range(PopulationSize):    
        np.random.shuffle(localIndex)
        np.random.shuffle(localRange)
        currentList=list(localIndex)[0:localRange[0]]
        container.append(currentList)
        
    return container

def MakeIndexEvolution(IndexPopulation,UpperBound,mutProb=0.5,recProb=0.5):
    """
    Parameters
    ----------
    IndexPopulation : list
        contains the index for feature generation.
    UpperBound : int
        Max integer value valid to be in the Index.
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
            currentIndexs[k][randomPosition]=np.random.randint(0,UpperBound)
            
    for j in range(nIndexs):
        if np.random.random()>recProb:
            rnIndividual=currentIndexs[np.random.randint(0,nIndexs)]
            recIndex=np.random.randint(0,len(rnIndividual))
            recInsertion=np.random.randint(0,len(currentIndexs[j]))
            currentIndexs[j].insert(recInsertion,rnIndividual[recIndex])
            
    rangeIndex=[ForceRange(indx,UpperBound) for indx in currentIndexs]
            
    return rangeIndex

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
    featureImportance : array
        performance of each feature used to build the model.

    """
    fitness=[]
    featureImportance=[]
    
    for ind in Population:
        
        _,score,importance=TrainModel(XData,YData,ind)
        fitness.append(score)
        featureImportance.append(importance)
        
    return fitness,featureImportance


def TrainOnGenerations(XData,YData,Generations,Population,UpperBound):
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
    UpperBound : int
        Max integer value valid to be in Index.

    Returns
    -------
    fitness : list
        performance of each individual in the population.
    currentPopulation : list
        contains the index of the las population in the iteration.
    Excluded : list
        List with the excluded values.
    """
    currentPopulation=MakeRandomSizePopulation(UpperBound,Population,[5,10])
    fitness,featureImportance=TrainOnPopulation(XData,YData,currentPopulation)
    Excluded=[]
    
    for imp,indx in zip(featureImportance,currentPopulation):
        Excluded=MemoryList(imp,indx,Excluded)
    
    for k in range(Generations):
        
        newPopulation=MakeIndexEvolution(currentPopulation,UpperBound)
        
        for k in range(Population):
            newPopulation[k]=ForceRange(newPopulation[k],UpperBound)
            newPopulation[k]=SingleIndexMemory(newPopulation[k],Excluded,UpperBound)

        newFitness,featureImportance=TrainOnPopulation(XData,YData,newPopulation)

        for imp,indx in zip(featureImportance,newPopulation):
            Excluded=MemoryList(imp,indx,Excluded)
        
        for k in range(Population):
            if newFitness[k]>fitness[k]:
                currentPopulation[k]=newPopulation[k]
                fitness[k]=newFitness[k]
        
    return fitness,currentPopulation,Excluded

###############################################################################
# Genetic Algorithm optimization 
###############################################################################

fitness,population,exc=TrainOnGenerations(XData,YData,5,20,maxFeatures)

modelSizes=[len(val) for val in population]
minsize=min(modelSizes)
maxsize=max(modelSizes)

widths=[0.5+(val-minsize)/(2*(maxsize-minsize)) for val in modelSizes]

plt.figure(figsize=(10,6))
plt.bar(np.arange(len(fitness)),fitness,widths)
plt.xlabel('Models',fontsize=14)
plt.ylabel('Score',fontsize=14)
ax=plt.gca()
PlotStyle(ax)

ExcludedNames1=[FeaturesNames[val] for val in exc]

scaler,Xtrain,Xtest,Ytrain,Ytest=MakeDataSets(XData,YData,population[np.argmax(fitness)])
model,score,importances=TrainModel(XData,YData,population[np.argmax(fitness)])

Xtest=scaler.transform(Xtest)
Ypreds=model.predict(Xtest)

plt.figure(figsize=(10,6))
plt.plot(Ytest,'b',label="Data")
plt.plot(Ypreds,'r',label="Forecast")
plt.xlabel('Time',fontsize=14)
plt.ylabel('Daily Min Temperature',fontsize=14)
plt.legend()
ax=plt.gca()
PlotStyle(ax)

localLabels=[FeaturesNames[val] for val in population[np.argmax(fitness)]]

plt.figure(figsize=(10,6))
plt.plot(importances,'b')
plt.xlabel('Features',fontsize=14)
plt.ylabel('Importances',fontsize=14)
ax=plt.gca()
ax.set_xticks(np.arange(len(localLabels)))
ax.set_xticklabels(localLabels)
ax.xaxis.set_tick_params(labelsize=13,rotation=90)
PlotStyle(ax)
