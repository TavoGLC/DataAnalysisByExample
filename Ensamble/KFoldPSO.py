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

from scipy.sparse import hstack
from sklearn import preprocessing as pr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "WA_Fn-UseC_-Telco-Customer-Churn.csv"

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

def MakeCategoricalBarPlot(Data,CategoricalFeature,PlotSize):
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
# Data processing 
###############################################################################

Data=pd.read_csv(DataDir)
Target="Churn"
Features=set(list(Data))-set(["customerID","Churn","TotalCharges"])
NumericalFeatures=["MonthlyCharges","tenure"]
CategoricalFeatures=list(Features-set(NumericalFeatures))
Data.fillna(0,inplace=True)

###############################################################################
# General Data Visualization 
###############################################################################

for val in CategoricalFeatures:
    MakeCategoricalBarPlot(Data,val,(8,6))

for k in range(len(NumericalFeatures)):
    plt.figure(figsize=(8,6))
    plt.hist(Data[NumericalFeatures[k]],bins=75)
    ax=plt.gca()
    ax.set_xlabel(NumericalFeatures[k],fontsize=14)
    PlotStyle(ax)
    
MakeCategoricalBarPlot(Data,Target,(8,6))

###############################################################################
#Class equalization  
###############################################################################

def ClassEqualization(Data,Category):
    """
    Parameters
    ----------
    Data : pandas dataframe
        Data for equalization.
    Category : string
        Category used to equalize the dataframe.

    Returns
    -------
    pandas dataframe
        Equalized dataframe.

    """
    
    CategoryUniques=Data[Category].unique()
    CategoryDictionary=UniqueToDictionary(CategoryUniques)
    
    categoryCounts=CountUniqueElements(CategoryUniques,np.array(Data[Category]))    
    minCounts=np.min(categoryCounts)
    nData=len(Data)
    
    localIndex=np.arange(0,nData)
    np.random.shuffle(localIndex)
    
    countContainer=[0 for k in range(len(CategoryUniques))]
    indexContainer=[]
    
    for k in localIndex:
        
        cVal=Data.iloc[k][Category]
        cLoc=CategoryDictionary[cVal]
        
        if countContainer[cLoc]<minCounts:
            countContainer[cLoc]=countContainer[cLoc]+1
            indexContainer.append(k)
            
    return Data.iloc[indexContainer]

EqualizedData=ClassEqualization(Data,Target)
MakeCategoricalBarPlot(EqualizedData,Target,(8,6))

###############################################################################
#Model building  
###############################################################################

def TrainLabelEncoders(Features,Data):
    """
    Parameters
    ----------
    Features : list, array 
        List or array with the categorical headers for OneHotEncoder.
    Data : pandas dataframe
        Data to be encoded.

    Returns
    -------
    labelEncoders : list
        list with trained onehotencoders.

    """
    labelEncoders=[]
    
    for feat in Features:
        localEncoder=pr.OneHotEncoder(handle_unknown='ignore')
        localData=np.array(Data[feat])
        localEncoder.fit(localData.reshape(-1,1))
        labelEncoders.append(localEncoder)
        
    return labelEncoders

#Wrapper function to generate the categorical data
def MakeEncodedData(Features,Data):
    
    localEncoders=TrainLabelEncoders(Features,Data)
    localData=np.array(Data[Features[0]])
    EncodedLabels=localEncoders[0].transform(localData.reshape(-1,1))
    
    for k in range(len(Features)-1):
        localData=np.array(Data[Features[k+1]])
        loopLabels=localEncoders[k+1].transform(localData.reshape(-1,1))
        EncodedLabels=hstack((EncodedLabels,loopLabels))
        
    return EncodedLabels

#Wrapper function to normalize the numerical data
def MakeNumericalData(Features,Data):
    
    localData=Data[Features]
    scaler=pr.MinMaxScaler()
    scaler.fit(localData)
    
    return scaler.transform(localData)

#Wrapper function to generate the data sets
def MakeDataSets(Data,CategoricalFeatures,NumericalFeatures,Target):
    
    YData=MakeEncodedData([Target],Data)
    YData=YData.toarray()
    CategoricalData=MakeEncodedData(CategoricalFeatures,Data)
    NumericalData=MakeNumericalData(NumericalFeatures,Data)
    XData=np.hstack((CategoricalData.toarray(),NumericalData))
    
    return XData,YData[:,1]

def CheckHyperparameters(Parameters):
    """
    Parameters
    ----------
    Parameters : list,array
        contains the hyperparameter values .

    Returns
    -------
    Parameters : list,array
        mantains the hyperparameters between certain boundaries.

    """
    Parameters=list(Parameters)
    Bounds=[[10,250],[5,100],[0,1],[0,0.5],[0,1]]
    
    for k in range(len(Parameters)):
        if Parameters[k]<Bounds[k][0] or Parameters[k]>Bounds[k][1]:
            Parameters[k]=np.mean(Bounds[k])
            
    Parameters[0]=np.int(Parameters[0])
    Parameters[1]=np.int(Parameters[1])
            
    return Parameters

#Wrapper funtion to format the hyperparamters 
def FormatHyperparameters(Parameters):
    hypNames=["n_estimators","max_depth","min_samples_split","min_samples_leaf","max_features"]
    hyp={}
    for name,val in zip(hypNames,Parameters):
        hyp[name]=val
        
    hyp["n_jobs"]=-2
    hyp["random_state"]=10
    return hyp

#Wrapper funtion to train the model
def TrainModel(Xtrain,Ytrain,Xtest,Ytest,HyperParams):
    
    localModel=RandomForestClassifier(**HyperParams)
    localModel.fit(Xtrain,Ytrain)
    localY=localModel.predict(Xtest)
    localScore=roc_auc_score(Ytest,localY)
    
    return localScore

#Wrapper function for KFold crossvalidation 
def TrainKFoldCVModel(XData,YData,HyperParams,splits):
    
    fitness=[]
    cKF=StratifiedKFold(n_splits=splits)
    
    for trainI,testI in cKF.split(XData,YData):
        Xtrain,Xtest=XData[trainI],XData[testI]
        Ytrain,Ytest=YData[trainI],YData[testI]
        fitness.append(TrainModel(Xtrain,Ytrain,Xtest,Ytest,HyperParams))
        
    return np.mean(fitness)

###############################################################################
#Particle Swarm Optimization
###############################################################################

def UpdateSwarm(Swarm,Velocity,BestIndividual,BestGlobal,InertiaC,SocialC,CognitiveC):
    """
    Parameters
    ----------
    Swarm : list,array
        Swarm particles positions.
    Velocity : list,array
        Swarm velocities.
    BestIndividual : list,array
        Best performance for each particle.
    BestGlobal : list, array
        Global best particle.
    InertiaC : float
        Inertia constant.
    SocialC : float
        Social constant.
    CognitiveC : float
        Cognitive constant.

    Returns
    -------
    newSwarm : list
        updated swarm positions.
    velocity : list
        swarm velocity.

    """
    newSwarm=copy.deepcopy(Swarm)
    velocity=[]
    
    for k in range(len(newSwarm)):
        inertia=InertiaC*np.asarray(Velocity[k])
        social=SocialC*np.random.random()*(np.asarray(BestGlobal)-np.asarray(newSwarm[k]))
        cognitive=CognitiveC*np.random.random()*(np.asarray(BestIndividual[k])-np.asarray(newSwarm[k]))
        vel=inertia+social+cognitive
        velocity.append(vel)
        newSwarm[k]=newSwarm[k]+vel
        
    return newSwarm,velocity

def EvaluateSwarmFitness(XData,YData,Swarm,Splits):
    """
    Parameters
    ----------
    XData : array
        Xdata.
    YData : array
        Ydata.
    Swarm : list,array
        swarm particle positions.
    Splits : int
        number of splits for kfold.

    Returns
    -------
    swarm : list,array
        swarm positions.
    fitness : list
        particle fitness.

    """
    fitness=[]
    swarm=[]
    
    for part in Swarm:
        cPart=CheckHyperparameters(part)
        cHyp=FormatHyperparameters(cPart)
        fitn=TrainKFoldCVModel(XData,YData,cHyp,Splits)
        swarm.append(cPart)
        fitness.append(fitn)
        
    return swarm,fitness

def KFoldCVPSO(XData,YData,Splits,SwarmSize,Iterations,Inertia=0.5,Social=0.25,Cognitive=0.25):
    """
    Parameters
    ----------
    XData : array
        Train Data.
    YData : array
        Train labels.
    Splits : int
        number of splits for k fold.
    SwarmSize : int
        number of particles in the swarm.
    Iterations : int
        Iterations for PSO.
    Inertia : float, optional
        Inertia Constant. The default is 0.5.
    Social : float, optional
        Social Constant. The default is 0.25.
    Cognitive : float, optional
        Cognitive Constant. The default is 0.25.

    Returns
    -------
    Swarm : list
        Optimized hyperparameters.
    loopFitness : list
        Performance of each hyperparameter combination.

    """

    Swarm=np.random.random((SwarmSize,5))
    Velocity=np.random.random((SwarmSize,5))
    
    bestSwarm,bestFitness=EvaluateSwarmFitness(XData,YData,Swarm,Splits)
    
    bestGFitness=np.max(bestFitness)
    bestGPart=bestSwarm[np.argmax(bestFitness)]
    
    for k in range(Iterations):
        
        Swarm,Velocity=UpdateSwarm(Swarm,Velocity,bestSwarm,bestGPart,Inertia,Social,Cognitive)
        Swarm,loopFitness=EvaluateSwarmFitness(XData,YData,Swarm,Splits)
        
        if np.max(loopFitness)>bestGFitness:
            bestGFitness=np.max(loopFitness)
            bestGPart=Swarm[np.argmax(loopFitness)]
        
        for k in range(SwarmSize):
            if loopFitness[k]>bestFitness[k]:
                bestFitness[k]=loopFitness[k]
                bestSwarm[k]=Swarm[k]
        
    return Swarm,loopFitness
            
        
###############################################################################
#Model performance  
###############################################################################
    
XData,YData=MakeDataSets(EqualizedData,CategoricalFeatures,NumericalFeatures,Target)

locScore=[]
locKFold=StratifiedKFold()

for trainI,testI in locKFold.split(XData,YData):
        Xtrain,Xtest=XData[trainI],XData[testI]
        Ytrain,Ytest=YData[trainI],YData[testI]
        basemodel=RandomForestClassifier(random_state=10)
        basemodel.fit(Xtrain,Ytrain)
        localY=basemodel.predict(Xtest)
        locScore.append(roc_auc_score(Ytest,localY))
        
baseScore=np.mean(locScore)

params,fitness=KFoldCVPSO(XData,YData,5,15,30)

improvement=[100*((val-baseScore)/baseScore) for val in fitness]

plt.figure(figsize=(15,6))
plt.bar(np.arange(len(fitness)),improvement)
plt.xlabel('Models',fontsize=14)
plt.ylabel('% Improvement',fontsize=14)
ax=plt.gca()
PlotStyle(ax)
