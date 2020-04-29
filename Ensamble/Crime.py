#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:12:54 2020

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

from scipy.sparse import hstack
from sklearn import preprocessing as pr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r'/media/tavoglc/storage/storage/LocalData/'
DataDir=GlobalDirectory + "Chicago_Crimes_2012_to_2017.csv"

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

Target="Arrest"

Data["Date"]=pd.to_datetime(Data["Date"],format='%m/%d/%Y %I:%M:%S %p')
Data["Updated On"]=pd.to_datetime(Data["Updated On"],format='%m/%d/%Y %I:%M:%S %p')

Data["Week"]=Data["Date"].dt.week
Data["Hour"]=Data["Date"].dt.hour
Data['DayOfWeek']=Data["Date"].dt.dayofweek
Data['Month']=Data["Date"].dt.month

Data["Week Update"]=Data["Updated On"].dt.week
Data["Hour Update"]=Data["Updated On"].dt.hour
Data['DayOfWeek Update']=Data["Updated On"].dt.dayofweek
Data['Month Update']=Data["Updated On"].dt.month

Data.fillna(0,inplace=True)
Data=Data.astype({'Location Description':str})

CategoricalFeatures=['Ward','Primary Type','Community Area','Description','Location Description','District','IUCR','Domestic',"Week","Hour",'DayOfWeek','Month',"Week Update","Hour Update",'DayOfWeek Update','Month Update']

###############################################################################
# General Data Visualization 
###############################################################################
    
MakeCategoricalPlot(Data,"Primary Type",(12,4))

MakeCategoricalPlot(Data,"Domestic",(8,6))

MakeCategoricalPlot(Data,"Ward",(12,6))

MakeCategoricalPlot(Data,"Community Area",(15,6))

MakeCategoricalPlot(Data,"Arrest",(8,6))

###############################################################################
#Class equalization  
###############################################################################

def ClassEqualization(Data,Category):
    """
    Randomly selects samples in data for the labels in Category have 
    the same number of samples.
    
    Parameters
    ----------
    Data : Pandas dtaframe
        Data to be equalized.
    Category : String 
        Column header of Data.

    Returns
    -------
    Pandas dataframe.
    
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

EqualizedData=ClassEqualization(Data,'Arrest')
MakeCategoricalPlot(EqualizedData,'Arrest',(8,6))

###############################################################################
#Model Building 
###############################################################################

def TrainLabelEncoders(Features,Data):
    """
    Train the label encoders for  the categorical data.
    Parameters
    ----------
    Features : List
        List of headers in the pandas dataframe to be encoded.
    Data : Pandas dataframe
        DESCRIPTION.

    Returns
    -------
    labelEncoders : List 
        list of trained OneHotEncoder objects.

    """
    labelEncoders=[]
    
    for feat in Features:
        localEncoder=pr.OneHotEncoder(handle_unknown='ignore')
        localData=np.array(Data[feat])
        localEncoder.fit(localData.reshape(-1,1))
        labelEncoders.append(localEncoder)
        
    return labelEncoders
        
def MakeEncodedData(Features,Data):
    """
    Parameters
    ----------
    Features : List
        List of headers in the pandas dataframe to be encoded.
    Data : 
        Pandas dataframe

    Returns
    -------
    EncodedLabels : sparsce array
        Sparce array with the enbcoded data.

    """
    
    localEncoders=TrainLabelEncoders(Features,Data)
    localData=np.array(Data[Features[0]])
    EncodedLabels=localEncoders[0].transform(localData.reshape(-1,1))
    
    for k in range(len(Features)-1):
        localData=np.array(Data[Features[k+1]])
        loopLabels=localEncoders[k+1].transform(localData.reshape(-1,1))
        EncodedLabels=hstack((EncodedLabels,loopLabels))
        
    return EncodedLabels
    
        
def TrainModel(Features,Data):
    """
    Parameters
    ----------
    Features : List
        List of headers in the pandas dataframe to be encoded.
    Data : 
        Pandas dataframe

    Returns
    -------
    float
        ROC auc score of the trained random forrest.

    """
    
    localData=MakeEncodedData(Features,Data)
    localTarget=MakeEncodedData(["Arrest"],Data)
    
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(localData,localTarget, test_size=0.15,train_size=0.85,random_state=23)
    localModel=RandomForestClassifier(n_jobs=-2)
    localModel.fit(Xtrain,Ytrain.toarray()[:,0])
    localY=localModel.predict(Xtest)
    
    return roc_auc_score(Ytest.toarray()[:,0],localY)


###############################################################################
#Model selection 
###############################################################################

def MakeRandomSizePopulation(MaxValue,Size):
    """
    Generate a list of list of random size. To be used as a random indexing 
    function
    Parameters
    ----------
    MaxValue : int
        Max integer value to be in a list.
    Size : int
        Number of lists to be in the container list.

    Returns
    -------
    container : list
        Contains a list of lists.

    """

    container=[]
    
    for k in range(Size):
        localIndex=np.arange(MaxValue)
        np.random.shuffle(localIndex)
        container.append(list(localIndex)[0:np.random.randint(1,int(MaxValue/2))])
        
    return container 

def TrainOnPopulation(TotalFeatures,Data,Size):
    """
    Parameters
    ----------
    TotalFeatures : List
        List with the names of the categorical values in the data.
    Data : Pandas dataframe
        Training data for the model.
    Size : int
        Number of models to be trained.

    Returns
    -------
    FitnessContainer : list
        Contains the ROC AUC score for each model.
    IndexContainer : list
        Contains the categorical features used for the model.

    """
    
    maxFeatures=len(TotalFeatures)
    PopulationIndex=MakeRandomSizePopulation(maxFeatures,Size)
    IndexContainer=[]
    FitnessContainer=[]
    
    for k in range(Size):
        localIndex=PopulationIndex[k]
        localFeatures=[TotalFeatures[val]for val in localIndex]
        FitnessContainer.append(TrainModel(localFeatures,Data))
        IndexContainer.append(localIndex)
        
    return FitnessContainer,IndexContainer

###############################################################################
#Performance visualization
###############################################################################

currentSize=30

Fitness,Categories=TrainOnPopulation(CategoricalFeatures,EqualizedData,currentSize)
modelSizes=[len(val) for val in Categories]
minsize=min(modelSizes)
maxsize=max(modelSizes)

widths=[0.5+(val-minsize)/(2*(maxsize-minsize)) for val in modelSizes]

plt.figure(8,figsize=(15,6))
plt.bar(np.arange(currentSize),Fitness,widths)
plt.xlabel('Models',fontsize=14)
plt.ylabel('ROC AUC',fontsize=14)
ax=plt.gca()
PlotStyle(ax)
