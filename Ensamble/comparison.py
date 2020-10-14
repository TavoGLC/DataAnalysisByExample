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


from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pr
from sklearn.model_selection import KFold

import xgboost as xgb
import lightgbm as lgb

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "SeoulBikeData.csv"

Data=pd.read_csv(DataDir)


def UniqueToDictionary(UniqueElements):
    
    localDictionary={}
    nElements=len(UniqueElements)
    
    for k in range(nElements):
        localDictionary[UniqueElements[k]]=k
        
    return localDictionary

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
    
    _,numberOfElements=Data[Labels].shape
    
    nrows,ncolumns=GetGridShape(numberOfElements)
    
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,13))
    
    for k in range(numberOfElements):
        
        if kind=="Scatter":
            axes[subPlotIndexs[k]].plot(Data[Labels[k]],'bo',alpha=0.5)
            axes[subPlotIndexs[k]].set_xlabel(Labels[k])
        elif kind=="Histogram":
            axes[subPlotIndexs[k]].hist(Data[Labels[k]],bins=75)
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

#Wrapper function for a simple bar plot
def MakeImportancesPlot(Data,FeaturesNames,PlotSize=(8,6)):

    Xaxis=np.arange(len(Data))
    plt.figure(figsize=PlotSize)
    barColors=[plt.cm.cividis(val) for val in np.linspace(0,1,num=len(Data))]
    plt.bar(Xaxis,Data,color=barColors,edgecolor=barColors)
    axis=plt.gca()
    axis.set_xticks(Xaxis)
    axis.set_xticklabels(FeaturesNames,rotation=85)
    PlotStyle(axis)

def MakeResidualsPlot(XData,YData,Model):
    """
    Parameters
    ----------
    XData : 2D array
        X data.
    YData : array
        Y data.
    Model : trained model object 
        Trained ensemble model (RandomForest, lightgbm, xgboost, catboost).

    Returns
    -------
    None.

    """
    
    if str(type(Model))=="<class 'xgboost.core.Booster'>":
        Ypred=Model.predict(xgb.DMatrix(XData))
    else:
        Ypred=Model.predict(XData)
        
    diffs=np.array(YData-Ypred)
    
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    subPlotIndexs=[(j,k) for j in range(2) for k in range(2)]
    
    vals,reg=stats.probplot(diffs)
    
    axes[0,0].plot(diffs)
    axes[0,0].set_ylabel("Difference (Data-Prediction)")
    axes[0,1].hist(diffs,bins=75)
    axes[0,1].set_xlabel("Difference (Data-Prediction)")
    
    axes[1,0].hist(np.abs(diffs),bins=75)
    axes[1,0].set_xlabel("Absolute Difference")
    axes[1,1].plot(vals[0],vals[1],'bo')
    axes[1,1].set_xlabel("Theoretical Quantiles")
    axes[1,1].set_ylabel("Ordered Values")
    
    for indx in subPlotIndexs:
        PlotStyle(axes[indx])
    plt.tight_layout()
    
###############################################################################
# Data frame functions
###############################################################################
    
def AddEncodedTemporalFeatures(DataFrame,TimeSeriesHeader):
    """
    Adds in place Temporal encoded series features
    Parameters
    ----------
    DataFrame : pandas dataframe
        Data to add the features.
    TimeSeriesHeader : string
        Column name of the time series in the DataFrame.

    Returns
    -------
    DataFrame : pandas data frame
        Data frame with the encoded features.

    """
    
    TimeSeries=pd.to_datetime(Data[TimeSeriesHeader],format='%d/%m/%Y')
    DataFrame["DayOfWeek"]=TimeSeries.dt.dayofweek.apply(int)
    DataFrame["Month"]=TimeSeries.dt.month.apply(int)
    DataFrame['Year']=TimeSeries.dt.year.apply(int)
    
    return DataFrame

def AddEncodedCategoricalFeatures(DataFrame,CategoricalFeatures):
    """
    Adds numerically encoded features in the dataframe
    Parameters
    ----------
    DataFrame : pandas dataframe
        Data to add the features.
    CategoricalFeatures : list
        List with the data headers of the categorical values.

    Returns
    -------
    DataFrame : pandas data frame
        Data frame with the encoded features.
    UniqueContainer : list
        List with the unique values in the data frame used to encode the data.

    """
    
    UniqueContainer=[]
    
    for val in CategoricalFeatures:
        currentUnique=DataFrame[val].unique()
        localDict=UniqueToDictionary(currentUnique)
        DataFrame[val+"Encoded"]=[localDict[sal] for sal in DataFrame[val]]
        UniqueContainer.append(currentUnique)
    
    return DataFrame,UniqueContainer
        

###############################################################################
# Data visualization
###############################################################################

Data=AddEncodedTemporalFeatures(Data,"Date")

CategoricalHeaders=['Seasons','Holiday','Functioning Day']

Data,_=AddEncodedCategoricalFeatures(Data,CategoricalHeaders)

NumericalHeaders=['Rented Bike Count','Hour','Temperature(C)','Humidity(%)',
                  'Wind speed (m/s)','Visibility (10m)','Dew point temperature(C)',
                  'Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)',
                  'DayOfWeek','Month','Year']

Target='Rented Bike Count'
TrainDataHeaders=[val for val in NumericalHeaders if val!=Target]
DataHeaders=TrainDataHeaders+[val+"Encoded" for val in CategoricalHeaders]

MakePlotGrid(Data,NumericalHeaders)
MakePlotGrid(Data,NumericalHeaders,kind="Histogram")
MakeCorrelationPlot(Data,NumericalHeaders)

TrainData=Data[DataHeaders]
Target=Data[Target]

###############################################################################
# Ensemble wrapper
###############################################################################

#Wrapper function for scikit RandomForestRegressor
def RFRegressor(Xtrain,Xtest,Ytrain,Ytest,Params):
    
    localRegressor=RandomForestRegressor(**Params)
    localRegressor.fit(Xtrain,Ytrain)
    localScore=np.mean((Ytest-localRegressor.predict(Xtest))**2)
    
    return localRegressor,localScore

#Wrapper function for lightgbm
def LGBRegressor(Xtrain,Xtest,Ytrain,Ytest,Params):
    
    localTrain=lgb.Dataset(Xtrain,label=Ytrain)
    localRegressor=lgb.train(Params,localTrain)
    localScore=np.mean((Ytest-localRegressor.predict(Xtest))**2)
    
    return localRegressor,localScore

#Wrapper function for xgboost
def XGBRegressor(Xtrain,Xtest,Ytrain,Ytest,Params):
    
    localTrain=xgb.DMatrix(Xtrain,label=Ytrain)
    Xtest=xgb.DMatrix(Xtest)
    localRegressor=xgb.train(Params,localTrain)
    localScore=np.mean((Ytest-localRegressor.predict(Xtest))**2)
    
    return localRegressor,localScore

#Wrapper function for Catboost
def CatRegressor(Xtrain,Xtest,Ytrain,Ytest,Params):
    
    localRegressor=CatBoostRegressor(**Params)
    localRegressor.fit(Xtrain,Ytrain)
    localScore=np.mean((Ytest-localRegressor.predict(Xtest))**2)
    
    return localRegressor,localScore

def TrainRegressor(XData,YData,Params,kind="RF"):
    """
    Parameters
    ----------
    XData : 2D array
        X data.
    YData : array
        Y data.
    Params : dict
        hyperparameters for the algorithm.
    kind : string, optional
        kind of algorithm to be used. The default is "RF".

    Returns
    -------
    localRegressor : Trained model object
        Trained model.
    score : float
        performance of the model.
    Xtest : array
        Test dataset.
    Ytest : array
        Test dataset.

    """
    
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(XData,YData,test_size=0.15,train_size=0.85,random_state=23)
    
    if kind=="RF":
        localRegressor,score=RFRegressor(Xtrain,Xtest,Ytrain,Ytest,Params)
    elif kind=="LGB":
        localRegressor,score=LGBRegressor(Xtrain,Xtest,Ytrain,Ytest,Params)
    elif kind=="XGB":
        localRegressor,score=XGBRegressor(Xtrain,Xtest,Ytrain,Ytest,Params)
    elif kind=="CAT":
        localRegressor,score=CatRegressor(Xtrain,Xtest,Ytrain,Ytest,Params)
        
    return localRegressor,score,Xtest,Ytest
    
###############################################################################
# Model Comparison
###############################################################################

RFModel,Rscore,Xtest,Ytest=TrainRegressor(TrainData,Target,{"n_estimators":100})
MakeImportancesPlot(RFModel.feature_importances_,DataHeaders)
MakeResidualsPlot(Xtest,Ytest,RFModel)

LGBModel,Lscore,_,_=TrainRegressor(TrainData,Target,{"objetive":"mse"},kind="LGB")
MakeImportancesPlot(LGBModel.feature_importance(),DataHeaders)
MakeResidualsPlot(Xtest,Ytest,LGBModel)

XGBModel,Xscore,_,_=TrainRegressor(TrainData,Target,{"objetive":"mse"},kind="XGB")
MakeImportancesPlot(XGBModel.get_score().values(),XGBModel.get_score().keys())
MakeResidualsPlot(Xtest,Ytest,XGBModel)

CATModel,Cscore,_,_=TrainRegressor(TrainData,Target,{"iterations":1000},kind="CAT")
MakeImportancesPlot(CATModel.get_feature_importance(),DataHeaders)
MakeResidualsPlot(Xtest,Ytest,CATModel)
