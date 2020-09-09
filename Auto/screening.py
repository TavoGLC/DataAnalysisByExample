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

from sklearn import preprocessing as pr
from sklearn import decomposition as dec

from sklearn import linear_model as lmod
from sklearn import discriminant_analysis as da
from sklearn import neighbors as ng
from sklearn import gaussian_process as gp
from sklearn import tree
from sklearn import ensemble as ens
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "winequality-red.csv"

Data=pd.read_csv(DataDir,delimiter=";")
DataHeaders=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
 'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

QualityHeader='quality'

def ToGoodBad(val):
    if val>5:
        return 1
    else:
        return 0
###############################################################################
# Data visualization
###############################################################################

MakePlotGrid(np.array(Data[DataHeaders]),DataHeaders,kind="Histogram")

MakeCorrelationPlot(Data,DataHeaders)

plt.figure()
plt.hist(Data[QualityHeader])
ax=plt.gca()
PlotStyle(ax)

Data["GoodBad"]=Data[QualityHeader].apply(ToGoodBad)

XData=np.array(Data[DataHeaders])
YData=np.array(Data["GoodBad"])

###############################################################################
#Preprocessing
###############################################################################

def DataNormalization(XData,YData,Index):
    """
    Parameters
    ----------
    XData : array 
        X data.
    YData : array
        Y data.
    Index : int
        scaler to be applied.

    Returns
    -------
    Scaler : sklearn scaler object
        trained scaler.
    Xtrain : array
        scaled X train data.
    Xtest : array
        scaled X test data.
    Ytrain : array
        scaled Y train data.
    Ytest : array
        scaled Y test data.
    """
    
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(XData,YData,test_size=0.15,train_size=0.85,random_state=23,stratify=YData)
    
    Scalers=[pr.MinMaxScaler,pr.StandardScaler,pr.MaxAbsScaler,pr.Normalizer,
             pr.QuantileTransformer,pr.PowerTransformer]
    
    if Index>=0 and Index<=4:
        Scaler=Scalers[Index]()
        Scaler.fit(Xtrain)
    elif Index==5:
        Scaler=Scalers[Index](method='yeo-johnson')
        Scaler.fit(Xtrain)
        
    Xtrain=Scaler.transform(Xtrain)
    Xtest=Scaler.transform(Xtest)
        
    return Scaler,Xtrain,Xtest,Ytrain,Ytest

def DimensionalityReduction(XData,Index):
    """
    Parameters
    ----------
    XData : array
        X data.
    Index : int
        kind of dimensionality to be applied.

    Returns
    -------
    dimred : sklearn object
        trained sklearn dimensionality reduction object.

    """
    
    components=XData.shape[1]
    ReductionAlgs=[dec.PCA,dec.SparsePCA,dec.TruncatedSVD,dec.FastICA,
                   dec.FactorAnalysis]
    
    if Index >= 0 and Index<=4:
        dimred=ReductionAlgs[Index](n_components=int(components/2))
        dimred.fit(XData)

    elif Index==5:
        dimred=dec.KernelPCA(n_components=int(components/2),kernel="rbf")
        dimred.fit(XData)
    elif Index==6:
        dimred=dec.KernelPCA(n_components=int(components/2),kernel="sigmoid")
        dimred.fit(XData)
    elif Index==7:
        dimred=dec.KernelPCA(n_components=int(components/2),kernel="poly")
        dimred.fit(XData)
    
    return dimred

def ClassificationAlgorithm(XData,YData,Index):
    """
    Parameters
    ----------
    XData : array
        X data.
    YData : array
        y dta.
    Index : int
        kind of algorithm to use.

    Returns
    -------
    algo : sklearn object
        trained classifier model.

    """
    
    ClassificationAlgs=[lmod.RidgeClassifier,lmod.ARDRegression,lmod.BayesianRidge,
                        lmod.ElasticNet,lmod.Lars,lmod.Lasso,
                        lmod.LassoLars,lmod.LogisticRegression,
                        lmod.OrthogonalMatchingPursuit,lmod.Perceptron,
                        lmod.SGDClassifier,da.LinearDiscriminantAnalysis,
                        da.QuadraticDiscriminantAnalysis,ng.KNeighborsClassifier,
                        gp.GaussianProcessClassifier,
                        tree.DecisionTreeClassifier,ens.RandomForestClassifier,
                        ens.ExtraTreesClassifier,ens.AdaBoostClassifier,ens.GradientBoostingClassifier,
                        svm.LinearSVC,svm.NuSVC]
    
    algo=ClassificationAlgs[Index]()
    algo.fit(XData,YData)
    
    return algo

def AlgorithmFitness(XData,YData,Index,dimreduction=False):
    """
    Parameters
    ----------
    XData : array
        X data.
    YData : array
        Y dta.
    Index : list
        list of integers, specify the normalization, dimensionality reduction and
        classification algorithm.
    dimreduction : bool, optional
        controls if dimensionality reduction is applied. The default is False.

    Returns
    -------
    float
        model performance roc auc.

    """
    
    scaler,xtrain,xtest,ytrain,ytest=DataNormalization(XData,YData,Index[0])
    
    if dimreduction:
        reduction=DimensionalityReduction(xtrain,Index[1])
        xtrain=reduction.transform(xtrain)
        xtest=reduction.transform(xtest)
        
    trainedModel=ClassificationAlgorithm(xtrain,ytrain,Index[2])
    Ypred=trainedModel.predict(xtest)
    
    return roc_auc_score(ytest,Ypred)

###############################################################################
#Random Number Generation
###############################################################################

def MakeRandomInput(Samples):
    """
    Parameters
    ----------
    Samples : int
        length of the list.

    Returns
    -------
    container : list
        list of lists with integer values.

    """
    
    norms=np.arange(0,6)
    reds=np.arange(0,8)
    algs=np.arange(0,22)
    container=[]
    
    for k in range(Samples):
        np.random.shuffle(norms)
        np.random.shuffle(reds)
        np.random.shuffle(algs)
        container.append([norms[0],reds[0],algs[0]])
        
    return container
    
###############################################################################
#Strategy 1 Random Search
###############################################################################

Configs01A=MakeRandomInput(20)
fitness01A=[AlgorithmFitness(XData,YData,inx) for inx in Configs01A]
plt.figure(figsize=(10,6))
plt.bar(np.arange(len(fitness01A)),fitness01A)
plt.xlabel('Models',fontsize=14)
plt.ylabel('Score',fontsize=14)
ax=plt.gca()
PlotStyle(ax)


Configs01B=MakeRandomInput(20)
fitness01B=[AlgorithmFitness(XData,YData,inx,dimreduction=True) for inx in Configs01B]
plt.figure(figsize=(10,6))
plt.bar(np.arange(len(fitness01B)),fitness01B)
plt.xlabel('Models',fontsize=14)
plt.ylabel('Score',fontsize=14)
ax=plt.gca()
PlotStyle(ax)


###############################################################################
#Strategy 2 Random Search & Threshold acceptance
###############################################################################

Configs02A=MakeRandomInput(40)
fitness02A=[]
Configs02AS=[]
for inx in Configs02A:
    fitness=AlgorithmFitness(XData,YData,inx)
    if fitness>0.75:
        fitness02A.append(fitness)
        Configs02AS.append(inx)

plt.figure(figsize=(10,6))
plt.bar(np.arange(len(fitness02A)),fitness02A)
plt.xlabel('Models',fontsize=14)
plt.ylabel('Score',fontsize=14)
ax=plt.gca()
PlotStyle(ax)


Configs02B=MakeRandomInput(40)
fitness02B=[]
Configs02BS=[]
for inx in Configs02B:
    fitness=AlgorithmFitness(XData,YData,inx,dimreduction=True)
    if fitness>0.75:
        fitness02B.append(fitness)
        Configs02BS.append(inx)

plt.figure(figsize=(10,6))
plt.bar(np.arange(len(fitness02B)),fitness02B)
plt.xlabel('Models',fontsize=14)
plt.ylabel('Score',fontsize=14)
ax=plt.gca()
PlotStyle(ax)
