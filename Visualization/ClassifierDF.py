#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License
Copyright (c) 2021 Octavio Gonzalez-Lugo 

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

import csv
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import ensemble as ens
from sklearn import preprocessing as pr
from sklearn import decomposition as dec
from sklearn import discriminant_analysis as da

###############################################################################
# Exporting Data Functions
###############################################################################

def DataExporter(Data,DataDir):
    '''
    Simple function to save data into a csv file 

    Parameters
    ----------
    Data : array-like
        Data to be saved.
    DataDir : raw string
        Location of the file to be saved.

    Returns
    -------
    None.

    '''
        
    with open(DataDir,'w',newline='') as output:
        
        writer=csv.writer(output)
            
        for k in range(len(Data)):
            writer.writerow(Data[k])

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
    

def MakeDesitionFunctionPlot(Data,Masks,Grids,TrainedModelFunction,Name):
    '''
    Returns a plot of the desition function of a given trained model. 

    Parameters
    ----------
    Data : array-like
        Data used to train the model.
    Masks : array-like
        Labels for each data point.
    Grids : array-like
        mesh for the mesh function.
    TrainedModelFunction : sklearn model
        Trained sklearn model.
    Name : str
        name of the file to be saved in the global directory.

    Returns
    -------
    None.

    '''
    
    GoodMask,BadMask=Masks
    xx,yy=Grids
    
    z=TrainedModelFunction.decision_function(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    
    DataExporter(z,GlobalDirectory+Name+'mesh.csv')
    
    plt.figure(figsize=(8,8))

    plt.plot(Data[GoodMask,0],Data[GoodMask,1],'bo')
    plt.plot(Data[BadMask,0],Data[BadMask,1],'ro')
    plt.pcolormesh(xx,yy,z,shading='auto',cmap='RdYlBu')
    ax=plt.gca()
    PlotStyle(ax)

    
###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "winequality-red.csv"

Data=pd.read_csv(DataDir,delimiter=";")
DataHeaders=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
 'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    
###############################################################################
# Data manipulations
###############################################################################

QualityHeader='quality'

#Wrapper function to binarize the labels. 
def ToGoodBad(val):
    if val>5:
        return 1
    else:
        return 0

Data["GoodBad"]=Data[QualityHeader].apply(ToGoodBad)

XData=np.array(Data[DataHeaders])
YData=np.array(Data["GoodBad"])

GoodMask=[j for j in range(len(YData)) if YData[j]==1]
BadMask=[j for j in range(len(YData)) if YData[j]==0]

gridSize=400
dmin,dmax=-1,1

xx,yy=np.meshgrid(np.linspace(dmin,dmax,gridSize),
                  np.linspace(dmin,dmax,gridSize))

###############################################################################
# Data Scaling
###############################################################################

Scaler=pr.MinMaxScaler()
Scaler.fit(XData)
XData=Scaler.transform(XData)

DimReduction=dec.PCA(n_components=2)
DimReduction.fit(XData)
XData=DimReduction.transform(XData)

DataExporter(XData,GlobalDirectory+'PCAData.csv')
DataExporter(np.array([YData,Data[QualityHeader]]),GlobalDirectory+'Mask.csv')

###############################################################################
# Model training 
###############################################################################

Models=[da.LinearDiscriminantAnalysis,da.QuadraticDiscriminantAnalysis,
        svm.LinearSVC,svm.NuSVC]

Names=['lda','qda','linearsvc','nusvc']

for nme,model in zip(Names,Models):
    
    cModel=model()
    cModel.fit(XData,YData)
    MakeDesitionFunctionPlot(XData,[GoodMask,BadMask],[xx,yy],cModel,nme)
    
    
    
