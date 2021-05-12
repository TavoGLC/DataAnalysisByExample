#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 00:59:58 2021

@author: tavoglc
"""

###############################################################################
# Loading packages 
###############################################################################

import csv
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.spatial import distance as ds

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

def MakeMatrixPlot(Data,Labels):
    """
    Parameters
    ----------
    Data : array-like
        Data to be analized.
    Labels : list
        List with the data headers for correlation analysis.

    Returns
    -------
    None.

    """
    
    plt.figure(figsize=(10,10))
    plt.imshow(Data)
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


CorrData=Data.corr()
CorrData.to_csv(GlobalDirectory+"Correlation.csv")
    
###############################################################################
# Plotting functions
###############################################################################

MakeMatrixPlot(Data.corr(),Data.corr().keys())

###############################################################################
# Loading the data
###############################################################################

UniqueQuality=np.sort(Data["quality"].unique())
QualityLabels=["Quality Score "+str(val) for val in UniqueQuality]

def GetSimilarityMatrix(Data,Mask,Similarity):
    """
    Parameters
    ----------
    Data : array-like
        Data to be analized.
    Mask : array-like
        values to be grouped for comparison.
    Similarity : function 
        Function for pairwise comparison.

    Returns
    -------
    ContainerMatrix : array
        Comparison data.

    """
    
    ContainerMatrix=np.zeros((len(Mask),len(Mask)))
    
    for k in range(len(Mask)):
        kvals= Data[Data["quality"]==Mask[k]].mean()
        for j in range(k,len(Mask)):
            jvals= Data[Data["quality"]==Mask[j]].mean()
            ContainerMatrix[k,j]=Similarity(kvals,jvals)
            ContainerMatrix[j,k]=Similarity(kvals,jvals)
            
    return ContainerMatrix

###############################################################################
# Loading the data
###############################################################################

SimilarityMeasures=[ds.euclidean,ds.correlation,ds.cosine,ds.canberra,ds.cityblock]
SimilarityNames=["euclidean","correlation","cosine","canberra","cityblock"]

for val,sal in zip(SimilarityMeasures,SimilarityNames):
    
    MatrixData0=GetSimilarityMatrix(Data,UniqueQuality,val)
    MakeMatrixPlot(MatrixData0,QualityLabels)
    DataExporter(MatrixData0,GlobalDirectory+sal+".csv")