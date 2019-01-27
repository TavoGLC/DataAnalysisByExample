# -*- coding: utf-8 -*-
"""

@author: TavoGLC
From:

-Mining biological databases: UniProt-

"""

###############################################################################
#                          Libraries to use  
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from matplotlib.gridspec import GridSpec
from scipy.spatial import distance as ds
from sklearn.decomposition import PCA,KernelPCA,SparsePCA

###############################################################################
#                    General Plot Functions  
###############################################################################

#Elimates the left and top lines and ticks in a matplotlib plot  
def PlotStyle(Axes,Title):
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.xaxis.set_tick_params(labelsize=14)
    Axes.yaxis.set_tick_params(labelsize=14)
    Axes.set_title(Title)
    
#
def PanelPlot(Figure,Data):
    
    cFig=Figure
    gridSp=GridSpec(2,2)
    DS1,DS2,DS3=Data
    
    ax1=cFig.add_subplot(gridSp[0,0])
    ax1.plot(DS1[:,0],DS1[:,1] ,'bo')
    PlotStyle(ax1,'Principal Component Analysis (PCA)')

    ax2=cFig.add_subplot(gridSp[0,1])
    ax2.plot(DS2[:,0],DS2[:,1] ,'bo')
    PlotStyle(ax2,'Sparse PCA')
    
    ax3=cFig.add_subplot(gridSp[1,0])
    ax3.plot(DS3[:,0],DS3[:,1] ,'bo')
    PlotStyle(ax3,'Kernel PCA')    
    
###############################################################################
#                          Libraries to use  
###############################################################################

##Global Directory
GlobalDir='Global data directory'

#Data directory
DataDir=GlobalDir+'\\'+'Data'

#Localization Data
LocDataFile=DataDir+'\\'+'LOCData.csv'

#GO data 
GODataFile=DataDir+'\\'+'GOData.csv'

###############################################################################
#                          Loading Data  
###############################################################################

#Loads the matrix generated in the previous post 
GOIndex=np.genfromtxt(GOIndexFile,delimiter=',')
GOData=np.genfromtxt(GODataFile,delimiter=',')

###############################################################################
#                          Libraries to use  
###############################################################################

#Performs 3 of the most common dimensionality reductions techniques 
def DimensionalityReduction(Data):
    
    cData=Data
    pca=PCA(n_components=2)
    kpca=KernelPCA(kernel='rbf',n_components=2)
    spca=SparsePCA(n_components=2)

    pData=pca.fit_transform(cData)
    sData=kpca.fit_transform(cData)
    kData=spca.fit_transform(cData)
    
    return pData,sData,kData

#Dimentionali reduced data 
LocReduced=DimensionalityReduction(LocData)
GOReduced=DimensionalityReduction(GOData)

###############################################################################
#                          Libraries to use  
###############################################################################

#Plotting the location data 
fig=plt.figure(1,figsize=(12,12))

PanelPlot(fig,LocReduced)

plt.suptitle('Location Data')

#Ploting the GO data 
fig=plt.figure(2,figsize=(12,12))

PanelPlot(fig,GOReduced)

plt.suptitle('Gene Ontology Data')

###############################################################################
#                          Libraries to use  
###############################################################################

#DBSCAN optimization 
def GetDBSCANParameters(Data):
    
    cData=Data
    nData=len(Data)
    
    #Estimation of the min number of samples 
    MinSamples=int(np.log(nData)) 
    
    Container=[]
    
    #Estimation of eps, calculates the distance between points and saves the minimmum distance 
    #of a given pair 
    for k in range(nData-1):
        
        cVal=cData[k]
        LoopContainer=[]
        
        for j in range(k+1,nData):
            
            LoopContainer.append(ds.euclidean(cVal,cData[j]))
            
        Container.append(np.min(LoopContainer))
    
    #Calculates the histogram of the distances data, as the distances histogram is skewed towards 0
    #we only take the 3 first intervals to calculate eps  
    Counts,Intervals=np.histogram(Container)
    EPS=np.mean(Intervals[0:3])
    
    return MinSamples,EPS
    
#Calculates the clusters and returns a dictionary of the cluster location 
def GetClusterDict(Data,EPS,MinSamples):
    
    cData=Data
    Clusters=DBSCAN(eps=EPS, min_samples=MinSamples).fit(cData)
    Labels=Clusters.labels_
    
    k=0
    LocalDict={}
    
    for val in Labels:
    
        LocalDict[k]=str(val)
        k=k+1
        
    return LocalDict, len(set(Labels))

#Given a dictionary takes the data of a given cluster 
def ClusterData(Data,CurrentDictionary,ClusterNumber):
    
    cData=[list(Data[k]) for k in range(len(Data)) if CurrentDictionary[k]==ClusterNumber]
    
    return np.array(cData)

#Plot of the different clusters in a dataset 
def ClusterPlot(Data,Axes,PlotTitle):
    
    cData=Data
    LocalSamp,LocalEPS=GetDBSCANParameters(cData)
    LDict,nClusters=GetClusterDict(cData,LocalEPS,LocalSamp)
    
    #Generates evenly spaced colors from the viridis colormap 
    LocalColors=plt.cm.viridis(np.linspace(0, 1,nClusters),alpha=0.75)
    
    for k in range(nClusters):
        
        LocalData=ClusterData(cData,LDict,str(k))
        
        #When a cluster have only one element the code will raise an index error, often that data point do not belong to any cluster
        #and DBSCAN will asign the -1 cluster number  
        try:
            
            Axes.plot(LocalData[:,0],LocalData[:,1],'o',markerfacecolor=LocalColors[k],markeredgecolor=LocalColors[k],markersize=14,label='Cluster '+str(k))
            
        except IndexError:
            
            pass
     
    PlotStyle(Axes,PlotTitle)

#Creates a panel of the cluster plots 
def PanelClusterPlot(Figure,Data):
    
    cFig=Figure
    gridSp=GridSpec(2,2)
    DS1,DS2,DS3=Data
    
    ax1=cFig.add_subplot(gridSp[0,0])
    ClusterPlot(DS1,ax1,'Principal Component Analysis (PCA)')

    ax2=cFig.add_subplot(gridSp[0,1])
    ClusterPlot(DS2,ax2,'Sparse PCA')
    
    ax3=cFig.add_subplot(gridSp[1,0])
    ClusterPlot(DS3,ax3,'Kernel PCA')

###############################################################################
#                          Libraries to use  
###############################################################################

#Plotting the location data 
fig=plt.figure(3,figsize=(12,12))

PanelClusterPlot(fig,LocReduced)

plt.suptitle('Location Data')

#Plotting the location data 
fig=plt.figure(4,figsize=(12,12))

PanelClusterPlot(fig,GOReduced)

plt.suptitle('GO Data')
