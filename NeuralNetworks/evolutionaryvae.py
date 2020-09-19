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
import gzip
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K 

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input,Activation,Dense,Layer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

###############################################################################
# Loading packages 
###############################################################################

globalSeed=768

from numpy.random import seed 
seed(globalSeed)

tf.compat.v1.set_random_seed(globalSeed)

###############################################################################
# Plotting functions
###############################################################################

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
    
def ImageStyle(Axes): 
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
    Axes.spines['bottom'].set_visible(False)
    Axes.spines['left'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.set_xticks([])
    Axes.set_yticks([])

def DisplaySetOfImages(PlotSize,Images,ImagesIndex):
    '''
    Parameters
    ----------
    PlotSize : tuple
        Size of the plot.
    Images : array
        Array of the images.
    ImagesIndex : list, array
        Contains the index of the images to be displayed.

    Returns
    -------
    None.

    '''
    
    NumberOfImages=len(ImagesIndex)
    nrows,ncolumns=GetGridShape(NumberOfImages)
    
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    
    fig,axes=plt.subplots(nrows,ncolumns,figsize=PlotSize)
    
    for val in enumerate(ImagesIndex):
        
        indxP,indxI=val
        currentImage=Images[indxI]
        cInx=subPlotIndexs[indxP]
        axes[cInx].imshow(currentImage,cmap='gray')
        
    [ImageStyle(axes[val]) for val in subPlotIndexs]            
    
###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/media/tavoglc/storage/storage/LocalData/"
TrainDataDir=GlobalDirectory+'train-images-idx3-ubyte.gz'
TestDataDir=GlobalDirectory+'t10k-images-idx3-ubyte.gz'

ImageSize=28
TrainSize=60000
TrainZip=gzip.open(TrainDataDir,'r')

TrainZip.read(16)
TrainBuff=TrainZip.read(ImageSize*ImageSize*TrainSize)

TrainData=np.frombuffer(TrainBuff,dtype=np.uint8).astype(np.float64)
TrainData=TrainData.reshape(TrainSize,ImageSize,ImageSize)
TrainData=TrainData/255
FlatData=TrainData.reshape((TrainSize,ImageSize*ImageSize))

###############################################################################
# Displaying the images
###############################################################################

ImagesIndex=np.arange(len(TrainData))
np.random.shuffle(ImagesIndex)

DisplaySetOfImages((10,10),TrainData,ImagesIndex[0:12])

###############################################################################
# Variational Autoencoder
###############################################################################
    
class KLDivergenceLayer(Layer):
    '''
    Custom layer to add the divergence loss to the final model
    '''
    
    def _init_(self,*args,**kwargs):
        self.is_placeholder=True
        super(KLDivergenceLayer,self)._init_(*args,**kwargs)
        
    def call(self,inputs):
        
        Mu,LogSigma=inputs
        klbatch=-0.5*K.sum(1+LogSigma-K.square(Mu)-K.exp(LogSigma),axis=-1)
        self.add_loss(K.mean(klbatch),inputs=inputs)
        
        return inputs

class Sampling(Layer):
    '''
    Custom Layer for sampling the latent space
    '''
    
    def call(self,inputs):
        
        Mu,LogSigma=inputs
        batch=tf.shape(Mu)[0]
        dim=tf.shape(Mu)[1]
        epsilon=K.random_normal(shape=(batch,dim))

        return Mu+(K.exp(0.5*LogSigma))*epsilon

#Wrapper function for the autoencoder loss
def AutoencoderLoss(Ytrue,Ypred):
    return K.sum(K.binary_crossentropy(Ytrue,Ypred),axis=-1)

def MakeCoder(InputShape,Units,Latent,UpSampling=False):
    '''
    Parameters
    ----------
    InputShape : tuple
        Data shape.
    Units : list
        List with the number of dense units per layer.
    Latent : int
        Size of the latent space.
    UpSampling : bool, optional
        Controls the behaviour of the function, False returns the encoder while True returns the decoder. 
        The default is False.

    Returns
    -------
    InputFunction : Keras Model input function
        Input Used to create the coder.
    localCoder : Keras Model Object
        Keras model.

    '''
    
    if UpSampling:
        denseUnits=Units[::-1]
        Name="Decoder"
    else:
        denseUnits=Units
        Name="Encoder"
    
    InputFunction=Input(shape=InputShape)
    nUnits=len(denseUnits)
    X=Dense(denseUnits[0])(InputFunction)
    X=Activation('relu')(X)
    
    for k in range(1,nUnits-1):
        X=Dense(denseUnits[k])(X)
        X=Activation('relu')(X)
    
    X=Dense(denseUnits[nUnits-1])(X)
    
    if UpSampling:
        Output=Activation('sigmoid')(X)
        localCoder=Model(inputs=InputFunction,outputs=Output,name=Name)
    else:    
        X=Activation('relu')(X)
        Mu=Dense(Latent)(X)
        LogSigma=Dense(Latent)(X)
        Mu,LogSigma=KLDivergenceLayer()([Mu,LogSigma])
        Output=Sampling()([Mu,LogSigma])
        localCoder=Model(inputs=InputFunction,outputs=[Mu,LogSigma,Output],name=Name)
    
    return InputFunction,localCoder

#Wrapper function to create the full variational autoencoder
def MakeVariationalAutoencoder(InputShape,Units,Latent):
    
    InputEncoder,Encoder=MakeCoder(InputShape,Units,Latent)
    InputDecoder,Decoder=MakeCoder((Latent,),Units,Latent,UpSampling=True)
    AEoutput=Decoder(Encoder(InputEncoder)[2])
    VAE=Model(inputs=InputEncoder,outputs=AEoutput)
    
    return Encoder,Decoder,VAE

###############################################################################
# Performance Measurement
###############################################################################
    
testUnits=[784,512,256,128,64]

def LossPerformance(XData,YData,Index):
    '''
    Parameters
    ----------
    XData : array
        Xdata.
    YData : array
        YData.
    Index : list
        List of integer values. Number of dense units per layer

    Returns
    -------
    float
        loss of the model.

    '''
    
    localEncoder,localDecoder,localVAE=MakeVariationalAutoencoder((784,),Index,2)
    localVAE.compile(optimizer=Adam(),loss=AutoencoderLoss)
    localVAE.fit(XData,YData,batch_size=128,epochs=25)
    
    return localVAE.evaluate(XData,YData)


def ClusterPerformance(XData,YData,Index):
    '''
    Parameters
    ----------
    XData : array
        Xdata.
    YData : array
        YData.
    Index : list
        List of integer values. Number of dense units per layer
    
    Returns
    -------
    performance : float
        silhouette coeficient obtained by appling kmeans clustering to the latent recostruction.

    '''
    
    localEncoder,localDecoder,localVAE=MakeVariationalAutoencoder((784,),Index,2)
    localVAE.compile(optimizer=Adam(),loss=AutoencoderLoss)
    localVAE.fit(XData,YData,batch_size=256,epochs=30)
    
    _,_,Latent=localEncoder.predict(XData)
    Clusters=KMeans(n_clusters=10,random_state=globalSeed)
    ClusterLabels=Clusters.fit_predict(Latent)
    performance=silhouette_score(Latent,ClusterLabels)
    
    return performance


###############################################################################
# Random Architecture generation
###############################################################################

def MakeArchitectureLibrary(LibrarySize):
    '''
    Parameters
    ----------
    LibrarySize : int
        Size of the architecture library.

    Returns
    -------
    container : list
        Contains a set of different architectures.

    '''
    
    minArchSize=3
    maxArchSize=10
    minUnits=4
    Units=np.arange(minUnits,(28*28)-1)
    container=[]
    
    for k in range(LibrarySize):
        currentSize=np.random.randint(minArchSize,maxArchSize)
        np.random.shuffle(Units)
        currentArch=[28*28]+list(Units[0:currentSize])
        currentArch=np.sort(currentArch)
        container.append(list(currentArch)[::-1])
        
    return container

def MakeIndexEvolution(IndexPopulation,mutProb=0.25):
    """
    Parameters
    ----------
    IndexPopulation : list
        contains the index for feature generation.
    mutProb : float (0,1)
        probability of mutation.

    Returns
    -------
    currentIndexs : list
        contains the modified indexs.
    """
    currentIndexs=copy.deepcopy(IndexPopulation)
    nIndexs=len(IndexPopulation)
    
    for k in range(nIndexs):
        
        if np.random.random()>mutProb:
            if len(currentIndexs[k])>4:
                randomPosition=np.random.randint(1,len(currentIndexs[k]))
                del currentIndexs[k][randomPosition]
            else:
                currentIndexs[k]=list(currentIndexs[k]+np.arange(0,len(currentIndexs[k])))
        
    return currentIndexs

def TrainOnPopulation(XData,YData,Population,Performance):
    """
    Parameters
    ----------
    XData : array
        X data.
    YData : array
        Y data.
    Population : list
        List with the different architectures.
    Performance : function
        Function that calculates the performance of the model

    Returns
    -------
    fitness : list
        contains the performance of each individual in the population.
    """
    fitness=[]
    for ind in Population:
        fitness.append(Performance(XData,YData,ind))
        
    return fitness

def TrainOnGenerations(XData,YData,Performance,Generations,PopulationSize,minimize=False):
    """
    Parameters
    ----------
    XData : array
        X data.
    YData : array
        Y data.
    Performance : function
        Function that calculates the performance of the model
    Generations : int
        Number of iterations.
    PopulationSize : int
        number of individuals per iteration.
    minimize : bool
        Algorithm behaviour, minimization or maximization of the objetive function.
        The default is false 

    Returns
    -------
    fitness : list
        performance of each individual in the population.
    currentPopulation : list
        contains the index of the las population in the iteration.
    """
    currentPopulation=MakeArchitectureLibrary(PopulationSize)
    fitness=TrainOnPopulation(XData,YData,currentPopulation,Performance)
    
    for k in range(Generations):
        
        newPopulation=MakeIndexEvolution(currentPopulation)
        newFitness=TrainOnPopulation(XData,YData,newPopulation,Performance)
        
        for k in range(PopulationSize):
            if minimize:
                if newFitness[k]<fitness[k]:
                    currentPopulation[k]=newPopulation[k]
                    fitness[k]=newFitness[k]
            else:
                if newFitness[k]>fitness[k]:
                    currentPopulation[k]=newPopulation[k]
                    fitness[k]=newFitness[k]
        
    return fitness,currentPopulation

###############################################################################
# Random Architecture generation
###############################################################################

Fitness,Population=TrainOnGenerations(FlatData,FlatData,ClusterPerformance,5,10,minimize=False)
NetworkDepth=[(len(val)/10) for val in Population]


plt.figure(2,figsize=(10,10))
plt.bar(np.arange(len(Fitness)),Fitness,NetworkDepth)
plt.xlabel('Architectures',fontsize=24)
plt.ylabel('Performance',fontsize=24)
ax=plt.gca()
PlotStyle(ax)
