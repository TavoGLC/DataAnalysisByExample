#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sun Dec  8 18:49:43 2019
MIT License
Copyright (c) 2019 Octavio Gonzalez-Lugo 
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
#                          Packages to use 
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, BatchNormalization, Activation

from keras.datasets import mnist

globalSeed=56
from numpy.random import seed
seed(globalSeed)
from tensorflow import set_random_seed
set_random_seed(globalSeed)

###############################################################################
#                    General Plot Functions  
###############################################################################

#Elimates the left and top lines and ticks in a matplotlib plot  
def PlotStyle(Axes,Title):
    
    """
    General plot style function
    """
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.xaxis.set_tick_params(labelsize=14)
    Axes.yaxis.set_tick_params(labelsize=14)
    Axes.set_title(Title)

###############################################################################
#                          Loading the dataset 
###############################################################################

(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

Xtrain = Xtrain.astype('float32') / 255.
Xtest = Xtest.astype('float32') / 255.
Xtrain = Xtrain.reshape((len(Xtrain), np.prod(Xtrain.shape[1:])))
Xtest = Xtest.reshape((len(Xtest), np.prod(Xtest.shape[1:])))

###############################################################################
#                          Vanilla Autoencoder 
###############################################################################
# Modified from https://blog.keras.io/building-autoencoders-in-keras.html

# this is the size of our encoded representations
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer=Adam(lr=0.00025), loss='mse')
autoencoder.fit(Xtrain, Xtrain,epochs=25,batch_size=256,shuffle=True)

baselinePerformance=autoencoder.evaluate(Xtest,Xtest)

###############################################################################
#                        Neural network generation
###############################################################################
latent_dim=2

def MakeEncoder(InputFunction,EncoderArchitecture):
    
    """
    Generates the encoder network using the functional API from keras 
    Its Intended as a wrapper function for TrainAutoencoder 
    
    InputFunction  Input function from the keras functional API 
    EncoderArchitecture A list with the number of dense units in the layer,
                        the lenght of the list is the number of layers in the 
                        network 
    
    """
    
    inputEncoder=InputFunction
    
    en=Dense(EncoderArchitecture[0])(inputEncoder)
    en=Activation('relu')(en)
    
    for j in range(len(EncoderArchitecture)-1):
        
        en=Dense(EncoderArchitecture[j+1])(en)
        en=Activation('relu')(en)
    
    en=Dense(latent_dim)(inputEncoder)
    output=Activation('relu')(en)
    
    Encoder=Model(inputEncoder,output,name='Encoder')
    
    return Encoder

def MakeDecoder(InputFunction,EncoderArchitecture):
    
    """
    Generates the decoder network using the functional API from keras 
    Its Intended as a wrapper function for TrainAutoencoder 
    
    InputFunction  Input function from the keras functional API 
    EncoderArchitecture A list with the number of dense units in the layer,
                        the lenght of the list is the number of layers in the 
                        network 
    
    """
    
    inputDecoder=InputFunction
    reversedArchitecture=EncoderArchitecture[::-1]
    
    dec=Dense(reversedArchitecture[0])(inputDecoder)
    dec=Activation('relu')(dec)
    
    for k in range(len(reversedArchitecture)-1):
        
        dec=Dense(reversedArchitecture[k+1])(dec)
        dec=Activation('relu')(dec)
    
    dec=Dense(784)(dec)
    output=Activation('sigmoid')(dec)
    
    Decoder=Model(inputDecoder,output,name='Decoder')
    
    return Decoder

def TrainAutoencoder(Architecture,TrainData):
    
    """
    Wrapper function to train the autoencoder network
    
    Architecture  A list with the number of dense units in the layer,
                  the lenght of the list is the number of layers in the 
                  network 
    TrainData     Data used to train the autoencoder network.
    
    """
    
    inputEncoder=Input(shape=(784,),name='InputEncoder')
    inputDecoder=Input(shape=(latent_dim,),name='InputDecoder')
    Encoder=MakeEncoder(inputEncoder,Architecture)
    Decoder=MakeDecoder(inputDecoder,Architecture)
    output=Decoder(Encoder(inputEncoder))
    Autoencoder=Model(inputEncoder,output,name='Autoencoder')
    
    Autoencoder.summary()
    Autoencoder.compile(loss='mse',optimizer=Adam(lr=0.00025))
    Autoencoder.fit(TrainData,TrainData,batch_size=256,epochs=25,shuffle=True)
    
    return Encoder, Autoencoder

###############################################################################
#                       Tournament approach
###############################################################################

def TournamentSearch(Population):
    
    """
    
    Generates a population of candidate architectures, train an autoencoder
    and returns the performance of each candidate solution. 
    
    Population  Number of candidate architectures to be generated 
    
    """
    
    localPopulation=Population
    localBound=784
    
    archContainer=[]
    fitness=[]
    params=[]
    
    for k in range(localPopulation):
        
        randomDepth=np.random.randint(2,10)
        randomArchitecture=np.random.randint(2,localBound,size=randomDepth)
        randomArchitecture=list(np.sort(randomArchitecture))
        localArch=randomArchitecture[::-1]
        
        _,localAutoencoder=TrainAutoencoder(localArch,Xtrain)
        
        fitness.append(localAutoencoder.evaluate(Xtest,Xtest))
        params.append(localAutoencoder.count_params())
        
        archContainer.append(localArch)
        
    return archContainer,fitness,params

###############################################################################
#                       Tournament performance 
###############################################################################

archs,performanceTournament,numberOfParams=TournamentSearch(50)
maxParams=np.max(numberOfParams)

performanceImprovement=[100*(1-(val/baselinePerformance)) for val in performanceTournament]
parameterImprovement=[(val/maxParams) for val in numberOfParams]

plt.figure(1,figsize=(10,10))
plt.bar(np.arange(len(performanceImprovement)),performanceImprovement,parameterImprovement)
plt.xlabel('Architectures',fontsize=24)
plt.ylabel('Performance Improvement',fontsize=24)
ax=plt.gca()
PlotStyle(ax,'')

###############################################################################
#                       Evolution approach
###############################################################################
    
def MakeArchitectureMutations(NetworkArchitectures):
    
    """
    
    Modify the candidate solutions and add new randomly generated candidate 
    solutions 
    
    NetworkArchitectures  List of list with the best candidate solutions  
    
    """
    
    localNA=NetworkArchitectures
    localBound=784
    nArchs=len(localNA)
    mutatedArchs=[]
    
    for arch in localNA:
        
        archDepth=len(arch)
        randIndex=np.random.randint(0,archDepth)
        bufferArch=arch.copy()
        bufferArch[randIndex]=np.random.randint(2,localBound)
        mutatedArchs.append(bufferArch)
    
    for k in range(nArchs):
        
        randomDepth=np.random.randint(2,10)
        randomArchitecture=list(np.random.randint(2,localBound,size=randomDepth))
        mutatedArchs.append(randomArchitecture)
        
    return mutatedArchs
        

def NetworkEvolution(Generations,Population):
    
    """
    
    Architecture optimization by evolutionary methods. 
    
    Generation  Number of iterations in the tournament search. 
    Population  Number of candidate architectures to be generated 
    
    """
    
    localGenerations=Generations
    localPopulation=Population
    
    archs,fitness,params=TournamentSearch(localPopulation)
    sortedIndex=np.argsort(np.array(fitness))
    selectedArchs=[archs[val] for val in sortedIndex[0:int(localPopulation/2)] ]
    
    bestArch=[archs[sortedIndex[0]]]
    bestFitness=[fitness[sortedIndex[0]]]
    bestParams=[params[sortedIndex[0]]]
    
    for k in range(localGenerations):
        
        archs=MakeArchitectureMutations(selectedArchs)
        fitness=[]
        params=[]
        
        for localArch in archs:
            
            _,localAutoencoder=TrainAutoencoder(localArch,Xtrain)
            fitness.append(localAutoencoder.evaluate(Xtest,Xtest))
            params.append(localAutoencoder.count_params())
            
        sortedIndex=np.argsort(np.array(fitness))
        
        bestArch.append(archs[sortedIndex[0]])
        bestFitness.append(fitness[sortedIndex[0]])
        bestParams.append(params[sortedIndex[0]])
        
        selectedArchs=[archs[val] for val in sortedIndex[0:int(localPopulation/2)] ]
    
    return bestArch,bestFitness,bestParams
    
###############################################################################
#                       Evolution performance 
###############################################################################

archs1,performanceDE,numberOfParams1=NetworkEvolution(10,6)
maxParams1=np.max(numberOfParams1)

performanceImprovement1=[100*(1-(val/baselinePerformance)) for val in performanceDE]
parameterImprovement1=[(val/maxParams1) for val in numberOfParams1]

plt.figure(2,figsize=(10,10))
plt.bar(np.arange(len(performanceImprovement1)),performanceImprovement1,parameterImprovement1)
plt.xlabel('Architectures',fontsize=24)
plt.ylabel('Performance Improvement',fontsize=24)
ax=plt.gca()
PlotStyle(ax,'')
