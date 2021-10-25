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

import time
import copy
import gzip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend as K 
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input,Activation,Dense,Layer,BatchNormalization
from keras.layers import Flatten,Reshape,concatenate,Lambda
from keras.layers import Conv2D,Conv2DTranspose

###############################################################################
# Custom configurations
###############################################################################

globalSeed=768

from numpy.random import seed 
seed(globalSeed)

tf.compat.v1.set_random_seed(globalSeed)

#This piece of code is only used if you have a Nvidia RTX or GTX1660 TI graphics card
#for some reason convolutional layers do not work poperly on those graphics cards 

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

###############################################################################
# Visualization functions
###############################################################################

def PlotStyle(Axes): 
    """
    Applies a general style to a plot 
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
    
###############################################################################
# Keras layers
###############################################################################

class KLDivergenceLayer(Layer):
    '''
    Custom KL loss layer
    '''
    
    def _init_(self,*args,**kwargs):
        self.is_placeholder=True
        super(KLDivergenceLayer,self)._init_(*args,**kwargs)
        
    def call(self,inputs):
        
        Mu,LogSigma=inputs
        klbatch=-0.5*(1*(2/784))*K.sum(1+LogSigma-K.square(Mu)-K.exp(LogSigma),axis=-1)
        self.add_loss(K.mean(klbatch),inputs=inputs)
        
        return inputs

class Sampling(Layer):
    '''
    Custom sampling layer
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}
    
    @tf.autograph.experimental.do_not_convert   
    def call(self,inputs,**kwargs):
        
        Mu,LogSigma=inputs
        batch=tf.shape(Mu)[0]
        dim=tf.shape(Mu)[1]
        epsilon=K.random_normal(shape=(batch,dim))

        return Mu+(K.exp(0.5*LogSigma))*epsilon

###############################################################################
# Variational autoencoder bottleneck
###############################################################################

#Wrapper function, creates a small Functional keras model 
#Bottleneck of the variational autoencoder
def MakeVariationalNetwork(Latent):
    
    InputFunction=Input(shape=(Latent,))
    Mu=Dense(Latent)(InputFunction)
    LogSigma=Dense(Latent)(InputFunction)
    Mu,LogSigma=KLDivergenceLayer()([Mu,LogSigma])
    Output=Sampling()([Mu,LogSigma])
    variationalBottleneck=Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,variationalBottleneck

###############################################################################
# Keras Utilituy functions
###############################################################################

def MakeConvolutionBlock(X, Convolutions):
    '''
    Parameters
    ----------
    X : keras functional layer
        Previous layer in the model.
    Convolutions : int
        Number of convolutional filters.

    Returns
    -------
    X : keras functiona layer 
        Block of layers added to the model.
    '''
    
    X=Conv2D(Convolutions, (3, 3), padding='same',use_bias=False)(X)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)

    return X

def MakeDenseBlock(x, Convolutions,Depth):
    '''
    Parameters
    ----------
    x : keras functional layer
        Previous layer in the block.
    Convolutions : int
        Number of convolutional filter to apply.
    Depth : int
        number of convolutional blocks to add.

    Returns
    -------
    concat_feat : keras functional layer 
        concatenated layers.

    '''
    
    concat_feat= x
    for i in range(Depth):
        x = MakeConvolutionBlock(concat_feat,Convolutions)
        concat_feat=concatenate([concat_feat,x])

    return concat_feat

def MakeDenseConvolutionalCoder(InputShape,Units,BlockDepth,UpSampling=False):
    '''
    Parameters
    ----------
    InputShape : tuple
        Input shape of the images.
    Units : Array-like
        Number of convolutional filters to apply per block.
    BlockSize : int
        Size of the concatenated convolutional block.
    UpSampling : bool, optional
        Controls the upsamplig or downsampling behaviour of the network.
        The default is False.

    Returns
    -------
    InputFunction : Keras functional model input
        input of the network.
    localCoder : Keras functional model
        Coder model, main body of the autoencoder.

    '''
    if UpSampling:
        denseUnits=Units[::-1]
        Name="Decoder"
    else:
        denseUnits=Units
        Name="Encoder"
    
    nUnits = len(denseUnits)
    
    InputFunction=Input(shape=InputShape)
    X=Conv2D(denseUnits[0], (3, 3), padding='same',use_bias=False)(InputFunction)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)

    for k in range(1,nUnits-1):
        
        X=MakeDenseBlock(X,denseUnits[k],BlockDepth)
        
        if UpSampling:
            X=Conv2DTranspose(denseUnits[k], (3, 3), padding='same',use_bias=False,strides=(2,2))(X)
        else:
            X=Conv2D(denseUnits[k], (3, 3), padding='same',use_bias=False,strides=(2,2))(X)
    
    if UpSampling:
        
        X=Conv2D(1, (3, 3), padding='same',use_bias=False)(X)
        X=BatchNormalization()(X)
        Output=Activation('sigmoid')(X)
        localCoder=Model(inputs=InputFunction,outputs=Output,name=Name)
        
    else:
        X=Conv2D(denseUnits[-1], (3, 3), padding='same',use_bias=False)(X)
        X=BatchNormalization()(X)
        Output=Activation('relu')(X)
        localCoder=Model(inputs=InputFunction,outputs=Output,name=Name) 

    return InputFunction,localCoder

def MakeConvolutionalBottleneck(InputShape,Latent,UpSampling=False):
    '''
    Parameters
    ----------
    InputShape : tuple
        input shape of the previous convolutional layer.
    Latent : int
        Dimentionality of the latent space.
    UpSampling : bool, optional
        Controls the sampling behaviour of the network.
        The default is False.

    Returns
    -------
    InputFunction : Keras functional model input
        input of the network.
    localCoder : Keras functional model
        Coder model, transition layer of the bottleneck.

    '''
    
    Units=[np.product(InputShape),Latent]
    
    if UpSampling:
        finalUnits=Units[::-1]
        InputFunction=Input(shape=(Latent,))
        X=Dense(finalUnits[0],use_bias=False)(InputFunction)
    
    else:
        finalUnits=Units
        InputFunction=Input(shape=InputShape)
        X=Flatten()(InputFunction)
        X=Dense(finalUnits[0],use_bias=False)(X)
                
    
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    X=Dense(finalUnits[1],use_bias=False)(X)
    X=BatchNormalization()(X)
    
    if UpSampling:
        X=Activation('relu')(X)
        Output=Reshape(InputShape)(X)
    else:
        Output=Activation('relu')(X)
        
    Bottleneck=Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,Bottleneck

###############################################################################
# Autoencoder model
###############################################################################

#Wrapper function joins the Coder function and the bottleneck function 
#to create a simple autoencoder
def MakeConvolutionalAutoencoder(InputShape,Units,BlockSize):
    
    InputEncoder,Encoder=MakeDenseConvolutionalCoder(InputShape,Units,BlockSize)
    EncoderOutputShape=Encoder.layers[-1].output_shape
    BottleneckInputShape=EncoderOutputShape[1::]
    InputBottleneck,Bottleneck=MakeConvolutionalBottleneck(BottleneckInputShape,2)
    ConvEncoderOutput=Bottleneck(Encoder(InputEncoder))
    
    ConvEncoder=Model(inputs=InputEncoder,outputs=ConvEncoderOutput)
    
    rInputBottleneck,rBottleneck=MakeConvolutionalBottleneck(BottleneckInputShape,2,UpSampling=True)
    InputDecoder,Decoder=MakeDenseConvolutionalCoder(BottleneckInputShape,Units,BlockSize,UpSampling=True)
    
    ConvDecoderOutput=Decoder(rBottleneck(rInputBottleneck))
    ConvDecoder=Model(inputs=rInputBottleneck,outputs=ConvDecoderOutput)
    
    
    ConvAEoutput=ConvDecoder(ConvEncoder(InputEncoder))
    ConvAE=Model(inputs=InputEncoder,outputs=ConvAEoutput)
    
    return InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,ConvAE

###############################################################################
# Variational autoencoder model
###############################################################################

# Wrapper functon, joins the autoencoder function with the custom variational
#layers to create an autoencoder
def MakeConvolutionalVariationalAutoencoder(InputShape,Units,BlockSize):
    
    InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,_=MakeConvolutionalAutoencoder(InputShape,Units,BlockSize)
    
    InputVAE,VAE=MakeVariationalNetwork(2)
    VAEencoderOutput=VAE(ConvEncoder(InputEncoder))
    ConvVAEencoder=Model(inputs=InputEncoder,outputs=VAEencoderOutput)
    
    VAEOutput=ConvDecoder(ConvVAEencoder(InputEncoder))
    ConvVAEAE=Model(inputs=InputEncoder,outputs=VAEOutput)
    
    return InputEncoder,InputDecoder,ConvVAEencoder,ConvDecoder,ConvVAEAE

###############################################################################
# Loading the data 
###############################################################################

def FormatData(DataDir,SetSize,ImageSize):
    '''
    
    Parameters
    ----------
    DataDir : string
        location of the data sets.
    SetSize : int
        number of samples in the dataset.
    ImageSize : int
        size of the image.

    Returns
    -------
    Data : numpy array
        dataset.

    '''
    loadZip = gzip.open(DataDir,'r')
    loadZip.read(16)
    Buffer = loadZip.read(ImageSize*ImageSize*SetSize)
    Data = np.frombuffer(Buffer,dtype=np.uint8).astype(np.float64)
    Data = Data.reshape((SetSize,ImageSize,ImageSize))
    Data = Data/255
    
    return Data

TrainDataDir = r'/media/tavoglc/Datasets/DABE/Kaggle/mnist/train-images-idx3-ubyte.gz'
TestDataDir = r'/media/tavoglc/Datasets/DABE/Kaggle/mnist/t10k-images-idx3-ubyte.gz'

ImageSize = 28
TrainSize = 60000
TestSize = 10000

TrainData = FormatData(TrainDataDir,TrainSize,ImageSize)
TestData = FormatData(TestDataDir,TestSize,ImageSize)
FullData = np.vstack((TrainData,TestData))

###############################################################################
#Evolutionary optimization
###############################################################################

#wrapper function, changes the final element in a list to ensure its always 
#equal to MinConvolutions
def FormatPopulation(Population,MinConvolutions):
    
    for particle in Population:
        particle[-1]=MinConvolutions
    
    return Population

def MakeIndexEvolution(IndexPopulation,Boundaries,MinConvolutions,MutationProbability=0.5):
    '''
    Parameters
    ----------
    IndexPopulation : array-like
        2D array with each individual in the population.
    Boundaries : array-like
        list or array of two elements, max and min values possible for an index
        to be found at index population.
    MinConvolutions : int
        min number of convolutional filters.
    MutationProbability : flooat, optional
        Individual mutation probability. The default is 0.5.

    Returns
    -------
    newIndexs : array like
        modified indexes in the population.

    '''
    
    lower,upper = Boundaries
    currentIndexs=copy.deepcopy(IndexPopulation)
    nIndexs=len(IndexPopulation)
    newIndexs = []
    
    for k in range(nIndexs):
        
        indexLocs = np.arange(nIndexs)
        np.random.shuffle(indexLocs)
        localIndex = currentIndexs[k]
        randomIndex = currentIndexs[indexLocs[0]]
        newInd = np.append(localIndex[0:2],randomIndex[1:3])
        newIndexs.append(newInd)
    
    for pop in newIndexs:
        if np.random.random()>MutationProbability:
            populationIndex = np.arange(len(currentIndexs[0]))
            np.random.shuffle(populationIndex)
            pop[populationIndex[0]] = np.random.randint(lower,upper)
            pop[populationIndex[1]] = np.random.randint(lower,upper)
        
    newIndexs = FormatPopulation(newIndexs,MinConvolutions) 
            
    return newIndexs

def TrainOnGenerations(XData,YData,FitnessFunction,Generations,Population,Boundaries,MinConvolutions=5,MutationProbability=0.5):
    '''
    Parameters
    ----------
    XData : array
        X data.
    YData : array
        Y data.
    FitnessFunction : python function
        Wrapper function to iterate through the individuals in the population.
    Generations : int
        Number of iterations.
    Population : int
        number of individuals per iteration.
    Boundaries : array-like
        list or array of two elements, max and min values possible for an index
        to be found at index population.
    MinConvolutions : int
        min number of convolutional filters.
    MutationProbability : flooat, optional
        Individual mutation probability. The default is 0.5.

    Returns
    -------
    fitness : list
        final performance of each individual in the population.
    currentPopulation : array like
        list of each individual in the population.

    '''
    lower,upper = Boundaries
    currentPopulation=np.random.randint(lower,upper,size=(Population,4))
    fitness=FitnessFunction(XData,YData,currentPopulation)
    
    for k in range(Generations):
        
        newPopulation=MakeIndexEvolution(currentPopulation,Boundaries,MinConvolutions)
        newFitness=FitnessFunction(XData,YData,newPopulation)
        
        for k in range(Population):
            if newFitness[k] < fitness[k]:
                currentPopulation[k]=newPopulation[k]
                fitness[k]=newFitness[k]
        
    return fitness,currentPopulation

###############################################################################
#Model training
###############################################################################

#Wrapper function to train a convolutional variational autoencoder
def NetworkFitness(Train,Test,Arch):
    
    Arch = np.sort(Arch)[::-1]
    
    lr=0.005
    minlr=0.00001
    epochs=15
    batch_size=64
    decay=2*(lr-minlr)/epochs

    _,_,ConvVAEEncoder,ConvVAEDecoder,ConvVAE=MakeConvolutionalVariationalAutoencoder((ImageSize,ImageSize,1),Arch,3)
    ConvVAE.summary()

    ConvVAE.compile(Adam(lr=lr,decay=decay),loss='binary_crossentropy')
    ConvVAE.fit(x=Train,y=Train,
                  batch_size=batch_size,epochs=epochs,validation_data=(Test,Test))

    repTrain=ConvVAEEncoder.predict(Train)
    repTest=ConvVAEEncoder.predict(Test)

   
    fig,axes=plt.subplots(2,2,figsize=(15,15))
    
    axes[0,0].plot(repTrain[:,0],repTrain[:,1],'bo',alpha=0.025,label = 'Total Parameters = ' +str(ConvVAE.count_params()))
    axes[0,0].legend(loc=3)
    axes[0,0].set_xlabel('Train Data')
    
    axes[0,1].plot(repTest[:,0],repTest[:,1],'bo',alpha=0.25,label = 'Architecture' + str(Arch) )
    axes[0,1].legend(loc=3)
    axes[0,1].set_xlabel('Test Data')
    
    axes[1,0].plot(ConvVAE.history.history['loss'],'b-',label = 'Loss')
    axes[1,0].plot(ConvVAE.history.history['val_loss'],'r-',label = 'Validation Loss')
    axes[1,0].legend(loc=3)
    
    PlotStyle(axes[0,0])
    PlotStyle(axes[1,0])
    PlotStyle(axes[0,1])
    ImageStyle(axes[1,1])
    
    time.sleep(90)
    
    return ConvVAE.evaluate(Test)

#Wrapper function to iterate through the individuals in the population
def EvolutiveFitness(Train,Test,Population):
    
    fitness = []
    for val in Population:
        performance = NetworkFitness(Train,Test,val)
        fitness.append(performance)
    
    return fitness

###############################################################################
#Model performance
###############################################################################

a,b = TrainOnGenerations(TrainData,TestData,EvolutiveFitness,5,5,[10,60],MinConvolutions=6)

plt.figure(figsize=(7,7))
plt.bar(np.arange(len(a)),a)
plt.ylabel('Reconstruction Loss')
plt.xlabel('Performance')
ax = plt.gca()
PlotStyle(ax)