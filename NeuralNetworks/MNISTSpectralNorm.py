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
import gzip
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from tensorflow import keras

from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Activation, Dense 
from tensorflow.keras.layers import Flatten, Reshape, concatenate, Layer 
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,BatchNormalization

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
    
    
###############################################################################
# Keras Utilituy functions
###############################################################################

def MakeConvolutionBlock(X, Convolutions,BatchNorm=True,SpectralNorm=True):
    
    '''
    Parameters
    ----------
    X : keras functional layer
        Previous layer in the model.
    Convolutions : int
        Number of convolutional filters.
    BatchNorm : bool, optional
        If True a batchnorm layer is added to the convolutional block. 
        The default is True.
    Spectral : bool, optional
        If true specvtral normalization is added to the convolutional layer. 
        The default is true.


    Returns
    -------
    X : keras functiona layer 
        Block of layers added to the model.
    '''

    if SpectralNorm:
        X = tfa.layers.SpectralNormalization(Conv2D(Convolutions, (3, 3), padding='same',use_bias=False))(X)
    else:
        X = Conv2D(Convolutions, (3, 3), padding='same',use_bias=False)(X)
    
    if BatchNorm:
        X = BatchNormalization()(X)
    
    X=Activation('relu')(X)

    return X

def MakeDenseBlock(x, Convolutions,Depth,BatchNorm=True,SpectralNorm=True):
    '''
    Parameters
    ----------
    x : keras functional layer
        Previous layer in the block.
    Convolutions : int
        Number of convolutional filter to apply.
    Depth : int
        number of convolutional blocks to add.
    BatchNorm : bool, optional
        If True a batchnorm layer is added to the convolutional block. 
        The default is True.
    Spectral : bool, optional
        If true specvtral normalization is added to the convolutional layer. 
        The default is true.

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

def MakeDenseConvolutionalCoder(InputShape,Units,BlockDepth,BatchNorm=True,SpectralNorm=True,UpSampling=False):
    '''
    Parameters
    ----------
    InputShape : tuple
        Input shape of the images.
    Units : Array-like
        Number of convolutional filters to apply per block.
    BlockDepth : int
        Size of the concatenated convolutional block.
    BatchNorm : bool, optional
        If True a batchnorm layer is added to the convolutional block. 
        The default is True.
    Spectral : bool, optional
        If true specvtral normalization is added to the convolutional layer. 
        The default is true.
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
    if SpectralNorm:
        X = tfa.layers.SpectralNormalization(Conv2D(denseUnits[0], (3, 3), padding='same',use_bias=False))(InputFunction)
    else:
        X = Conv2D(denseUnits[0], (3, 3), padding='same',use_bias=False)(InputFunction)
    
    if BatchNorm:
        X=BatchNormalization()(X)

    X=Activation('relu')(X)

    for k in range(1,nUnits-1):
        
        X=MakeDenseBlock(X,denseUnits[k],BlockDepth,BatchNorm=BatchNorm,SpectralNorm=SpectralNorm)
        
        if UpSampling:
            if SpectralNorm:
                X=tfa.layers.SpectralNormalization(
                    Conv2DTranspose(denseUnits[k], (3, 3), padding='same',use_bias=False,strides=(2,2)))(X)
            else:
                X=Conv2DTranspose(denseUnits[k], (3, 3), padding='same',use_bias=False,strides=(2,2))(X)
                
        else:
            if SpectralNorm:
                X=tfa.layers.SpectralNormalization(
                    Conv2D(denseUnits[k], (3, 3), padding='same',use_bias=False,strides=(2,2)))(X)
            else:
                X=Conv2D(denseUnits[k], (3, 3), padding='same',use_bias=False,strides=(2,2))(X)
    
    if UpSampling:
        if SpectralNorm:
            X=X=tfa.layers.SpectralNormalization(
                Conv2D(1, (3, 3), padding='same',use_bias=False))(X)
            
        else:
            X=Conv2D(1, (3, 3), padding='same',use_bias=False)(X)
        
        if BatchNorm:
            X=BatchNormalization()(X)
            
        Output=Activation('sigmoid')(X)
        localCoder=Model(inputs=InputFunction,outputs=Output,name=Name)
        
    else:
        if SpectralNorm:
            X = tfa.layers.SpectralNormalization(Conv2D(denseUnits[-1], (3, 3), padding='same',use_bias=False))(X)
        else:
            X=Conv2D(denseUnits[-1], (3, 3), padding='same',use_bias=False)(X)
        
        if BatchNorm:    
            X=BatchNormalization()(X)
            
        Output=Activation('relu')(X)
        localCoder=Model(inputs=InputFunction,outputs=Output,name=Name) 

    return InputFunction,localCoder

    
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
        klbatch=-0.5*(0.000001)*K.sum(1+LogSigma-K.square(Mu)-K.exp(LogSigma),axis=-1)
        self.add_loss(K.mean(klbatch),inputs=inputs)
        self.add_metric(klbatch,name='kl_loss',aggregation='mean')
        
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
# Autoencoder utility functions
###############################################################################

def MakeBottleneck(InputShape,Latent,UpSampling=False):
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
# Autoencoder Model
###############################################################################

#Wrapper function joins the Coder function and the bottleneck function 
#to create a simple autoencoder
def MakeAutoencoder(CoderFunction,InputShape,Units,BlockSize,BatchNorm=True,SpectralNorm=True):
    
    InputEncoder,Encoder=CoderFunction(InputShape,Units,BlockSize,BatchNorm=BatchNorm,SpectralNorm=SpectralNorm)
    EncoderOutputShape=Encoder.layers[-1].output_shape
    BottleneckInputShape=EncoderOutputShape[1::]
    InputBottleneck,Bottleneck=MakeBottleneck(BottleneckInputShape,2)
    ConvEncoderOutput=Bottleneck(Encoder(InputEncoder))
    
    ConvEncoder=Model(inputs=InputEncoder,outputs=ConvEncoderOutput)
    
    rInputBottleneck,rBottleneck=MakeBottleneck(BottleneckInputShape,2,UpSampling=True)
    InputDecoder,Decoder=CoderFunction(BottleneckInputShape,Units,BlockSize,UpSampling=True,BatchNorm=BatchNorm,SpectralNorm=SpectralNorm)
    
    ConvDecoderOutput=Decoder(rBottleneck(rInputBottleneck))
    ConvDecoder=Model(inputs=rInputBottleneck,outputs=ConvDecoderOutput)
    
    ConvAEoutput=ConvDecoder(ConvEncoder(InputEncoder))
    ConvAE=Model(inputs=InputEncoder,outputs=ConvAEoutput)
    
    return InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,ConvAE

###############################################################################
# Variational Autoencoder model
###############################################################################


# Wrapper functon, joins the autoencoder function with the custom variational
#layers to create an autoencoder
def MakeVariationalAutoencoder(CoderFunction,InputShape,Units,BlockSize,BatchNorm=True,SpectralNorm=True):
    
    InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,_=MakeAutoencoder(CoderFunction,InputShape,Units,BlockSize,BatchNorm=BatchNorm,SpectralNorm=SpectralNorm)
    
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
# Model training 
###############################################################################

#Wrapper function to train a convolutional variational autoencoder
def NetworkFitness(Train,Test,Arch,ModelCoder,BatchNorm=True,SpectralNorm=True):
    
    Arch = np.sort(Arch)[::-1]
    
    lr=0.005
    minlr=0.0001
    epochs=15
    batch_size=64
    decay=2*(lr-minlr)/epochs
    
    input_shape = (28,28,1)

    _,_,Encoder,Decoder,AE = MakeVariationalAutoencoder(ModelCoder,input_shape,Arch,3,BatchNorm=BatchNorm,SpectralNorm=SpectralNorm)
    Encoder.summary()
    Decoder.summary()
    AE.summary()

    AE.compile(Adam(lr=lr,decay=decay),loss='binary_crossentropy')
    history = AE.fit(x=Train,y=Train,batch_size=batch_size,epochs=epochs,
                     validation_data=(Test,Test))

    repTrain=Encoder.predict(Train)
    repTest=Encoder.predict(Test)
    
    fitness = AE.evaluate(Test)
    params = AE.count_params()


    fig,axes=plt.subplots(2,2,figsize=(15,15))
    
    axes[0,0].plot(repTrain[:,0],repTrain[:,1],'bo',alpha=0.025,label = 'Total Parameters = ' +str(params))
    axes[0,0].legend(loc=3)
    axes[0,0].set_xlabel('Train Data')
    
    axes[0,1].plot(repTest[:,0],repTest[:,1],'bo',alpha=0.25,label = 'Architecture' + str(Arch) )
    axes[0,1].legend(loc=3)
    axes[0,1].set_xlabel('Test Data')
    
    axes[1,0].plot(history.history['loss'],'b-',label = 'Loss')
    axes[1,0].plot(history.history['val_loss'],'r-',label = 'Validation Loss')
    axes[1,0].legend(loc=3)
    
    axes[1,1].plot(history.history['kl_loss'],'b-',label = 'KL Loss')
    axes[1,1].plot(history.history['val_kl_loss'],'r-',label = 'KL Validation Loss')
    axes[1,1].legend(loc=3)
    
    [PlotStyle(val) for val in axes.ravel()]
    
    plt.savefig(r'/media/tavoglc/Datasets/DABE/Kaggle/mnist/fig'+str(str(datetime.datetime.now().time())[0:5])+'.png')
    
    #time.sleep(90)
    
    return fitness,params

###############################################################################
# Model performance 
###############################################################################

TestArch = np.random.randint(4,25,size=(30,4))

performanceA = []

for val in TestArch:
    a,b = NetworkFitness(TrainData,TestData,val,MakeDenseConvolutionalCoder,BatchNorm=False,SpectralNorm = False)
    performanceA.append(a)

performanceA = np.array(performanceA)

performanceB = []

for val in TestArch:
    a,b = NetworkFitness(TrainData,TestData,val,MakeDenseConvolutionalCoder,BatchNorm=True,SpectralNorm = False)
    performanceB.append(a)

performanceB = np.array(performanceB)

performanceC = []

for val in TestArch:
    a,b = NetworkFitness(TrainData,TestData,val,MakeDenseConvolutionalCoder,BatchNorm=False,SpectralNorm = True)
    performanceC.append(a)

performanceC = np.array(performanceC)

performanceD = []

for val in TestArch:
    a,b = NetworkFitness(TrainData,TestData,val,MakeDenseConvolutionalCoder,BatchNorm=True,SpectralNorm = True)
    performanceD.append(a)

performanceD = np.array(performanceD)

performanceData = np.vstack((performanceA[:,0],performanceB[:,0],performanceC[:,0],performanceD[:,0]))

plt.figure(figsize=(8,7))
plt.plot(performanceData,color='gray')
plt.plot(np.median(performanceData,axis=1),color='red')
plt.xticks([0,1,2,3],['Base line','Batch Normalization','Spectral Normalization','Batch and Spectral Normalization'],rotation=80)
ax = plt.gca()
PlotStyle(ax)
