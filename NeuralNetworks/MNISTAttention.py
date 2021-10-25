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
import datetime
import numpy as np
import tensorflow as tf
import statsmodels.api as sm
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from keras import backend as K 
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input,Activation,Dense,Layer,BatchNormalization
from keras.layers import Flatten,Reshape,concatenate,Lambda
from keras.layers import Conv2D,Conv2DTranspose,Dropout

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

def MakeLatentSpaceWalk(ImageSize,Samples,DecoderModel):
    '''
    Parameters
    ----------
    ImageSize : int
        Size of the original image.
    Samples : int
        number of samples to take.
    DecoderModel : keras trained model
        Decoder part of an autoencoder, performs the reconstruction from
        the bottleneck.

    Returns
    -------
    figureContainer : numpy array
        Contains the decode images froim the latent space walk.

    '''

    figureContainer=np.zeros((ImageSize*Samples,ImageSize*Samples))

    gridx = np.linspace(-5,5,Samples)
    gridy = np.linspace(-5,5,Samples)

    for i,yi in enumerate(gridx):
        for j,xi in enumerate(gridy):
            zsample = np.array([[xi,yi]])
            xDec = DecoderModel.predict(zsample)
            digit = xDec[0].reshape(ImageSize,ImageSize)
            figureContainer[i*ImageSize:(i+1)*ImageSize,j*ImageSize:(j+1)*ImageSize] = digit
            
    return figureContainer

###############################################################################
# Keras Utilituy functions
###############################################################################

def MakeConvolutionBlock(X, Convolutions,BatchNorm=True,Drop=True,SpAttention=True,Act='relu'):
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
    Drop : bool, optional
        If true a Droput layer is added to the model. The default is True.
    SpAttention : bool, optional
        If true a SpatialAttention layer is added to the model. The default is True.
    Act : string, optional
        Controls the kind of activation to be used. The default is 'relu'.

    Returns
    -------
    X : keras functiona layer 
        Block of layers added to the model.

    '''
    X = Conv2D(Convolutions, (3,3), padding='same',use_bias=False)(X)
    
    if SpAttention:
        X = SpatialAttention()(X)
        
    if BatchNorm:
        X = BatchNormalization()(X)
        
    if Drop:
        X = Dropout(0.2)(X)
    
    X=Activation(Act)(X)

    return X

def MakeDenseBlock(x, Convolutions,Depth,**kwargs):
    '''
    Parameters
    ----------
    x : keras functional layer
        Previous layer in the block.
    Convolutions : int
        Number of convolutional filter to apply.
    Depth : int
        number of convolutional blocks to add.
    **kwargs 
        keyword arguments from MakeConvolutionBlock.

    Returns
    -------
    concat_feat : keras functional layer 
        concatenated layers.

    '''
    
    concat_feat= x
    for i in range(Depth):
        x = MakeConvolutionBlock(concat_feat,Convolutions,**kwargs)
        concat_feat=concatenate([concat_feat,x])

    return concat_feat

def MakeDenseConvolutionalCoder(InputShape,Units,BlockDepth,UpSampling=False,**kwargs):
    '''
    Parameters
    ----------
    InputShape : tuple
        Input shape of the images.
    Units : Array-like
        Number of convolutional filters to apply per block.
    BlockDepth : int
        Size of the concatenated convolutional block.
    UpSampling : bool, optional
        Controls the upsamplig or downsampling behaviour of the network.
        The default is False.
    **kwargs 
        keyword arguments from MakeConvolutionBlock.

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
    X = Conv2D(denseUnits[0], (3, 3), padding='same',use_bias=False)(InputFunction)
    X=Activation('relu')(X)

    for k in range(1,nUnits-1):
        
        X=MakeDenseBlock(X,denseUnits[k],BlockDepth,**kwargs)
        
        if UpSampling:
            X=Conv2DTranspose(denseUnits[k], (3, 3), padding='same',use_bias=False,strides=(2,2))(X)
        else:
            X=Conv2D(denseUnits[k], (3, 3), padding='same',use_bias=False,strides=(2,2))(X)
    
    if UpSampling:
        
        X=Conv2D(1, (3, 3), padding='same',use_bias=False)(X)
        X=BatchNormalization()(X)
        Output=Activation('sigmoid')(X)
        
    else:
        X=Conv2D(denseUnits[-1], (3, 3), padding='same',use_bias=False)(X)
        X=BatchNormalization()(X)
        Output=Activation('relu')(X)
        
    localCoder=Model(inputs=InputFunction,outputs=Output,name=Name)

    return InputFunction,localCoder


###############################################################################
# Keras Layers
###############################################################################
class SpatialAttention(Layer):
    '''
    Custom Spatial attention layer
    '''
    
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__()
        self.kwargs = kwargs

    def build(self, input_shapes):
        self.conv = Conv2D(filters=1, kernel_size=5, strides=1, padding='same')

    def call(self, inputs):
        pooled_channels = tf.concat(
            [tf.math.reduce_max(inputs, axis=3, keepdims=True),
            tf.math.reduce_mean(inputs, axis=3, keepdims=True)],
            axis=3)

        scale = self.conv(pooled_channels)
        scale = tf.math.sigmoid(scale)

        return inputs * scale


class KLDivergenceLayer(Layer):
    '''
    Custom KL loss layer
    '''
    def __init__(self,shrinkage,*args,**kwargs):
        self.shrinkage = shrinkage
        self.is_placeholder=True
        super(KLDivergenceLayer,self).__init__(*args,**kwargs)
        
    def call(self,inputs):
        
        Mu,LogSigma=inputs
        klbatch=-0.5*self.shrinkage*K.sum(1+LogSigma-K.square(Mu)-K.exp(LogSigma),axis=-1)
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
# Variational bottleneck
###############################################################################

#Wrapper function, creates a small Functional keras model 
#Bottleneck of the variational autoencoder    
def MakeVariationalNetwork(Latent,shrinkage):
    
    InputFunction=Input(shape=(Latent,))
    Mu=Dense(Latent)(InputFunction)
    LogSigma=Dense(Latent)(InputFunction)
    Mu,LogSigma=KLDivergenceLayer(shrinkage)([Mu,LogSigma])
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
def MakeAutoencoder(CoderFunction,InputShape,Units,BlockSize,**kwargs):
    
    InputEncoder,Encoder=CoderFunction(InputShape,Units,BlockSize,**kwargs)
    EncoderOutputShape=Encoder.layers[-1].output_shape
    BottleneckInputShape=EncoderOutputShape[1::]
    InputBottleneck,Bottleneck=MakeBottleneck(BottleneckInputShape,2)
    ConvEncoderOutput=Bottleneck(Encoder(InputEncoder))
    
    ConvEncoder=Model(inputs=InputEncoder,outputs=ConvEncoderOutput)
    
    rInputBottleneck,rBottleneck=MakeBottleneck(BottleneckInputShape,2,UpSampling=True)
    InputDecoder,Decoder=CoderFunction(BottleneckInputShape,Units,BlockSize,UpSampling=True,**kwargs)
    
    ConvDecoderOutput=Decoder(rBottleneck(rInputBottleneck))
    ConvDecoder=Model(inputs=rInputBottleneck,outputs=ConvDecoderOutput)
    
    ConvAEoutput=ConvDecoder(ConvEncoder(InputEncoder))
    ConvAE=Model(inputs=InputEncoder,outputs=ConvAEoutput)
    
    return InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,ConvAE

###############################################################################
# Variational Autoencoder Model
###############################################################################

# Wrapper functon, joins the autoencoder function with the custom variational
#layers to create an autoencoder
def MakeVariationalAutoencoder(CoderFunction,InputShape,Units,BlockSize,shrinkage,**kwargs):
    
    InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,_=MakeAutoencoder(CoderFunction,InputShape,Units,BlockSize,**kwargs)
    
    InputVAE,VAE=MakeVariationalNetwork(2,shrinkage)
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
#Model training 
###############################################################################

#Wrapper function to train a convolutional variational autoencoder
def NetworkFitness(Train,Test,Arch,ModelCoder,shrinkage,**kwargs):
    
    Arch = np.sort(Arch)[::-1]
    
    lr=0.005
    minlr=0.0001
    epochs=25
    batch_size=64
    decay=2*(lr-minlr)/epochs
    
    input_shape = (28,28,1)

    _,_,Encoder,Decoder,AE = MakeVariationalAutoencoder(ModelCoder,input_shape,Arch,3,shrinkage,**kwargs)
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
    
    figcont = MakeLatentSpaceWalk(input_shape[0],30,Decoder)
            
    fig = plt.figure(figsize = (16,8))
    
    gs = plt.GridSpec(4,8)
    
    ax0 = plt.subplot(gs[0:2,0:2])
    ax0.plot(repTrain[:,0],repTrain[:,1],'bo',alpha=0.025)
    ax0.title.set_text('Train Data')
    
    ax1 = plt.subplot(gs[0:2,2:4])
    ax1.plot(repTest[:,0],repTest[:,1],'bo',alpha=0.25)
    ax1.title.set_text('Test Data' )
    
    ax2 = plt.subplot(gs[2,0:4])
    ax2.plot(history.history['loss'],'b-',label = 'Loss')
    ax2.plot(history.history['val_loss'],'r-',label = 'Validation Loss')
    ax2.legend(loc=0)
    
    ax3 = plt.subplot(gs[3,0:4])
    ax3.plot(history.history['kl_loss'],'b-',label = 'KL Loss')
    ax3.plot(history.history['val_kl_loss'],'r-',label = 'KL Validation Loss')
    ax3.legend(loc=0)
    
    
    ax4 = plt.subplot(gs[:,4:])
    ax4.imshow(figcont)
    ax4.title.set_text('Architecture = ' + str(Arch) +' Total Parameters = ' +str(params))
    
    [PlotStyle(val) for val in [ax0,ax1,ax2,ax3]]
    ImageStyle(ax4)
    
    plt.savefig(r'/media/tavoglc/Datasets/DABE/Kaggle/mnist/fig'+str(str(datetime.datetime.now().time())[0:5])+'.png')
    plt.close(fig)
    time.sleep(90)
    
    return fitness[0]

###############################################################################
# Training the network under differnt configurations
###############################################################################

MainArch = [35,25,15,5]

activationNames = ['elu','relu','selu','sigmoid']
configs = [{'BatchNorm':bn, 'Drop':dr, 'SpAttention':spat,'Act':act} for act in activationNames for spat in [True,False] for bn in [True,False] for dr in [True,False]]

performanceA = []

for val in configs:
    a = NetworkFitness(TrainData,TestData,MainArch,MakeDenseConvolutionalCoder,0.001,**val)
    performanceA.append(a)


###############################################################################
#Applying a linear model to the fitness results
###############################################################################

names = ['BatchNorm','Dropout','SpatialAttention','Activation_elu','Activation_relu','Activation_selu','Activation_sigmoid']
container = []

for conf in configs:
    initial = [int(conf[ky]) for ky in ['BatchNorm', 'Drop', 'SpAttention']]
    for k,val in enumerate(activationNames):
        current=[0,0,0,0]
        if conf['Act']==val:
            current[k]=1
            break
            
    initial.extend(current)
    container.append(initial)
    
    
linearModel = sm.OLS(performanceA,np.array(container))
results = linearModel.fit()
results.summary(xname=names)


