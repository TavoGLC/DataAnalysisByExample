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

import io
import time
import copy
import gzip
import datetime
import numpy as np
import seaborn as sns
import tensorflow as tf
import statsmodels.api as sm
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image

from tensorflow import keras

from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Activation, Dense, Layer, BatchNormalization
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add,Conv2DTranspose
from tensorflow.keras.layers import Flatten, Reshape, Dropout, LayerNormalization, GlobalAveragePooling2D

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
    Axes.xaxis.set_tick_params(labelsize=10)
    Axes.yaxis.set_tick_params(labelsize=10)
    
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
# Keras Mixer 
#Modified from https://keras.io/examples/vision/convmixer/
###############################################################################

#Wrapper function convolutional mixer block
def ActivationBlock(X):
    
    X = Activation("gelu")(X)
    X = BatchNormalization()(X)
    
    return X

#Extract the patches from the image
def ConvStem(X, filters: int, patch_size: int):
    
    X = Conv2D(filters, kernel_size=patch_size, strides=patch_size)(X)
    X = ActivationBlock(X)
    
    return X


#main body of the ConvMixer
def AddConvMixerBlock(X, filters: int, kernel_size: int):
    # Depthwise convolution.
    X0 = X
    X = DepthwiseConv2D(kernel_size=kernel_size, padding="same")(X)
    X = Add()([ActivationBlock(X), X0])  # Residual.

    # Pointwise convolution.
    X = Conv2D(filters, kernel_size=1)(X)
    X = ActivationBlock(X)

    return X

def MakeMixerUnit(X, filters=256, depth=8, kernel_size=5, patch_size=2):
    
    """
    ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """

    # Extract patch embeddings.
    X = ConvStem(X, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        X = AddConvMixerBlock(X, filters, kernel_size)

    # Classification block.
    outputs = GlobalAveragePooling2D()(X)

    return outputs


def ConvMixerCoder(InputShape,Units,InitialDepth,UpSampling=False):
    
    '''
    Parameters
    ----------
    InputShape : tuple
        Input shape of the network.
    Units : array-like
        Contains the dimentionality of the embedding dimentions.
    InitialDepth : int
        Initial depth of the Mixer unit.
    UpSampling : bool, optional
        Controls the upsamplig or downsampling behaviour of the network.
        The default is False.
    '''
    
    if UpSampling:
        convUnits=Units[::-1]
        depths = np.arange(len(convUnits))[::-1]
    else:
        convUnits=Units
        depths = np.arange(len(convUnits))
    
    CoderInput = Input(shape=InputShape)
    X = MakeMixerUnit(CoderInput,filters=convUnits[0],depth=InitialDepth)
    currentShape=np.sqrt(convUnits[0]).astype(int)
    X = Reshape((currentShape,currentShape,1))(X)
    
    for k in range(1,len(convUnits)):
        X=MakeMixerUnit(X,filters=convUnits[k],depth=InitialDepth+depths[k])
        currentShape=np.sqrt(convUnits[k]).astype(int)
        X = Reshape((currentShape,currentShape,1))(X)
        
    output = Activation('sigmoid')(X)
    coderModel = Model(inputs=CoderInput,outputs=output)
    
    return CoderInput,coderModel

inpshape=(2,2,1)
Units = [784,256,64,16,4]
md =ConvMixerCoder(inpshape,Units,1,UpSampling=True)


###############################################################################
# Keras layers
###############################################################################

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
#Variational autoencoder bottleneck 
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
# Differen learning rate schedules
###############################################################################

def scheduler01(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr / 2
    
def scheduler02(epoch, lr):
    if epoch < 20:
        return (5/6)*lr
    else:
        return lr / 2

def scheduler03(epoch, lr):
    return lr * tf.math.exp(-0.1)

def scheduler04(epoch, lr):
    if epoch < 5:
        return (2/3)*lr
    else:
        return lr/1000000

lrWeight = 1-(1/(1+np.exp(-1*np.linspace(-10,10,num=42))))
baselr=0.005

def scheduler05(epoch, lr):
    return baselr*lrWeight[epoch]

###############################################################################
# Model training 
###############################################################################
#Wrapper function to train a convolutional variational autoencoder
def NetworkFitness(Train,Test,Arch,ModelCoder,shrinkage,schedule,**kwargs):
    
    Arch = np.sort(Arch)[::-1]
    
    lr=0.005
    minlr=0.0001
    epochs=40
    batch_size=64
    decay=2*(lr-minlr)/epochs
    
    input_shape = (28,28,1)

    _,_,Encoder,Decoder,AE = MakeVariationalAutoencoder(ModelCoder,input_shape,Arch,1,shrinkage,**kwargs)
    Encoder.summary()
    Decoder.summary()
    AE.summary()
    
    callback = tf.keras.callbacks.LearningRateScheduler(schedule)
    AE.compile(Adam(lr=lr,decay=decay),loss='binary_crossentropy')
    history = AE.fit(x=Train,y=Train,batch_size=batch_size,epochs=epochs,
                     validation_data=(Test,Test),callbacks=[callback])

    repTrain=Encoder.predict(Train)
    repTest=Encoder.predict(Test)
    
    fitness = AE.evaluate(Test)
    params = AE.count_params()
    
    figcont = MakeLatentSpaceWalk(input_shape[0],30,Decoder)
            
    fig = plt.figure(figsize = (16,8))
    
    gs = plt.GridSpec(6,12)
    
    ax0 = plt.subplot(gs[0:3,6:9])
    ax0.plot(repTrain[:,0],repTrain[:,1],'ko',alpha=0.025)
    ax0.title.set_text('Train Data')
    
    ax1 = plt.subplot(gs[0:3,9:12])
    ax1.plot(repTest[:,0],repTest[:,1],'ko',alpha=0.25)
    ax1.title.set_text('Test Data' )
    
    ax2 = plt.subplot(gs[3,6::])
    ax2.plot(history.history['loss'],'k-',label = 'Loss')
    ax2.plot(history.history['val_loss'],'r-',label = 'Validation Loss')
    ax2.legend(loc=0)
    
    ax3 = plt.subplot(gs[4,6::])
    ax3.plot(history.history['kl_loss'],'k-',label = 'KL Loss')
    ax3.plot(history.history['val_kl_loss'],'r-',label = 'KL Validation Loss')
    ax3.legend(loc=0)
    
    ax4 = plt.subplot(gs[5,6::])
    ax4.plot(np.arange(len(history.history['lr'])),history.history['lr'],'k-',label = 'Learning Rate')
    ax4.set_xlabel('Epochs')
    ax4.legend(loc=0)
    
    ax5 = plt.subplot(gs[:,0:6])
    ax5.imshow(figcont,cmap='inferno')
    ax5.title.set_text('Architecture = ' + str(Arch) +' Total Parameters = ' +str(params))
    
    [PlotStyle(val) for val in [ax0,ax1,ax2,ax3,ax4]]
    ImageStyle(ax5)
    
    ax2.spines['bottom'].set_visible(False)
    ax2.set_xticks([])
    ax3.spines['bottom'].set_visible(False)
    ax3.set_xticks([])
        
    plt.tight_layout()
    
    plt.savefig(r'/media/tavoglc/Datasets/DABE/Kaggle/mnist/fig'+str(str(datetime.datetime.now().time())[0:5])+'.png')
    plt.close(fig)
    time.sleep(90)
    
    return history
   
###############################################################################
# Model perfomance
###############################################################################

schedules = [scheduler01,scheduler02,scheduler03,scheduler04,scheduler05]

data = []

for sch in schedules:
    localData = NetworkFitness(TrainData,TestData,[784,16*16,64,4],ConvMixerCoder,0.00001,sch)
    data.append(localData)
    
    
lengths = []

for val in data:
    lengths.append(sum(np.abs(np.diff(val.history['loss']))))
    
    
plt.figure(figsize=(7,7))
plt.bar(np.arange(len(lengths)),lengths)
plt.ylabel('Loss path lenght')
plt.xlabel('schedule')
ax = plt.gca()
PlotStyle(ax)    
