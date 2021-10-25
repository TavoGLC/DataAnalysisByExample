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
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

from tensorflow import keras

from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Activation, Dense, Layer, BatchNormalization
from tensorflow.keras.layers import Flatten, Reshape, Dropout, LayerNormalization, GlobalAveragePooling1D

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
#Variational autoencoder bottleneck
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
# Keras Autoencoder body functions
###############################################################################

class Patches(Layer):
    '''
    Taken from
    https://keras.io/examples/vision/mlp_image_classification/
    '''
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        
    @tf.autograph.experimental.do_not_convert
    def call(self, images,**kwargs):
        
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches


class MLPMixerLayer(Layer):
    '''
    Taken from
    https://keras.io/examples/vision/mlp_image_classification/
    '''
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                Dense(units=num_patches),
                tfa.layers.GELU(),
                Dense(units=num_patches),
                Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                Dense(units=num_patches),
                tfa.layers.GELU(),
                Dense(units=hidden_units),
                Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = LayerNormalization(epsilon=1e-6)
    
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs,**kwargs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x


def MakeMixerBlock(inputs,blocks,patch_size,num_patches,embedding_dim,dropout_rate):
    '''
    Parameters
    ----------
    inputs : keras layer
        Input of the mixer block.
    blocks : keras sequential model
        mixer blocks.
    patch_size : int
        size of the image patch, same for each dimention.
    num_patches : int
        number of patches per image.
    embedding_dim : int
        size of the embedding dimention in the mixer block.
    dropout_rate : float
        droput rate in the mixer block.

    Returns
    -------
    representation : keras layer 
        DESCRIPTION.

    '''
    
    patches = Patches(patch_size, num_patches)(inputs)
    x = Dense(units=embedding_dim)(patches)
    x = blocks(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(rate=dropout_rate)(x)
    reshapeDim = np.sqrt(embedding_dim).astype(int)
    representation = Reshape((reshapeDim,reshapeDim,1))(x)
    
    return representation

def MakeMixerCoder(InputShape,Units,NumBlocks,DropoutRate=0.2,PatchSize=4,UpSampling=False):
    '''
    Parameters
    ----------
    InputShape : tuple
        Input shape of the network.
    Units : array-like
        Contains the dimentionality of the embedding dimentions.
    NumBlocks : int
        Number of mixer blocks.
    DropoutRate : float, optional
        Dropout rate of the mixer block. The default is 0.2.
    PatchSize : int, optional
        size of the segmented patch in the image. The default is 4.
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
        EmbeddingDimentions=Units[::-1]
    else:
        EmbeddingDimentions=Units
        
    currentSize = np.sqrt(EmbeddingDimentions[0]).astype(int)
    num_patches = (currentSize//PatchSize)**2
    InputFunction = Input(shape = InputShape)
    MBlocks = keras.Sequential(
        [MLPMixerLayer(num_patches, EmbeddingDimentions[0], DropoutRate) for _ in range(NumBlocks)]
        )
    
    X = MakeMixerBlock(InputFunction,MBlocks,PatchSize,num_patches,EmbeddingDimentions[0],DropoutRate)

    for k in range(1,len(EmbeddingDimentions)):
        
        currentSize = np.sqrt(EmbeddingDimentions[k-1]).astype(int)
        num_patches = (currentSize//PatchSize)**2
        
        MBlocks =  keras.Sequential(
            [MLPMixerLayer(num_patches, EmbeddingDimentions[k], DropoutRate) for _ in range(NumBlocks)]
            )
        X = MakeMixerBlock(X,MBlocks,PatchSize,num_patches,EmbeddingDimentions[k],DropoutRate)

    if UpSampling:
        Output = Activation('sigmoid')(X)
        localCoder = Model(inputs=InputFunction,outputs=Output)
        
    else:
        localCoder = Model(inputs=InputFunction,outputs=X)
    
    return InputFunction,localCoder

###############################################################################
# Autoencoder model
###############################################################################
#Wrapper function joins the Coder function and the bottleneck function 
#to create a simple autoencoder
def MakeMixerAutoencoder(InputShape,Units,BlockSize):
    
    InputEncoder,Encoder=MakeMixerCoder(InputShape,Units,BlockSize)
    EncoderOutputShape=Encoder.layers[-1].output_shape
    BottleneckInputShape=EncoderOutputShape[1::]
    InputBottleneck,Bottleneck=MakeBottleneck(BottleneckInputShape,2)
    ConvEncoderOutput=Bottleneck(Encoder(InputEncoder))
    
    ConvEncoder=Model(inputs=InputEncoder,outputs=ConvEncoderOutput)
    
    rInputBottleneck,rBottleneck=MakeBottleneck(BottleneckInputShape,2,UpSampling=True)
    InputDecoder,Decoder=MakeMixerCoder(BottleneckInputShape,Units,BlockSize,UpSampling=True)
    
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
def MakeMixerVariationalAutoencoder(InputShape,Units,BlockSize):
    
    InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,_=MakeMixerAutoencoder(InputShape,Units,BlockSize)
    
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
# Variational mixer autoencoder
###############################################################################
#Wrapper function to train a convolutional variational autoencoder
def NetworkFitness(Train,Test,Arch):
    
    Arch = np.sort(Arch)[::-1]
    
    lr=0.005
    minlr=0.0001
    epochs=15
    batch_size=64
    decay=2*(lr-minlr)/epochs
    
    input_shape = (28,28,1)

    _,_,mEncoder,mDecoder,mAE = MakeMixerVariationalAutoencoder(input_shape,Arch,3)
    mAE.summary()

    mAE.compile(Adam(lr=lr,decay=decay),loss='binary_crossentropy')
    history = mAE.fit(x=Train,y=Train,
                  batch_size=batch_size,epochs=epochs,validation_data=(Test,Test))

    repTrain=mEncoder.predict(Train)
    repTest=mEncoder.predict(Test)
    
    fitness = mAE.evaluate(Test)
    params = mAE.count_params()


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
    
    PlotStyle(axes[0,0])
    PlotStyle(axes[1,0])
    PlotStyle(axes[0,1])
    ImageStyle(axes[1,1])
    
    plt.savefig(r'/media/tavoglc/Datasets/DABE/Kaggle/mnist/fig'+str(str(datetime.datetime.now().time())[0:5])+'.png')
    
    time.sleep(90)
    
    return fitness,params

###############################################################################
# Autoencoder performance
###############################################################################

fitness = []
params = []
    
for k in range(5):
    
    embds = [784,(20-k)**2,(15-k)**2,(10-k)**2,16]
    fitns,par = NetworkFitness(TrainData,TestData,embds)
    fitness.append(fitns)
    params.append(par)


plt.figure(figsize=(7,7))
plt.bar(np.arange(len(fitness)),fitness)
plt.ylabel('Reconstruction Loss')
plt.xlabel('Performance')
ax = plt.gca()
PlotStyle(ax)

plt.figure(figsize=(7,7))
plt.bar(np.arange(len(params)),params)
plt.ylabel('Parameters')
plt.xlabel('Performance')
ax = plt.gca()
PlotStyle(ax)