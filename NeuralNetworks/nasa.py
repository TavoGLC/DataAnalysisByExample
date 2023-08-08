#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License
Copyright (c) 2022 Octavio Gonzalez-Lugo 
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

import os 
import time
import numpy as np
import matplotlib.pyplot as plt

from typing import Sequence,Tuple

import jax
import optax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from sklearn.model_selection import train_test_split

###############################################################################
# Visualization functions
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
    Axes.xaxis.set_tick_params(labelsize=12)
    Axes.yaxis.set_tick_params(labelsize=12)

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
# Loading packages 
###############################################################################
class CoderCONV(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    depth: Sequence[int]
    
    Name: str 
    UpSampling: bool = True
    train: bool = True
                    
    @nn.compact
    def __call__(self,inputs):
        x = inputs
        for k,val in enumerate(self.Units):
            for ii in range(self.depth[k]):
                x = nn.Conv(val,self.Ksize,padding='SAME',use_bias=False,
                            name=self.Name+'_conv_conv_'+str(k)+str(ii))(x)
                x = nn.BatchNorm(use_running_average=not self.train,
                                 name = self.Name+'_conv_norm_'+str(k)+str(ii))(x)
                x = nn.leaky_relu(x)
                
            if self.UpSampling:
                x = nn.ConvTranspose(val,self.Ksize,padding='SAME',
                                     strides=self.Strides,use_bias=False,
                                     name=self.Name+'_convUp_conv_'+str(k)+str(ii))(x)
                x = nn.BatchNorm(use_running_average=not self.train,
                                 name = self.name+'_conv_normUp_'+str(k)+str(ii))(x)
                x = nn.leaky_relu(x)
            else:
                x = nn.Conv(val,self.Ksize,padding='SAME',strides=self.Strides,
                            use_bias=False,
                            name=self.Name+'_convDn_conv_'+str(k)+str(ii))(x)
                x = nn.BatchNorm(use_running_average=not self.train,
                                 name = self.Name+'_conv_normDn_'+str(k)+str(ii))(x)
                x = nn.leaky_relu(x)
            
        return x

class CoderMLP(nn.Module):
    
    Units: Sequence[int]
    Name: str 
    train: bool = True
    
    @nn.compact
    def __call__(self,inputs):
        x = inputs
        for k,feat in enumerate(self.Units):
            x = nn.Dense(feat,use_bias=False,name = self.Name+'_layer_'+str(k))(x)
            x = nn.BatchNorm(use_running_average=not self.train,name = self.Name+'_norm_'+str(k))(x)
            x = nn.leaky_relu(x)
        return x

###############################################################################
# Loading packages 
###############################################################################

class EncoderMLP(nn.Module):
    
    Units: Sequence[int]
    train: bool = True
    
    def setup(self):
        self.encoder = CoderMLP(self.Units[1::],'encoder_nlp',train=self.train)
        self.mean = nn.Dense(self.Units[-1], name='mean')
        self.logvar = nn.Dense(self.Units[-1], name='logvar')
    
    def __call__(self, inputs):
        
        x = inputs
        mlpencoded = self.encoder(x)
        mean_x = self.mean(mlpencoded)
        logvar_x = self.logvar(mlpencoded)
        
        return mean_x, logvar_x

class DecoderMLP(nn.Module):
    
    Units: Sequence[int]
    train: bool = True
    
    def setup(self):
        self.decoder = CoderMLP(self.Units[0:-1],'decoder_mlp',train=self.train)
        self.out = nn.Dense(self.Units[-1],use_bias=False, name='out_mlp')
        self.outnorm = nn.BatchNorm(use_running_average=not self.train,name = 'outnorm_mlp')
    
    def __call__(self, inputs):
        x = inputs
        decoded_1 = self.decoder(x)
        
        out =self.out(decoded_1)
        out = self.outnorm(out)
        out = nn.leaky_relu(out)
        
        return out

###############################################################################
# Loading packages 
###############################################################################

class CONVEncoder(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    InputShape: Tuple[int]
    
    depth: int
    BatchSize: int 
    train: bool = True 
    
    def setup(self):
        
        depths = [1 if self.depth-k+1<=1 else self.depth-k+1 for k,_ in enumerate(self.Units)]
        
        self.localConv = CoderCONV(self.Units,self.Ksize,self.Strides,depths,'encoder_conv',UpSampling=False,train=self.train)
        self.divFactor = 2**(len(self.Units)-1)
        
        self.targetShape = [val//self.divFactor for val in self.InputShape[0:-1]] + [self.Units[-1]]
        
        self.localShape = np.prod(np.array(self.targetShape))
        self.EncUnits = [self.localShape,self.localShape//4,self.localShape//16,2]
        self.localEncoder = EncoderMLP(self.EncUnits,train=self.train)
        
    def __call__(self,inputs):
        
        x = inputs
        x = self.localConv(x)
        x = x.reshape((x.shape[0],-1))
        mean_x,logvar_x = self.localEncoder(x)
        
        return mean_x,logvar_x

class CONVDecoder(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    InputShape: Tuple[int]
    
    outchannels: int
    depth: int
    BatchSize: int
    train: bool = True 
    
    def setup(self):
        
        depths = [1 if self.depth-k+1<=1 else self.depth-k+1 for k,_ in enumerate(self.Units)]
        depths = depths[::-1]
        
        self.localConv = CoderCONV(self.Units[1::],self.Ksize,self.Strides,depths[1::],'decoder_conv',UpSampling=True,train=self.train)
        self.divFactor = 2**(len(self.Units)-1)
        
        self.finalShape = [self.BatchSize]+[val//self.divFactor for val in self.InputShape[0:-1]] + [self.Units[-1]]
        
        self.localShape = np.prod(np.array(self.finalShape[1::]))
        self.DecUnits = [2,self.localShape//16,self.localShape//4,self.localShape]
        
        self.localDecoder = DecoderMLP(self.DecUnits,train=self.train)
        
        self.outnorm = nn.BatchNorm(use_running_average=not self.train,name = 'outnorm_conv')
        self.outConv = nn.Conv(self.outchannels,self.Ksize,padding='SAME',use_bias=False,name='decoder_conv_out')
        
    #@nn.compact
    def __call__(self,inputs):
        
        x = inputs
        x = self.localDecoder(x)
        x = jnp.reshape(jnp.array(x),self.finalShape)
        x = self.localConv(x)
        x = self.outConv(x)
        x = self.outnorm(x)
        x = nn.sigmoid(x)
        
        return x

###############################################################################
# Loading packages 
###############################################################################

def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std

class ConvVAE(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    InputShape: Tuple[int]
    
    outchannels: int
    depth: int
    BatchSize: int
    train: bool = True 
    
    def setup(self):
        self.encoder = CONVEncoder(self.Units,self.Ksize,self.Strides,
                                   self.InputShape,self.depth,self.BatchSize,
                                   self.train)
        self.decoder = CONVDecoder(self.Units[::-1],self.Ksize,self.Strides,
                                   self.InputShape,self.outchannels,
                                   self.depth,self.BatchSize,
                                   self.train)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

###############################################################################
# Loading packages 
###############################################################################

sh = 10**-4

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * sh * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

def MainLoss(Model,params,batchStats,z_rng ,batch):
  
    block, newbatchst = Model().apply({'params': params, 'batch_stats': batchStats}, batch, z_rng,mutable=['batch_stats'])
    recon_x, mean, logvar = block
    kld_loss = kl_divergence(mean, logvar).mean()
    loss_value = optax.l2_loss(recon_x, batch).mean()
    total_loss = loss_value + kld_loss
    
    return total_loss,newbatchst['batch_stats']

def TrainModel(TrainData,TestData,Loss,params,batchStats,rng,epochs=10,batch_size=64,lr=0.005):
    
    totalSteps = epochs*(TrainData.shape[0]//batch_size) + epochs
    stepsPerCycle = totalSteps//4

    esp = [{"init_value":lr/10, 
            "peak_value":(lr)/((k+1)), 
            "decay_steps":int(stepsPerCycle*0.75), 
            "warmup_steps":int(stepsPerCycle*0.25), 
            "end_value":lr/10} for k in range(4)]
    
    Scheduler = optax.sgdr_schedule(esp)

    localOptimizer = optax.adam(learning_rate=Scheduler)
    optState = localOptimizer.init(params)
    
    @jax.jit
    def step(params,batchStats ,optState, z_rng, batch):
        
        (loss_value,batchStats), grads = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch)
        updates, optState = localOptimizer.update(grads, optState, params)
        params = optax.apply_updates(params, updates)
        
        return params,batchStats, optState, loss_value
    
    @jax.jit
    def getloss(params,batchStats, z_rng, batch):
        (loss_value,_), _ = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch)
        return loss_value
    
    trainloss = []
    testloss = []
    
    for epoch in range(epochs):
        
        st = time.time()
        batchtime = []
        losses = []
        
        for k in range(0,len(TrainData),batch_size):

            stb = time.time()
            batch = TrainData[k:k+batch_size]
        
            rng, key = random.split(rng)
            params,batchStats ,optState, lossval = step(params,batchStats,optState,key,batch)
            losses.append(lossval)
            batchtime.append(time.time()-stb)

        
        valloss = []
        for i in range(0,len(TestData),batch_size):
            
            rng, key = random.split(rng)
            val_batch = TestData[i:i+batch_size]
            valloss.append(getloss(params,batchStats,key,val_batch))
        
        mbatch = 1000*np.mean(batchtime)
        meanloss = np.mean(losses)
        meanvalloss = np.mean(valloss)
        
        trainloss.append(meanloss)
        testloss.append(meanvalloss)
        np.random.shuffle(TrainData)
    
        end = time.time()
        output = 'Epoch = '+str(epoch) + ' Time per epoch = ' + str(round(end-st,3)) + 's  Time per batch = ' + str(round(mbatch,3)) + 'ms' + ' Train Loss = ' + str(meanloss) +' Test Loss = ' + str(meanvalloss)
        print(output)
        
    return trainloss,testloss,params,batchStats

##############################################################################
# Data loading 
###############################################################################

trainPath = '/media/tavo/storage/nasa_npy/data'
trainfiles = os.listdir(trainPath)
filesPath = np.array([trainPath+'/'+val for val in trainfiles])

def ProcessData(paths,index,fshape=(32,32,64,1)):
    
    mins = []
    maxs = []
    for val in filesPath:
        
        dta = np.load(val)
        mins.append(dta[index].min())
        mins.append(dta[index+7].min())
        
        maxs.append(dta[index].max())
        maxs.append(dta[index+7].max())
        
    gmin = np.min(mins)
    gmax = np.max(maxs)
    
    cont = []
    
    for val in filesPath:
        
        dta = np.load(val)
        adata = (dta[index].ravel() - gmin)/(gmax-gmin)
        adata = list(adata)
        aadata = adata + [-1 for k in range(np.prod(fshape)-len(adata))]
        bdata = (dta[index+7].ravel()-gmin)/(gmax-gmin)
        bdata = list(bdata)
        bbdata = bdata + [-1 for k in range(np.prod(fshape)-len(bdata))]
        
        cont.append(np.array(aadata).reshape(fshape))
        cont.append(np.array(bbdata).reshape(fshape))
    
    cont = np.stack(cont)
    
    return cont

_,_,trainSamps,testSamps = train_test_split(filesPath,np.arange(len(filesPath)), test_size=0.25, random_state=42)

finaldata = ProcessData(filesPath,2)

###############################################################################
# Loading packages 
###############################################################################

batchSize = 16
InputShape = (32,32,64,1)
outChannels = 1
depth = 4

mainUnits = [32,36,36,36,32]

trainDataI = trainSamps[0:batchSize*(trainSamps.shape[0]//batchSize)]
testDataI = testSamps[0:batchSize*(testSamps.shape[0]//batchSize)]

trainData = finaldata[trainDataI]
testData = finaldata[testDataI]

###############################################################################
# Loading packages 
###############################################################################

def VAEModel():
    return ConvVAE(mainUnits,(3,3,3),(2,2,2),InputShape,outChannels,depth,batchSize)
     
def loss(params,batchStats,z_rng ,batch):
    return MainLoss(VAEModel,params,batchStats,z_rng ,batch)
     
rng = random.PRNGKey(0)
rng, key = random.split(rng)
     
finalShape = tuple([batchSize]+list(InputShape))
init_data = jnp.ones(finalShape, jnp.float32)
initModel = VAEModel().init(key, init_data, rng)
    
params0 = initModel['params']
batchStats = initModel['batch_stats']

trloss,tstloss,params0,batchStats = TrainModel(trainData,testData,loss,params0,
                                               batchStats,rng,lr=0.01,
                                               epochs=50,batch_size=batchSize)

finalParams = {'params':params0,'batch_stats':batchStats}

plt.figure(figsize=(15,7))
plt.plot(trloss,'k-',label = 'Loss')
plt.plot(tstloss,'r-',label = 'Validation Loss')
plt.legend()
ax = plt.gca()
PlotStyle(ax)

###############################################################################
# Loading packages 
###############################################################################

def TransformData(Model,Data,Bsize):

    VariationalRep = []
    rng = random.PRNGKey(0)
    
    for k in range(0,len(Data),Bsize):
        
        batch = Data[k:k+Bsize]
        mu,logvar = Model(batch)
        varfrag = reparameterize(rng,mu,logvar)
        VariationalRep.append(varfrag)
        rng, key = random.split(rng)
            
    VariationalRep = np.vstack(VariationalRep)
    
    return VariationalRep

localparams = {'params':finalParams['params']['encoder'],'batch_stats':finalParams['batch_stats']['encoder']}

def EncoderModel(batch):
    return CONVEncoder(mainUnits,(3,3,3),(2,2,2),InputShape,depth,batchSize,train=False).apply(localparams, batch)

VariationalRep = TransformData(EncoderModel,finaldata,batchSize)

plt.figure(figsize=(15,10))
plt.scatter(VariationalRep[:,0],VariationalRep[:,1])
ax = plt.gca()
PlotStyle(ax)

###############################################################################
# Loading packages 
###############################################################################

decoderparams = {'params':finalParams['params']['decoder'],'batch_stats':finalParams['batch_stats']['decoder']}

def DecoderModel(batch):
    return CONVDecoder(mainUnits,(3,3,3),(2,2,2),InputShape,1,depth,batchSize,train=False).apply(decoderparams, batch)

x = np.linspace(VariationalRep[:,0].min(), VariationalRep[:,0].max(), 8)
y = np.linspace(VariationalRep[:,1].min(), VariationalRep[:,1].max(), 8)
xv, yv = np.meshgrid(x, y)

xy = np.stack((xv.ravel(),yv.ravel())).T

decoded = []

for k in range(0,len(xy),batchSize):
    
    dbatch = DecoderModel(xy[k:k+batchSize])
    for val in dbatch:    
        decoded.append(val.ravel()[0:180*360].reshape(180,360,1))

decoded = np.stack(decoded)

fig,axs = plt.subplots(8,8,figsize=(15,9))

for k,val in enumerate(axs.ravel()):
    val.imshow(decoded[k])
    ImageStyle(val)
    
fig.tight_layout()
