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

import re
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.cluster import DBSCAN
from sklearn import preprocessing as pr

from wordcloud import WordCloud
from scipy.spatial import distance as ds

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense,Activation,BatchNormalization

globalSeed=50
from numpy.random import seed
seed(globalSeed)
from tensorflow import set_random_seed
set_random_seed(globalSeed)

###############################################################################
#                          Data Location
###############################################################################

GlobalDirectory= r'/media/tavoglc/storage/storage/Medium/Life sciences/Mining Pubmed/Data/'
AbstractDataDir=GlobalDirectory+'Abstracts.txt'

###############################################################################
#                          Plot Style Functions 
###############################################################################

#Elimates the left and top lines and ticks in a matplotlib plot 
def PlotStyle(Axes,Title,xlabel,ylabel):
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.tick_params(labelsize=18)
    Axes.set_xlabel(xlabel,fontsize = 20) 
    Axes.set_ylabel(ylabel,fontsize = 20) 
    Axes.set_title(Title,fontsize= 25)

###############################################################################
#                          Data Preparation
###############################################################################

#Open a text file and split it by lines 
def GetFileLines(Dir):
    
    with open(Dir,'r') as file:
        Lines=[]
        for lines in file.readlines():
            Lines.append(lines)
    
    return Lines

#Select each abstract entry in the text
def MakeAbstractIndexs(AbstractData):
    
    cData=AbstractData

    DataDelimiters=[j for j in range(len(cData)) if re.match('(.*)PMID(.*)',cData[j])]
    DataDelimiters.insert(0,0)

    DataToIndex={}

    for k in range(len(DataDelimiters)-1):
    
        DataToIndex[k]=[j for j in range(DataDelimiters[k],DataDelimiters[k+1])]
        
    return DataToIndex

#Select and clean the abstract text, removes the \n and special characters
def GetMainText(AbstractSlice):
    
    cData=AbstractSlice
    blankSpaces=[j for j in range(len(cData)) if cData[j]=='\n']
    differences=[blankSpaces[k]-blankSpaces[k+1] for k in range(len(blankSpaces)-1)]
    
    try:
        
        order=np.argsort(differences)
        currentAbstract=cData[blankSpaces[order[0]]+1:blankSpaces[order[0]+1]]
        mainText=''.join(currentAbstract)
        mainText=re.sub(r'[\n]',' ',mainText)
        mainText=re.sub(r'[^\w\s]',' ',mainText)
    
        return mainText.lower()
    
    except IndexError:
         
         return 'no abstract'

#For each abstract in the data set selects the unique terms in all the data set
def UniqueDataSetTokens(TextData):
  
  cData=TextData
  nData=len(cData)
    
  def SplitAndReduce(TargetString):
    return list(set(TargetString.split()))
  
  container=SplitAndReduce(cData[0])
  
  for k in range(1,nData):
    container=container+SplitAndReduce(cData[k])
    if k%100==0:  
      container=list(set(container))
  
  return container

###############################################################################
#                          Unique Terms Selection
###############################################################################

AbstractData=np.array(GetFileLines(AbstractDataDir))
AbstractToIndex=MakeAbstractIndexs(AbstractData)
mainTexts=[GetMainText(AbstractData[AbstractToIndex[k]]) for k in range(len(AbstractToIndex))]

#Unique terms in the data set
mainTextToken=UniqueDataSetTokens(mainTexts)


#Clean the data from terms with only one character, numbers, or russian letters
filteredToken=[]

for val in mainTextToken:
    
    try:
        float(val)
    except ValueError:
        
        if len(val)==1 or len(val)==2:
            pass
        else:
            
            if re.match('[а-яА-Я]',val):
                pass
            else:
                filteredToken.append(val)

#remove the stopwords in the text
s=stopwords.words('english')
reducedToken=list(set(filteredToken)-set(s))

tokenToLocation={}

for k in range(len(reducedToken)):
    tokenToLocation[reducedToken[k]]=k

tokenLenghts=[len(val) for val in reducedToken]

plt.figure(1,figsize=(10,5))
plt.hist(tokenLenghts,bins=30,alpha=0.75)
ax=plt.gca()
PlotStyle(ax,'','Terms Length','Frequency')

###############################################################################
#                          Data Location
###############################################################################

#Calculate the frequency of each unique term in each abstract
def TokenFrequencies(TargetString,Token):
    
    if type(TargetString)==list:
        cTarget=TargetString
    else:
        cTarget=TargetString.split()
  
    cTokenDictionary=Token
    nToken=len(cTokenDictionary)
    Container=[0 for k in range(nToken)]
  
    for val in cTarget:
        try: 
            cLocation=cTokenDictionary[val]
            Container[cLocation]=Container[cLocation]+1 
        except KeyError: 
            pass
    
    return np.array(Container)

#Selects the most common terms 
frequency=[TokenFrequencies(abstract,tokenToLocation) for abstract in mainTexts]
mostCommonTerms=[reducedToken[np.argmax(val)] for val in frequency]
frequency=np.array(frequency)

tokenFullCounts=frequency.sum(axis=0)

plt.figure(2,figsize=(10,5))
plt.plot(tokenFullCounts)
ax=plt.gca()
ax.set_xlim([0,56000])
ax.set_ylim([0,16000])
PlotStyle(ax,'','Unique Terms', 'Terms Count')

#Selects the mosto common terms in the dataset 
discriminator=tokenFullCounts.max()*0.035
greaterThanDiscriminatorIndex=[j for j in range(tokenFullCounts.size) if tokenFullCounts[j]>discriminator] 

#Normalization of the dataset 
filteredCounts=frequency[:,greaterThanDiscriminatorIndex]
dataScaler=pr.MinMaxScaler()
dataScaler.fit(filteredCounts)
scaledCounts=dataScaler.transform(filteredCounts)

###############################################################################
#                          Word Cloud of the Most Common Terms 
###############################################################################

commonText=' '.join(mostCommonTerms)

wcloud=WordCloud(max_words=1500,background_color="white",colormap='magma',width=1600, height=800,min_font_size=8)
wcloud.generate(commonText)

plt.figure(3,figsize=(20,20))
plt.imshow(wcloud,interpolation='gaussian')
plt.axis("off")

###############################################################################
#                          Autoencoder 
###############################################################################

inputShape=scaledCounts[0].shape
Autoencoder=Sequential()

Autoencoder.add(Dense(100,input_shape=inputShape,use_bias=False))
Autoencoder.add(BatchNormalization())
Autoencoder.add(Activation('elu'))

Autoencoder.add(Dense(20,use_bias=False))
Autoencoder.add(BatchNormalization())
Autoencoder.add(Activation('elu'))

Autoencoder.add(Dense(2, name="bottleneck",use_bias=False))
Autoencoder.add(BatchNormalization())
Autoencoder.add(Activation('linear'))

Autoencoder.add(Dense(20,use_bias=False))
Autoencoder.add(BatchNormalization())
Autoencoder.add(Activation('elu'))

Autoencoder.add(Dense(100,use_bias=False))
Autoencoder.add(BatchNormalization())
Autoencoder.add(Activation('elu'))

Autoencoder.add(Dense(inputShape[0],use_bias=False))
Autoencoder.add(BatchNormalization())
Autoencoder.add(Activation('linear'))

decay=0.001
Autoencoder.compile(loss='mean_squared_error', optimizer = Adam(lr=0.01,decay=decay))
Autoencoder.fit(scaledCounts, scaledCounts, batch_size=24, epochs=30, verbose=1)

Encoder=Model(Autoencoder.input, Autoencoder.get_layer('bottleneck').output)
Zenc=Encoder.predict(scaledCounts)

###############################################################################
#                          Clusterin of the Encoded Data Set
###############################################################################

encoderDistances=ds.pdist(Zenc,metric='euclidean')
distanceHistogram=np.histogram(encoderDistances,bins=75)

clusters=DBSCAN(eps=distanceHistogram[1][1], min_samples=10).fit(Zenc)

clusterDataLabel=clusters.labels_
clustersLabels=np.unique(clusterDataLabel)
nClusters=clustersLabels.size
nColors=plt.cm.magma(np.linspace(0, 1,nClusters),alpha=0.75)

#Visualization of the clusters
plt.figure(5,figsize=(12,12))

for k in range(nClusters):
    
    index=[j for j in range(clusterDataLabel.size) if clusterDataLabel[j]==clustersLabels[k]]
    plt.plot(Zenc[index,0],Zenc[index,1],'o',markerfacecolor=nColors[k],markeredgecolor=nColors[k])
    
ax=plt.gca()
PlotStyle(ax,'','Encoded Dimension 0','Encoded Dimension 1')

#Creates  a word cloud for eacyh cluster found 

for k in range(nClusters):
    
    index=[j for j in range(clusterDataLabel.size) if clusterDataLabel[j]==clustersLabels[k]]
    clusterTerms=np.array(mostCommonTerms)[index]
    commonText=' '.join(clusterTerms)

    wcloud=WordCloud(max_words=1500,background_color="white",colormap='magma',width=1600, height=800,min_font_size=8)
    wcloud.generate(commonText)

    plt.figure(k+5+1,figsize=(10,10))
    plt.imshow(wcloud,interpolation='gaussian')
    plt.axis("off")
