# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 23:21:01 2018

MIT License
Copyright (c) 2020 Octavio Gonzalez-Lugo 

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

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing as pr

from keras.models import Sequential, Model
from keras.layers import Dense,Activation
from keras.optimizers import Adam

###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "PricesData.csv"

###############################################################################
# Plot Functions
###############################################################################

def PlotStyle(Axes): 
    """
    General style used in all the plots 
    
    Axes -> matplotlib axes object
    """ 
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=12)
    Axes.yaxis.set_tick_params(labelsize=12)


def ImageStyle(Axes): 
    """
    General style used in all the images
    
    Axes -> matplotlib axes object
    """ 
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(False)
    Axes.spines['left'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.set_xticks([])
    Axes.set_yticks([])

def MakeNetworkWeightsPanel(NetworkWeights):
    """
    Makes a panel with three rows an k columns, each column 
    contains a visualization for each layer, an histogram and 
    color map for the matrix and a line plot for the biases
    
    NetworkWeights -> list, output of model.get_weights() where model 
                      is a keras model 
    """
    nrows=3
    ncolumns=int(len(NetworkWeights)/2)

    indexs=[[(j,k) for j in range(nrows)] for k in range(ncolumns)]

    fig,axes=plt.subplots(nrows,ncolumns,figsize=(14,7))

    for k in range(ncolumns):
    
        aIndex,bIndex,cIndex=indexs[k]
    
        mappable=axes[aIndex].imshow(NetworkWeights[2*k])
        plt.colorbar(mappable,ax=axes[aIndex])
    
        axes[bIndex].hist(NetworkWeights[2*k].flatten(),bins=50)
        axes[cIndex].plot(NetworkWeights[2*k+1].flatten(),'bo-')
    
        ImageStyle(axes[aIndex])
        PlotStyle(axes[bIndex])
        PlotStyle(axes[cIndex])

    plt.tight_layout()

###############################################################################
# Pretreatement functions
###############################################################################

#Centra los datos a promedio 0 y desviacion estadndar unitaria
def Normalization(SeriesData):
    """
    Time series normalization
    
    SeriesData -> List,array or data frame with the data
    """     
    cData=np.reshape(np.array(SeriesData),(-1,1))
    Scaler=pr.StandardScaler()
    FitScaler=Scaler.fit(cData)
    
    return FitScaler.transform(cData)

#Itera a lo largo de la serie de datos 
def SeriesToData(SeriesData,IntervalSize,Forecast):
    """
    Generates a trainig and target datasets
    
    SeriesData   -> List,array or data frame with the data
    IntervalSize -> int, Interval used for the sliding window
    Forecast     -> int, time steps ahead from the las item in the window 
                    to be forecasted
    """ 
    nSteps=len(SeriesData)
    AContainer=[]
    BContainer=[]
    
    for k in range(nSteps-IntervalSize-Forecast-1):
        
        AContainer.append(SeriesData[k:k+IntervalSize])
        BContainer.append(SeriesData[k+IntervalSize+Forecast])
    
    return np.array(AContainer),np.array(BContainer)

###############################################################################
# Neural Network Generation 
###############################################################################

#Centra los datos a promedio 0 y desviacion estadndar unitaria
def NeuralGenerator(Shape,Nodes):
    """
    Generates a keras deep neural network
    
    Shape   -> int, fragment size from the sliding window 
    Nodes   -> int, Architecture of the neural network 
    """ 
    NeuralNet=Sequential()
    NeuralNet.add(Dense(Shape, input_shape=(Shape,)))
    NeuralNet.add(Activation('linear'))
    
    for val in Nodes:
        
        NeuralNet.add(Dense(val))
        NeuralNet.add(Activation('elu'))
    
    NeuralNet.add(Dense(1, name='Series'))
    NeuralNet.add(Activation('linear'))    
    
    return NeuralNet

#Centra los datos a promedio 0 y desviacion estadndar unitaria
def NeuralTrain(SeriesXVals,SeriesYVals,NetworkStructure):
    """
    Trains a keras neural network 
    
    SeriesXVals      -> array, Training data 
    SeriesYVals      -> array, Training target
    NetworkStructure -> list,array  Architecture of the neural network 
    """ 
    nEpochs=25
    net=NeuralGenerator(len(SeriesXVals[0]),NetworkStructure)
    decayRate=0.0000000001/nEpochs
    
    net.compile(loss='mean_squared_error', optimizer = Adam(lr=0.001,decay=decayRate))
    net.fit(SeriesXVals,SeriesYVals, batch_size=64, epochs=nEpochs, verbose=1,shuffle=True)
    TrainedModel=Model(net.input, net.get_layer('Series').output)
    
    return TrainedModel

###############################################################################
# Data visualization
###############################################################################
    
Data=np.genfromtxt(DataDir,delimiter=',')
ScaledData=Normalization(Data[1:,1]).reshape(1,-1)[0]

plt.figure(1)
plt.plot(ScaledData)
ax=plt.gca()
PlotStyle(ax)

TrainSeries=ScaledData[0:int(0.9*len(ScaledData))]
TestSeries=ScaledData[int(0.9*len(ScaledData)):len(ScaledData)]

fig,axes=plt.subplots(1,2,figsize=(10,5),sharex=False,sharey=True)

axes[0].plot(TrainSeries)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Price")
axes[1].plot(TestSeries)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Price")

PlotStyle(axes[0])
PlotStyle(axes[1])

###############################################################################
# Training the network 
###############################################################################

FragmentSize=35
Forecast=1
CurrentArchitecture=[25,12,6,3]

Xtrain,Ytrain=SeriesToData(TrainSeries,FragmentSize,Forecast)
Xtest,Ytest=SeriesToData(TestSeries,FragmentSize,Forecast)

SeriesModel=NeuralTrain(Xtrain, Ytrain, CurrentArchitecture)

Ypred=SeriesModel.predict(Xtest)

plt.figure(3)
plt.plot(Ypred)
plt.plot(Ytest,'r')
ax=plt.gca()
ax.set_xlabel("Time")
ax.set_ylabel("Price")
PlotStyle(ax)

SeriesModel.compile(loss="mse",optimizer=Adam())
BasePerformance=SeriesModel.evaluate(Xtest,Ytest)

###############################################################################
# Visualizing the weights 
###############################################################################

CurrentNetworkWeights=SeriesModel.get_weights()

MakeNetworkWeightsPanel(CurrentNetworkWeights)

###############################################################################
# Trimming Functions
###############################################################################

def GetWeightsRanges(WeightsShapes):
    """
    Returns a list with the range of weights on each layer
    
    Percentage         -> float, range [0,1] fraction of all the weights
                          to be trimmed 
    WeightsShapes       -> list, list with the shapes of every layer in the network 
    """
    weightRanges=[0]
    for val in WeightsShapes:
        last=weightRanges[-1]
        if len(val)==2:
            weightRanges.append(last+val[0]*val[1])
        else:
            weightRanges.append(last+val[0])
    
    return weightRanges

def MakeRandomTrimmIndex(Percentage,TotalWeights):
    """
    Returns a randomly generated index where the weights will be trimmed 
    
    Percentage         -> float, range [0,1] fraction of all the weights
                          to be trimmed 
    TotalWeights       -> float, total number of weights
    """
    nTrimmed=int(Percentage*TotalWeights)
    index=np.arange(TotalWeights)
    trimmIndex=np.random.choice(index,nTrimmed)
    trimmIndex=np.sort(trimmIndex)
    
    return trimmIndex

def BoundaryToElement(Value,Boundaries):
    """
    Returns the index where a value is between the k and k+1 boundary
    Value      -> int, value to be search
    Boundaries -> list, contains the boundaries of each element 
    """
    responce=-1
    for k in range(len(Boundaries)-1):
        
        if Value>=Boundaries[k] and Value<Boundaries[k+1]:
            responce=k
            break
    return responce 

def OrderIndexs(Index,Ranges):
    """
    Arranges the trimming index for each element in the weights list
    Index  -> List, contains the location of the weights to be trimmed 
    Ranges ->List, boundaries of each element in the weights list
    """
    nRanges=len(Ranges)
    nElements=nRanges-1
    indexPerElement=[[] for k in range(nElements)]
    
    for val in Index:
        iElement=BoundaryToElement(val,Ranges)
        indexPerElement[iElement].append(val)
    
    return indexPerElement

def TrimmByIndex(Index,NetworkWeights):
    """
    Creates a new weights list with the trimmed weights
    Index  -> list, Locations of the weights to be trimmed 
    NetworkWeights ->List, output from kerasmodel.get_weights()
    """
    weightsShapes=[val.shape for val in NetworkWeights]
    ranges=GetWeightsRanges(weightsShapes)
    indexPerElement=OrderIndexs(Index,ranges)
    newWeights=[]
    
    for k in range(len(NetworkWeights)):
        
        if len(indexPerElement[k])==0:
            localArray=NetworkWeights[k].copy()
            newWeights.append(localArray)
            
        else:
            localArray=NetworkWeights[k].copy()
            localArray=localArray.flatten()
            for val in indexPerElement[k]:
                localArray[val-ranges[k]]=0
            localArray=np.reshape(localArray,weightsShapes[k])
            newWeights.append(localArray)
        
    return newWeights

def RandomTrimming(Percentage,NetworkWeights):
    """
    Creates a new weights list with randomly trimmed weights
    Percentage  -> float, relative amount of the weights to be trimmed 
    NetworkWeights ->List, output from kerasmodel.get_weights()
    """
    weightsShapes=[val.shape for val in NetworkWeights]
    ranges=GetWeightsRanges(weightsShapes)
    index=MakeRandomTrimmIndex(Percentage,ranges[-1])
    
    return TrimmByIndex(index,NetworkWeights)

def GetTrimmPerformance(NetworkWeights,X,Y):
    """
    Wrapper function to calculate the performace of the trimming 
    operation 
    
    NetworkWeights -> list, list of numpy arrays with the network weights
    X              -> array, test data
    Y              -> array, test data
    """
    localNetwork=NeuralGenerator(FragmentSize,CurrentArchitecture)
    localNetwork.set_weights(NetworkWeights)
    localNetwork.compile(loss="mse",optimizer=Adam())
    
    return localNetwork.evaluate(X,Y)

###############################################################################
# Trimming Performance 
###############################################################################

percentages=np.logspace(0.0001,0.05,50)
percentages=np.array([np.log10(val) for val in percentages])

performances=[]

for val in percentages:
    newWeights=RandomTrimming(val,CurrentNetworkWeights)
    performances.append(1-(GetTrimmPerformance(newWeights,Xtest,Ytest)/BasePerformance))

plt.figure(5)
plt.plot(100*np.array(percentages),performances,'bo-')
ax=plt.gca()
ax.set_xlabel("Trimm %")
ax.set_ylabel("Performance")
PlotStyle(ax)

###############################################################################
# Simulated annealing trimming 
###############################################################################

ranges=GetWeightsRanges([val.shape for val in CurrentNetworkWeights])
TotalWeights=ranges[-1]

def ObjetiveFunction(Index):
    """
    Wrapper function for the objetive function
    Index -> List with the weight locations to be trimmed.
    """
    newWeights=TrimmByIndex(Index,CurrentNetworkWeights)    
    return GetTrimmPerformance(newWeights,Xtest,Ytest)

def AcceptanceProbability(Cost,NewCost,Temperature):
    """
    Probability to accept a proposed state
    Cost        -> float, cost of the previous state 
    NewCost     -> float, cost of the current state 
    Temperature -> float, current temperature in the optimizer
    """
    if NewCost<Cost:
        return 1
    else:
        return np.exp(-(NewCost-Cost)/Temperature)

def Temperature(Fraction):
    """
    Temperature relaxation function 
    """
    return max(0.01,min(1,1-Fraction))

def ForceRange(Index):
    """
    Forces all the values to be between the [0,TotalWeights] range 
    Index -> List with the weight locations to be trimmed.
    """
    for k in range(len(Index)):
        if Index[k]>=TotalWeights:
            Index[k]=int(np.random.choice(np.arange(0,TotalWeights-1),1))
    return Index

def RandomNeighbour(State,Fraction):
    """
    Makes a small change to randomly selected items in the State 
    State    -> List with the weight locations to be trimmed.
    Fraction -> float, optimizer temperature
    """
    nToModify=int(len(State)/4)
    indexToModify=np.arange(0,len(State))
    np.random.shuffle(indexToModify)
    newState=State.copy()
    
    for val in indexToModify[0:nToModify]:
        delta=(TotalWeights*Fraction/10)*(np.random.random())
        newState[val]=int(newState[val]+delta)
    
    return ForceRange(newState)
    

def SimulatedAnnealing(StartState,maxSteps=300):
    """
    Weight trimming optimization by simulated annealing
    StartState -> List with the weight locations to be trimmed.
    maxSteps   -> int, number of steps taken by the optimizer 
    """    
    state=StartState
    cost=ObjetiveFunction(StartState)
    states,costs=[state],[cost]
    
    for k in range(maxSteps):
        fraction=k/float(maxSteps)
        T=Temperature(fraction)
        newstate=RandomNeighbour(state,fraction)
        newcost=ObjetiveFunction(newstate)
        if AcceptanceProbability(cost,newcost,T)>np.random.random():
            state,cost=newstate,newcost
            states.append(state)
            costs.append(cost)
            
    return states,costs

###############################################################################
# Simulated annealing trimming 
###############################################################################

StartGuess=np.random.choice(np.arange(0,TotalWeights),size=125)
States,Costs=SimulatedAnnealing(StartGuess,400)

NewWeights=TrimmByIndex(States[np.argmin(Costs)],CurrentNetworkWeights)

TrimmNetwork=NeuralGenerator(FragmentSize,CurrentArchitecture)
TrimmNetwork.set_weights(NewWeights)
TrimmNetwork.compile(loss="mse",optimizer=Adam())

YtrimTrain=TrimmNetwork.predict(Xtrain)
YtrimTest=TrimmNetwork.predict(Xtest)

fig,axes=plt.subplots(1,2,figsize=(10,5),sharex=False,sharey=True)

axes[0].plot(Ytrain)
axes[0].plot(YtrimTrain,'r-',alpha=0.75)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Price")
axes[1].plot(Ytest)
axes[1].plot(YtrimTest,'r-',alpha=0.75)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Price")

PlotStyle(axes[0])
PlotStyle(axes[1])
