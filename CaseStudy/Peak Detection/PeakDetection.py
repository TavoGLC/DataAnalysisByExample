# -*- coding: utf-8 -*-
"""

@author: TavoGLC
From:

-A simple peak detection method-

"""

###############################################################################
#                          Libraries to use  
###############################A################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib.gridspec import GridSpec

¡###############################################################################
#                    General plot functions 
###############################################################################

#Elimates the left and top lines and ticks in a matplotlib plot 
def PlotStyle(Axes,Title):
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.set_title(Title)

###############################################################################
#                               Loading the data 
###############################################################################

#Directories where data is located 
DataDir='Data directory'
DataFile=DataDir+'\\'+'FileName.csv'

###############################################################################
#                          Data Visual Evaluation
###############################################################################

Data=np.genfromtxt(DataFile,delimiter=';')

TimeData=Data[:,0]
TimeStep=TimeData[1]-TimeData[0]

SignalData=Data[:,1]

plt.figure(1)

plt.plot(TimeData,SignalData)
ax=plt.gca()
PlotStyle(ax,' Full Recording ')

###############################################################################
#                         Base Line Removal Function  
###############################################################################

nData=len(SignalData)
nSample=int(0.15*nData) #Sample size 
gap=100

#Finds a change in the base line and returns the location of the change
def CropPoint(DataList):
    
    response=-1
    
    for k in range(len(DataList)-gap):
        
        DISC=np.abs((DataList[k]-DataList[k+gap])/DataList[k])
        
        if DISC>2:
            
            response=k
            
            break
        
        else:
            
            pass
        
    return response 

###############################################################################
#                            Croping the data 
###############################################################################

#Moving average of the selected sample to remove the baseline 
AveragesFoward=[np.mean(SignalData[k:k+gap]) for k in range(nSample)]
AveragesBackward=[np.mean(SignalData[len(SignalData)-gap-k:len(SignalData)-k]) for k in range(nSample)]

#Croping points 
FowardCrop=CropPoint(AveragesFoward)+gap
BackwardCrop=nData-(CropPoint(AveragesBackward)+gap)
CropedSignal=SignalData[FowardCrop:BackwardCrop]

CropedTime=[TimeStep*k for k in range(len(CropedSignal))]

plt.figure(2)
plt.plot(CropedTime,CropedSignal)
ax=plt.gca()
PlotStyle(ax,' Baseline removed')

###############################################################################
#                             Peak segmentation  
###############################################################################

#Location of a value in a list, only the location of the first time that the value is encounter 
def Location(List,Value):
    
    for k in range(len(List)):
        
        if List[k]==Value:
            
            response=k
            break
            
        else:
            
            pass
        
    return response 

#Aproximate peak location
def AproximateLocations(DataList):
    
    cData=DataList
    cnData=len(cData)
    cMx=np.max(cData)
    
    Locations=[k for k in range(cnData) if cData[k]<=cMx and (0.9*cMx)<cData[k]] #Similar values to the max value
    ListPosition=[k for k in range(len(Locations)-1) if 500<Locations[k+1]-Locations[k]] #Clustering of near data 
    
    LocationsMask=[]
    LocationsMask.append(Locations[0:ListPosition[0]]) #Clustering of unique peaks 
    
    for k in range(1,len(ListPosition)):
            
        LocationsMask.append(Locations[ListPosition[k-1]:ListPosition[k]])
            
    LocationsMask.append(Locations[ListPosition[-1]:len(Locations)])
    
    return LocationsMask

#Peak location 
def PeakLocations(DataList):
    
    cData=DataList
    aproxLocations=AproximateLocations(cData)
    maxLocations=[]
    
    for val in aproxLocations:
        
        cfragment=cData[val]
        cMax=np.max(cfragment)
        MaxValuePosition=Location(cfragment,cMax)
        maxLocations.append(val[MaxValuePosition])
        
    return maxLocations

#Slices the exitation signal recordings inside the data 
def DataSlicer(DataList):
    
    cData=DataList
    peakLocations=PeakLocations(cData)
    DataSlices=[]
    
    for val in peakLocations:
        
        DataSlices.append(cData[val-750:val+5500])#Slices the data using a fixed data size 
        
    return DataSlices
            
###############################################################################
#                              Peak Visualization  
###############################################################################

#Individual peaks 
RecordLibrary=DataSlicer(CropedSignal)

nSignals=len(RecordLibrary)
Rows=2

#Identifies if there are an even or an odd number of peaks 
DISC=nSignals%Rows

if DISC==0:
    
    Colums=int(nSignals/2)
    
else:
    
    Colums=int(1+(nSignals/2))

#Axes indexs for the subplots 
Indexs=[(i,j) for i in range(Rows) for j in range(Colums)]

fig=plt.figure(3,figsize=(14,5))

gridSp=GridSpec(Rows,Colums)
record=0

for Ind in Indexs:
    
    cSignal=RecordLibrary[record]
    cTime=[TimeStep*k for k in range(len(cSignal))]
    
    cax=fig.add_subplot(gridSp[Ind])
    cax.plot(cTime,cSignal,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
    
    PlotStyle(cax,'')
    
    record=record+1
    
plt.tight_layout()

###############################################################################
#                         Mean Peak Signal  
###############################################################################

MeanSignal=np.mean(RecordLibrary,axis=0) #Mean signal of the recording 
TimeVals=[k*TimeStep for k in range(len(MeanSignal))]

plt.figure(4,figsize=(7,7))

plt.plot(TimeVals,MeanSignal,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])
ax=plt.gca()
PlotStyle(ax,'Mean Excitation Signal')
