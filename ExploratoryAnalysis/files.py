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

import os 
import numpy as np
import pandas as pd 
import itertools as it 
import matplotlib.pyplot as plt

###############################################################################
# Plotting utility functions
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
    Axes.xaxis.set_tick_params(labelsize=13)
    Axes.yaxis.set_tick_params(labelsize=13)

###############################################################################
# Load data function
###############################################################################

def GetTxtFile(Dir):
    '''
    Finds and loads a txt file 
    Parameters
    ----------
    Dir : str
        file directory.

    Returns
    -------
    container : list
        list of strings, each item in the list its a file line.

    '''
    
    cDir=Dir
    
    with open(cDir) as file:
        container=file.readlines()
    
    return container

###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/scrapped/"
FileNames=os.listdir(GlobalDirectory)

nFiles=len(FileNames)

print(nFiles)

FileNames=FileNames
###############################################################################
# Correct Download
###############################################################################

Header='           PRECIP  EVAP   TMAX   TMIN\n'

correctDownload=[]
headerLocation=[]

for j,file in enumerate(FileNames):
    currentFile=GetTxtFile(GlobalDirectory+file)
    disc=[currentFile[k]==Header for k in range(18)]
    if any(disc):
        correctDownload.append(j)
        headerLocation.append(np.argmax(disc))

print(len(correctDownload))

###############################################################################
# Correct size
###############################################################################

correctSize=[]
sizeheaderLocation=[]

for k,loc in zip(correctDownload,headerLocation):
    currentFile=GetTxtFile(GlobalDirectory+FileNames[k])
    if len(currentFile)>380:
        correctSize.append(k)
        sizeheaderLocation.append(loc)

print(len(correctSize))        

###############################################################################
# Enough data content
###############################################################################

missingdata=[]
filesize=[]
correctData=[]
correctHeaderloc=[]

for k,loc in zip(correctSize,sizeheaderLocation):
    
    currentFile=GetTxtFile(GlobalDirectory+FileNames[k])
    currentData=np.array([currentFile[j].split() for j in range(loc+1,len(currentFile)) if len(currentFile[j].split())==5])
    disc=sum(1 for val in currentData.ravel() if val=="Nulo")/(len(currentData.ravel())+1)
    
    if disc < 0.25:
        correctData.append(k)
        correctHeaderloc.append(loc)
        
    missingdata.append(disc)
    filesize.append(np.product(currentData.shape))
    
print(len(correctData))

plt.figure()
plt.hist(missingdata,bins=50,label="Missing Data Fraction")
plt.legend()
plt.xlabel("Missing Data Fraction")
plt.ylabel("Frequency")
ax=plt.gca()
PlotStyle(ax)

###############################################################################
# Time range
###############################################################################

scrapedDF=pd.DataFrame()
scrapedDF["nameindex"]=correctData
scrapedDF["headerloc"]=correctHeaderloc
container=[]

for k,loc in zip(correctData,correctHeaderloc):
    
    currentFile=GetTxtFile(GlobalDirectory+FileNames[k])
    currentData=np.array([currentFile[j].split() for j in range(loc+3,len(currentFile)-1) if len(currentFile[j].split())==5])
    TimeSeries=pd.to_datetime(currentData[:,0],format='%d/%m/%Y')
    container.append([TimeSeries.min().year,TimeSeries.max().year])
    
container=np.array(container)

scrapedDF["dateminyear"]=container[:,0]
scrapedDF["datemaxyear"]=container[:,1]
scrapedDF["daterange"]=container[:,1]-container[:,0]

plt.figure()
plt.plot(scrapedDF["dateminyear"].value_counts(),'bo',label="Minimum record avaliable")
plt.plot(scrapedDF["datemaxyear"].value_counts(),'ro',label="Maximum trecord avaliable")
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.legend()
ax=plt.gca()
PlotStyle(ax)

plt.figure()
plt.plot(scrapedDF["daterange"].value_counts(),'bo',label="Records time range")
plt.xlabel("Time Range (Years)")
plt.ylabel("Frequency")
plt.legend()
ax=plt.gca()
PlotStyle(ax)

###############################################################################
# Time range
###############################################################################

Counts=scrapedDF["daterange"].value_counts()
CountsLabels=Counts.keys()

#Wrapper function to count the number of records pear yeard range
def RangeSum(fac):
    return sum(Counts[val] for val in CountsLabels if val>fac)

plt.figure()
plt.plot([RangeSum(k) for k in range(150)])
plt.xlabel("Time Range (Years)")
plt.ylabel("Number of Records")
ax=plt.gca()
PlotStyle(ax)

###############################################################################
# Missing data per file and year
###############################################################################

def MakeBufferDF(FileDir,loc):
    '''
    Parameters
    ----------
    FileDir : str
        File location.
    loc : int
        number of rows to be skiped. End of the file header 

    Returns
    -------
    BufferDF : pandas data frame
        Format a data file and return it as a data frame.

    '''
    
    currentFile=GetTxtFile(FileDir)
    currentData=np.array([currentFile[j].split() for j in range(loc+3,len(currentFile)-3) if len(currentFile[j].split())==5])
    TimeSeries=pd.to_datetime(currentData[:,0],format='%d/%m/%Y')
    BufferDF=pd.DataFrame(currentData)
    BufferDF[0]=TimeSeries.year
    
    return BufferDF

def GetYearlyMissingData(DataFrame,YearIndex):
    '''
    Parameters
    ----------
    DataFrame : pandas dataframe
        Contains the records of a file. Ouput of MakebufferDF
    YearIndex : array-like
        Array of integers, in the same range as the date column.

    Returns
    -------
    container : list
        number of missing records in the file per year.

    '''
    
    container=[]
    for val in YearIndex:
        count=0
        for k in range(1,5):
            try:
                localCount=DataFrame[DataFrame[0]==val][k].value_counts()["Nulo"]
            except KeyError:
                localCount=0
                
            count=count+localCount
        container.append(count)
    
    return container

YearIndex=[1960+k for k in range(62)]
container=[]
for k,loc in zip(correctData,correctHeaderloc):
    
    currentDir=GlobalDirectory+FileNames[k]
    bufferDF=MakeBufferDF(currentDir,loc)
    data=GetYearlyMissingData(bufferDF,YearIndex)
    container.append(data)
    
plt.figure()
plt.plot(np.array(container).mean(axis=0))
plt.xticks([k for k in range(0,62,10)],[k+1960 for k in range(0,62,10)],rotation=45)
plt.xlabel("Time (Years)")
plt.ylabel("Mean Number of Missing Records")
ax=plt.gca()
PlotStyle(ax)

plt.figure()
plt.plot(np.array(container).mean(axis=1))
plt.xlabel("File")
plt.ylabel("Mean Number of Missing Records")
ax=plt.gca()
PlotStyle(ax)

###############################################################################
# Missing time fragments
###############################################################################

def GetMissingFragmentSize(FileDir,loc):
    '''
    Parameters
    ----------
    FileDir : str
        File location.
    loc : int
        number of rows to be skiped. End of the file header 

    Returns
    -------
    int
        Total missing days in the file

    '''
    
    currentFile=GetTxtFile(FileDir)
    currentData=np.array([currentFile[j].split() for j in range(loc+3,len(currentFile)-3) if len(currentFile[j].split())==5])
    TimeSeries=pd.to_datetime(currentData[:,0],format='%d/%m/%Y')
    Deltas=[(TimeSeries[val+1]-TimeSeries[val]).days for val in range(len(TimeSeries)-1)]
    MissingFragments=[val for val in Deltas if val>1]
    
    return sum(MissingFragments)

container=[]

for k,loc in zip(correctData,correctHeaderloc):
    
    container.append(GetMissingFragmentSize(GlobalDirectory+FileNames[k],loc))
    
plt.figure()
plt.plot(container)
plt.xlabel("Files")
plt.ylabel("Missing Days")
ax=plt.gca()
PlotStyle(ax)    

plt.figure()
plt.hist(container,bins=50)
plt.xlabel("Missing Days")
plt.ylabel("Frequency")
ax=plt.gca()
PlotStyle(ax)    

###############################################################################
# Merging files
###############################################################################

def MakeAndFormatDF(FileDir,loc):
    '''
    Formats a file into a data frame 
    Parameters
    ----------
    FileDir : str
        Location of the file.
    loc : int
        number of rows to be skiped. End of the file header 

    Returns
    -------
    BufferDF : pandas data frame
        Format a data file and return it as a data frame.

    '''
    
    currentFile=GetTxtFile(FileDir)
    fileName=FileDir[FileDir.find("/",28)+1:FileDir.find(".")]
    
    currentData=np.array([currentFile[j].split() for j in range(loc+3,len(currentFile)-3) if len(currentFile[j].split())==5])
    currentData[currentData=="Nulo"]=np.nan
    TimeSeries=pd.to_datetime(currentData[:,0],format='%d/%m/%Y')
    BufferDF=pd.DataFrame()
    BufferDF['date']=TimeSeries
    
    colNames=["precip", "evap", "tmax","tmin"]
    
    for k,nme in enumerate(colNames):
        BufferDF[nme+fileName]=np.array(currentData[:,k+1],dtype=np.float32)
    
    BufferDF.set_index("date",inplace=True)
    
    return BufferDF

def MakeMasterDataFrame(YearRange,dataIndex,headerIndex,Files):
    '''
    Merges a set of files using the date as the index 
    Parameters
    ----------
    YearRange : list
        Two element list with the start and end year range.
    dataIndex : list
        location in a list of correctly downloaded files.
    headerIndex : list
        Lines to be skipped for each file.
    Files : list
        List of strings with the name of the files.

    Returns
    -------
    MasterDataFrame : pandas dataframe
        Merged data frame.

    '''
    StartYear,EndYear = YearRange
    minDate=StartYear+"-01-01"
    endDate=EndYear+"-12-31"
    dIndex=pd.date_range(minDate,endDate,freq="D")
    MasterDataFrame=pd.DataFrame()
    MasterDataFrame["date"]=dIndex
    MasterDataFrame["Year"]=dIndex.year
    MasterDataFrame["month"]=dIndex.month
    MasterDataFrame["day"]=dIndex.day
    MasterDataFrame["dayofweek"]=dIndex.dayofweek
    
    MasterDataFrame.set_index("date",inplace=True)
    
    for k, loc in zip(dataIndex,headerIndex):
        
        currentDir=GlobalDirectory+Files[k]
        bufferDF=MakeAndFormatDF(currentDir,loc)
        
        if bufferDF.index.max().year>int(StartYear):
            if bufferDF.isna().sum().sum()/np.prod(bufferDF.shape)<0.2:    
                MasterDataFrame=MasterDataFrame.join(bufferDF,on="date")
        
    return MasterDataFrame

###############################################################################
# Missing data fraction
###############################################################################

yearsIndex=[k for k in range(1960,2025,5)]
container=[]
siz=[]

for k in range(len(yearsIndex)-1):
    
    mdf=MakeMasterDataFrame([str(yearsIndex[k]),str(yearsIndex[k+1])],correctData,correctHeaderloc,FileNames)
    container.append(mdf.isna().sum().sum()/np.prod(mdf.shape))
    siz.append(np.prod(mdf.shape))
    

plt.figure()
plt.plot(container,label="Missing Records")
plt.plot([val/max(siz) for val in siz],label="Relative Records Size")
labels=[str(yearsIndex[k])+"-"+str(yearsIndex[k+1]) for k in range(len(yearsIndex)-1)]
plt.xticks(np.arange(len(container)),labels,rotation=45)
plt.xlabel("Time Range")
plt.ylabel("Missing Records Fraction")
plt.legend()
ax=plt.gca()
PlotStyle(ax)

plt.figure()
plt.plot(container,[val/max(siz) for val in siz],'bo')
plt.xlabel("Relative Missing Records")
plt.ylabel("Relative Records Size")
ax=plt.gca()
PlotStyle(ax)

###############################################################################
# Missing data fraction
###############################################################################

yearsIndex=[k for k in range(1980,2020,3)]
container=[]
siz=[]

for k in range(len(yearsIndex)-1):
    
    mdf=MakeMasterDataFrame([str(yearsIndex[k]),str(yearsIndex[k+1])],correctData,correctHeaderloc,FileNames)
    container.append(mdf.isna().sum().sum()/np.prod(mdf.shape))
    siz.append(np.prod(mdf.shape))
    
labels=[str(yearsIndex[k])+"-"+str(yearsIndex[k+1]) for k in range(len(yearsIndex)-1)]
plt.figure()
plt.plot(container,label="Missing Records")
plt.plot([val/max(siz) for val in siz],label="Relative Records Size")
plt.xticks(np.arange(len(container)),labels,rotation=45)
plt.xlabel("Time Range")
plt.ylabel("Records Fraction")
plt.legend()
ax=plt.gca()
PlotStyle(ax)

plt.figure()
plt.plot(container,[val/max(siz) for val in siz],'bo')
plt.xlabel("Relative Missing Records")
plt.ylabel("Relative Records Size")

ax=plt.gca()
PlotStyle(ax)

###############################################################################
# Missing data fraction
###############################################################################

yearsIndex=[k for k in range(1980,1987,1)]
container=[]
siz=[]

for k in range(len(yearsIndex)-1):
    
    mdf=MakeMasterDataFrame([str(yearsIndex[k]),str(yearsIndex[k+1])],correctData,correctHeaderloc,FileNames)
    container.append(mdf.isna().sum().sum()/np.prod(mdf.shape))
    siz.append(np.prod(mdf.shape))
    
labels=[str(yearsIndex[k])+"-"+str(yearsIndex[k+1]) for k in range(len(yearsIndex)-1)]
plt.figure()
plt.plot(container,label="Missing Records")
plt.plot([val/max(siz) for val in siz],label="Relative Records Size")
plt.xticks(np.arange(len(container)),labels,rotation=45)
plt.xlabel("Time Range")
plt.ylabel("Records Fraction")
plt.legend()
ax=plt.gca()
PlotStyle(ax)

plt.figure()
plt.plot(container,[val/max(siz) for val in siz],'bo')
plt.xlabel("Relative Missing Records")
plt.ylabel("Relative Records Size")
ax=plt.gca()
PlotStyle(ax)
