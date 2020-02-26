#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:55:43 2020

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
# Loading packages 
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

from Bio import SeqIO
from io import StringIO

###############################################################################
# Data Location
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
ProteomeDataDir=GlobalDirectory+"HumanProteome.fasta"

###############################################################################
# Global Definitions
###############################################################################

Aminoacids=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X','U']
Classification={'R':'0','H':'0','K':'0','D':'1','E':'1','S':'2','T':'2','N':'2','Q':'2','C':'3','U':'3','G':'3','P':'3','A':'4','I':'4','L':'4','M':'4','F':'4','W':'4','Y':'4','V':'4'}
UniqueClasses=['0','1','2','3','4']

###############################################################################
# General Funcions 
###############################################################################

#Funcion para la obtencion de las secuencias en formato de texto
def GetSequences(Directory):
    
    """
    Opens a fasta file and parse all the sequences in the file
    returns a list with all the sequences
    
    Directory -> location of the data 
    """
    
    with open(Directory) as file:
        
        seqData=file.read()
        
    Seq=StringIO(seqData)
    SeqList=list(SeqIO.parse(Seq,'fasta'))
    
    return SeqList  

def PlotStyle(Axes):
    
    """
    General style used in all the plots 
    """
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=14)
    Axes.yaxis.set_tick_params(labelsize=14)

###############################################################################
# Unique elements functions
###############################################################################

def SplitString(String,ChunkSize):
    
    """
    Preprossecing function, takes a string and split it with a 
    sliding window returns a list with the fragments of text
    
    String -> string to be splited
    ChunkSize -> size of the fragment taken by the sliding window
    """
      
    if ChunkSize==1:
        Splitted=[val for val in String]
    
    else:
        nCharacters=len(String)
        Splitted=[String[k:k+ChunkSize] for k in range(nCharacters-ChunkSize)]
        
    return Splitted

def UniqueToDictionary(UniqueElements):
    
    """
    Returns a dictionary that maps the unique character to 
    an integer to be used as an index.
    
    UniqueElements -> Lis  of the unique elements. 
    """
    
    localDictionary={}
    nElements=len(UniqueElements)
    
    for k in range(nElements):
        localDictionary[UniqueElements[k]]=k
        
    return localDictionary

def CountUniqueElements(UniqueElements,ProcessedString):
    
    """
    Counts the frequency of the unique elements in a splited or 
    processed string. Returns a list with the frequency of the 
    unique elements. 
    
    UniqueElements -> List with the unique elements
    ProcessedString -> List obtained from the SplitString function
    """
    
    nUnique=len(UniqueElements)
    localCounter=[0 for k in range(nUnique)]
    UniqueDictionary=UniqueToDictionary(UniqueElements)
    
    for val in ProcessedString:
        try:
            localPosition=UniqueDictionary[val]
            localCounter[localPosition]=localCounter[localPosition]+1
        except KeyError:
            pass
    return localCounter

def GetUniqueElements(DataBase,FragmentSize):
    
    """
    Get the unique elements (single character or a string fragment) 
    in a list of strings, returns a numpy array with the unique 
    elements 
    
    DataBase -> List with all the sequences 
    FragmentSize -> Number of characters on each element
    
    """
    
    Container=[]
    counter=0
    
    for val in DataBase:
        
        try:
            newList=set(SplitString(str(val.seq),FragmentSize))
        except AttributeError:
            newList=set(SplitString(val,FragmentSize))
    
        if counter%250==0:
            Container=list(np.unique(Container))
    
        Container=Container+list(newList)
        
    return np.unique(Container)

def SeqToClassification(Sequence):
    
    """
    Changes the characters in a sequence to a set of characters 
    according to a classic classification of aminoacids
    
    Sequence -> Sequence to be modified.
    """
    
    container=[]
    
    for val in Sequence:
        
        try:
            container.append(Classification[val])
        
        except KeyError:    
            pass
        
    return ''.join(container)

###############################################################################
# Analysis of the proteome data 
###############################################################################

ProteomeData=GetSequences(ProteomeDataDir)
NumberOfProteins=len(ProteomeData)

print(NumberOfProteins)

RevProteins=sum([1 for val in ProteomeData if val.id[0:2]=="sp"])

print(RevProteins)

print(((NumberOfProteins-RevProteins)/NumberOfProteins)*100)

SequenceLength=[len(val.seq) for val in ProteomeData]

plt.figure(1)
plt.plot(SequenceLength)
plt.xlabel('Proteins',fontsize=16)
plt.ylabel('Sequence size',fontsize=16)
ax=plt.gca()
PlotStyle(ax)

plt.figure(2)
plt.hist(SequenceLength,bins=np.logspace(0.001,4.7))
plt.xlabel('Sequence size',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
ax=plt.gca()
PlotStyle(ax)

plt.figure(3)
plt.hist(SequenceLength,bins=np.logspace(0.001,3))
plt.xlabel('Sequence size',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
ax=plt.gca()
PlotStyle(ax)

###############################################################################
# Unique elements counts
###############################################################################

Unique1Counts=[]

for val in ProteomeData:
    
    cSequence=str(val.seq)
    procesedSeq=SplitString(cSequence,1)
    Unique1Counts.append(CountUniqueElements(Aminoacids,procesedSeq))
    
Unique1Counts=np.array(Unique1Counts)

plt.figure(4)
plt.bar(np.arange(len(Aminoacids)),Unique1Counts.sum(axis=0))
plt.ylabel('Frequency',fontsize=16)
ax=plt.gca()
ax.set_xticks(np.arange(len(Aminoacids)))
ax.set_xticklabels(Aminoacids)
PlotStyle(ax)


SortedIndex=np.argsort(Unique1Counts.sum(axis=0))
InvSorted=SortedIndex[::-1][0:5]

for val in InvSorted:
    print(Aminoacids[val])

###############################################################################
# Number of unique elements 
###############################################################################

NormalizedUniqueElements=[]
for val in ProteomeData:    
    innerContainer=[]
    preString=str(val.seq)
    for k in range(0,15):
        cString=SplitString(preString,k)
        if len(cString)==0:
            innerContainer.append(1)
        else:
            unique=np.unique(cString)
            innerContainer.append(len(unique)/len(cString))
    NormalizedUniqueElements.append(innerContainer)

NormalizedUniqueElements=np.array(NormalizedUniqueElements)
plt.figure(5)
plt.plot(NormalizedUniqueElements.mean(axis=0))
plt.xlabel('Fragment size',fontsize=16)
plt.ylabel('Normalized Unique Element',fontsize=16)
ax=plt.gca()
PlotStyle(ax)

###############################################################################
# Frequency of unique elements (two characters)
###############################################################################

Unique3=GetUniqueElements(ProteomeData,3)
Unique3Counts=[]

for val in ProteomeData:
    
    cSequence=str(val.seq)
    procesed2Seq=SplitString(cSequence,3)
    Unique3Counts.append(CountUniqueElements(Unique3,procesed2Seq))
    
Unique3Counts=np.array(Unique3Counts)

plt.figure(6)
plt.bar(np.arange(len(Unique3)),Unique3Counts.sum(axis=0))
plt.ylabel('Frequency',fontsize=16)
ax=plt.gca()
PlotStyle(ax)

SortedIndex=np.argsort(Unique3Counts.sum(axis=0))
InvSorted=SortedIndex[::-1][0:5]

for val in InvSorted:
    print(Unique3[val])

###############################################################################
# Number of unique elements modified sequences 
###############################################################################

NormalizedClUniqueElements=[]
ClassifiedProteome=[]

for val in ProteomeData:    
    innerContainer=[]
    preString=SeqToClassification(val.seq)
    ClassifiedProteome.append(preString)
    for k in range(0,15):        
        cString=SplitString(preString,k)
        if len(cString)==0:
            innerContainer.append(1)
        else:
            unique=np.unique(cString)
            innerContainer.append(len(unique)/len(cString))
    NormalizedClUniqueElements.append(innerContainer)

NormalizedClUniqueElements=np.array(NormalizedClUniqueElements)

plt.figure(7)
plt.plot(NormalizedClUniqueElements.mean(axis=0))
plt.xlabel('Fragment size',fontsize=16)
plt.ylabel('Normalized Unique Element',fontsize=16)
ax=plt.gca()
PlotStyle(ax)

###############################################################################
# Unique elements counts classified
###############################################################################

UniqueCl1Counts=[]

for val in ClassifiedProteome:
    
    cSequence=str(val)
    procesedSeq=SplitString(cSequence,1)
    UniqueCl1Counts.append(CountUniqueElements(UniqueClasses,procesedSeq))
    
UniqueCl1Counts=np.array(UniqueCl1Counts)

plt.figure(8)
plt.bar(np.arange(len(UniqueClasses)),UniqueCl1Counts.sum(axis=0))
plt.ylabel('Frequency',fontsize=16)
ax=plt.gca()
ax.set_xticks(np.arange(len(UniqueClasses)))
ax.set_xticklabels(['Positive' ,'Negative' ,'Polar' ,'Special' ,'Hydrophobic'],rotation=45)
PlotStyle(ax)


###############################################################################
# Frequency of unique classified elements (six characters)
###############################################################################

Unique6Cl=GetUniqueElements(ClassifiedProteome,5)
Unique6ClCounts=[]

for val in ClassifiedProteome:
    
    cSequence=str(val)
    procesed2Seq=SplitString(cSequence,5)
    Unique6ClCounts.append(CountUniqueElements(Unique6Cl,procesed2Seq))
    
Unique6ClCounts=np.array(Unique6ClCounts)

plt.figure(9)
plt.bar(np.arange(len(Unique6Cl)),Unique6ClCounts.sum(axis=0))
plt.ylabel('Frequency',fontsize=16)
ax=plt.gca()
PlotStyle(ax)

SortedIndex=np.argsort(Unique6ClCounts.sum(axis=0))
InvSorted=SortedIndex[::-1][0:5]

for val in InvSorted:
    print(Unique6Cl[val])
