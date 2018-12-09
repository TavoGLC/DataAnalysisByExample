# -*- coding: utf-8 -*-
"""

@author: TavoGLC
From:

-Mining biological databases: UniProt-

"""

###############################################################################
#                          Libraries to use  
###############################################################################

import re
import numpy as np
import requests as r

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

###############################################################################
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
#                         Working Directories  
###############################################################################

#Global Data Directory
GlobalDir= 'Global data directory'

#List of selected proteins 
TargetListFile=GlobalDir+'\\'+'TargetList.csv'

#Uniprot saved files directory 
UniprotDir=GlobalDir+'\\'+'Uniprot Files'

#Localization and GO terms directory
DefDir=GlobalDir+'\\'+'Definitions'

#Localization file
SubcelFile=DefDir+'\\'+'subcell.txt'

#GO file 
GOFile=DefDir+'\\'+'keywlist.txt'

###############################################################################
#                          Loading selected proteins  
###############################################################################

TargetData=np.genfromtxt(TargetListFile,delimiter=',',dtype='|S10')
TargetList=[val.decode() for val in TargetData]

###############################################################################
#                          Downloading uniprot data
###############################################################################

#Check if an uniprot has already been downloaded 
def FindFile(Dir):
    
    try:
        open(Dir)
        
        return True
    
    except FileNotFoundError:
        
        return False

#Download an uniprot entree saving it with the identifier name  
def UniprotSave(Identifier,OutputDir):
    
    cID=Identifier
    
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+cID+".txt"
    response = r.post(currentUrl)
    cData=''.join(response.text)
    
    nData=cData.splitlines()
    nLines=len(nData)
    
    NewDir=OutputDir
    
    with open(NewDir,'w',newline='\n') as output:
        
        for k in range(nLines):
            
            output.write(nData[k]+'\n')
        
#Iterates through the selected proteins list
def IdentifiersIterator(Identifiers):
    
    cList=Identifiers
    
    for val in cList:
        
        OutDir=UniprotDir+'\\'+val+'.txt'
        
        if FindFile(OutDir)==False:
            
        
            UniprotSave(val,OutDir)
            
        else:
            
            pass
        
    return ':3 done' 

IdentifiersIterator(TargetList)

###############################################################################
#                          Load text file function  
###############################################################################

#Open a text file and split it by lines 
def GetFileLines(Dir):
    
    with open(Dir,'r') as file:
        
        Lines=[]
        
        for lines in file.readlines():
            
            Lines.append(lines)
    
    return Lines

###############################################################################
#                          Protein sequence lenght  
###############################################################################

#Finds the protein sequence lenght inside an uniprot file 
def GetSize(Identifier):
    
    cID=Identifier
    cDir=UniprotDir+'\\'+cID+'.txt'
    
    cLines=GetFileLines(cDir)  #Load the data 
    nLines=len(cLines)
    
    SeqLoc=[j for j in range(nLines) if re.match('(.*)'+'SEQUENCE'+'(.*)',cLines[j])] #Find the location of the SEQUENCE pattern 
    SeqLine=cLines[SeqLoc[-1]].split()
    
    return int(SeqLine[2])

#   
ProteinSizes=[GetSize(val) for val in TargetList]

plt.figure(1,figsize=(7,7))
plt.plot(ProteinSizes,path_effects=[path_effects.SimpleLineShadow(alpha=0.2,rho=0.2),
                       path_effects.Normal()])

ax=plt.gca()

PlotStyle(ax,'Protein Sequence Lenght')

###############################################################################
#                          Pattern mathching functions   
###############################################################################

#Iterates through an uniprot file and a term list 
def FileIterator(File,TermList):
    
    cF=GetFileLines(File)
    nLines=len(cF)
    nTerms=len(TermList)
    Terms=np.zeros(nTerms)
    
    for k in range(nLines):
        
        cLine=cF[k]
        
        for j in range(nTerms):
            
            if re.match('(.*)'+TermList[j]+'(.*)',cLine):  #Creates a pattern based on a provided list
                
                Terms[j]=1
                
            else:
                
                continue
                
    return Terms

#Iterates through the selected proteins and selects only the terms present in the selected proteins 
def MatrixTerms(Selected,Terms):
    
    nSelected=len(Selected)
    nTerms=len(Terms)
    Data=np.zeros((nSelected,nTerms))
    
    for k in range(nSelected):
        
        cEntrie=Selected[k]
        cDir=UniprotDir+'\\'+cEntrie+'.txt'
        vec=FileIterator(cDir,Terms)
        Data[k,:]=vec
    
    Index=[]
    nData=[]
    
    for j in range(nTerms):
        
        cDisc=sum(Data[:,j])
        
        if cDisc==0:
            
            continue
        
        else:
            
            Index.append(j)
            nData.append(Data[:,j])
        
    return Index, np.transpose(nData)    
        
#Loadign location and GO terms data 
LocationData=GetFileLines(SubcelFile)
GOData=GetFileLines(GOFile)
Locations=[line[5:len(line)-2] for line in LocationData if line[0:2]=='ID']
GOT=[line[5:15] for line in GOData if line[0:2]=='GO']
GON=[line[16:len(line)] for line in GOData if line[0:2]=='GO']

#Location and GO terms frequencie data  
LMatrix=MatrixTerms(TargetList,Locations)
GOMatrix=MatrixTerms(TargetList,GOT)

###############################################################################
#                          Analysis Visualization  
###############################################################################

#Selecting only present values 
SelectedLocations=[Locations[val] for val in LMatrix[0]]
SelectedGO=[GON[val] for val in GOMatrix[0]]

#Frequency of each value 
LocationsFreqs=np.sum(LMatrix[1],axis=0)
GOFreqs=np.sum(GOMatrix[1],axis=0)

#Location frequency plot

plt.figure(2,figsize=(10,7))

xVals=np.arange(len(LocationsFreqs))
plt.bar(xVals,LocationsFreqs,align='center',alpha=0.75)
plt.xticks(xVals,SelectedLocations,rotation=90)

ax=plt.gca()
PlotStyle(ax,'Locations')

#GO terms frequency plot

plt.figure(3,figsize=(22,7))

xVals=np.arange(len(GOFreqs))
plt.bar(xVals,GOFreqs,align='center',alpha=0.75)
plt.xticks(xVals,SelectedGO,rotation=90)

ax=plt.gca()
PlotStyle(ax,'GO Terms')
