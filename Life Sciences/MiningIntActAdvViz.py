# -*- coding: utf-8 -*-
"""

@author: TavoGLC
From:

-Mining biological databases: IntAct-

"""

#####################################################################
#                        Libraries to use  
#####################################################################

import re
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
####################################################################
#                         Data Directory 
####################################################################

GlobalDir='Global data directory'
DataDir=GlobalDir+'\\'+'Data'

TargetListDir=GlobalDir+'\\'+'TargetList.csv'

####################################################################
#                   Data saving functions
####################################################################

#Elimates the left and top lines and ticks in a matplotlib plot 
def CleanStyle(Axes,Title):
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.spines['bottom'].set_visible(False)
    Axes.spines['left'].set_visible(False)
    Axes.set_xticks([])
    Axes.set_yticks([])
    Axes.set_title(Title)

####################################################################
#                   Data manipulation functions
####################################################################

#Makes a flat list
def Flatten(ListOfLists):
    
    return [item for sublist in ListOfLists for item in sublist]

#Return the first value of a list of lists 
def Firsts(ListOfLists):
    
    return [val[0] for val in ListOfLists]

#Check if the ID is a valid EBI ID 
def ValidQ(IdValue):
    
    if re.match('(.*)EBI-(.*)',IdValue):
        
        cResponse=True
        
    else:
        
        cResponse=False
        
    return cResponse 


#Generates a directory to save a file, name will be the Uniprot identifier 
def GenCSVDir(ParentDirectory,Name):
    
    return ParentDirectory+'\\'+Name+'.csv'

#Gets the ieraction data 
def GetCurrentFile(Name):
    
    cData=np.genfromtxt(GenCSVDir(DataDir,Name),delimiter=',',dtype=None,encoding='utf-8')
    #nList=[str(val.decode()) for val in cData]
    
    return cData

#Gets the ieraction data
def MakeIDEdit(Name,UniprotID):
    
    if Name[0:6]==UniprotID:
        
        NewName=Name[7:len(Name)]
        
    else:
        
        NewName=Name
    
    return NewName

#Loading Uniprot identifiers 
targetList=np.genfromtxt(TargetListDir,dtype="|U6")

#Returns the valid nodes of a given target
def NodesData(TargetList):
    
    Container=[]
    
    for ids in TargetList:
        
        cData=np.atleast_1d(GetCurrentFile(ids))
        lContainer=[]
        
        for val in cData:
            
            lContainer.append(MakeIDEdit(val,ids))
            
        Container.append(lContainer)
        
    return Container

####################################################################
#                        Data filtering
####################################################################

#Load all the nodes 
GlobalData=NodesData(targetList)

#Removes duplicated nodes 
TotalNodes=np.unique(Flatten(GlobalData))

#Validates the nodes 
ValidNodes=[val for val in TotalNodes if ValidQ(val)==True]
RepNodes=[val for val in Firsts(GlobalData) if ValidQ(val)==True]

#Creates a dictionary from EBI ID to node number 
EBiToNumber={}
NumberToEBi={}
k=0

for val in ValidNodes:
    
    EBiToNumber[val]=k
    NumberToEBi[k]=val
    k=k+1
    
####################################################################
#                  Graph generation functions
####################################################################

#creates the interaction graph
def MakeGraph(GraphData,DataNodes):
    
    localData=GraphData
    localNodes=DataNodes

    NodesList=[k for k in range(len(localNodes))]

    localG=nx.Graph()
    localG.add_nodes_from(NodesList)

    for List in localData:
    
        for val in List[1:len(List)]:
        
            try:
            
                localG.add_edge(EBiToNumber[List[0]],EBiToNumber[val])
            
            except KeyError:
            
                pass
            
    #Removes isolated nodes
    localG.remove_nodes_from(list(nx.isolates(localG)))

    return localG

#Calculate the Eigencentrality and returns a dictionary of nodes sizes
def CentralityMask(Graph):
    
    cGraph=Graph
    cent=nx.eigenvector_centrality(cGraph)
    
    MaxCent=max(cent.values())
    MinCent=min(cent.values())
    
    MaxSize,MinSize=300,10
    newDict={}
    
    for key,value in cent.items():
        
        newDict[key]=MinSize+((value-MinCent)/MaxCent)*MaxSize
    
    return newDict

#Return a list of nodes and sizes for the graph 
def MakeNodesList(NodesList,MaskDictionary):
    
    LocalList=NodesList
    LocalMask=MaskDictionary
    NodesC=[]
    SizeC=[]
    
    for val in LocalList:
        
        try:
            
            SizeC.append(LocalMask[val])
            NodesC.append(val)
    
        except KeyError:
            
            pass
        
    return NodesC,SizeC
            
####################################################################
#                     Nodes segmentation
####################################################################

#Generates the graph and the graphlayout         
G=MakeGraph(GlobalData,ValidNodes)
pos=nx.spring_layout(G,iterations=50)
centMask=CentralityMask(G)

#Select source and target nodes interactios 
IntSource=[EBiToNumber[val] for val in RepNodes]
IntTargetsName=list(set(ValidNodes) - set(RepNodes))
IntTarget=[EBiToNumber[val] for val in IntTargetsName]

#Nodes sizes
sNodes,sSizes=MakeNodesList(IntSource,centMask)
tNodes,tSizes=MakeNodesList(IntTarget,centMask)

####################################################################
#                    Graph visualization
####################################################################

plt.figure(1,figsize=(12,12))

nx.draw_networkx_edges(G,pos,alpha=0.15)

nx.draw_networkx_nodes(G,pos,nodelist=sNodes,node_size=sSizes,node_color='r',alpha=0.5)
nx.draw_networkx_nodes(G,pos,nodelist=tNodes,node_size=tSizes,node_color='b',alpha=0.2)

ax=plt.gca()
CleanStyle(ax,'')

####################################################################
#                       Graph clustering 
####################################################################

#Number of clusters calculation 
LaplacianMatrix=nx.laplacian_matrix(G)
eigVals,eigVecs=np.linalg.eig(LaplacianMatrix.toarray())
difs=[eigVals[k]-eigVals[k+1] for k in range(len(eigVals)-1)]
maxDifPos=[j for j in range(len(difs)) if difs[j]==max(difs)]
nClusters=int(maxDifPos[0])

#Spectral clustering calculations
AdjacencyMatrix=nx.to_numpy_matrix(G)
SCMethod=SpectralClustering(nClusters,gamma=1,affinity='rbf',assign_labels="discretize", n_init=100)
SCMethod.fit(AdjacencyMatrix)

####################################################################
#                 Graph clusters visualization
####################################################################

colors=fColors=plt.cm.viridis(np.linspace(0, 1,nClusters),alpha=0.75)

NodesOrder=G.nodes()
Clusters=SCMethod.labels_

for k in range(nClusters):
    
    plt.figure(k+2,figsize=(6,6))

    nx.draw_networkx_edges(G,pos,alpha=0.15)
    NodesList=[nod for nod,clus in zip(NodesOrder,Clusters) if clus==k]
    nx.draw_networkx_nodes(G,pos,nodelist=NodesList,node_size=50,node_color=colors[k],alpha=0.35)
    ax=plt.gca()
    CleanStyle(ax,'')

    
