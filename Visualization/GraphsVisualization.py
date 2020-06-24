"""
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
# Adding the graphs
###############################################################################
"""
The following code is intended to be used at the start of the program 
before making any changes in the scene. 
"""

###############################################################################
# Loading packages 
###############################################################################

import bpy 
import numpy as np

###############################################################################
# Data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory + "Covid19Proteome.fasta"

###############################################################################
# Loading the data
###############################################################################

def GetFileSequences(Dir):
    """
    Parameters
    ----------
    Dir : string
        file location.

    Returns
    -------
    names : list
        list with the names of all the proteins.
    sequences : list
        list with the sequences of all the proteins.

    """
    
    with open(Dir,'r') as file:
        
        names=[]
        sequences=[]
        currentSeq=''
        
        for line in file.readlines():
            
            if line[0]==">":
                names.append(line[0:-1])
                sequences.append(currentSeq)
                currentSeq=''
            else:
                currentSeq=currentSeq+line[0:-1]
        
        sequences.append(currentSeq)
        del sequences[0]
    
    return names,sequences

###############################################################################
# General Functions 
###############################################################################

#Calculates the distance between two points
def Distance(PointA,PointB):
    return np.sqrt(sum([(val-sal)**2 for val,sal in zip(PointA,PointB)]))

#Calculates the midpoint between two points
def MidPointLocation(PointA,PointB):
    return tuple([(val+sal)/2 for val,sal in zip(PointA,PointB)])

#Calculates the slope between two points
def Rotation(PointA,PointB):
    return np.arctan((PointA[1]-PointB[1])/(PointA[0]-PointB[0]))-(np.pi/2)

def JoinObjectsByName(ObjectNames):
    """
    Parameters
    ----------
    ObjectNames : list
        list with the geometry names.

    Returns
    -------
    string
        name of the final object.
    """

    scene=bpy.context.scene
    totalObs=[bpy.data.objects[val] for val in ObjectNames]
    ctx=bpy.context.copy()
    ctx["active_object"]=totalObs[0]
    ctx["selected_objects"]=totalObs
    ctx["selected_editable_bases"]=[scene.object_bases[ob.name] for ob in totalObs]
    bpy.ops.object.join(ctx)

    return ObjectNames[0]

###############################################################################
# Grid Functions 
###############################################################################

def GetSquaredGridSize(numberOfElements):
    """
    Parameters
    ----------
    numberOfElements : int
        number of elements to create a grid.

    Returns
    -------
    nrows : int
        number of rows in the grid.
    ncolumns : columns
        number of columns in the grid.

    """

    squaredUnique=int(np.sqrt(numberOfElements))
    
    if squaredUnique*squaredUnique==numberOfElements:
        nrows,ncolumns=squaredUnique,squaredUnique
    elif squaredUnique*(squaredUnique+1)<numberOfElements:
        nrows,ncolumns=squaredUnique+1,squaredUnique+1
    else:
        nrows,ncolumns=squaredUnique,squaredUnique+1
        
    return nrows,ncolumns

def GetSquaredGridCenters(numberOfElements,xExtend,yExtend,centerLocation):
    
    nrows,ncolumns=GetSquaredGridSize(numberOfElements)
    xEdges=np.linspace(-xExtend+centerLocation[0],xExtend+centerLocation[0],num=ncolumns+1)
    yEdges=np.linspace(-yExtend+centerLocation[1],yExtend+centerLocation[1],num=nrows+1)
    centerLocations=[]

    for k in range(nrows):
        for j in range(ncolumns):
            centerPoint=tuple([0.5*(xEdges[j]+xEdges[j+1]),0.5*(yEdges[k]+yEdges[k+1]),centerLocation[2]])
            centerLocations.append(centerPoint)
    
    return centerLocations[::-1]

###############################################################################
# Naming Functions
###############################################################################

#Generates a list with the names of the objects
def MakeObjectNames(ObjectName,NumberOfObjects):

    NamesContainer=[]
    NamesContainer.append(ObjectName)

    for k in range(1,NumberOfObjects):
        if k<=9:
            NamesContainer.append(ObjectName+".00"+str(k))
        elif k>9 and k<=99:
            NamesContainer.append(ObjectName+".0"+str(k))
        elif k>99 and k<=999:
            NamesContainer.append(ObjectName+"."+str(k))
        else:
            NamesContainer.append(ObjectName+"."+str(k))

    return NamesContainer

###############################################################################
# Graph functions 
###############################################################################

def MakeAdjacencyMatrix(splitSequence):
    """
    Parameters
    ----------
    splitSequence : list
        list with each element in the sequence.

    Returns
    -------
    AdjacencyMatrix : array
        Adjacency matrix of the sequence graph.

    """
    uniqueVals=np.unique(splitSequence)
    nNodes=len(uniqueVals)
    ResToLocation={}

    for k in range(nNodes):
        ResToLocation[uniqueVals[k]]=k

    AdjacencyMatrix=np.zeros((nNodes,nNodes))
    
    for k in range(len(splitSequence)-1):
        currentVal=splitSequence[k]
        nextVal=splitSequence[k+1]
        AdjacencyMatrix[ResToLocation[currentVal],ResToLocation[nextVal]]=AdjacencyMatrix[ResToLocation[currentVal],ResToLocation[nextVal]]+1
        AdjacencyMatrix[ResToLocation[nextVal],ResToLocation[currentVal]]=AdjacencyMatrix[ResToLocation[nextVal],ResToLocation[currentVal]]+1

    return AdjacencyMatrix

#Calculates the degree matric from the adjacency matrix
def MakeDegreeMatrix(AdjacencyMatrix):
    return  np.diag((AdjacencyMatrix.sum(axis=0)+AdjacencyMatrix.sum(axis=1)-np.diag(AdjacencyMatrix))**(-1/2))

def GetSpectralLayoutPositions(AdjacencyMatrix,DegreeMatrix,Extend,LayoutCenter):
    """
    Parameters
    ----------
    AdjacencyMatrix : array
        2D array, adjacency matrix of the graph.
    DegreeMatrix : array
        2D array, degree matrix of the graph.
    Extend : float
        factor to expand or contract the size of the graph.
    LayoutCenter : tuple, list
        center of the graph in blender global coordinates.

    Returns
    -------
    list
        location of the nodes in the graph.

    """
    Lap=np.eye(DegreeMatrix.shape[0])-DegreeMatrix.dot(AdjacencyMatrix.dot(DegreeMatrix))
    eivals,eivects=np.linalg.eig(Lap)
    eiOrder=np.argsort(eivals)
    eicoords=eivects[eiOrder[1:3]]
    minA,minB=np.min(eicoords,axis=1)
    maxA,maxB=np.max(eicoords,axis=1)
    rangeA=maxA-minA
    rangeB=maxB-minB
    Xcent,Ycent,Zcent=LayoutCenter
    Coords=[]
    
    for val,sal in zip(eicoords[0],eicoords[1]):
        Xval=2*((val-minA)/rangeA)-1
        Yval=2*((sal-minB)/rangeB)-1
        Coords.append([(Extend*Xval)+Xcent,(Extend*Yval)+Ycent,Zcent])
        
    return Coords

def MakeLayoutPositions(splitSequence,extend,LayoutCenter,LayoutType):
    """
    Parameters
    ----------
    splitSequence : list
        list with the sequence elements.
    extend : float
        constant that controls the size of the graph .
    LayoutCenter : tuple, list
        center of the graph in blender global coordinates.
    LayoutType : string
        type of layout to be used, options are Circular,Random,Spectral.

    Returns
    -------
    nNodes : int
        number of nodes in the graph.
    NodesPositions : array,list
        Positions of the nodes in blender global coordinates.

    """
    NodesPositions=[]
    nNodes=len(np.unique(splitSequence))

    if LayoutType=="Circular":

        degreeSlice=(2*np.pi)/nNodes
        Xcent,Ycent,Zcent=LayoutCenter

        for k in range(nNodes):
    
            degree=k*degreeSlice
            Xpos=extend*np.cos(degree)
            Ypos=extend*np.sin(degree)
            NodesPositions.append([Xpos+Xcent,Ypos+Ycent,Zcent])
    
    elif LayoutType=="Random":
        
        Xcent,Ycent,Zcent=LayoutCenter
        Xvals=[extend*val*(-1)**np.random.randint(10) for val in np.random.random(nNodes)]
        Yvals=[extend*val*(-1)**np.random.randint(10) for val in np.random.random(nNodes)]

        for k in range(nNodes):
            NodesPositions.append([Xvals[k]+Xcent,Yvals[k]+Ycent,Zcent])

    elif LayoutType=="Spectral":
        AdMatrix=MakeAdjacencyMatrix(splitSequence)
        DegMatrix=MakeDegreeMatrix(AdMatrix)
        NodesPositions=GetSpectralLayoutPositions(AdMatrix,DegMatrix,extend,LayoutCenter)

    return nNodes,NodesPositions


def AddGraphNodes(splitSequence,extend,LayoutCenter,LayoutType,GraphName,NodeScaling):
    """
    Parameters
    ----------
    splitSequence : list
        list with the sequence elements.
    extend : float
        constant that controls the size of the graph .
    LayoutCenter : tuple, list
        center of the graph in blender global coordinates.
    LayoutType : string
        type of layout to be used, options are Circular,Random,Spectral.
    GraphName : string
        Name of the graph.
    NodeScaling : float
        Constant that controlls the size of the nodes.

    Returns
    -------
    NodesNames : list
        List with the names of all the nodes.
    NodeNameToPosition : dict
        Maps node name to position.
    """
    nNodes,NodesPositions=MakeLayoutPositions(splitSequence,extend,LayoutCenter,LayoutType)
    NodesNames=MakeObjectNames(GraphName+'Node',nNodes)
    NodeNameToPosition={}
    localScale=extend/NodeScaling
    for k in range(nNodes):

        bpy.ops.mesh.primitive_uv_sphere_add(location=(NodesPositions[k][0],NodesPositions[k][1],NodesPositions[k][2]))
        bpy.data.objects["Sphere"].scale=(localScale,localScale,localScale)
        NodeNameToPosition[NodesNames[k]]=(NodesPositions[k][0],NodesPositions[k][1],NodesPositions[k][2])
        bpy.data.objects["Sphere"].data.name=NodesNames[k]
        bpy.data.objects["Sphere"].name=NodesNames[k]
    
    return NodesNames,NodeNameToPosition

def InsertAndChangeText(TextLocation,TextLabel):
    """
    Parameters
    ----------
    TextLocation : tuple
        loaction of the text in blender global coordinates.
    TextLabel : string
        Name of the label.

    Returns
    -------
    None.

    """
    
    bpy.ops.object.text_add(location=TextLocation,enter_editmode=True)
    for k in range(4):
        bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
    for char in TextLabel:
        bpy.ops.font.text_insert(text=char)
    bpy.context.object.data.align_x='CENTER'
    bpy.context.object.data.align_y='CENTER'
    bpy.ops.object.editmode_toggle()

#Wrapper function to add the nodes lables
def AddNodesLabels(NodesLabels,NodesNames,NodesLocations,GraphName,LabelScale,padding):

    nNodes=len(NodesNames)
    LabelNames=MakeObjectNames(GraphName+'Labels',nNodes)

    for k in range(len(NodesNames)):

        nodeLocation=NodesLocations[NodesNames[k]]
        newLocation=[val+np.sign(val)*padding for val in nodeLocation]
        InsertAndChangeText(tuple(newLocation),NodesLabels[k])
        bpy.data.objects['Text'].scale=(LabelScale,LabelScale,LabelScale)
        bpy.data.objects['Text'].name=LabelNames[k]
    
    return LabelNames

def AddEdgesFromSequence(splitSequence,extend,EdgeScaling,NodesNames,NodesLocations,GraphName):
    """
    Parameters
    ----------
    plitSequence : list
        list with the sequence elements.
    extend : float
        constant that controls the size of the graph .
    EdgeScaling : float
        constant that controlls the size of the edge.
    NodesNames : list
        List with the geometry names of the nodes.
    NodesLocations : dict
        maps node name to node location.
    GraphName : string
        name of the graph.

    Returns
    -------
    EdgeNames : list
        List witn the geometry names of the edges.

    """

    SequenceUnique=np.unique(splitSequence)
    
    numberOfNodes=len(SequenceUnique)
    numberOfEdges=len(splitSequence)
    
    FromNumberToName={}
    FromLetterToNumber={}

    for k in range(numberOfNodes):
        FromNumberToName[k]=NodesNames[k]
        FromLetterToNumber[SequenceUnique[k]]=k

    EdgeNames=MakeObjectNames(GraphName+'Edge',numberOfEdges-1)
    localScale=extend/EdgeScaling

    for k in range(numberOfEdges-1):
    
        nodeA=FromNumberToName[FromLetterToNumber[splitSequence[k]]]
        nodeB=FromNumberToName[FromLetterToNumber[splitSequence[k+1]]]

        pointA=NodesLocations[nodeA]
        pointB=NodesLocations[nodeB]

        midpoint=MidPointLocation(pointA,pointB)
        edgeLength=Distance(pointA,pointB)/2
        rotation=Rotation(pointA,pointB)
        bpy.ops.mesh.primitive_cube_add(location=midpoint)
        bpy.data.objects["Cube"].scale=(localScale,edgeLength,localScale)
        bpy.data.objects["Cube"].rotation_euler=(0,0,rotation)
        bpy.data.objects["Cube"].data.name=EdgeNames[k]
        bpy.data.objects["Cube"].name=EdgeNames[k]
    
    return EdgeNames

#Wrapper function to add the graph
def AddSequenceGraph(splitSequence,extend,LayoutCenter,LayoutType,GraphName,NodeScaling,EdgeScaling,Labels=False):

    UniqueElements=np.unique(splitSequence)
    NodesNames,NodesLocations=AddGraphNodes(splitSequence,extend,LayoutCenter,LayoutType,GraphName,NodeScaling)
    EdgeNames=AddEdgesFromSequence(splitSequence,extend,EdgeScaling,NodesNames,NodesLocations,GraphName)

    if Labels:
        
        LabelScaling=(1.5*extend)/NodeScaling
        Padding=1.5*LabelScaling
        LabelNames=AddNodesLabels(UniqueElements,NodesNames,NodesLocations,GraphName,LabelScaling,Padding)
        
        return LabelNames,NodesNames,EdgeNames

    else:
        return NodesNames,EdgeNames

###############################################################################
# Material Functions
###############################################################################

def AddTypeAMaterial(GeometryName,RGBData):
    """
    Simple RGB and BSDF principled node mixed.
    Parameters
    ----------
    GeometryName : string
        Geometry name of the object.
    RGBData : tuple, list
        RGB values for the rgb shader output.

    Returns
    -------
    None.

    """

    r,g,b=RGBData
    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (r,g,b,1)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs[4].default_value = 0.5
    bsdf.inputs[7].default_value = 1
    bsdf.inputs[15].default_value = 1
    currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[0])
    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(bsdf.outputs[0],materialOutput.inputs[0])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

def AddTypeBMaterial(GeometryName,RGBData):
    """
    Small glass and transparency effect
    Parameters
    ----------
    GeometryName : string
        Geometry name of the object.
    RGBData : tuple, list
        RGB values for the rgb shader output.
        
    Returns
    -------
    None.

    """

    currentMaterial = bpy.data.materials.new(name="TypeB"+GeometryName)
    currentMaterial.use_nodes = True

    nodes = currentMaterial.node_tree.nodes
    r,g,b=RGBData
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (r,g,b,1)

    diffuse=nodes.get("Diffuse BSDF")
    diffuse.inputs[1].default_value = 1

    currentMaterial.node_tree.links.new(rgb.outputs[0],diffuse.inputs[0])

    bsdf = nodes.new("ShaderNodeBsdfGlass")
    bsdf.inputs[1].default_value = 1

    mixer = nodes.new("ShaderNodeMixShader")
    mixer.inputs[0].default_value = 0.6

    currentMaterial.node_tree.links.new(diffuse.outputs[0],mixer.inputs[1])
    currentMaterial.node_tree.links.new(bsdf.outputs[0],mixer.inputs[2])

    materialOutput=nodes.get("Material Output")

    currentMaterial.node_tree.links.new(mixer.outputs[0],materialOutput.inputs[0])

    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

#Wrapper function to add a black type A material
def AddTypeAMaterialBlack(GeometryName):
    return AddTypeAMaterial(GeometryName,(0,0,0))

#Wrapper function to add a blue type B material
def AddTypeBMaterialBlue(GeometryName):
    return AddTypeAMaterial(GeometryName,(0,0,1))

###############################################################################
# Removing the original cube
###############################################################################

bpy.data.objects["Cube"].select=True
bpy.ops.object.delete()

###############################################################################
# Moving the camera
###############################################################################

bpy.data.objects["Camera"].location=(0,0,7)
bpy.data.objects["Camera"].rotation_euler=(0,0,0)

###############################################################################
# Changing the horizon 
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.world.horizon_color = (1,1,1)

###############################################################################
# Adding the graphs
###############################################################################

sequenceNames,sequences=GetFileSequences(DataDir)

nSeqs=len(sequenceNames)
cLocations=GetSquaredGridCenters(nSeqs,3,1.75,(0,0,0))

for k in range(nSeqs):
    
    loopSeq=sequences[k]
    loopSplit=[val for val in loopSeq]
    loopNodeNames,loopEdgeNames=AddSequenceGraph(loopSplit,0.35,cLocations[k],"Circular",'Test'+str(k),15,125,Labels=False)
    NodeName=JoinObjectsByName(loopNodeNames)
    EdgeName=JoinObjectsByName(loopEdgeNames)
    AddTypeBMaterialBlue(NodeName)
    AddTypeAMaterialBlack(EdgeName)
