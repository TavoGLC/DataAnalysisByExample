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

######################################################################
#                 Intended Use
######################################################################

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

SequenceExample="SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"

GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory+"cartoon.obj"

###############################################################################
# General Functions 
###############################################################################

def SplitString(String,ChunkSize):
    """
    Parameters
    ----------
    String : String
        String to be divided.
    ChunkSize : int
        Size of the fragment taken.

    Returns
    -------
    Splitted : list
        List with the fragments of ChunkSize size taken from String.
    """
    if ChunkSize==1:
        Splitted=[val for val in String]
    
    else:
        nCharacters=len(String)
        Splitted=[String[k:k+ChunkSize] for k in range(nCharacters-ChunkSize)]
        
    return Splitted

def UniqueToDictionary(UniqueElements):
    """
    Parameters
    ----------
    UniqueElements : list,array
        Container of the unique elements.

    Returns
    -------
    localDictionary : dictionary
        Maps element to an index value.
    """
    localDictionary={}
    nElements=len(UniqueElements)
    
    for k in range(nElements):
        localDictionary[UniqueElements[k]]=k
        
    return localDictionary

def CountUniqueElements(UniqueElements,ProcessedString):
    """
    Parameters
    ----------
    UniqueElements : list,array
        List of elements to calculate its frequency in ProcessedString.
    ProcessedString : list,array
        List of strings.

    Returns
    -------
    localCounter : list
        List with the frequencies of Unique elements in ProcessedString.
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

#Calculates the distance between two points
def Distance(PointA,PointB):
    return np.sqrt(sum([(val-sal)**2 for val,sal in zip(PointA,PointB)]))

#Calculates the midpoint between two points
def MidPointLocation(PointA,PointB):
    return tuple([(val+sal)/2 for val,sal in zip(PointA,PointB)])

#Calculates the slope between two points
def Rotation(PointA,PointB):
    return np.arctan((PointA[1]-PointB[1])/(PointA[0]-PointB[0]))-(np.pi/2)

###############################################################################
# Naming Functions
###############################################################################

#Generates a list with the names of the objects
def MakeObjectNames(ObjectName,NumberOfObjects):
    """
    Parameters
    ----------
    ObjectName : String
        Name of the object to be created.
    NumberOfObjects : int
        Number of names to be created.

    Returns
    -------
    NamesContainer : list
        List of strings with the names of the objects.

    """
    NamesContainer=[]
    NamesContainer.append(ObjectName)

    for k in range(1,NumberOfObjects):
        if k<=9:
            NamesContainer.append(ObjectName+".00"+str(k))
        elif k>9 and k<=99:
            NamesContainer.append(ObjectName+".0"+str(k))
        elif k>99 and k<=999:
            NamesContainer.append(ObjectName+"."+str(k))

    return NamesContainer

###############################################################################
# Adding the text block
###############################################################################

def InsertTextBlock(TextLocation,nLines,scale,Text,ObjectName):
    """
    Parameters
    ----------
    TextLocation : tuple
        Global center location of the text block.
    nLines : int
        Number of lines used to write Text.
    scale : float (0,1)
        Scale of the text object.
    Text : String
        Text to be written.
    ObjectName : String
        Name of the text object.

    Returns
    -------
    None.
    """
    totalCharacters=len(Text)
    characterPerLine=int(totalCharacters/nLines)
    remainingCharacters=totalCharacters-(nLines*characterPerLine)
    
    bpy.ops.object.text_add(location=TextLocation,enter_editmode=True)
    for k in range(4):
        bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
    for k in range(nLines):
        currentText=Text[(k*characterPerLine):(k+1)*characterPerLine]
        for char in currentText:
            bpy.ops.font.text_insert(text=char)
        bpy.ops.font.line_break()
    if remainingCharacters!=0 and remainingCharacters>0:
        currentText=Text[-remainingCharacters:]
        for char in currentText:
            bpy.ops.font.text_insert(text=char)
    bpy.context.object.data.align_x='CENTER'
    bpy.context.object.data.align_y='CENTER'
    bpy.ops.object.editmode_toggle()
    bpy.data.objects['Text'].scale=(scale,scale,scale)
    bpy.data.objects['Text'].name=ObjectName

###############################################################################
# Graph functions 
###############################################################################

def MakeCircularLayout(nNodes,radious,LayoutCenter,GraphName):
    """
    Parameters
    ----------
    nNodes : int
        Number of nodes in the graph
    radious : float
        Radious of the circular layout.
    LayoutCenter : tuple
        X,Y,Z location of the graph.
    GraphName : String
        Name of the graph object.

    Returns
    -------
    NodesNames : list
        List with the geometry names of all the nodes.
    NodeNameToPosition : dict
        Dicttionary with the location of each node.
    """
    NodesNames=MakeObjectNames(GraphName+'Node',nNodes)
    NodeNameToPosition={}
    localScale=radious/15
    degreeSlice=(2*np.pi)/nNodes
    Xcent,Ycent,Zcent=LayoutCenter

    for k in range(nNodes):
    
        degree=k*degreeSlice
        Xpos=radious*np.cos(degree)
        Ypos=radious*np.sin(degree)
        bpy.ops.mesh.primitive_uv_sphere_add(location=(Xpos+Xcent,Ypos+Ycent,Zcent))
        bpy.data.objects["Sphere"].scale=(localScale,localScale,localScale)
        NodeNameToPosition[NodesNames[k]]=(Xpos+Xcent,Ypos+Ycent,Zcent)
        bpy.data.objects["Sphere"].data.name=NodesNames[k]
        bpy.data.objects["Sphere"].name=NodesNames[k]

    return NodesNames,NodeNameToPosition

def MakeRandomLayout(nNodes,extend,LayoutCenter,GraphName):
    """
    Parameters
    ----------
    nNodes : int
        Number of nodes in the graph
    extend : float
        Extension of the random layout
    LayoutCenter : tuple
        X,Y,Z location of the graph.
    GraphName : String
        Name of the graph object.

    Returns
    -------
    NodesNames : list
        List with the geometry names of all the nodes.
    NodeNameToPosition : dict
        Dicttionary with the location of each node.

    """   
    NodesNames=MakeObjectNames(GraphName+'Node',nNodes)
    NodeNameToPosition={}
    Xcent,Ycent,Zcent=LayoutCenter
    localScale=extend/15
    Xvals=[extend*val*(-1)**np.random.randint(10) for val in np.random.random(nNodes)]
    Yvals=[extend*val*(-1)**np.random.randint(10) for val in np.random.random(nNodes)]

    for k in range(nNodes):
    
        Xpos=Xvals[k]
        Ypos=Yvals[k]
        bpy.ops.mesh.primitive_uv_sphere_add(location=(Xpos+Xcent,Ypos+Ycent,Zcent))
        bpy.data.objects["Sphere"].scale=(localScale,localScale,localScale)
        NodeNameToPosition[NodesNames[k]]=(Xpos+Xcent,Ypos+Ycent,Zcent)
        bpy.data.objects["Sphere"].data.name=NodesNames[k]
        bpy.data.objects["Sphere"].name=NodesNames[k]

    return NodesNames,NodeNameToPosition

#Wrapper function to create the graph layout
def AddGraphLayout(nNodes,extend,LayoutCenter,GraphName,layout='Circular',):
    if layout=='Circular':
        return MakeCircularLayout(nNodes,extend,LayoutCenter,GraphName)
    elif layout=='Random':
        return MakeRandomLayout(nNodes,extend,LayoutCenter,GraphName)
    else:
        return 'Not valid Layout'

def AddEdgesFromSequence(Sequence,NodesNames,NodesLocations,extend,GraphName):
    """
    Parameters
    ----------
    Sequence : String
        Sequence to be used to create the graph.
    NodesNames : List
        List of strings with the names of the geometries.
    NodesLocations : Dict
        Dictionary with the location of the nodes.
    extend : float
        Extension of the graph layout.
    GraphName : String
        Name of the graph object.

    Returns
    -------
    EdgeNames : Lits
        List with the names of all the edges in the graph.
    """
    SequenceSplitted=[val for val in SequenceExample]
    SequenceUnique=np.unique(SequenceSplitted)
    
    numberOfNodes=len(SequenceUnique)
    numberOfEdges=len(SequenceSplitted)
    
    FromNumberToName={}
    FromLetterToNumber={}

    for k in range(numberOfNodes):
        FromNumberToName[k]=NodesNames[k]
        FromLetterToNumber[SequenceUnique[k]]=k

    EdgeNames=MakeObjectNames(GraphName+'Edge',numberOfEdges-1)
    localScale=extend/100

    for k in range(numberOfEdges-1):
    
        nodeA=FromNumberToName[FromLetterToNumber[SequenceSplitted[k]]]
        nodeB=FromNumberToName[FromLetterToNumber[SequenceSplitted[k+1]]]

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

def AddNodesMaterials(NodesNames):
    """
    Parameters
    ----------
    NodesNames : List
        List with the name of all the nodes.

    Returns
    -------
    None.
    """
    for val in NodesNames:

        currentMaterial = bpy.data.materials.new(name='NodeMaterial'+val)
        currentMaterial.use_nodes = True

        nodes = currentMaterial.node_tree.nodes

        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = (0,0,1,1)

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

        bpy.data.objects[val].data.materials.append(currentMaterial)

def AddEdgeMaterials(EdgeNames):
    """
    Parameters
    ----------
    EdgeNames : List
        List with the name of all the edges.

    Returns
    -------
    None.
    """
    for val in EdgeNames:

        currentMaterial = bpy.data.materials.new(name='EdgeMaterial'+val)
        currentMaterial.use_nodes = True
    
        nodes = currentMaterial.node_tree.nodes

        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = (0,0,0,1)

        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs[4].default_value = 0.5
        bsdf.inputs[7].default_value = 1
        bsdf.inputs[15].default_value = 1
    
        currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[0])

        materialOutput=nodes.get("Material Output")
        currentMaterial.node_tree.links.new(bsdf.outputs[0],materialOutput.inputs[0])

        bpy.data.objects[val].data.materials.append(currentMaterial)

#Wrapper function to add a graph
def AddGraph(Sequence,extent,LayoutCenter,GraphName,layout='Circular'):
    
    SequenceSplitted=[val for val in Sequence]
    SequenceUnique=np.unique(SequenceSplitted)
    nNodes=len(SequenceUnique)
    NodesNames,NodesLocations = AddGraphLayout(nNodes,extent,LayoutCenter,GraphName,layout=layout)
    EdgeNames=AddEdgesFromSequence(Sequence,NodesNames,NodesLocations,extent,GraphName)
    AddNodesMaterials(NodesNames)
    AddEdgeMaterials(EdgeNames)

###############################################################################
# BarPlot Functions
###############################################################################

def MakeBarPlot(BarsData,ticksLabels,PlotName,location,extend):
    """
    Parameters
    ----------
    BarsData : list,array
        Data for the bar plot.
    ticksLabels : list,array
        list with the tick labels for the x axis in the bar plot.
    PlotName : string
        name for the geometries created for the plot.
    location : tuple
        Global location for the bar plot.
    extend : float
        extend of the bar plot.

    Returns
    -------
    barNames : list
        list with the bars geometry names.
    tickNames : lits
        list with the ticks geometry names..

    """
    nBars=len(BarsData)
    maxVal=max(BarsData)
    NormData=[val/maxVal for val in BarsData]
    step=(2*extend)/(nBars-1)
    xBarLocations=[-(extend)+(k*step)+location[0] for k in range(nBars)]
    barNames=MakeObjectNames(PlotName+"Bar",nBars)
    tickNames=MakeObjectNames(PlotName+"Tick",nBars)

    for k in range(nBars):
        currentBarHeight=(extend/2)*NormData[k]
        currentBarLocation=currentBarHeight-extend+location[1]
        bpy.ops.mesh.primitive_cube_add(radius=1,location=(xBarLocations[k],currentBarLocation,0))
        bpy.data.objects["Cube"].scale=(extend/40,currentBarHeight,extend/40)
        bpy.data.objects["Cube"].data.name=barNames[k]
        bpy.data.objects["Cube"].name=barNames[k]

        bpy.ops.object.text_add(location=(xBarLocations[k],-extend-location[1]-0.45,0),enter_editmode=True)
        for j in range(4):
            bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
        for char in ticksLabels[k]:
            bpy.ops.font.text_insert(text=char)
        
        bpy.context.object.data.align_x='CENTER'
        bpy.context.object.data.align_y='CENTER'
        bpy.ops.object.editmode_toggle()
        bpy.data.objects['Text'].scale=(extend/10,extend/10,extend/10)
        bpy.data.objects['Text'].name=tickNames[k]

    return barNames,tickNames

###############################################################################
# Material functions 
###############################################################################

def AddRGBMaterials(GeometryNames,RGBData):
    """
    Parameters
    ----------
    GeometryNames : list
        List with the geometry namesto apply the following material.
    RGBData : tuple, list, len(RGBData)==3
        tuple or list with the rgb values.

    Returns
    -------
    None.
    """
    r,g,b=RGBData
    for val in GeometryNames:

        currentMaterial = bpy.data.materials.new(name='EdgeMaterial'+val)
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

        bpy.data.objects[val].data.materials.append(currentMaterial)

###############################################################################
# Removing the original cube
###############################################################################

bpy.data.objects["Cube"].select=True
bpy.ops.object.delete()

###############################################################################
# Moving the camera
###############################################################################

bpy.data.objects["Camera"].location=(0,0,6)
bpy.data.objects["Camera"].rotation_euler=(0,0,0)

###############################################################################
# Changing the horizon 
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.world.horizon_color = (1,1,1)

###############################################################################
# Making the Dashboard
###############################################################################

#inserting the protein structure
bpy.ops.import_scene.obj(filepath=DataDir)
bpy.context.scene.objects.active=bpy.data.objects["cartoon"]
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.remove_doubles()
bpy.ops.object.editmode_toggle()
bpy.data.objects["cartoon"].scale=(0.03,0.03,0.03)
bpy.data.objects["cartoon"].location=(1.5,0.7,0)
AddRGBMaterials(["cartoon"],(0,0,1))

#Adding the sequence in a text block
InsertTextBlock((-2,0,0),20,0.125,SequenceExample,"Sequence")
AddRGBMaterials(["Sequence"],(0,0,0))

#Adding the sequence graph representation 
AddGraph(SequenceExample,0.6,(-0.5,0.75,0),"Circular")
AddGraph(SequenceExample,0.6,(-0.5,-0.75,0),"Random",layout="Random")

#Adding the sequence aminoacid frequency
SequenceSplit=SplitString(SequenceExample,1)
UniqueSequenceAA=np.unique(SequenceSplit)
Frequencies=CountUniqueElements(UniqueSequenceAA,SequenceSplit)

BarNames,TickNames=MakeBarPlot(Frequencies,UniqueSequenceAA,"BarPlot",(1.3,-0.15,0),1)
AddRGBMaterials(BarNames,(0,0,1))
AddRGBMaterials(TickNames,(0,0,0))
