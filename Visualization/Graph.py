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

nNodes=25
edgeList={}

for k in range(nNodes):
    cvector=[j for j in range(nNodes) if j!=k]
    edgeList[k]=int(np.random.choice(cvector))

nEdges=len(edgeList)

###############################################################################
# General Functions 
###############################################################################

#Calculates the distance between two points
def Distance(PointA,PointB):
    return np.sqrt(sum([(val-sal)**2 for val,sal in zip(PointA,PointB)]))

#Calculates the midpoint between two points
def MidPointLocation(PointA,PointB):
    return tuple([(val+sal)/2 for val,sal in zip(PointA,PointB)])

def Rotation(PointA,PointB):
    return np.arctan((PointA[1]-PointB[1])/(PointA[0]-PointB[0]))-(np.pi/2)

###############################################################################
# Naming Functions
###############################################################################

#Generates a list with the names of the objects
def MakeObjectNames(ObjectName,NumberOfObjects):
    """
    ObjectName -> Base name of the object created
    NumberOfObjects -> Number of objects to be created
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

#Wrapper function to create a list of shpere object names
def MakeNodesNames(NumberOfElements):
    return MakeObjectNames("Sphere",NumberOfElements)

#Wrapper function to create a list of cube object names
def MakeEdgeNames(NumberOfElements):
    return MakeObjectNames("Cube",NumberOfElements)

###############################################################################
# Geometry Functions
###############################################################################

def MakeCircularLayout(nNodes,radious):
    """
    Add the nodes geometries for the graph circular layout
    
    nNodes  -> Number of nodes in the graph
    radious -> Radious of the circle in the circular layout
    """
    NodesNames=MakeNodesNames(nNodes)
    NodeNameToPosition={}
    degreeSlice=(2*np.pi)/nNodes

    for k in range(nNodes):
    
        degree=k*degreeSlice
        Xpos=radious*np.cos(degree)
        Ypos=radious*np.sin(degree)
        bpy.ops.mesh.primitive_uv_sphere_add(location=(Xpos,Ypos,0))
        bpy.data.objects[NodesNames[k]].scale=(0.05,0.05,0.05)
        NodeNameToPosition[NodesNames[k]]=(Xpos,Ypos,0)

    return NodesNames,NodeNameToPosition

def MakeRandomLayout(nNodes,extend):
    """
    Add the nodes geometries for the graph circular layout
    
    nNodes -> Number of nodes in the graph
    extend -> Range for the edges to be spreaded 
    """    
    NodesNames=MakeNodesNames(nNodes)
    NodeNameToPosition={}
    Xvals=[extend*val*(-1)**np.random.randint(10) for val in np.random.random(nNodes)]
    Yvals=[extend*val*(-1)**np.random.randint(10) for val in np.random.random(nNodes)]

    for k in range(nNodes):
    
        Xpos=Xvals[k]
        Ypos=Yvals[k]
        bpy.ops.mesh.primitive_uv_sphere_add(location=(Xpos,Ypos,0))
        bpy.data.objects[NodesNames[k]].scale=(0.05,0.05,0.05)
        NodeNameToPosition[NodesNames[k]]=(Xpos,Ypos,0)

    return NodesNames,NodeNameToPosition

def AddGraphLayout(nNodes,extend,layout='Circular'):
    """
    Wrapper function to create the graph layout
    """
    if layout=='Circular':
        return MakeCircularLayout(nNodes,extend)
    elif layout=='Random':
        return MakeRandomLayout(nNodes,extend)
    else:
        return 'Not valid Layout'


def AddEdgesFromIncidenceList(IncidenceList,NodesNames,NodesLocations):
    """
    Add the edge geometries for the graph
    
    IncidenceList   -> Dictionary with the nodes that shared an edge 
    NodesNames      -> List with the name of the nodes 
    NodesLocations  -> Dictionary that contains the location of the nodes 
    """    
    
    numberOfNodes=len(NodesNames)
    numberOfEdges=len(IncidenceList)
    
    FromNumberToName={}
    for k in range(numberOfNodes):
        FromNumberToName[k]=NodesNames[k]

    EdgeNames=MakeEdgeNames(numberOfEdges)
    EdgeLocations={}

    for k in range(numberOfEdges):
    
        nodeA=FromNumberToName[k]
        nodeB=FromNumberToName[IncidenceList[k]]

        pointA=NodesLocations[nodeA]
        pointB=NodesLocations[nodeB]

        midpoint=MidPointLocation(pointA,pointB)
        edgeLength=Distance(pointA,pointB)/2
        rotation=Rotation(pointA,pointB)
        EdgeLocations[EdgeNames[k]]=midpoint
        bpy.ops.mesh.primitive_cube_add(location=midpoint)
        bpy.data.objects[EdgeNames[k]].scale=(0.01,edgeLength,0.01)
        bpy.data.objects[EdgeNames[k]].rotation_euler=(0,0,rotation)
    
    return EdgeNames,EdgeLocations


def AddNodesMaterials(NodesNames):
    """
    Add the nodes materials
    NodesNames      -> List with the name of the nodes 
    """  

    for val in NodesNames:

        currentMaterial = bpy.data.materials.new(name='NodeMaterial'+val)
        currentMaterial.use_nodes = True

        nodes = currentMaterial.node_tree.nodes

        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = (np.random.random(),np.random.random(),np.random.random(),1)

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
    Add the nodes materials
    EdgeNames      -> List with the name of the edges
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
    
        currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[1])

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
# Adding the nodes
###############################################################################

NodesNames,NodesLocations = AddGraphLayout(nNodes,1.2,layout='Circular')

###############################################################################
# Adding the edges
###############################################################################

EdgeNames,EdgeLocations=AddEdgesFromIncidenceList(edgeList,NodesNames,NodesLocations)

###############################################################################
# Changing the horizon 
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.world.horizon_color = (1,1,1)

###############################################################################
# Adding the materials 
###############################################################################

AddNodesMaterials(NodesNames)
AddEdgeMaterials(EdgeNames)
