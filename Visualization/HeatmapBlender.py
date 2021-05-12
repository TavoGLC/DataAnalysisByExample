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
import bmesh
import colorsys
import numpy as np

###############################################################################
# Utility functions
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
        else:
            NamesContainer.append(ObjectName+"."+str(k))
    return NamesContainer

def GetCentersLocations(NumberOfItems,stretch):
    """
    Parameters
    ----------
    NumberOfItems : int
        Row or solumn length.
    stretch : float
        boundaries of the visualization square

    Returns
    -------
    centerLocations : list
        List with the center locations of each cube in the heatmap

    """
    
    grid=np.linspace(stretch,-stretch,NumberOfItems+1)
    
    centerLocations=[]
    indexLocations=[]

    for k in range(NumberOfItems):
        
        midY=(grid[k+1]+grid[k])/2
        
        for i in range(NumberOfItems):
            
            midX=(grid[i+1]+grid[i])/2
            currentLoc=(midX,0,midY)
            centerLocations.append(currentLoc)
            
    return centerLocations

###############################################################################
# Text functions
###############################################################################

def InsertAndChangeText(TextLocation,TextLabel,Name,scale,rotation):
    """
    Parameters
    ----------
    TextLocation : tuple
        Location of the text object
    TextLabel : string
        Text to add to the text object
    Name : string
        Name of the text object
    scale : float
        Scale of the text object
    rotation : tuple
        Rotation of the text object

    Returns
    -------
    None

    """
    
    bpy.ops.object.text_add(location=TextLocation,enter_editmode=True)
    for k in range(4):
        bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
    for char in TextLabel:
        bpy.ops.font.text_insert(text=char)
    bpy.ops.object.editmode_toggle()
    bpy.data.objects['Text'].data.name=Name
    bpy.data.objects['Text'].name=Name
    bpy.data.objects[Name].rotation_euler=rotation
    bpy.data.objects[Name].scale=(scale,scale,scale)

###############################################################################
# Data functions
###############################################################################

def MinMaxNormalization(DisplacementData):
"""
    Parameters
    ----------
    DisplacementData : array
        Data for normalization
    
    Returns
    -------
    disp : array
        Normalized data 

    """
    
    disp=(DisplacementData-DisplacementData.min())/(DisplacementData.max()-DisplacementData.min())
    return disp

###############################################################################
# Geometry functions
###############################################################################

def AddGeometryByKind(Kind,Name,Location,Scale):
    '''
    Add a mesh object to the scee and change it's name 
    Parameters
    ----------
    Kind : string
        Name of the mesh to be added. Takes the values of Sphere, Cube, Cylinder and 
        Torus
    Name : string
        New name of the geometry.
    Location : tuple
        Location of the object.
    Scale : tuple
        Scale of the object.

    Returns
    -------
    None.

    '''

    if Kind=="Sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(location=Location,segments=64,ring_count=32)
    elif Kind=="Cube":
        bpy.ops.mesh.primitive_cube_add(location=Location)
    elif Kind=="Cylinder":
        bpy.ops.mesh.primitive_cylinder_add(location=Location,vertices=64)
    elif Kind=="Torus":
        bpy.ops.mesh.primitive_torus_add(location=Location,major_segments=96,minor_segments=24)

    bpy.data.objects[Kind].scale=Scale
    bpy.data.objects[Kind].data.name=Name
    bpy.data.objects[Kind].name=Name

def AddElementsForMatrix(Locations,Scale,Kind,Name):

    """
        Parameters
    ----------
    Locations : list,array-like
        Location of the objects.
    Scale : float
        Scale of the geometry
    Kind : str
        Kind of geometry to be added.
    Name : str
        Name of the geometry

    Returns
    -------
    objectNames : list
        Names of the geometries.
    """

    objectNames=MakeObjectNames('GeometryForMatrix'+Name,len(Locations))
    nObjects=len(Locations)

    try:
        ScaleArray=len(Locations)==len(Scale)
    except TypeError:
        ScaleArray=False
    
    for k in range(nObjects):

        if ScaleArray:
            AddGeometryByKind(Kind,objectNames[k],tuple(Locations[k]),(Scale[k],Scale[k],Scale[k]))
        else:
            AddGeometryByKind(Kind,objectNames[k],tuple(Locations[k]),(Scale,Scale,Scale))
    
    return objectNames

###############################################################################
# Material functions
###############################################################################

def AddPolystyreneTypeMaterial(GeometryName,RGBData):
    """
    Polystyrene type material
    ----------
    GeometryName : string
        Geometry name of the object.
    RGBData : tuple, list
        RGB values for the rgb shader output.

    Returns
    -------
    None.

    """

    #Color

    r,g,b=RGBData
    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (r,g,b,1)

    #Bump

    texture=nodes.new("ShaderNodeTexCoord")

    voronoitex=nodes.new("ShaderNodeTexVoronoi")

    multiplication=nodes.new("ShaderNodeMath")
    multiplication.inputs[0].default_value = 5
    multiplication.inputs[1].default_value = 20

    bump=nodes.new("ShaderNodeBump")
    bump.invert = True
    bump.inputs[1].default_value = 0.2

    currentMaterial.node_tree.links.new(texture.outputs[0],voronoitex.inputs[0])
    currentMaterial.node_tree.links.new(multiplication.outputs[0],voronoitex.inputs[1])
    currentMaterial.node_tree.links.new(voronoitex.outputs[1],bump.inputs[2])

    #Shader

    fresnel=nodes.new("ShaderNodeFresnel")
    fresnel.inputs[0].default_value = 1.550 

    diffuse=nodes.new("ShaderNodeBsdfDiffuse")
    currentMaterial.node_tree.links.new(rgb.outputs[0],diffuse.inputs[0])

    glossy=nodes.new("ShaderNodeBsdfGlossy")
    glossy.inputs[1].default_value = 0.6

    mix=nodes.new("ShaderNodeMixShader")

    currentMaterial.node_tree.links.new(fresnel.outputs[0],mix.inputs[0])
    currentMaterial.node_tree.links.new(diffuse.outputs[0],mix.inputs[1])
    currentMaterial.node_tree.links.new(glossy.outputs[0],mix.inputs[2])

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(mix.outputs[0],materialOutput.inputs[0])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

def AddEmeraldTypeMaterial(GeometryName,RGBData):
    """
    Emeral Type material
    ----------
    GeometryName : string
        Geometry name of the object.
    RGBData : tuple, list
        RGB values for the rgb shader output.

    Returns
    -------
    None.

    """

    #Color

    r,g,b=RGBData
    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (r,g,b,1)

    glass=nodes.new("ShaderNodeBsdfGlass")

    volumeabs=nodes.new("ShaderNodeVolumeAbsorption")
    volumeabs.inputs[1].default_value = 13.4
    currentMaterial.node_tree.links.new(rgb.outputs[0],volumeabs.inputs[0])

    
    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(glass.outputs[0],materialOutput.inputs[0])
    currentMaterial.node_tree.links.new(volumeabs.outputs[0],materialOutput.inputs[1])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

###############################################################################
# Settings for the render
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 3
bpy.context.scene.world.horizon_color = (1,1,1)
bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 1200
bpy.context.scene.render.tile_x = 384
bpy.context.scene.render.tile_y = 256
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 750

###############################################################################
# Settings for the scene
###############################################################################

bpy.data.objects['Cube'].data.name='StartCube'
bpy.data.objects['Cube'].name='StartCube'
bpy.data.objects['StartCube'].select=True
bpy.data.objects['StartCube'].scale=(6,6,6)
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.bevel(offset=0.15,vertex_only=False)
bpy.ops.object.editmode_toggle()
bpy.ops.object.shade_smooth()
bpy.data.objects['StartCube'].select=False

bpy.data.objects["Camera"].location=(0,4,0)
bpy.data.objects["Camera"].rotation_euler=(-np.pi/2,np.pi,0)

bpy.data.objects['Lamp'].data.use_nodes=True
bpy.data.objects['Lamp'].data.node_tree.nodes['Emission'].inputs[1].default_value=500
bpy.data.objects['Lamp'].data.type='AREA'
bpy.data.objects['Lamp'].location=(0,0,4.5)
bpy.data.objects["Lamp"].rotation_euler=(0,0,0)
bpy.data.objects['Lamp'].scale=(50,50,50)

###############################################################################
# Heat map data
###############################################################################
GlobalDirectory=r"/home/tavoglc/LocalData/"
DataDir=GlobalDirectory+'Correlation.csv'

Data=np.genfromtxt(DataDir,delimiter=',',dtype=['|S50','<f8', '<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8','<f8'])
labels=[Data[k][0].decode() for k in range(1,len(Data))]

Data=np.genfromtxt(DataDir,delimiter=',')

MatrixData=Data[1::,1::]

maxScale=0.075
nItems=MatrixData.shape[0]
delta=0.1
stretch=nItems*maxScale+delta

center=GetCentersLocations(nItems,stretch)
names = AddElementsForMatrix(center,maxScale,"Cube","Test")

for nme,dta in zip(names,MatrixData.ravel()):
    AddPolystyreneTypeMaterial(nme,[dta,dta,dta])

###############################################################################
# Labels
###############################################################################

labelCenterlocationsY=[center[k*nItems+1] for k in range(nItems)]
labelCenterlocationsX=[center[k] for k in range(nItems)]

scale=0.075
xpos=stretch+7*scale+0.1
ypos=0

for k,nme in enumerate(labels):
    InsertAndChangeText((xpos,ypos,labelCenterlocationsY[k][2]),nme,nme+"y",scale,(np.pi/2,0,np.pi))
    AddPolystyreneTypeMaterial(nme+"y",[0,0,0])
    InsertAndChangeText((labelCenterlocationsX[k][0],ypos,labelCenterlocationsY[0][2]+2*scale),nme,nme+"x",scale,(np.pi/2,-np.pi/4,np.pi))
    AddPolystyreneTypeMaterial(nme+"x",[0,0,0])
