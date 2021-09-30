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

    GeometryNames = ["Sphere","Cube","Cylinder","Torus","Suzanne"]
    GeometryFunctions = [bpy.ops.mesh.primitive_uv_sphere_add,bpy.ops.mesh.primitive_cube_add
                        ,bpy.ops.mesh.primitive_cylinder_add,bpy.ops.mesh.primitive_torus_add
                        ,bpy.ops.mesh.primitive_monkey_add]
    
    NameToLocation = {}

    for val,name in enumerate(GeometryNames):
        NameToLocation[name] = val

    try:
        
        funcIndex = NameToLocation[Kind]
        GeometryFunctions[funcIndex](location=Location)
        bpy.data.objects[Kind].scale = Scale
        bpy.data.objects[Kind].data.name = Name
        bpy.data.objects[Kind].name = Name
        
        if Kind == "Torus":
            bpy.data.objects[Name].rotation_euler = (np.pi/2,0,0)

        if Kind == "Suzanne":
            bpy.data.objects[Name].rotation_euler = (0,0,np.pi)
            
    except ValueError:
        print("Not recognized type")


def AddElementsFromLocations(Locations,Scale,Kind,Name):
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
            AddGeometryByKind(Kind,objectNames[k],tuple(Locations[k]),Scale[k])
        else:
            AddGeometryByKind(Kind,objectNames[k],tuple(Locations[k]),(Scale,Scale,Scale))
    
    return objectNames

###############################################################################
# Text Geometry functions
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
# Material functions
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
    currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[3])
    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(bsdf.outputs[0],materialOutput.inputs[0])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

def AddGoldMaterial(GeometryName,RGBData):
    """
    Stone Type material
    ----------
    GeometryName : string
        Geometry name of the object.
    RGBData : tuple, list
        RGB values for the rgb shader output.

    Returns
    -------
    None.

    """
        
    #Nodes handler
    currentMaterial = bpy.data.materials.new(name='Gold'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes

    r,g,b=RGBData
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (r,g,b,1)

    rgb2 = nodes.new("ShaderNodeRGB")
    rgb2.outputs[0].default_value = (r/2,g/2,b/2,1)

    mixRGB = nodes.new('ShaderNodeMixRGB')

    currentMaterial.node_tree.links.new(rgb.outputs[0],mixRGB.inputs[1])
    currentMaterial.node_tree.links.new(rgb2.outputs[0],mixRGB.inputs[2])

    layerWeight = nodes.new('ShaderNodeLayerWeight')
    layerWeight.inputs[0].default_value = 0.15
    
    currentMaterial.node_tree.links.new(layerWeight.outputs[1],mixRGB.inputs[0])

    glossyBSDF = nodes.new('ShaderNodeBsdfGlossy')
    currentMaterial.node_tree.links.new(mixRGB.outputs[0],glossyBSDF.inputs[0])

    powerNode = nodes.new('ShaderNodeMath')
    powerNode.operation = 'POWER'

    currentMaterial.node_tree.links.new(mixRGB.outputs[0],glossyBSDF.inputs[0])
    currentMaterial.node_tree.links.new(powerNode.outputs[0],glossyBSDF.inputs[1])

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(glossyBSDF.outputs[0],materialOutput.inputs[0])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

def TwoPointLinearColorMap(startPoint,endPoint,points):
    """
    Parameters
    ----------
    startPoint : tuple,array-like
        Start location in the RGB color space.
    endPoint : tuple, array-like
        End location in the RGB color space.
    points : int
        Number of sample points between the startPoint and endPoint 

    Returns
    -------
    objectNames : array
        contains an linearly sampled RGB colors between the startPoint and the endPoint
    """
    
    rs,gs,bs=startPoint
    re,ge,be=endPoint
    r=np.linspace(rs,re,num=points)
    g=np.linspace(gs,ge,num=points)
    b=np.linspace(bs,be,num=points)
    
    return np.vstack((r,g,b)).T

###############################################################################
# Render Settings
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 5
bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 1200
bpy.context.scene.render.tile_x = 384
bpy.context.scene.render.tile_y = 256
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 750

###############################################################################
# Scene settings
###############################################################################

bpy.data.objects['Cube'].data.name = 'StartCube'
bpy.data.objects['Cube'].name = 'StartCube'
bpy.data.objects['StartCube'].select_set(state = True)
bpy.data.objects['StartCube'].scale = (6,6,6)
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.bevel(offset=0.15)
bpy.ops.object.editmode_toggle()
bpy.ops.object.shade_smooth()
bpy.data.objects['StartCube'].select_set(state = True)

bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, location=(0, 0, 0))
bpy.data.objects['Plane'].scale = (2,2,1)
bpy.data.objects['Plane'].rotation_euler[0] = np.pi/2

bpy.data.objects["Camera"].location = (0,3,0)
bpy.data.objects["Camera"].rotation_euler = (-np.pi/2,np.pi,0)

bpy.data.objects['Light'].data.use_nodes = True
bpy.data.objects['Light'].data.node_tree.nodes['Emission'].inputs[1].default_value = 0.5
bpy.data.objects['Light'].data.type = 'AREA'
bpy.data.objects['Light'].location = (0,0,4.5)
bpy.data.objects["Light"].rotation_euler = (0,0,0)
bpy.data.objects['Light'].scale = (50,50,50)

###############################################################################
# Plotting functions
###############################################################################

#Wrapper function to scale the data. 
def ScaleData(data,scale):
    scaled = (data - data.min())/(data.max() - data.min())
    return scale*scaled/2

def AddRadarPlotElements(data,scale):
    """
    Parameters
    ----------
    data : array-like
        data for the plot
    scale : float
        Controls the size of the geometries.

    Returns
    -------
    objectNames : array
        Contains the names of the different geometries for the plot.
    """

    data = ScaleData(data,scale)
    num_vars = len(data)
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    scales=[tuple([0.005,0.005,val]) for val in data]   
    locs = [(val*np.sin(sal),0.5,val*np.cos(sal)) for val,sal in zip(data,theta)]

    objNames = AddElementsFromLocations(locs,scales,"Cube","bars")

    for nme,ang,loc in zip(objNames,theta,locs):

        bpy.data.objects[nme].rotation_euler = (np.random.random()/7,ang,np.random.random()/7)

    return objNames

###############################################################################
# Visualization 
###############################################################################

data = np.random.random(500)

GeomNames = AddRadarPlotElements(data,0.75)

colors = TwoPointLinearColorMap((0.6,0.2,0),(0.8,1,0.71),len(data))

[AddGoldMaterial(nme,colr) for nme,colr in zip(GeomNames,colors)]

PlotLabel = 'bar radar plot example'
InsertAndChangeText((0,0.25,-0.85),PlotLabel,'PlotLabel',0.05,(np.pi/2,0,np.pi))
AddTypeAMaterial('PlotLabel',(0,0,0))
