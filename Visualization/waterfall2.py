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
# Use
###############################################################################
"""
The following code is intended to be used at the start of the program 
before making any changes to the scene. 
"""

###############################################################################
# Loading packages 
###############################################################################

import bpy
import bmesh
import numpy as np

###############################################################################
# Grid functions
###############################################################################

def Add2DSurface(Data,GeometryName='MyPlane'):
    '''
    Parameters
    ----------
    Data : array-like
        Data for geometry modification.
    Name : str
        Name of the geometry to modify.
    scale : float, optional
        Max value of geometry deformation. The default is 1.

    Returns
    -------
    None.

    '''

    data = Data[:,0:3]
    ndivisions = int(len(Data)/2)-1
    bpy.ops.mesh.primitive_grid_add(size=2, x_subdivisions = ndivisions, y_subdivisions = 1,enter_editmode=False, location=(0, 0, 0))
    bpy.data.objects['Grid'].data.name = GeometryName
    bpy.data.objects['Grid'].name = GeometryName

    bm=bmesh.new()
    bm.from_mesh(bpy.data.objects[GeometryName].data)

    for j,vrts in enumerate(bm.verts):
        vrts.co = tuple(data[j])
    
    bm.to_mesh(bpy.data.objects[GeometryName].data)

###############################################################################
# Material functions 
###############################################################################

def AddDispersionMaterial(GeometryName,RGBData):
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
    onlyR = tuple([r,0,0,1])
    onlyG = tuple([0,g,0,1])
    onlyB = tuple([0,0,b,1])


    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes

    math01 = nodes.new("ShaderNodeMath")
    math01.operation = "POWER"

    glassBSDF01 = nodes.new("ShaderNodeBsdfGlass")
    glassBSDF01.inputs[0].default_value = onlyR
    currentMaterial.node_tree.links.new(math01.outputs[0],glassBSDF01.inputs[1])

    glassBSDF02 = nodes.new("ShaderNodeBsdfGlass")
    glassBSDF02.inputs[0].default_value = onlyG
    currentMaterial.node_tree.links.new(math01.outputs[0],glassBSDF02.inputs[1])

    glassBSDF03 = nodes.new("ShaderNodeBsdfGlass")
    glassBSDF03.inputs[0].default_value = onlyB
    currentMaterial.node_tree.links.new(math01.outputs[0],glassBSDF03.inputs[1])

    math02 = nodes.new("ShaderNodeMath")
    currentMaterial.node_tree.links.new(math02.outputs[0],glassBSDF02.inputs[2])

    math03 = nodes.new("ShaderNodeMath")
    currentMaterial.node_tree.links.new(math02.outputs[0],math03.inputs[1])
    currentMaterial.node_tree.links.new(math03.outputs[0],glassBSDF01.inputs[2])

    addShader01 = nodes.new("ShaderNodeAddShader")
    currentMaterial.node_tree.links.new(glassBSDF01.outputs[0],addShader01.inputs[0])
    currentMaterial.node_tree.links.new(glassBSDF02.outputs[0],addShader01.inputs[1])

    addShader02 = nodes.new("ShaderNodeAddShader")
    currentMaterial.node_tree.links.new(addShader01.outputs[0],addShader02.inputs[0])
    currentMaterial.node_tree.links.new(glassBSDF03.outputs[0],addShader02.inputs[1])

    volumeAbs = nodes.new("ShaderNodeVolumeAbsorption")

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(addShader02.outputs[0],materialOutput.inputs[0])
    currentMaterial.node_tree.links.new(volumeAbs.outputs[0],materialOutput.inputs[1])

    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

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

    currentMaterial.node_tree.links.new(bump.outputs[0],diffuse.inputs[2])
    currentMaterial.node_tree.links.new(bump.outputs[0],glossy.inputs[1])
    currentMaterial.node_tree.links.new(bump.outputs[0],fresnel.inputs[1])

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(mix.outputs[0],materialOutput.inputs[0])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

###############################################################################
# Data handling 
###############################################################################

#Wrapper function to get the normalization coeficients
def GetNormalizationData(Data):
    return np.min(Data),np.max(Data)-np.min(Data)

def NormalizeData(Data,minVal,RangeVal):
    normData = 2*((np.array(Data)-minVal)/RangeVal) - 1
    return normData

#Wrapper function to change data to blender coordinates
def DataToCoordinates(Data,center):
    return np.array([[val+center[0],center[1],sal+center[2],1] for val,sal in zip(Data[0],Data[1])])

def ScaleData(Data,Extent,center):
    """"
    Parameters
    ----------
    Data : 2D array
        X and y data of the geometry
    Extent : array-like 
        controlls the x and y size of the plot 
    center : array
        Global location ofthe geometry

    Returns
    -------
    array
        global coordinates for the geometry
    """

    Xdata,Ydata = Data
    Xextent,Yextent = Extent

    xnorm0,xnorm1 = GetNormalizationData(Xdata)
    ynorm0,ynorm1 = GetNormalizationData(Ydata)

    Xnorm,Ynorm = Xextent*NormalizeData(Xdata,xnorm0,xnorm1),Yextent*NormalizeData(Ydata,ynorm0,ynorm1)
    Coordinates = DataToCoordinates([Xnorm,Ynorm],center)

    return Coordinates

def FormatCoordinates(CoordinatesData,MaxValue,center):
    """"
    Parameters
    ----------
    CoordinatesData : 2D array
        Global coordinates of the plot
    MaxValue : float
        Max value to scale and add the base of the plot
    center : array
        Global location ofthe geometry

    Returns
    -------
    array
        global coordinates for the geometry 
    """

    
    totalData = len(CoordinatesData)
    container = []
    for k in range(totalData):

        coordinate = CoordinatesData[k]
        base = [coordinate[0],coordinate[1],center[2]-MaxValue,1]
        container.append(base)

    container = np.array(container)
    newData = np.vstack((CoordinatesData,container))

    return newData

def AddWaterfallPlot(Xdata,Ydata,extent,center,PlotName='TestPlot'):
    """"
    Parameters
    ----------
    Xdata : array-like
        X data for the plot
    Ydata : array-like
        Y data for the plot
    extent : array-like 
        controlls the x and y size of the plot 
    center : array
        Global location ofthe geometry
    PlotName : string, optional
        Name of the geometry, default is TestPlot

    Returns
    -------
    array
        global coordinates for the geometry
    """


    divisions = np.linspace(-0.5,0.5,len(Ydata))
    dataContainer = []
    maxContainer = []

    for div,val in zip(divisions,Ydata):

        newCenter = [center[0],center[1]+div,center[2]]
        currentData = ScaleData([Xdata,val],extent,newCenter)
        currentMax = currentData[:,2].max()
        dataContainer.append(currentData)
        maxContainer.append(currentMax)
    
    globalMax = np.max(maxContainer)
    names = []

    for k,dta in enumerate(dataContainer):
        coords = FormatCoordinates(dta,globalMax,center)
        localName = PlotName + str(k)
        Add2DSurface(coords,GeometryName=localName)
        names.append(localName)
    
    return names

###############################################################################
# Render Settings
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 5
bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 1200
bpy.context.scene.render.tile_x = 128
bpy.context.scene.render.tile_y = 128
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 350

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

bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, location=(0, 0, -0.2))
bpy.data.objects['Plane'].scale = (2,2,1)
#bpy.data.objects['Plane'].rotation_euler[0] = np.pi/2
bpy.data.objects['Plane'].data.name = 'StartPlane'
bpy.data.objects['Plane'].name = 'StartPlane'

bpy.data.objects["Camera"].location = (-1.5,3,1)
bpy.data.objects["Camera"].rotation_euler = (-1.91986,np.pi,0.488692)

bpy.data.objects['Light'].data.use_nodes = True
bpy.data.objects['Light'].data.node_tree.nodes['Emission'].inputs[1].default_value = 0.75
bpy.data.objects['Light'].data.type = 'AREA'
bpy.data.objects['Light'].location = (0,0,4.5)
bpy.data.objects["Light"].rotation_euler = (0,0,0)
bpy.data.objects['Light'].scale = (50,50,50)

###############################################################################
# Scene settings
###############################################################################

XData = np.linspace(0,3*np.pi,num=50)
mu = 5
YData = [np.random.random(50) + val for val in np.linspace(1,5,num=10)]

nmes = AddWaterfallPlot(XData,YData,[0.92,0.15],(0,0,0))

[AddDispersionMaterial(nm,[0.2,0.5,0.05*k]) for k,nm in enumerate(nmes)]
AddPolystyreneTypeMaterial("StartPlane",[0.1,0.1,0.1])