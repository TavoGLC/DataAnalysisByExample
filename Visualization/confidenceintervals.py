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
# Curve functions
###############################################################################

def AddCurveFromCoords(Coords,objName="MyCurve"):
    """
    Adds a curve following the Coords
    Parameters
    ----------
    Coords : list,array-like
        Global coordinates for the curve object.
    objName : string, optional
        Name of the curve object. The default is "MyCurve".

    Returns
    -------
    None.

    """

    curv=bpy.data.curves.new(objName,"CURVE")
    curvob=bpy.data.objects.new(objName,curv)

    scene=bpy.context.scene
    scene.collection.objects.link(curvob)
    scene.collection.objects[objName].select_set(state = True)

    line=curv.splines.new("NURBS")

    line.points.add(len(Coords)-1)

    for index,point in enumerate(Coords):
        line.points[index].co=point

    curv.dimensions="3D"
    curv.use_path=True
    curv.bevel_object=bpy.data.objects[objName+'line']
    bpy.data.curves[objName].use_fill_deform=True
    line.use_endpoint_u=True

###############################################################################
# Grid functions
###############################################################################

def Modify2DSurface(Data,GeometryName='MyPlane'):
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
    ndivisions = int(len(Data)/2)
    bpy.ops.mesh.primitive_grid_add(size=2, x_subdivisions = ndivisions, y_subdivisions = 1,enter_editmode=False, location=(0, 0, 0))
    bpy.data.objects['Grid'].data.name = GeometryName
    bpy.data.objects['Grid'].name = GeometryName

    bm=bmesh.new()
    bm.from_mesh(bpy.data.objects[GeometryName].data)\

    for j,vrts in enumerate(bm.verts):
        vrts.co = tuple(data[j])

    bm.to_mesh(bpy.data.objects[GeometryName].data)

###############################################################################
# Material functions 
###############################################################################
def AddCeramicTypeMaterial(GeometryName,RGBData):
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

    #Nodes handler
    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes

    #Outer definitions
    textureCoordinate=nodes.new("ShaderNodeTexCoord")
    textureCoordinate.object = bpy.data.objects[GeometryName]

    noiseTexture=nodes.new("ShaderNodeTexNoise")
    currentMaterial.node_tree.links.new(textureCoordinate.outputs[0],noiseTexture.inputs[0])

    bump=nodes.new("ShaderNodeBump")
    currentMaterial.node_tree.links.new(noiseTexture.outputs[1],bump.inputs[2])

    #Color
    r,g,b=RGBData
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (r,g,b,1)

    fresnel01=nodes.new("ShaderNodeFresnel")
    currentMaterial.node_tree.links.new(bump.outputs[0],fresnel01.inputs[1])

    diffusebsdf=nodes.new("ShaderNodeBsdfDiffuse")
    currentMaterial.node_tree.links.new(rgb.outputs[0],diffusebsdf.inputs[0])

    glossybsdf01=nodes.new("ShaderNodeBsdfGlossy")
    currentMaterial.node_tree.links.new(bump.outputs[0],glossybsdf01.inputs[2])

    mix01=nodes.new("ShaderNodeMixShader")
    currentMaterial.node_tree.links.new(fresnel01.outputs[0],mix01.inputs[0])
    currentMaterial.node_tree.links.new(diffusebsdf.outputs[0],mix01.inputs[1])
    currentMaterial.node_tree.links.new(glossybsdf01.outputs[0],mix01.inputs[2])

    fresnel02=nodes.new("ShaderNodeFresnel")
    currentMaterial.node_tree.links.new(bump.outputs[0],fresnel02.inputs[1])
    glossybsdf02=nodes.new("ShaderNodeBsdfGlossy")
    currentMaterial.node_tree.links.new(bump.outputs[0],glossybsdf02.inputs[2])

    mix02=nodes.new("ShaderNodeMixShader")
    currentMaterial.node_tree.links.new(fresnel02.outputs[0],mix02.inputs[0])
    currentMaterial.node_tree.links.new(mix01.outputs[0],mix02.inputs[1])
    currentMaterial.node_tree.links.new(glossybsdf02.outputs[0],mix02.inputs[2])
    
    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(mix02.outputs[0],materialOutput.inputs[0])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

def AddWaterTypeMaterial(GeometryName,RGBData):
    """
    Water-like material
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

    text01=nodes.new('ShaderNodeTexNoise')
    text01.inputs[1].default_value=5
    text01.inputs[2].default_value=1.25

    rgb1 = nodes.new("ShaderNodeRGB")
    rgb1.outputs[0].default_value = (r,g,b,1)

    glossy=nodes.new('ShaderNodeBsdfGlossy')

    currentMaterial.node_tree.links.new(rgb1.outputs[0],glossy.inputs[0])
    currentMaterial.node_tree.links.new(text01.outputs[1],glossy.inputs[1])

    text02=nodes.new('ShaderNodeTexNoise')
    text02.inputs[1].default_value=15
    text02.inputs[2].default_value=3

    rgb2 = nodes.new("ShaderNodeRGB")
    rgb2.outputs[0].default_value = (r,1,b,1)

    glass=nodes.new('ShaderNodeBsdfGlass')

    currentMaterial.node_tree.links.new(rgb2.outputs[0],glass.inputs[0])
    currentMaterial.node_tree.links.new(text02.outputs[1],glass.inputs[1])

    mix=nodes.new('ShaderNodeMixShader')

    currentMaterial.node_tree.links.new(glossy.outputs[0],mix.inputs[1])
    currentMaterial.node_tree.links.new(glass.outputs[0],mix.inputs[2])

    text03=nodes.new('ShaderNodeTexNoise')
    text03.inputs[1].default_value=15
    text03.inputs[2].default_value=3

    mult=nodes.new('ShaderNodeMath')
    mult.operation='MULTIPLY'
    mult.inputs[1].default_value=1.2

    currentMaterial.node_tree.links.new(text03.outputs[1],mult.inputs[0])

    materialOutput=nodes.get("Material Output")

    currentMaterial.node_tree.links.new(mix.outputs[0],materialOutput.inputs[0])
    currentMaterial.node_tree.links.new(mult.outputs[0],materialOutput.inputs[2])

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
    """

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

def MakeIntervalsData(Data,Extent,intervals,center):
    """

    Parameters
    ----------
    Data : 2D array
        X and y data of the geometry
    Extent : array-like 
        controlls the x and y size of the plot 
    intervals : array
        confidence interval for each data point, assumes symmetry
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

    upperInterval = Ydata + intervals
    lowerInterval = Ydata - intervals

    upperScaled = Yextent*NormalizeData(upperInterval,ynorm0,ynorm1)
    lowerScaled = Yextent*NormalizeData(lowerInterval,ynorm0,ynorm1)
    Xnorm = Xextent*NormalizeData(Xdata,xnorm0,xnorm1)

    UpperCoords = DataToCoordinates([Xnorm,upperScaled],center)
    LowerCoords = DataToCoordinates([Xnorm,lowerScaled],center)

    fullCoordinates = np.vstack((UpperCoords,LowerCoords))

    return fullCoordinates

def AddLinePlot(Xdata,Ydata, IntervalData = None, Extent=[0.75,0.75], linewidth = 0.02 , location=(0,0,0), LineName='TestPlot'):
    """
    Parameters
    ----------
    Xdata : array
        X axis data for the plot 
    Ydata : array
        Y axis data for the plot 
    IntervalData : array, optional
        confidence interval for each data point, assumes symmetry, by default None
    Extent : list, optional
        controlls the x and y size of the plot , by default [0.75,0.75]
    linewidth : float, optional
        size of the line, by default 0.02
    location : tuple, optional
        location of the geometry in global coordinates, by default (0,0,0)
    LineName : str, optional
        name of the geometry object, by default 'TestPlot'
    """

    Coords = ScaleData([Xdata,Ydata],Extent,location)

    extrudeElementName = LineName+'line'
    bpy.ops.curve.primitive_nurbs_circle_add(radius=linewidth, enter_editmode=False, location=(10, 10, 10))
    bpy.data.objects['NurbsCircle'].scale=(linewidth,linewidth,linewidth)
    bpy.data.objects['NurbsCircle'].data.name = extrudeElementName
    bpy.data.objects['NurbsCircle'].name = extrudeElementName

    AddCurveFromCoords(Coords,objName=LineName)

    if type(IntervalData) != type(None):

        intLoc = tuple([location[0],location[1]-0.05,location[2]])
        intData = MakeIntervalsData([Xdata,Ydata],Extent,IntervalData,location)
        Modify2DSurface(intData)
        bpy.data.objects["MyPlane"].location = (0,-0.025,0)

###############################################################################
# Render Settings
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 3
bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 1200
bpy.context.scene.render.tile_x = 384
bpy.context.scene.render.tile_y = 256
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 500

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
bpy.data.objects['Plane'].data.name = 'StartPlane'
bpy.data.objects['Plane'].name = 'StartPlane'

bpy.data.objects["Camera"].location = (0,3,0)
bpy.data.objects["Camera"].rotation_euler = (-np.pi/2,np.pi,0)

bpy.data.objects['Light'].data.use_nodes = True
bpy.data.objects['Light'].data.node_tree.nodes['Emission'].inputs[1].default_value = 0.75
bpy.data.objects['Light'].data.type = 'AREA'
bpy.data.objects['Light'].location = (0,0,4.5)
bpy.data.objects["Light"].rotation_euler = (0,0,0)
bpy.data.objects['Light'].scale = (50,50,50)

###############################################################################
# Scene settings
###############################################################################

nData = 100
Xdata = np.linspace(0,2*np.pi,num=nData)
Ydata = np.exp(-0.5*Xdata)*np.sin(Xdata)-(0.25*np.random.random(nData))

AddLinePlot(Xdata,Ydata,Extent=[0.75,0.25],linewidth=0.075,location=(0,0.1,0),IntervalData=0.5*np.random.random(nData))
AddCeramicTypeMaterial('TestPlot',(0.975,0.01,0.025))
AddWaterTypeMaterial('MyPlane',(0.92,0.08,0.92))
