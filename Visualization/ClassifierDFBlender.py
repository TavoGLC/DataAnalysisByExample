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

def ValueToMaterialIndex(value,GridValues):
    '''
    Parameters
    ----------
    value : float
        Value to be found..
    GridValues : list
        Boundaries for the value to be indexed by.

    Returns
    -------
    responce : int
        Index of a given value.

    '''
    responce=0
    for k in range(len(GridValues)-1):
        if GridValues[k]<value and GridValues[k+1]>=value:
            responce=k
            break
    return responce 

###############################################################################
# Data
###############################################################################

def MinMaxNormalization(DisplacementData):
    '''
    Scales the displacement data between 0 and 1 
    ----------
    DisplacementData : array
        Data 
    
    Returns
    -------
    disp : array
        Normalized data
    '''
    
    minVal=DisplacementData.min()
    RangeVal=DisplacementData.max()-minVal
    disp=(DisplacementData-minVal)/RangeVal
    return disp

###############################################################################
# Scatter plot
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

def AddElementsByLocation(Locations,Extend,Kind,Name,Scale):
    '''
    Parameters
    ----------
    Locations : list,array-like
        Location of the objects.
    Extend : float
        Controlls how close the objects are between them.
    Kind : str
        Kind of geometry to be added.
    Name : str
        Name of the added geometry. 
    EqualScales : bool, optional
        Controlls if the scale of the geometry is equal on each spatial component.
        The default is True.

    Returns
    -------
    NewNames : list
        Names of the generated geometries.

    '''
    localLocations=Locations
    localLocations=localLocations*Extend
    nObjects=len(localLocations)
    NewNames=MakeObjectNames(Name+Kind,nObjects)

    try:
        ScaleArray=len(Locations)==len(Scale)
    except TypeError:
        ScaleArray=False
    
    for k in range(nObjects):

        if ScaleArray:
            AddGeometryByKind(Kind,NewNames[k],tuple(localLocations[k]),(Scale[k],Scale[k],Scale[k]))
        else:
            AddGeometryByKind(Kind,NewNames[k],tuple(localLocations[k]),(Scale,Scale,Scale))
    
    return NewNames

###############################################################################
# Density plot
###############################################################################

def Modify2DSurface(DisplacementData,GeometryName,scale=1):
    '''
    Parameters
    ----------
    DisplacementData : array-like
        Data for geometry modification.
    Name : str
        Name of the geometry to modify.
    scale : float, optional
        Max value of geometry deformation. The default is 1.

    Returns
    -------
    None.

    '''

    disp=MinMaxNormalization(DisplacementData).flatten()
    bm=bmesh.new()
    bm.from_mesh(bpy.data.objects[GeometryName].data)
    matrixWorld=bpy.data.objects[GeometryName].matrix_world

    Faces = bm.faces
    Vertexs=[matrixWorld*v.co for v in bm.verts]

    for face in Faces:
        normal=face.normal
        faceVerts=face.verts
        for verts in faceVerts:
            i=verts.index
            d=normal*disp[i]*scale
            verts.co=Vertexs[i]+d

    bm.to_mesh(bpy.data.objects[GeometryName].data)

###############################################################################
# Colormaps 
###############################################################################

def TwoPointLinearColorMap(startPoint,endPoint,points):
    '''
    Parameters
    ----------
    startPoint : list, tuple,
        Start point for the interpolatuion in the rgb color space.
    endPoint : list, tuple,
        Start point for the interpolatuion in the rgb color space.
    points : int
        Number of divisions between the start and end points.

    Returns
    -------
    r : float
        r channel value.
    g : float
        g channel value.
    b : float
        b channel value.

    '''
    
    rs,gs,bs=startPoint
    re,ge,be=endPoint
    r=np.linspace(rs,re,num=points)
    g=np.linspace(gs,ge,num=points)
    b=np.linspace(bs,be,num=points)
    
    return r,g,b

###############################################################################
# Node Materials
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
    text01.inputs[1].default_value=15
    text01.inputs[2].default_value=1.25

    rgb1 = nodes.new("ShaderNodeRGB")
    rgb1.outputs[0].default_value = (r,g,b,1)

    glossy=nodes.new('ShaderNodeBsdfGlossy')

    currentMaterial.node_tree.links.new(rgb1.outputs[0],glossy.inputs[0])
    currentMaterial.node_tree.links.new(text01.outputs[1],glossy.inputs[1])

    text02=nodes.new('ShaderNodeTexNoise')
    text02.inputs[1].default_value=14
    text02.inputs[2].default_value=5

    rgb2 = nodes.new("ShaderNodeRGB")
    rgb2.outputs[0].default_value = (r,1,b,1)

    glass=nodes.new('ShaderNodeBsdfGlass')

    currentMaterial.node_tree.links.new(rgb2.outputs[0],glass.inputs[0])
    currentMaterial.node_tree.links.new(text02.outputs[1],glass.inputs[1])

    mix=nodes.new('ShaderNodeMixShader')

    currentMaterial.node_tree.links.new(glossy.outputs[0],mix.inputs[1])
    currentMaterial.node_tree.links.new(glass.outputs[0],mix.inputs[2])

    text03=nodes.new('ShaderNodeTexNoise')
    text03.inputs[1].default_value=14
    text03.inputs[2].default_value=5

    mult=nodes.new('ShaderNodeMath')
    mult.operation='MULTIPLY'
    mult.inputs[1].default_value=1.2

    currentMaterial.node_tree.links.new(text03.outputs[1],mult.inputs[0])

    materialOutput=nodes.get("Material Output")

    currentMaterial.node_tree.links.new(mix.outputs[0],materialOutput.inputs[0])
    currentMaterial.node_tree.links.new(mult.outputs[0],materialOutput.inputs[2])

    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

def AddColorMap(GeometryName,MaterialFunction,ColorMap,Component):
    '''
    Parameters
    ----------
    GeometryName : str
        Name of the geometry to add a material.
    MaterialFunction : function
        Function that adds a material to a geometry, musdt take two arguments, 
        the geometry name and the rgb color 
    ColorMap : list,array like
        RGB tuples of the elements in the colormap.
    Component :int
        Controlls the component taken to apply the materials. 
    Returns
    -------
    None.

    '''

    r,g,b=ColorMap
    GridSize=len(r)
    GridValues=np.linspace(0,1,GridSize+1)

    for k in range(GridSize):
        MaterialFunction(GeometryName,[r[k],g[k],b[k]])

    Values=np.array([val.center[Component] for val in bpy.data.objects[GeometryName].data.polygons.values()])
    Values=MinMaxNormalization(Values)

    for k in range(len(bpy.data.objects[GeometryName].data.polygons.items())):
        
        value=Values[k]
        currentIndex=ValueToMaterialIndex(value,GridValues)
        bpy.data.objects[GeometryName].data.polygons[k].material_index=currentIndex


###############################################################################
# Settings for the render
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 5
bpy.context.scene.world.horizon_color = (1,1,1)
bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 800
bpy.context.scene.render.tile_x = 128
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

bpy.data.objects["Camera"].location=(0.125,-4,0.125)
bpy.data.objects["Camera"].rotation_euler=(np.pi/2,0,0)

bpy.data.objects['Lamp'].data.use_nodes=True
bpy.data.objects['Lamp'].data.node_tree.nodes['Emission'].inputs[1].default_value=500
bpy.data.objects['Lamp'].data.type='AREA'
bpy.data.objects['Lamp'].location=(0,0,4.5)
bpy.data.objects["Lamp"].rotation_euler=(0,0,0)
bpy.data.objects['Lamp'].scale=(50,50,50)

###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/media/tavoglc/storage/storage/LocalData/"
ScatterDataDir=GlobalDirectory+'PCAData.csv'

ScatterData=np.genfromtxt(ScatterDataDir,delimiter=',')

xData=ScatterData[:,0]
yData=ScatterData[:,1]
Data=np.array([[val,0,sal] for val,sal in zip(xData,yData)])

MaskDir=GlobalDirectory+'Mask.csv'
MaskData=np.genfromtxt(MaskDir,delimiter=',')
ClassData=MaskData[0,:]
QualityData=MaskData[1,:]

ClassA=[k for k in range(len(ClassData)) if ClassData[k]==0]
ClassB=[k for k in range(len(ClassData)) if ClassData[k]==1]

ClassAShapes=MinMaxNormalization(np.array([QualityData[val] for val in ClassA]))
ClassBShapes=MinMaxNormalization(np.array([QualityData[val] for val in ClassB]))

###############################################################################
# Scatter plot
###############################################################################

ClassANames=AddElementsByLocation(Data[ClassA,:],1,"Sphere",'ClassA',0.02)
ClassBNames=AddElementsByLocation(Data[ClassB,:],1,"Sphere",'ClassB',0.02)

for nme in ClassANames:
    AddWaterTypeMaterial(nme,[1,0,0])

for nme in ClassBNames:
    AddTypeAMaterial(nme,[0,0,1])

###############################################################################
# 
###############################################################################

MeshDataDir=GlobalDirectory+'nusvcmesh.csv'
MeshData=np.genfromtxt(MeshDataDir,delimiter=',')

densityGrid=MeshData.shape[0]

bpy.ops.mesh.primitive_grid_add(x_subdivisions=densityGrid,y_subdivisions=densityGrid,radius=1,location=(0,0.1,0))
bpy.data.objects["Grid"].data.name='DensityGrid'
bpy.data.objects["Grid"].name='DensityGrid'
bpy.data.objects['DensityGrid'].rotation_euler=(np.pi/2,0,0)

ColorMapSize=50

DivergentA=TwoPointLinearColorMap((1,0,0),(1,1,1),ColorMapSize)
DivergentB=TwoPointLinearColorMap((1,1,1),(0,0,1),ColorMapSize)
DivergentColorMap=tuple([np.hstack((val,sal)) for val,sal in zip(DivergentA,DivergentB)])

Modify2DSurface(MeshData,'DensityGrid',scale=0.1)
AddColorMap('DensityGrid',AddTypeAMaterial,DivergentColorMap,2)
