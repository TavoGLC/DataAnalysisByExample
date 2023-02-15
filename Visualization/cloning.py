#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
tCopyright (c) 2022 Octavio Gonzalez-Lugo 
o use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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

import bpy
import bmesh
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from mpl_toolkits.basemap import Basemap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

###############################################################################
# Materials
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

    disp = DisplacementData
    bm=bmesh.new()
    bm.from_mesh(bpy.data.objects[GeometryName].data)
    matrixWorld=bpy.data.objects[GeometryName].matrix_world

    Faces = bm.faces
    Vertexs=[matrixWorld@v.co for v in bm.verts]

    dists=[]

    for face in Faces:
        normal=face.normal
        faceVerts=face.verts
        for verts in faceVerts:
            i=verts.index
            d=normal*disp[i]*scale
            verts.co=Vertexs[i]+d

    bm.to_mesh(bpy.data.objects[GeometryName].data)

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

def AddColorMap(GeometryName,scale,ColorMap,MaterialFunction):
    '''
    
    Parameters
    ----------
    GeometryName : str
        Name of the geometry to add a material.
    scale : float
        Scale of the geometry, used to determine the kind of material to be used.
    ColorMap : list,array like
        RGB tuples of the elements in the colormap.
    glow : bool, optional
        Adds an emmiting material when set to True. The default is False.
    Returns
    -------
    None.
    '''

    r,g,b=ColorMap
    GridSize=len(r)
    GridValues=np.linspace(0,scale,GridSize+1)

    for k in range(GridSize):

        MaterialFunction(GeometryName,[r[k],g[k],b[k]])

    for k in range(len(bpy.data.objects[GeometryName].data.polygons.items())):
        
        value=bpy.data.objects[GeometryName].data.polygons[k].center[2]
        currentIndex=ValueToMaterialIndex(value,GridValues)
        bpy.data.objects[GeometryName].data.polygons[k].material_index=currentIndex

###############################################################################
# Materials
###############################################################################

def AddGlassTypeMaterial(GeometryName,RGBData):
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

    r,g,b=RGBData

    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes

    ligthPath=nodes.new('ShaderNodeLightPath')
    divide=nodes.new('ShaderNodeMath')
    divide.operation='DIVIDE'

    currentMaterial.node_tree.links.new(ligthPath.outputs[7],divide.inputs[0])
    currentMaterial.node_tree.links.new(ligthPath.outputs[8],divide.inputs[1])

    ramp=nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position=0.928
    ramp.color_ramp.interpolation='EASE'
    currentMaterial.node_tree.links.new(divide.outputs[0],ramp.inputs[0])

    mix=nodes.new('ShaderNodeMixRGB')
    mix.inputs[1].default_value=(1,1,1,1)
    currentMaterial.node_tree.links.new(ramp.outputs[0],mix.inputs[0])

    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (r,g,b,1)
    currentMaterial.node_tree.links.new(rgb.outputs[0],mix.inputs[1])

    glass=nodes.new('ShaderNodeBsdfGlass')
    glass.inputs[2].default_value=2
    currentMaterial.node_tree.links.new(mix.outputs[0],glass.inputs[0])
    
    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(glass.outputs[0],materialOutput.inputs[0])

    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

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

###############################################################################
# Render Settings
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 3
bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 1200
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 250

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
bpy.data.objects['Plane'].scale = (9,9,1)
bpy.data.objects['Plane'].data.name = 'StartPlane'
bpy.data.objects['Plane'].name = 'StartPlane'

bpy.data.objects["Camera"].location = (0,0,3)
bpy.data.objects["Camera"].rotation_euler = (0,0,0)

bpy.data.objects['Light'].data.use_nodes = True
bpy.data.objects['Light'].data.node_tree.nodes['Emission'].inputs[1].default_value = 0.25
bpy.data.objects['Light'].data.type = 'AREA'
bpy.data.objects['Light'].location = (0.5,0.5,4.5)
bpy.data.objects["Light"].rotation_euler = (0 ,0,0)
bpy.data.objects['Light'].scale = (1,1,1)

###############################################################################
# Scene settings
###############################################################################

import matplotlib
matplotlib.use('Agg')

import seaborn as sns
sns.set_theme(style="ticks")

rs = np.random.RandomState(11)
x = rs.gamma(2, size=1000)
y = -.5 * x + rs.normal(size=1000)

fg = sns.jointplot(x=x, y=y, kind="hex", color="#4CB391")

canvas0 = FigureCanvas(fg.figure)
s, (width, height) = canvas0.print_to_buffer()
X0 = Image.frombytes("RGBA", (width, height), s)
image_array = np.asarray(X0)/255
image_array = image_array.reshape(-1,4)
flat_image_array = 1-image_array.mean(axis=1)

###############################################################################
# Scene settings
###############################################################################

PlotScale = 0.2
cMapGridSize = 512

bpy.ops.mesh.primitive_grid_add(x_subdivisions=width-1,y_subdivisions=height-1,location=(0,0,0))
bpy.data.objects["Grid"].data.name='plot'
bpy.data.objects["Grid"].name='plot'

Modify2DSurface(flat_image_array,'plot',PlotScale)

cmap=TwoPointLinearColorMap((1,1,1),(1,0,0),cMapGridSize)
AddColorMap('plot',0.2,cmap,AddGlassTypeMaterial)

bpy.data.objects["plot"].rotation_euler = (np.pi ,0,0)
