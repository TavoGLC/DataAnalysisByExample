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
import bmesh
import colorsys
import numpy as np

###############################################################################
# Data
###############################################################################

def MakeMaldebrontFractal(gridSize,maxIter):
    '''
    Parameters
    ----------
    gridSize : int
        number of divisions on the grid.
    maxIter : int
        max number of iterations taken to determine if a given point belogs or
        not to the Maldebront set.

    Returns
    -------
    Ns : 2D array
        relative colouring of the fractal.

    '''

    M=np.zeros((gridSize,gridSize))
    Ns=np.zeros((gridSize,gridSize))
    xvals=np.linspace(0.25,.5,gridSize)
    yvals=np.linspace(0.25,.5,gridSize)

    for u,x in enumerate(xvals):
        for v,y in enumerate(yvals):
            z=0
            c=complex(x,y)
            for i in range(maxIter):
                z=z*z+c
                if abs(z)>2:
                    M[u,v]=1
                    Ns[u,v]=i+1-np.log(np.log2(abs(z)))
                    break
                
    return Ns

def MakeJuliaSet(gridSize,maxIter):
    '''
    Parameters
    ----------
    gridSize : int
        number of divisions on the grid.
    maxIter : int
        max number of iterations taken to determine if a given point belogs or
        not to the Julia set.

    Returns
    -------
    Ns : 2D array
        relative colouring of the fractal.

    '''
        
    c=complex(-0.1,0.65)    
    Julia=np.zeros((gridSize,gridSize))
    Shades=np.zeros((gridSize,gridSize))    
    xvals=np.linspace(-1.5,1.5,gridSize)
    yvals=np.linspace(-1.5,1.5,gridSize)
    
    for k,x in enumerate(xvals):
        for j,y in enumerate(yvals):
            z=complex(x,y)
            for i in range(maxIter):
                z=z**2+c
                if abs(z)>(10):
                    Julia[k,j]=1
                    break
            shade=1-np.sqrt(i/maxIter)
            Shades[k,j]=shade
            
    return Shades

#Wrapper function for MinMax normalization of the data
def MinMaxNormalization(DisplacementData):
    minVal=DisplacementData.min()
    rangeVals=DisplacementData.max()-minVal
    disp=(DisplacementData-minVal)/(rangeVals)
    return disp.flatten()

###############################################################################
# Moving the camera
###############################################################################

def EvenSpacedColorMap(N):
    '''
    Parameters
    ----------
    N : int
        Number of evenly spaced colors to be created.

    Returns
    -------
    r : float
        r channel value.
    g : float
        g channel value.
    b : float
        b channel value.

    '''
    HSV_tuples=[(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    rgbList=list(map(lambda x: colorsys.hsv_to_rgb(*x),HSV_tuples))
    r=[val[0] for val in rgbList]
    g=[val[1] for val in rgbList]                  
    b=[val[2] for val in rgbList]
    
    return r,g,b

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
# Moving the camera
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

def AddTypeBMaterial(GeometryName,RGBData):
    """
    Adds a glowing material
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

    bsdf=nodes.new("ShaderNodeBsdfPrincipled")
    currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[0])

    emission=nodes.new("ShaderNodeEmission")
    emission.inputs[1].default_value=0.75
    currentMaterial.node_tree.links.new(rgb.outputs[0],emission.inputs[0])

    mix=nodes.new("ShaderNodeMixShader")
    currentMaterial.node_tree.links.new(bsdf.outputs[0],mix.inputs[1])
    currentMaterial.node_tree.links.new(emission.outputs[0],mix.inputs[2])

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(mix.outputs[0],materialOutput.inputs[0])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

def AddColorMap(GeometryName,scale,ColorMap,glow=False):
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

        if glow:
            AddTypeBMaterial(GeometryName,[r[k],g[k],b[k]])
        else:
            AddTypeAMaterial(GeometryName,[r[k],g[k],b[k]])

    for k in range(len(bpy.data.objects[GeometryName].data.polygons.items())):
        
        value=bpy.data.objects[GeometryName].data.polygons[k].center[2]
        currentIndex=ValueToMaterialIndex(value,GridValues)
        bpy.data.objects[GeometryName].data.polygons[k].material_index=currentIndex

###############################################################################
# Moving the camera
###############################################################################

def Add2DSurface(location,subdivisions,Name):
    '''
    Parameters
    ----------
    location : tuple
        Location to add the surface.
    subdivisions : int
        Number of subdivisions in the surface.
    Name : str
        Name of the geometry.

    Returns
    -------
    None.

    '''

    bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdivisions,y_subdivisions=subdivisions,radius=1,location=location)
    bpy.data.objects["Grid"].data.name=Name
    bpy.data.objects["Grid"].name=Name


def Modify2DSurface(DisplacementData,Name,scale=1):
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

    disp=MinMaxNormalization(DisplacementData)
    bm=bmesh.new()
    bm.from_mesh(bpy.data.objects[Name].data)
    matrixWorld=bpy.data.objects[Name].matrix_world

    Faces = bm.faces
    Vertexs=[matrixWorld*v.co for v in bm.verts]

    for face in Faces:
        normal=face.normal
        faceVerts=face.verts
        for verts in faceVerts:
            i=verts.index
            d=normal*disp[i]*scale
            verts.co=Vertexs[i]+d

    bm.to_mesh(bpy.data.objects[Name].data)

###############################################################################
# Removing the original cube
###############################################################################

bpy.data.objects["Cube"].select=True
bpy.ops.object.delete()

###############################################################################
# Moving the camera
###############################################################################

bpy.data.objects["Camera"].location=(0,0,2.25)
bpy.data.objects["Camera"].rotation_euler=(0,0,0)
bpy.context.scene.render.resolution_x = 1080

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.world.horizon_color = (1,1,1)

###############################################################################
# Moving the camera
###############################################################################

MaldebrontGridSize=1000
PlotScale=0.25
MaxIter=200
cMapGridSize=200

#cmap=EvenSpacedColorMap(cMapGridSize)
cmap=TwoPointLinearColorMap((1,1,1),(1,0,0),cMapGridSize)
z=MakeJuliaSet(MaldebrontGridSize,MaxIter)

Add2DSurface((0,0,0),MaldebrontGridSize,'test')
Modify2DSurface(z,'test',PlotScale)
AddColorMap("test",PlotScale,cmap,glow=False)
