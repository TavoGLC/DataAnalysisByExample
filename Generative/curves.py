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
import numpy as np

###############################################################################
# Coordinates utility functions
###############################################################################

def ListScaling(Coords):
    """
    Min-Max normaliztion for a list of values 
    Parameters
    ----------
    Coords : list,array-like
        List of values.

    Returns
    -------
    norms : list
        Min-Max normalization of Coords.
    """

    minVec,maxVec=min(Coords),max(Coords)
    vecRange=maxVec-minVec
    if vecRange==0:
        vecRange=1
    norms=[(val-minVec)/vecRange for val in Coords]

    return norms

def CoordsScaling(Coords,extend=2,center=(0,0,0)):
    """
    Normalization of a 2d array 
    Parameters
    ----------
    Coords : list,array-like
        Coordinates to be normalized.
    extend : float, optional
        Controls the extend in the normalization. The default is 2.
    center : tuple,list, optional
        Controls the center of the normalization. The default is (0,0,0).

    Returns
    -------
    scaled : list
        Scaled coordinates.

    """

    Xs=ListScaling([val[0] for val in Coords])
    Ys=ListScaling([val[1] for val in Coords])
    Zs=ListScaling([val[2] for val in Coords])
    Ws=[val[3] for val in Coords]

    scaled=[(extend*x+center[0],extend*y+center[1],extend*z+center[2],w) for x,y,z,w in zip(Xs,Ys,Zs,Ws)]

    return scaled

###############################################################################
# Coordinates generation
###############################################################################

def CoordinatesPoints(xFunction,yFunction,limit):
    """
    Calculates the raw values for the patterns
    Parameters
    ----------
    xFunction : function
        x-axis evaluating function.
    yFunction : function
        x-axis evaluating function.
    limit : float
        limit of the meshgrid.

    Returns
    -------
    Coords : list
        Pattern coordinates.

    """

    points=np.linspace(-limit,limit,num=100)
    xx, yy = np.meshgrid(points, points, sparse=True)
    z1 = xFunction(xx,yy)
    z2 = yFunction(xx,yy)

    xData=z1.ravel()
    yData=z2.ravel()
    zs=[0 for _ in range(z1.size)]
    ws=[1 for _ in range(z1.size)]

    Coords=[(x,y,z,w) for x,y,z,w in zip(xData,yData,zs,ws)]

    return Coords

###############################################################################
# Wrapper functions
###############################################################################

def Wrapper01(x,y):
    return x-np.cos(x/y)

def Wrapper02(x,y):
    return y+np.sin(y*x)

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
    scene.objects.link(curvob)
    scene.objects.active=curvob

    line=curv.splines.new("NURBS")

    line.points.add(len(Coords)-1)

    for index,point in enumerate(Coords):
        line.points[index].co=point

    curv.dimensions="3D"
    curv.use_path=True
    curv.bevel_object=bpy.data.objects["NurbsCircle"]
    bpy.data.curves[objName].use_fill_deform=True
    line.use_endpoint_u=True

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

#Wrapper function to add a black type A material
def AddTypeAMaterialRed(GeometryName):
    return AddTypeAMaterial(GeometryName,(1,0,0))

#Wrapper function to add a blue type B material
def AddTypeAMaterialBlue(GeometryName):
    return AddTypeAMaterial(GeometryName,(0,0,1))


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

    r,g,b,a=RGBData
    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes
    
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (r,g,b,a)

    bsdf=nodes.new("ShaderNodeBsdfPrincipled")
    currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[0])

    emission=nodes.new("ShaderNodeEmission")
    emission.inputs[1].default_value=1
    currentMaterial.node_tree.links.new(rgb.outputs[0],emission.inputs[0])

    mix=nodes.new("ShaderNodeMixShader")
    currentMaterial.node_tree.links.new(bsdf.outputs[0],mix.inputs[1])
    currentMaterial.node_tree.links.new(emission.outputs[0],mix.inputs[2])

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(mix.outputs[0],materialOutput.inputs[0])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)


#Wrapper function to add a blue type B material
def AddTypeBMaterialRed(GeometryName):
    return AddTypeBMaterial(GeometryName,(1,0,0,1))

###############################################################################
# Removing the original cube
###############################################################################

bpy.data.objects["Cube"].select=True
bpy.ops.object.delete()

###############################################################################
# Adding extruding element
###############################################################################

localScale=0.001
bpy.ops.curve.primitive_nurbs_circle_add(radius=1, view_align=False, enter_editmode=False, location=(0, 0, 0))
bpy.data.objects['NurbsCircle'].scale=(localScale,localScale,localScale)

###############################################################################
# Moving the camera
###############################################################################

bpy.data.objects["Camera"].location=(0,0,5)
bpy.data.objects["Camera"].rotation_euler=(0,0,0)

###############################################################################
# Changing the horizon 
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.world.horizon_color = (1,1,1)

###############################################################################
# Adding the curves 
###############################################################################

glow=False

name1="StartCurveA"
name2="StartCurveB"
pps=CoordinatesPoints(Wrapper01,Wrapper02,10)
points=CoordsScaling(pps,center=(-0.95,-1,0))
points2=CoordsScaling(pps,center=(-1.05,-1,0))
AddCurveFromCoords(points,objName=name1)
AddCurveFromCoords(points2,objName=name2)

if glow:
    
    bpy.data.objects["NurbsCircle"].select=False
    bpy.data.objects["Lamp"].select=True
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_plane_add(location=(0,0,-0.0005))
    bpy.data.objects["Plane"].scale=(4,4,4)
    bpy.context.scene.world.horizon_color = (0,0,0)
    AddTypeBMaterialRed(name1)

else:
    AddTypeAMaterialBlue(name1)
    AddTypeAMaterialRed(name2)
