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
        elif k>99 and k<=999:
            NamesContainer.append(ObjectName+"."+str(k))

    return NamesContainer

###############################################################################
# Functions
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


def AddRandomElements(Locations,Extend,Kind,EqualScales=True):
    '''
    Parameters
    ----------
    Locations : list,array-like
        Location of the objects.
    Extend : float
        Controlls how close the objects are between them.
    Kind : str
        Kind of geometry to be added.
    EqualScales : bool, optional
        Controlls if the scale of the geometry is equal on each spatial component.
        The default is True.

    Returns
    -------
    NewNames : list
        Names of the generated geometries.

    '''

    Locations=(Locations-0.5)/0.5
    Locations=Locations*Extend
    nObjects=len(Locations)
    NewNames=MakeObjectNames("Random"+Kind,nObjects)

    for k in range(nObjects):
        if EqualScales:
            scle=np.random.random()/2
            LoopScale=(scle,scle,scle)
        else:
            LoopScale=tuple(np.random.random(3))

        AddGeometryByKind(Kind,NewNames[k],tuple(Locations[k]),LoopScale)
    
    return NewNames


###############################################################################
# Functions
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

def AddGlassTypeMaterial(GeometryName,RGBData):
    '''
    Glass shader
    Parameters
    ----------
    GeometryName : str
        Name of the geometry to add the material.
    RGBData : array-like
        Color specification for the shader.

    Returns
    -------
    None.

    '''

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

def AddDoubleTypeMaterial(GeometryName,RGBDataA,RGBDataB):
    '''
    Doble shader, adds a glow effect to a geometry. 
    Parameters
    ----------
    GeometryName : str
        Name of the geometry to add the material.
    RGBDataA : array-like
        Color specification for the shader.
    RGBDataB : array-like
        Color specification for the shader.
    
    Returns
    -------
    None.

    '''

    ra,ga,ba=RGBDataA
    rb,gb,bb=RGBDataB

    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes

    geometry=nodes.new('ShaderNodeNewGeometry')

    ramp=nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color=(rb,gb,bb,1)
    ramp.color_ramp.elements[0].position=0.250
    ramp.color_ramp.elements[1].color=(ra,ga,ba,1)
    ramp.color_ramp.elements[1].position=0.76
    currentMaterial.node_tree.links.new(geometry.outputs[2],ramp.inputs[0])

    emission=nodes.new('ShaderNodeEmission')
    emission.inputs[1].default_value=1
    currentMaterial.node_tree.links.new(ramp.outputs[0],emission.inputs[0])

    lightPath=nodes.new('ShaderNodeLightPath')
    mixShaderFinal=nodes.new('ShaderNodeMixShader')
    
    currentMaterial.node_tree.links.new(lightPath.outputs[0],mixShaderFinal.inputs[0])
    currentMaterial.node_tree.links.new(emission.outputs[0],mixShaderFinal.inputs[1])
    
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (ra,ga,ba,1)

    bsdf=nodes.new('ShaderNodeBsdfDiffuse')
    currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[0])

    bsdfGlossy=nodes.new('ShaderNodeBsdfGlossy')
    mixShaderFresnel=nodes.new('ShaderNodeMixShader')
    currentMaterial.node_tree.links.new(bsdf.outputs[0],mixShaderFresnel.inputs[1])
    currentMaterial.node_tree.links.new(bsdfGlossy.outputs[0],mixShaderFresnel.inputs[2])

    #FresnelGroup 
    bump=nodes.new('ShaderNodeBump')
    geometryFresnel=nodes.new('ShaderNodeNewGeometry')

    mixRGBFresnel=nodes.new('ShaderNodeMixRGB')
    currentMaterial.node_tree.links.new(bump.outputs[0],mixRGBFresnel.inputs[1])
    currentMaterial.node_tree.links.new(geometryFresnel.outputs[4],mixRGBFresnel.inputs[2])
    
    fresnel=nodes.new('ShaderNodeFresnel')
    currentMaterial.node_tree.links.new(mixRGBFresnel.outputs[0],fresnel.inputs[1])
    currentMaterial.node_tree.links.new(fresnel.outputs[0],mixShaderFresnel.inputs[0])
    currentMaterial.node_tree.links.new(mixShaderFresnel.outputs[0],mixShaderFinal.inputs[2])

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(mixShaderFinal.outputs[0],materialOutput.inputs[0])

    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

###############################################################################
# Settings for the scene
###############################################################################

bpy.data.objects['Cube'].data.name='StartCube'
bpy.data.objects['Cube'].name='StartCube'
bpy.data.objects['StartCube'].select=True
bpy.data.objects['StartCube'].scale=(6,6,6)

bpy.data.objects["Camera"].location=(4,-3.25,3)

bpy.data.objects['Lamp'].data.use_nodes=True
bpy.data.objects['Lamp'].data.node_tree.nodes['Emission'].inputs[1].default_value=500
bpy.data.objects['Lamp'].data.type='AREA'
bpy.data.objects['Lamp'].location=(0,0,3.5)
bpy.data.objects["Lamp"].rotation_euler=(0,0,0)
bpy.data.objects['Lamp'].scale=(100,100,100)

###############################################################################
# Settings for the render
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 5
bpy.context.scene.world.horizon_color = (1,1,1)
bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 1200
bpy.context.scene.render.tile_x = 384
bpy.context.scene.render.tile_y = 256
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 750

###############################################################################
# Data
###############################################################################

Data=np.random.random((100,3))

###############################################################################
# Adding the geometries
###############################################################################

NewNames=AddRandomElements(Data,1,"Cylinder",EqualScales=False)

Rs,Gs,Bs=TwoPointLinearColorMap((0.75,0.75,0),(0,1,1),len(NewNames))

for k,nme in enumerate(NewNames):
    AddGlassTypeMaterial(nme,[Rs[k],Gs[k],Bs[k]],[Bs[k],Bs[k],Bs[k]])