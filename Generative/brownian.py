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

###############################################################################
#                 Intended Use
###############################################################################

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
# Data
###############################################################################

def Brownian(X0,dt,steps):
    """
    Brownian motion 
    Parameters
    ----------
    X0 : float
        initial condition.
    dt : float
        time step size.
    steps : int
        numbr of steps in the simulation.

    Returns
    -------
    container : list
        one dimensional brownian motion evolution trough time.

    """
    
    container=[]
    container.append(X0)
    
    for _ in range(steps-1):
        X0=X0+np.sqrt(dt)*np.random.standard_normal()
        container.append(X0)
    
    return container

#Wrapper function for three dimensional brownian motion
def Brownian3D(X0,dt,steps):
    
    X1,X2,X3=X0
    br1=Brownian(X1,dt,steps)
    br2=Brownian(X2,dt,steps)
    br3=Brownian(X3,dt,steps)
    
    return np.array([br1,br2,br3]).T

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

bpy.data.objects["Camera"].location=(0.125,-3.5,0.125)
bpy.data.objects["Camera"].rotation_euler=(np.pi/2,0,0)

bpy.data.objects['Lamp'].data.use_nodes=True
bpy.data.objects['Lamp'].data.node_tree.nodes['Emission'].inputs[1].default_value=500
bpy.data.objects['Lamp'].data.type='AREA'
bpy.data.objects['Lamp'].location=(0,0,4.5)
bpy.data.objects["Lamp"].rotation_euler=(0,0,0)
bpy.data.objects['Lamp'].scale=(50,50,50)

###############################################################################
# Adjusting the data 
###############################################################################

steps=30
locations=Brownian3D([1,1.3,2],0.5,steps)
locations=MinMaxNormalization(locations)
locations=locations-locations.mean(axis=0)
locations=locations*3.0
weights=np.ones((steps,1))
data=np.hstack((locations,weights))
AddCurveFromCoords(data,objName='test')

###############################################################################
# Settings for the force field
###############################################################################

bpy.data.objects['test'].select=True
bpy.ops.object.forcefield_toggle()
bpy.context.object.field.shape = 'SURFACE'
bpy.context.object.field.strength = -5
bpy.context.object.field.flow = 2
bpy.data.objects['test'].select=False

###############################################################################
# Settings for the particle system
###############################################################################

bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1,size=0.05,location=(locations[1,0],locations[1,1],locations[1,2]))
bpy.ops.object.particle_system_add()
bpy.data.particles["ParticleSettings"].count = 10000
bpy.data.particles["ParticleSettings"].effector_weights.gravity = 0
bpy.data.particles["ParticleSettings"].lifetime = 200
bpy.data.objects['Icosphere'].select=False

bpy.ops.mesh.primitive_cube_add(radius=1, view_align=False, enter_editmode=False, location=(10, 10, 0))
bpy.data.objects['Cube'].data.name='ParticleCube'
bpy.data.objects['Cube'].name='ParticleCube'
bpy.data.objects['ParticleCube'].scale=(0.05,0.05,0.05)

bpy.data.particles["ParticleSettings"].render_type = 'OBJECT'
bpy.data.particles["ParticleSettings"].dupli_object = bpy.data.objects["ParticleCube"]
bpy.data.particles["ParticleSettings"].particle_size = 0.1


AddTypeAMaterial('ParticleCube',(0.3,0.4,0.7))
