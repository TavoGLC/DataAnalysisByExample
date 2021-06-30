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
import json
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
    scene.objects.link(curvob)
    scene.objects.active=curvob

    line=curv.splines.new("NURBS")

    line.points.add(len(Coords))

    for index,point in enumerate(Coords):
        location=[point[0],point[1],0,1]
        line.points[index].co=location
    
    bpy.data.objects[objName].data.splines[0].use_cyclic_u=True
    bpy.data.objects[objName].data.extrude=0.01

###############################################################################
# Material functions
###############################################################################

def TwoPointLinearColorMap(startPoint,endPoint,points):
    """
    Linear interpolation between two points in the rgb color space
    ----------
    startPoint : array-like
        array of three elements, start point of the interpolation.
    endPoint : array-like
        array of three elements, end point of the interpolation.
    points : int
        Number of interpolation points between the start and end point 

    Returns
    -------
    None.
    """

    
    rs,gs,bs=startPoint
    re,ge,be=endPoint
    r=np.linspace(rs,re,num=points)
    g=np.linspace(gs,ge,num=points)
    b=np.linspace(bs,be,num=points)
    
    return np.vstack((r,g,b,np.ones(points))).T

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
    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value =tuple(RGBData)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs[4].default_value = 0.5
    bsdf.inputs[7].default_value = 1
    bsdf.inputs[15].default_value = 1

    currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[0])
    currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[3]
    )
    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(bsdf.outputs[0],materialOutput.inputs[0])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

###############################################################################
# Geo data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/"
FileDir=GlobalDirectory+"GeoJSONEstados"

with open(FileDir) as f:
    GeoData=json.load(f)

nItems=GeoData["totalFeatures"]
GeoContainer=[]
names=[]

for k in range(nItems):
    
    bufferData=GeoData["features"][k]["geometry"]["coordinates"][0][0]
    names.append(GeoData["features"][k]["properties"]["nom_ent"])
    GeoContainer.append(bufferData)
    bufferData=np.array(bufferData)

mins=np.min([np.array(val).min(axis=0) for val in GeoContainer],axis=0)
maxs=np.max([np.array(val).max(axis=0) for val in GeoContainer],axis=0)

GeoNormalization=[]

for val in GeoContainer:
    bufferData=np.array(val)
    bufferData=(bufferData-mins)/(maxs-mins)
    GeoNormalization.append(bufferData)

means=np.mean([val.mean(axis=0) for val in GeoNormalization],axis=0)

GeoAsArrays=[]
for val in GeoNormalization:
    bufferData=val-means
    GeoAsArrays.append(bufferData)

###############################################################################
# Settings for the render
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 2
bpy.context.scene.world.horizon_color = (0.1,0.1,0.1)
bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 1200
bpy.context.scene.render.tile_x = 384
bpy.context.scene.render.tile_y = 256
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 750

###############################################################################
# Settings for the scene
###############################################################################

bpy.data.objects["Camera"].location=(0,0.1,1.5)
bpy.data.objects["Camera"].rotation_euler=(0,0,0)

bpy.data.objects['Lamp'].data.use_nodes=True
bpy.data.objects['Lamp'].data.node_tree.nodes['Emission'].inputs[1].default_value=100
bpy.data.objects['Lamp'].data.type='AREA'
bpy.data.objects['Lamp'].location=(0,1.6,0)
bpy.data.objects["Lamp"].rotation_euler=(-np.pi/2,0,0)
bpy.data.objects['Lamp'].scale=(7,7,7)

###############################################################################
# Auxiliary geometry for the scene 
###############################################################################

bpy.data.objects['Cube'].data.name='StartCube'
bpy.data.objects['Cube'].name='StartCube'
bpy.data.objects['StartCube'].select=True
bpy.data.objects['StartCube'].scale=(2,2,2)
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.bevel(offset=0.15,vertex_only=False)
bpy.ops.object.editmode_toggle()
bpy.ops.object.shade_smooth()
bpy.data.objects['StartCube'].select=False

bpy.ops.mesh.primitive_plane_add(radius=1, view_align=False, enter_editmode=False, location=(0, 0, 0))
bpy.data.objects['Plane'].scale=(2,1,1)

###############################################################################
# Adding geographical data
###############################################################################

for index,val in enumerate(GeoAsArrays):
    data=val[np.arange(0,len(val),13)]
    AddCurveFromCoords(data,objName=names[index])

for val in names:
    bpy.context.scene.objects.active=bpy.data.objects[val]
    bpy.data.objects[val].select=True
    bpy.ops.object.convert(target='MESH')
    bpy.data.objects[val].select=False

###############################################################################
# Adding materials to each geographical object
###############################################################################

colors=TwoPointLinearColorMap((0.5,0.5,0.5),(0.75,0.75,0.75),len(names))

for col,val in zip(colors,names):
    AddTypeAMaterial(val,col)

###############################################################################
# Auxiliary geometry for the particle systems 
###############################################################################

for k,val in enumerate(names):
    
    helperGeometrynme='zParticle'+val
    particleSystemnme="Particle"+val
    bpy.ops.mesh.primitive_cube_add(location=(4, 4, k))
    bpy.data.objects['Cube'].data.name=helperGeometrynme
    bpy.data.objects['Cube'].name=helperGeometrynme

    bpy.context.scene.objects.active=bpy.data.objects[val]
    bpy.ops.object.particle_system_add()
    bpy.data.particles["ParticleSettings"].name=particleSystemnme
    bpy.data.particles[particleSystemnme].physics_type = 'NO'
    bpy.data.particles[particleSystemnme].emit_from = 'VOLUME'
    bpy.data.particles[particleSystemnme].distribution = 'RAND'
    bpy.data.particles[particleSystemnme].lifetime = 200
    bpy.data.particles[particleSystemnme].frame_end = 50
    bpy.data.particles[particleSystemnme].count = np.random.randint(50,1000)
    bpy.data.particles[particleSystemnme].render_type = 'OBJECT'
    bpy.data.particles[particleSystemnme].dupli_object = bpy.data.objects[helperGeometrynme]
    bpy.data.particles[particleSystemnme].particle_size = 0.0025


for val in names:
    helperGeometrynme='zParticle'+val
    col=list(np.random.random(3))+[1]
    AddTypeAMaterial(helperGeometrynme,col)
