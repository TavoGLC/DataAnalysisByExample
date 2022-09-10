
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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

###############################################################################
# Helper functions
###############################################################################

def JoinObjectsByName(ObjectNames):
    """
    Parameters
    ----------
    ObjectNames : list
        list with the geometry names.
    Returns
    -------
    string
        name of the final object.
    """

    scene=bpy.context.scene
    totalObs=[bpy.data.objects[val] for val in ObjectNames]
    ctx=bpy.context.copy()
    ctx["active_object"]=totalObs[0]
    ctx["selected_objects"] = ctx["selected_editable_objects"] =totalObs
    bpy.ops.object.join(ctx)

    return ObjectNames[0]

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

bpy.data.objects["Camera"].location = (0.5,0.5,1.5)
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

data = pd.read_csv('/media/tavo/storage/data/temp/fulldata.csv')
data = data[data['lengthofday'] != 0]

qryData = data.groupby('qry').mean()
qryData = qryData[qryData['cases']<1500]
MaxSize=0.015
MinSize=0.0025

lats = np.array(qryData['lat'])
long = np.array(qryData['long'])
cses = np.array(qryData['cases'])

feature = np.array(qryData['lat'])

lats = (lats-lats.min())/(lats.max()-lats.min())
long = (long-long.min())/(long.max()-long.min())

cses = (cses-cses.min())/(cses.max()-cses.min())
cses = MaxSize*cses + MinSize

feature = (feature-feature.min())/(feature.max()-feature.min())

blockSize = 100
for k in range(0,len(lats),blockSize):

    localLats = lats[k:k+blockSize]
    locaLong = long[k:k+blockSize]
    localCases = cses[k:k+blockSize]
    localFeature = feature[k:k+blockSize]

    localColors = []

    for j,block in enumerate(zip(localLats,locaLong,localCases,localFeature)):
        val,sal,xal,yal = block
        bpy.ops.mesh.primitive_uv_sphere_add(enter_editmode=False, align='WORLD', location=(sal, val, np.random.random()/10), scale=(xal, xal, xal))
        bpy.data.objects["Sphere"].data.name='Sphere'+str(j)
        bpy.data.objects["Sphere"].name='Sphere'+str(j)
        localColors.append(plt.cm.viridis(yal)[0:3])

    [AddTypeAMaterial('Sphere'+str(i),localColors[i]) for i in range(len(localLats))]
    innerName = JoinObjectsByName(['Sphere'+str(i) for i in range(len(localLats))])

    bpy.data.objects[innerName].data.name='Block'+str(k//blockSize)
    bpy.data.objects[innerName].name='Block'+str(k//blockSize)

finalName = JoinObjectsByName(['Block'+str(i) for i in range(1+len(lats)//blockSize)])
