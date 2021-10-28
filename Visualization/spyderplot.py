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

def ModifyCylinder(Data,GeometryName='MyPlane'):
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

    data = Data
    ndivisions = int(len(Data))/2
    bpy.ops.mesh.primitive_cylinder_add(vertices=ndivisions)
    bpy.data.objects['Cylinder'].data.name = GeometryName
    bpy.data.objects['Cylinder'].name = GeometryName

    bm=bmesh.new()
    bm.from_mesh(bpy.data.objects[GeometryName].data)

    for j,vrts in enumerate(bm.verts):
        vrts.co = tuple(data[j])
    
    bm.to_mesh(bpy.data.objects[GeometryName].data)

###############################################################################
# Text Geometry functions
###############################################################################

def InsertAndChangeText(TextLocation,TextLabel,Name,scale,rotation):
    """
    Parameters
    ----------
    TextLocation : tuple
        Location of the text object
    TextLabel : string
        Text to add to the text object
    Name : string
        Name of the text object
    scale : float
        Scale of the text object
    rotation : tuple
        Rotation of the text object

    Returns
    -------
    None

    """
    
    bpy.ops.object.text_add(location=TextLocation,enter_editmode=True)
    for k in range(4):
        bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
    for char in TextLabel:
        bpy.ops.font.text_insert(text=char)
    bpy.ops.object.editmode_toggle()
    bpy.data.objects['Text'].data.name=Name
    bpy.data.objects['Text'].name=Name
    bpy.data.objects[Name].rotation_euler=rotation
    bpy.data.objects[Name].scale=(scale,scale,scale)

###############################################################################
# Material functions
###############################################################################

def AddGlassLikeAMaterial(GeometryName,RGBData):
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
    finalRGB = tuple([r,g,b,1])
    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes

    lightPath = nodes.new("ShaderNodeLightPath")

    glassBSDF = nodes.new("ShaderNodeBsdfGlass")
    glassBSDF.inputs[0].default_value = finalRGB

    transparentBSDF = nodes.new("ShaderNodeBsdfTransparent")

    mix01 = nodes.new("ShaderNodeMixShader")
    #currentMaterial.node_tree.links.new(lightPath.outputs[2],mix01.inputs[0])
    currentMaterial.node_tree.links.new(glassBSDF.outputs[0],mix01.inputs[1])
    currentMaterial.node_tree.links.new(transparentBSDF.outputs[0],mix01.inputs[2])
    
    mix02 = nodes.new("ShaderNodeMixShader")
    currentMaterial.node_tree.links.new(lightPath.outputs[1],mix02.inputs[0])
    currentMaterial.node_tree.links.new(mix01.outputs[0],mix02.inputs[1])

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(mix02.outputs[0],materialOutput.inputs[0])
    
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

###############################################################################
# Render Settings
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 5
bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 1200
bpy.context.scene.render.tile_x = 384
bpy.context.scene.render.tile_y = 256
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 550

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

bpy.data.objects["Camera"].location = (0,3,0)
bpy.data.objects["Camera"].rotation_euler = (-np.pi/2,np.pi,0)

bpy.data.objects['Light'].data.use_nodes = True
bpy.data.objects['Light'].data.node_tree.nodes['Emission'].inputs[1].default_value = 0.5
bpy.data.objects['Light'].data.type = 'AREA'
bpy.data.objects['Light'].location = (0,0,4.5)
bpy.data.objects["Light"].rotation_euler = (0,0,0)
bpy.data.objects['Light'].scale = (50,50,50)

###############################################################################
# Plotting functions
###############################################################################

#Wrapper function to scale the data. 
def ScaleData(data,scale):
    scaled = (data - data.min())/(data.max() - data.min())
    return scale*scaled

#Wrapper function to scale and format data
def MakeCartesianCoords(data,scale):

    num_vars = len(data)
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    XData = [scale*val*np.sin(sal) for val,sal in zip(data,theta)]
    YData = [scale*val*np.cos(sal) for val,sal in zip(data,theta)]

    return XData,YData

#Changes scaled data to global coordinates
def FormatData(XData,YData,center):

    coordinates = []
    for val,sal in zip(XData,YData):
        coord1 = [val+center[0],sal+center[1],center[2]]
        coord2 = [val+center[0],sal+center[1],center[2]-0.01]
        coordinates.append(coord1)
        coordinates.append(coord2)

    return np.array(coordinates)

#Wrapper function ot add a radar plot
def AddRadarPlot(data,scale,center,offset=0.5,name='TestData'):

    scaled = ScaleData(data,1)
    XData,YData = MakeCartesianCoords(scaled,scale)
    coords = FormatData(YData,XData,center)
    ModifyCylinder(coords,GeometryName=name)
    bpy.data.objects[name].rotation_euler[0] = np.pi/2
    bpy.data.objects[name].location[1] = offset
    


###############################################################################
# Visualization 
###############################################################################

data = np.random.random(15)
data2 = np.random.random(15)
data3 = np.random.random(15)

plotName = "spyderplot01"
plotName2 = "spyderplot02"
plotName3 = "spyderplot03"

AddRadarPlot(data,0.5,(0,0,0),name=plotName,offset=0.1)
AddRadarPlot(data2,0.5,(0,0,0),name=plotName2,offset=0.2)
AddRadarPlot(data3,0.5,(0,0,0),name=plotName3,offset=0.3)

AddGlassLikeAMaterial(plotName,[.95,0,.008])
AddGlassLikeAMaterial(plotName2,[.95,0,0.98])
AddGlassLikeAMaterial(plotName3,[.009,0.98,0.0098])