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

gridSize=250

x = np.linspace(-5, 5, gridSize)
y = np.linspace(-5, 5, gridSize)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)

###############################################################################
# Removing the original cube
###############################################################################

bpy.data.objects["Cube"].select=True
bpy.ops.object.delete()

###############################################################################
# Moving the camera
###############################################################################

bpy.data.objects["Camera"].location=(4.62,-3.81,3.38)

###############################################################################
# Moving the camera
###############################################################################

def MakeObjectNames(ObjectName,NumberOfObjects):
    """
    ObjectName -> Base name of the object created
    NumberOfObjects -> Number of objects to be created
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

#Wrapper function to create a list of text object names
def MakeTextNames(NumberOfElements):
    return MakeObjectNames("Text",NumberOfElements)

#Wrapper function to create a list of cube object names
def MakeCubeNames(NumberOfElements):
    return MakeObjectNames("Cube",NumberOfElements)

def InsertAndChangeText(TextLocation,TextLabel,rotation=False):
    """
    Inserts a text object at TextLocation and change it to TextLabel
    TextLocation -> vector, location of the text to be added
    TextLabel    -> string, text to be written
    rotation     -> bool, rotate the text for the z axis 
    """
    bpy.ops.object.text_add(location=TextLocation,enter_editmode=True)
    for k in range(4):
        bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
    for char in TextLabel:
        bpy.ops.font.text_insert(text=char)
    bpy.ops.object.editmode_toggle()
    bpy.ops.transform.resize(value=(0.15,0.15,0.15))

    if rotation==True:
        bpy.ops.transform.rotate(value=1.5708,axis=(1,0,0))
        bpy.ops.transform.rotate(value=1.5708,axis=(0,0,1))

def MinMaxNormalization(DisplacementData):
    """
    Scales the displacement data between 0 and 1 
    DisplacementData -> numpy array, z values to be plotted. 
    """
    disp=(DisplacementData-DisplacementData.min())/(DisplacementData.max()-DisplacementData.min())
    return disp.flatten()

def LocalColorMap(N):
    """
    Creates an evenly spaced range of colors 
    N -> int, number of colors in the range 
    """
    HSV_tuples=[(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    rgbList=list(map(lambda x: colorsys.hsv_to_rgb(*x),HSV_tuples))
    r=[val[0] for val in rgbList]
    g=[val[1] for val in rgbList]                  
    b=[val[2] for val in rgbList]                  
    return r,g,b


def ValueToMaterialIndex(value,GridValues):
    """
    Returns the index of a given value 
    value      -> float, value to be found.
    GridValues -> list, array, boundaries for the value to be indexed by. 
    """
    responce=0
    for k in range(len(GridValues)-1):
        if GridValues[k]<value and GridValues[k+1]>=value:
            responce=k
            break
    return responce 

def Add2DSurface(DisplacementData):
    """
    Add and modify a mesh to show a 2d plot 
    DisplacementData -> numpy array, z values to be plotted. 
    """
    subdivisions=DisplacementData[0,:].size
    disp=MinMaxNormalization(DisplacementData)

    bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdivisions,y_subdivisions=subdivisions,radius=1,location=(0,0,0))
    bm=bmesh.new()
    bm.from_mesh(bpy.data.objects["Grid"].data)
    matrixWorld=bpy.data.objects["Grid"].matrix_world

    Faces = bm.faces
    Vertexs=[matrixWorld*v.co for v in bm.verts]

    for face in Faces:
        normal=face.normal
        faceVerts=face.verts
        for verts in faceVerts:
            i=verts.index
            d=normal*disp[i]
            verts.co=Vertexs[i]+d

    bm.to_mesh(bpy.data.objects["Grid"].data)

def AddPlotAxis(x,y,z,nticks):
    """
    Adds the axis of the 2D plot 
    x      -> list, array, contains the x values of the plot
    y      -> list, array, contains the y values of the plot
    z      -> list, array, contains the z values of the plot
    nticks -> number of ticks to be displayed
    """
    bpy.ops.mesh.primitive_cube_add(location=(1.051,0,0))
    bpy.ops.mesh.primitive_cube_add(location=(0,-1.051,0))
    bpy.ops.mesh.primitive_cube_add(location=(-1,-1.05,0.5))

    bpy.data.objects["Cube"].scale=(0.01,1,0.01)
    bpy.data.objects["Cube.001"].scale=(1,0.01,0.01)
    bpy.data.objects["Cube.002"].scale=(0.01,0.01,0.5)

    xticks=np.linspace(np.min(x),np.max(x),nticks)
    yticks=np.linspace(np.min(y),np.max(y),nticks)
    zticks=np.linspace(np.min(z),np.max(z),nticks)

    xylocations=np.linspace(-1,1,nticks)
    zlocations=np.linspace(0,1,nticks)

    xticks=[round(val,3) for val in xticks]
    yticks=[round(val,3) for val in yticks]
    zticks=[round(val,3) for val in zticks]

    for k in range(nticks):
        
        xlocation=(1.2,xylocations[k],0)
        ylocation=(xylocations[k],-1.2,0)
        zlocation=(-1,-1.45,zlocations[k])
        InsertAndChangeText(xlocation,str(xticks[k]))
        InsertAndChangeText(ylocation,str(yticks[k]))
        InsertAndChangeText(zlocation,str(zticks[k]),rotation=True)

def AddColorMap(GeometryName,GridSize):
    """
    Adds a color map to a surface
    GeometryName      -> string, name of the geometry to be modified 
    GridSize          -> int, number of subdivisions in the colormap 
    """
    GridValues=np.linspace(0,1,GridSize+1)
    r,g,b=LocalColorMap(GridSize)

    for k in range(GridSize):
        
        currentMaterial=bpy.data.materials.new(name="surface"+str(k))
        currentMaterial.use_nodes=True 
        nodes = currentMaterial.node_tree.nodes
        
        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = (r[k],g[k],b[k],1)

        diffuse=nodes.get("Diffuse BSDF")
        diffuse.inputs[1].default_value = 1

        currentMaterial.node_tree.links.new(rgb.outputs[0],diffuse.inputs[0])

        bsdf = nodes.new("ShaderNodeBsdfGlass")
        bsdf.inputs[1].default_value = 1
        currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[0])

        mixer = nodes.new("ShaderNodeMixShader")
        mixer.inputs[0].default_value = 0.6

        currentMaterial.node_tree.links.new(diffuse.outputs[0],mixer.inputs[1])
        currentMaterial.node_tree.links.new(bsdf.outputs[0],mixer.inputs[2])

        materialOutput=nodes.get("Material Output")

        currentMaterial.node_tree.links.new(mixer.outputs[0],materialOutput.inputs[0])

        bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

    for k in range(len(bpy.data.objects[GeometryName].data.polygons.items())):
        
        value=bpy.data.objects[GeometryName].data.polygons[k].center[2]
        currentIndex=ValueToMaterialIndex(value,GridValues)
        bpy.data.objects[GeometryName].data.polygons[k].material_index=currentIndex



def AddBlackMaterials(GeometryNames):
    """
    Add a black material to each geometry in Geometry names
    GeometryNames      -> List with the name of the geometries
    """  
    for val in GeometryNames:

        currentMaterial = bpy.data.materials.new(name='EdgeMaterial'+val)
        currentMaterial.use_nodes = True
    
        nodes = currentMaterial.node_tree.nodes

        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = (0,0,0,1)

        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs[4].default_value = 0.5
        bsdf.inputs[7].default_value = 1
        bsdf.inputs[15].default_value = 1
    
        currentMaterial.node_tree.links.new(rgb.outputs[0],bsdf.inputs[1])

        materialOutput=nodes.get("Material Output")
        currentMaterial.node_tree.links.new(bsdf.outputs[0],materialOutput.inputs[0])

        bpy.data.objects[val].data.materials.append(currentMaterial)


###############################################################################
# Moving the camera
###############################################################################

nticks=3
Add2DSurface(z)
AddPlotAxis(x,y,z,nticks)

###############################################################################
# Changing the horizon 
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.world.horizon_color = (1,1,1)

AxisNames=MakeCubeNames(3)+MakeTextNames(3*nticks)

AddBlackMaterials(AxisNames)
AddColorMap("Grid",10)
