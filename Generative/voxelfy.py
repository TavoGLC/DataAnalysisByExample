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
import colorsys
import numpy as np

from PIL import Image

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

def GetCentersLocations(NumberOfItems,stretch,random_norm=100):
    """
    Parameters
    ----------
    NumberOfItems : int
        Row or solumn length.
    stretch : float
        boundaries of the visualization square

    Returns
    -------
    centerLocations : list
        List with the center locations of each cube in the heatmap

    """

    xItems,yItems = NumberOfItems
    xStretch,yStretch = stretch
    
    xGrid=np.linspace(xStretch,-xStretch,xItems+1)
    yGrid=np.linspace(yStretch,-yStretch,yItems+1)
    
    centerLocations=[]

    for k in range(yItems):
        
        midY=(yGrid[k+1]+yGrid[k])/2
        
        for i in range(xItems):
            
            midX=(xGrid[i+1]+xGrid[i])/2
            currentLoc=(midX,np.random.random()/random_norm,midY)
            centerLocations.append(currentLoc)
            
    return centerLocations

###############################################################################
# Geometry functions
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

    GeometryNames = ["Sphere","Cube","Cylinder","Torus","Suzanne"]
    GeometryFunctions = [bpy.ops.mesh.primitive_uv_sphere_add,bpy.ops.mesh.primitive_cube_add
                        ,bpy.ops.mesh.primitive_cylinder_add,bpy.ops.mesh.primitive_torus_add
                        ,bpy.ops.mesh.primitive_monkey_add]
    
    NameToLocation = {}

    for val,name in enumerate(GeometryNames):
        NameToLocation[name] = val

    try:
        
        funcIndex = NameToLocation[Kind]
        GeometryFunctions[funcIndex](location=Location)
        bpy.data.objects[Kind].scale = Scale
        bpy.data.objects[Kind].data.name = Name
        bpy.data.objects[Kind].name = Name
        
        if Kind == "Torus":
            bpy.data.objects[Name].rotation_euler = (np.pi/2,0,0)

        if Kind == "Suzanne":
            bpy.data.objects[Name].rotation_euler = (0,0,np.pi)
            
    except ValueError:
        print("Not recognized type")

def AddElementsFromLocations(Locations,Scale,Kind,Name):

    """
        Parameters
    ----------
    Locations : list,array-like
        Location of the objects.
    Scale : float
        Scale of the geometry
    Kind : str
        Kind of geometry to be added.
    Name : str
        Name of the geometry

    Returns
    -------
    objectNames : list
        Names of the geometries.
    """

    objectNames=MakeObjectNames('GeometryForMatrix'+Name,len(Locations))
    nObjects=len(Locations)
    
    for k in range(nObjects):

        AddGeometryByKind(Kind,objectNames[k],tuple(Locations[k]),(Scale,Scale,Scale))
    
    return objectNames

###############################################################################
# Material functions
###############################################################################

def AddTypeAnodizedMetalMaterial(GeometryName,RGBData):
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

    bsdfA = nodes.new("ShaderNodeBsdfAnisotropic")
    bsdfA.distribution = 'BECKMANN'
    bsdfA.inputs[1].default_value = 0.1
    bsdfA.inputs[2].default_value = 0.5
    currentMaterial.node_tree.links.new(rgb.outputs[0],bsdfA.inputs[0])

    bsdfG = nodes.new("ShaderNodeBsdfGlossy")
    bsdfG.distribution = 'BECKMANN'
    bsdfG.inputs[1].default_value = 0.1
    currentMaterial.node_tree.links.new(rgb.outputs[0],bsdfG.inputs[0])

    mixShader = nodes.new("ShaderNodeMixShader")
    mixShader.inputs[0].default_value = 0.5
    currentMaterial.node_tree.links.new(bsdfA.outputs[0],mixShader.inputs[1])
    currentMaterial.node_tree.links.new(bsdfG.outputs[0],mixShader.inputs[2])

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(mixShader.outputs[0],materialOutput.inputs[0])
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

bpy.data.objects["Camera"].location = (0,4,0)
bpy.data.objects["Camera"].rotation_euler = (-np.pi/2,-np.pi,0)

bpy.data.objects['Light'].data.use_nodes = True
bpy.data.objects['Light'].data.node_tree.nodes['Emission'].inputs[1].default_value = 0.25
bpy.data.objects['Light'].data.type = 'AREA'
bpy.data.objects['Light'].location = (0.5,0.5,4.5)
bpy.data.objects["Light"].rotation_euler = (0 ,0,0)
bpy.data.objects['Light'].scale = (1,1,1)

###############################################################################
# Visualization settings
###############################################################################

def Pixelate(image, window):
    
    n, m, channels = image.shape
    n, m = n - n % window[0], m - m % window[1]
    pixelImage = np.zeros((n//window[0], m//window[1], channels))
    
    for jj,j in enumerate(range(0, n, window[0])):
        for kk,k in enumerate(range(0, m, window[1])):
            pixelImage[jj,kk] = np.mean(image[j:j+window[0],k:k+window[1]],axis=(0,1)).astype(np.uint8)
            
    return pixelImage

im = Image.open(r"/media/tavo/storage/marissa-lewis-Fm17vn1lmAQ-unsplash.jpg") 

div = 8
window = (60,60)

im = np.array(im)
piximg = Pixelate(im,window)
piximg = np.array(piximg) // div * div + div // 2

ravpix = piximg.reshape(-1,3)/255

###############################################################################
# Visualization settings
###############################################################################

maxScale = 0.025
nItems = piximg.shape[0:2][::-1]
delta = 0.2

if nItems[0]>nItems[1]:
    maxScale_x = maxScale
    maxScale_y = maxScale*(nItems[1]/nItems[0])
elif nItems[0]<nItems[1]:
    maxScale_y = maxScale
    maxScale_x = maxScale*(nItems[0]/nItems[1])
else:
    maxScale_x = maxScale
    maxScale_y = maxScale

xStretch = nItems[0]*maxScale_x+delta
yStretch = nItems[1]*maxScale_y+delta
stretch = [xStretch,yStretch]

center = GetCentersLocations(nItems,stretch)
names = AddElementsFromLocations(center,maxScale,"Cube","Test")

for nme,clr in zip(names,ravpix):
    AddTypeAMaterial(nme,clr)
