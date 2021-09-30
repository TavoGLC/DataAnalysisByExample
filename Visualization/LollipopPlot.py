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

def JoinGeometryByNames(GeometryNames):
    """
    Function to join different geometries by its names

    Parameters
    ----------
    GeometryNames : list, array-like
        list of strings that contains the names of the geometries to be merged
    """

    scene = bpy.context.scene 
    objectsToJoin=[]
    for nme in GeometryNames:
        objectsToJoin.append(bpy.data.objects[nme])

    joinDict = {}
    joinDict["object"] = joinDict["active_object"] = objectsToJoin[0]
    joinDict["selected_objects"] = joinDict["selected_editable_objects"] = objectsToJoin
    bpy.ops.object.join(joinDict)


###############################################################################
# Geometry functions
###############################################################################

def AddSimpleGeometryByKind(Kind,Name,Location,Scale=None):
    '''
    Add a mesh object to the scene and change it's name 
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
        bpy.data.objects[Kind].data.name = Name
        bpy.data.objects[Kind].name = Name

        if Scale != None:
            bpy.data.objects[Kind].scale = Scale
    
    except ValueError:
        print("Not recognized type")

def AddLollipop(Length=1, location=(0,0,0), headRotation = (np.pi/2,0,0), kind="Sphere" ,name="Lollipop"):
    """
    Add a simple lollipop geometry

    Parameters
    ----------
    Length : float, optional
        Lenght of the lollipop, by default 1
    location : tuple, optional
        location of the geometry, by default (0,0,0)
    headRotation : tuple, optional
        Rotatio, of the lillipop, useful ponly for torus and suzanne geometries, by default (np.pi/2,0,0)
    kind : str, optional
        Kind of geometry on top of the lollipop, by default "Sphere"
    name : str, optional
        namer of the final geometry, by default "Lollipop"
    """

    names = ["Cube",kind]

    bpy.ops.mesh.primitive_cube_add(location=(0,0,0))
    bpy.data.objects['Cube'].scale = (0.0025,0.0025,Length)
    bpy.data.objects['Cube'].rotation_euler = (0,0,np.random.random())

    geoArgs=(0,0,Length)

    AddSimpleGeometryByKind(kind,kind,geoArgs)
    bpy.data.objects[kind].scale = (0.005,0.005,0.005)

    JoinGeometryByNames(names)
    bpy.data.objects['Cube'].data.name = name
    bpy.data.objects['Cube'].name = name
    bpy.data.objects[name].location = location


def AddElementsFromLocations(Locations,Kind,Name,Scale):

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

        AddSimpleGeometryByKind(Kind,objectNames[k],tuple(Locations[k]),Scale)
    
    return objectNames

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

def AddHolograficMaterial(GeometryName,RGBData):
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
    
    #Nodes handler
    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes

    r,g,b=RGBData
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (min([r+0.1,1]),min([g+0.1,1]),min([b+0.1,1]),1)

    rgb2 = nodes.new("ShaderNodeRGB")
    rgb2.outputs[0].default_value = (r,g,b,1)

    layerWeight = nodes.new('ShaderNodeLayerWeight')
    layerWeight.inputs[0].default_value = 0.15

    addNode = nodes.new('ShaderNodeMath')
    currentMaterial.node_tree.links.new(layerWeight.outputs[0],addNode.inputs[0])
    currentMaterial.node_tree.links.new(layerWeight.outputs[1],addNode.inputs[1])

    transparentBSDF = nodes.new('ShaderNodeBsdfTransparent')
    emissionBSDF = nodes.new('ShaderNodeEmission')
    emissionBSDF.inputs[1].default_value = 0.5

    mixShader01 = nodes.new('ShaderNodeMixShader')

    currentMaterial.node_tree.links.new(rgb.outputs[0],transparentBSDF.inputs[0])
    currentMaterial.node_tree.links.new(rgb.outputs[0],emissionBSDF.inputs[0])

    currentMaterial.node_tree.links.new(addNode.outputs[0],mixShader01.inputs[0])
    currentMaterial.node_tree.links.new(transparentBSDF.outputs[0],mixShader01.inputs[1])
    currentMaterial.node_tree.links.new(emissionBSDF.outputs[0],mixShader01.inputs[2])

    wireframe = nodes.new('ShaderNodeWireframe')
    wireframe.inputs[0].default_value = 0.2
    mixShader02 = nodes.new('ShaderNodeMixShader')

    currentMaterial.node_tree.links.new(wireframe.outputs[0],mixShader02.inputs[0])
    currentMaterial.node_tree.links.new(mixShader01.outputs[0],mixShader02.inputs[1])
    currentMaterial.node_tree.links.new(emissionBSDF.outputs[0],mixShader02.inputs[2])

    lightPath = nodes.new('ShaderNodeLightPath')
    mixShader03 = nodes.new('ShaderNodeMixShader')
    transparentBSDF01 = nodes.new('ShaderNodeBsdfTransparent')

    currentMaterial.node_tree.links.new(lightPath.outputs[0],mixShader03.inputs[0])
    currentMaterial.node_tree.links.new(mixShader02.outputs[0],mixShader03.inputs[1])
    currentMaterial.node_tree.links.new(transparentBSDF01.outputs[0],mixShader03.inputs[2])

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(mixShader03.outputs[0],materialOutput.inputs[0])

    mixShader04 = nodes.new("ShaderNodeMixShader")
    emissionBSDF01 = nodes.new('ShaderNodeEmission')
    emissionBSDF01.inputs[1].default_value = 0.5
    volumeAbs = nodes.new('ShaderNodeVolumeAbsorption')

    currentMaterial.node_tree.links.new(rgb2.outputs[0],transparentBSDF01.inputs[0])
    currentMaterial.node_tree.links.new(rgb2.outputs[0],emissionBSDF01.inputs[0])

    currentMaterial.node_tree.links.new(volumeAbs.outputs[0],mixShader04.inputs[1])
    currentMaterial.node_tree.links.new(emissionBSDF01.outputs[0],mixShader04.inputs[2])

    currentMaterial.node_tree.links.new(mixShader04.outputs[0],materialOutput.inputs[1])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)

def TwoPointLinearColorMap(startPoint,endPoint,points):
    """
    Parameters
    ----------
    startPoint : tuple,array-like
        Start location in the RGB color space.
    endPoint : tuple, array-like
        End location in the RGB color space.
    points : int
        Number of sample points between the startPoint and endPoint 

    Returns
    -------
    objectNames : array
        contains an linearly sampled RGB colors between the startPoint and the endPoint
    """
    
    rs,gs,bs=startPoint
    re,ge,be=endPoint
    r=np.linspace(rs,re,num=points)
    g=np.linspace(gs,ge,num=points)
    b=np.linspace(bs,be,num=points)
    
    return np.vstack((r,g,b)).T

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
bpy.context.scene.cycles.samples = 750

###############################################################################
# Settings for the scene
###############################################################################

bpy.data.objects['Cube'].data.name = 'StartCube'
bpy.data.objects['Cube'].name = 'StartCube'
bpy.data.objects['StartCube'].select_set(state = True)
bpy.data.objects['StartCube'].scale = (6,6,6)
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.bevel(offset=0.15,vertex_only=False)
bpy.ops.object.editmode_toggle()
bpy.ops.object.shade_smooth()
bpy.data.objects['StartCube'].select_set(state = False)

bpy.ops.mesh.primitive_plane_add(size = 1, enter_editmode=False, location=(0, 0, 0))
bpy.data.objects['Plane'].scale = (2,2,1)
bpy.data.objects['Plane'].rotation_euler[0] = np.pi/2

bpy.data.objects["Camera"].location = (0,2.85,0)
bpy.data.objects["Camera"].rotation_euler = (-np.pi/2,np.pi,0)

bpy.data.objects['Light'].data.use_nodes = True
bpy.data.objects['Light'].data.node_tree.nodes['Emission'].inputs[1].default_value = 1
bpy.data.objects['Light'].data.type = 'AREA'
bpy.data.objects['Light'].location = (0,0,4.5)
bpy.data.objects["Light"].rotation_euler = (0,0,0)
bpy.data.objects['Light'].scale = (50,50,50)

###############################################################################
# Plotting functions
###############################################################################

def ScaleData(Data,Extent,center):
    """
    Scales the data in a [-1,1] range 
    Parameters
    ----------
    Data : array-like
        2D array, x and y data for the plot
    Extent : array-like
        Two element array with floats in the range [0,1].
        Controls the extent of the plot in the x and y axis.
    center : array-like
        location of the center of the plor in global blender 
        coordinates

    Returns
    -------
    array
        Contains the normalized x and y data 
    """

    X_data, Y_data = Data
    X_Extent, Y_Extent = Extent

    X_data = (X_data - np.min(X_data))/(np.max(X_data) - np.min(X_data))
    X_data = 2*X_data - 1
    X_data = X_Extent*X_data -center[0]

    Y_data = (Y_data - np.min(Y_data))/(np.max(Y_data) - np.min(Y_data))
    Y_data = 2*Y_data - 1
    Y_data = Y_Extent*Y_data -center[2]

    return X_data,Y_data

#Wrapper function to add all the elemnts for a lollipo plot
def AddLollipops(Data,Extent,center):

    X_data,Y_data = ScaleData(Data,Extent,center)
    lollipopNames = MakeObjectNames('Lollipop',len(X_data))
    [AddLollipop(Length=sal,kind='Sphere',location=(val,center[1],sal-Extent[1]),name=nme) for val,sal,nme in zip(X_data,Y_data,lollipopNames)]

    return lollipopNames

###############################################################################
#Lollipop plot
###############################################################################

npoints = 100
X_locations = np.linspace(-6,6,npoints)
Y_locations = np.exp(X_locations/3)*np.sin(4*X_locations)

nmes = AddLollipops([X_locations,Y_locations],[0.75,0.15],(0,0.05,0))
colors = TwoPointLinearColorMap((0.7,0.7,0.7),(0.25,0.5,0.6),npoints)

[AddHolograficMaterial(nme,col) for nme,col in zip(nmes,colors)]
