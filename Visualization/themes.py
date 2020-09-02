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
import numpy as np
###############################################################################
# Data
###############################################################################

maxData=100

barPlotData=np.random.random(int(maxData/10))
barPlotTicks=["Class" +str(val) for val in range(int(maxData/10))]

xScatterData=np.random.random(maxData)
yScatterData=np.random.randn(maxData)

xLineData=np.arange(int(maxData/8))
yLineData=np.random.random(int(maxData/8))

###############################################################################
# Geometry utility functions
###############################################################################

#Generates a list with the names of the objects
def MakeObjectNames(ObjectName,NumberOfObjects):
    """
    Parameters
    ----------
    ObjectName : string
        Name of the object to be created.
    NumberOfObjects : int
        Number of objects.

    Returns
    -------
    NamesContainer : list
        List of strings with the name of the objects.

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
        else:
            NamesContainer.append(ObjectName+"."+str(k))

    return NamesContainer

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
    ctx["selected_objects"]=totalObs
    ctx["selected_editable_bases"]=[scene.object_bases[ob.name] for ob in totalObs]
    bpy.ops.object.join(ctx)

    return ObjectNames[0]

###############################################################################
# Visualization utility functions
###############################################################################

def MakeTickLabels(data,ticksnumber=4):
    """
    Parameters
    ----------
    data : list, array
        data of the plot.
    ticksnumber : int, optional
        Nuber of ticks in the plot. The default is 4.

    Returns
    -------
    tickLocations : list
        locations of the ticks in local coordinates.
    tickLabels : list
        list with the tick labels
    """

    dmin,dmax=min(data),max(data)
    tickLabels=[]
    tickLocations=[]
    step=(dmax-dmin)/ticksnumber
    delta=1/(ticksnumber)
    
    for k in range(ticksnumber+1):
        tickVal=dmin+(k*step)
        tickLoc=k*delta
        tickLabels.append(str(round(tickVal,2)))
        tickLocations.append(tickLoc)
        
    return tickLocations,tickLabels

#Wrapper function to change Hex color values to rgb 
def FromHexToRGB(hexName):
    hexName=hexName.lstrip("#")
    return [int(hexName[i:i+2],16)/255 for i in (0,2,4)]
    
#Calculates the distance between two points
def Distance(PointA,PointB):
    return np.sqrt(sum([(val-sal)**2 for val,sal in zip(PointA,PointB)]))

#Calculates the midpoint between two points
def MidPointLocation(PointA,PointB):
    return tuple([(val+sal)/2 for val,sal in zip(PointA,PointB)])

#Calculates the slope between two points
def Rotation(PointA,PointB):
    return np.arctan((PointA[1]-PointB[1])/(PointA[0]-PointB[0]))-(np.pi/2)

def AddTextAtLocation(Text,Location,Scale,GeometryName):
    """
    Parameters
    ----------
    Text : String
        Text to add.
    Location : tuple
        location of the text in global coordinates.
    Scale : float
        factor to scale the text.
    GeometryName : string
        name of the geometry object created.

    Returns
    -------
    None.

    """

    bpy.ops.object.text_add(location=Location,enter_editmode=True)
    for j in range(4):
        bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
    for char in Text:
        bpy.ops.font.text_insert(text=char)
    
    bpy.context.object.data.align_x='CENTER'
    bpy.context.object.data.align_y='CENTER'
    bpy.ops.object.editmode_toggle()
    bpy.data.objects['Text'].scale=(Scale/15,Scale/15,Scale/15)
    bpy.data.objects['Text'].name=GeometryName

#Wrapper to change a text geometry object to a mesh
def FromTextToMesh(GeometryObject):

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.scene.objects.active=GeometryObject
    GeometryObject.select=True
    bpy.ops.object.convert(target="MESH",keep_original=False)
    GeometryObject.select=False

###############################################################################
# Data Visualization functions
###############################################################################

def MakeBarPlot(BarsData,ticksLabels,PlotName,location,extend):
    """
    Parameters
    ----------
    BarsData : list,array
        Data for the bar plot.
    ticksLabels : list,array
        list with the tick labels for the x axis in the bar plot.
    PlotName : string
        name for the geometries created for the plot.
    location : tuple
        Global location for the bar plot.
    extend : float
        extend of the bar plot.

    Returns
    -------
    barNames : list
        list with the bars geometry names.
    tickNames : lits
        list with the ticks geometry names..

    """
    nBars=len(BarsData)
    maxVal=max(BarsData)
    NormData=[val/maxVal for val in BarsData]
    step=(2*extend)/(nBars-1)
    xBarLocations=[-(extend)+(k*step)+location[0] for k in range(nBars)]
    barGeometryNames=MakeObjectNames(PlotName+"Bar",nBars)
    tickGeometryNames=MakeObjectNames(PlotName+"Tick",nBars)
    padding=extend/20

    for k in range(nBars):
        currentBarHeight=(extend/2)*NormData[k]
        currentBarLocation=currentBarHeight-extend+location[1]
        bpy.ops.mesh.primitive_cube_add(radius=1,location=(xBarLocations[k],currentBarLocation,0))
        bpy.data.objects["Cube"].scale=(extend/40,currentBarHeight,extend/40)
        bpy.data.objects["Cube"].data.name=barGeometryNames[k]
        bpy.data.objects["Cube"].name=barGeometryNames[k]

        AddTextAtLocation(ticksLabels[k],(xBarLocations[k],-extend+location[1]-padding,0),extend,tickGeometryNames[k])
    
    barGeometry=JoinObjectsByName(barGeometryNames)

    [FromTextToMesh(bpy.data.objects[name]) for name in tickGeometryNames]
    tickGeometry=JoinObjectsByName(tickGeometryNames)

    return barGeometry,tickGeometry

def MakeScatterPlot(XData,YData,PlotName,location,extend,joined=False):
    """
    
    Parameters
    ----------
    XData : list,array
        Data of the x axis.
    YData : list,array
        Data of the y axis.
    PlotName : string
        Name for the mesh.
    location : tuple
        location of the plot in global coordinates.
    extend : float
        factor to control the size of the plot.
    joined : bool, optional
        Controls the behabiour of the plot, scatter or line plot. 
        The default is False.

    Returns
    -------
    list
        list with the geometry names of all the objects in the plot 
        if Joined == True then it also contains a lost with the line
        names

    """
    
    ndata=len(XData)
    xMin,xMax=np.min(XData),np.max(XData)
    xNorm=[(val-xMin)/(xMax-xMin) for val in XData]
    yMin,yMax=np.min(YData),np.max(YData)
    yNorm=[(val-yMin)/(yMax-yMin) for val in YData]
    dotGeometryNames=MakeObjectNames(PlotName+"scatter",ndata)

    for val in enumerate(zip(xNorm,yNorm)):
        k,locs=val
        newLocation=(locs[0]+location[0],locs[1]+location[1],location[2])
        bpy.ops.mesh.primitive_uv_sphere_add(location=newLocation)
        bpy.data.objects["Sphere"].scale=(extend/60,extend/60,extend/60)
        bpy.data.objects["Sphere"].data.name=dotGeometryNames[k]
        bpy.data.objects["Sphere"].name=dotGeometryNames[k]

    dotName=JoinObjectsByName(dotGeometryNames)
    
    if joined:

        linesNames=MakeObjectNames(PlotName+"lines",ndata-1)
        lineWidth=extend/200

        for k in range(ndata-1):
            current=(location[0]+xNorm[k],location[1]+yNorm[k],location[2])
            foward=(location[0]+xNorm[k+1],location[1]+yNorm[k+1],location[2])

            midpoint=MidPointLocation(current,foward)
            lineLength=Distance(current,foward)/2
            rotation=Rotation(current,foward)
            bpy.ops.mesh.primitive_cube_add(location=midpoint)
            bpy.data.objects["Cube"].scale=(lineWidth,lineLength,lineWidth)
            bpy.data.objects["Cube"].rotation_euler=(0,0,rotation)
            bpy.data.objects["Cube"].data.name=linesNames[k]
            bpy.data.objects["Cube"].name=linesNames[k]
        
        lineGeometryName=JoinObjectsByName(linesNames)

    xTicksLocations,xTicksNames=MakeTickLabels(XData)
    xTickNames=MakeObjectNames(PlotName+"xTicks",len(xTicksNames))
    yTicksLocations,yTicksNames=MakeTickLabels(YData)
    yTickNames=MakeObjectNames(PlotName+"yTicks",len(yTicksNames))

    padding=extend/10
    
    for val in enumerate(zip(xTicksLocations,xTicksNames)):
        k,tickvals=val
        xLoc=(location[0]+tickvals[0],location[1]-padding,location[2])
        AddTextAtLocation(tickvals[1],xLoc,extend,xTickNames[k])

    for val in enumerate(zip(yTicksLocations,yTicksNames)):
        k,tickvals=val
        yLoc=(location[0]-padding,location[1]+tickvals[0],location[2])
        AddTextAtLocation(tickvals[1],yLoc,extend,yTickNames[k])

    [FromTextToMesh(bpy.data.objects[name]) for name in xTickNames]
    [FromTextToMesh(bpy.data.objects[name]) for name in yTickNames]
    xTickGeometry=JoinObjectsByName(xTickNames)
    yTickGeometry=JoinObjectsByName(yTickNames)

    if joined:
        return dotName,lineGeometryName,xTickGeometry,yTickGeometry

    else:
        return dotName,xTickGeometry,yTickGeometry




###############################################################################
# Material functions
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

    r,g,b,a=RGBData
    currentMaterial = bpy.data.materials.new(name='TypeA'+GeometryName)
    currentMaterial.use_nodes = True
    nodes = currentMaterial.node_tree.nodes
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (r,g,b,a)

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
    emission.inputs[1].default_value=0.5
    currentMaterial.node_tree.links.new(rgb.outputs[0],emission.inputs[0])

    mix=nodes.new("ShaderNodeMixShader")
    currentMaterial.node_tree.links.new(bsdf.outputs[0],mix.inputs[1])
    currentMaterial.node_tree.links.new(emission.outputs[0],mix.inputs[2])

    materialOutput=nodes.get("Material Output")
    currentMaterial.node_tree.links.new(mix.outputs[0],materialOutput.inputs[0])
    bpy.data.objects[GeometryName].data.materials.append(currentMaterial)


###############################################################################
# Initial setup
###############################################################################

bpy.data.objects["Cube"].select=True
bpy.ops.object.delete()

bpy.data.objects["Camera"].location=(0,0,7)
bpy.data.objects["Camera"].rotation_euler=(0,0,0)

bpy.context.scene.render.engine = 'CYCLES'

glow=True

if glow:
    bpy.ops.mesh.primitive_plane_add(location=(0,0,-0.0))
    bpy.data.objects["Plane"].scale=(4,4,4)

###############################################################################
# Adding the graphs
###############################################################################

bars,barTicks=MakeBarPlot(barPlotData,barPlotTicks,"test",(1,1,0),1.5)

scatterDots,scatterXticks,scatterYticks=MakeScatterPlot(xScatterData,yScatterData,"test",(-2.25,-1.25,0),1.75)

lineDots,lineLines,lineXticks,lineYticks=MakeScatterPlot(xLineData,yLineData,"testline",(-2.25,0.25,0),1.75,joined=True)

###############################################################################
# Applying Styles
###############################################################################

HEXCodes=["#ef3d59","#344ec5","#573f4c","#933469","#fff066","#247ba0"]
grayHEX="#808080"
dataHEX="#fff066"
tickHEX="#933469"

horizonRGB=(0,0,0)
dataElementsRGBA=tuple(FromHexToRGB(dataHEX)+[1])
dataElementsRGBAL=tuple(FromHexToRGB(dataHEX)+[0.2])
tickElementRGBA=tuple(FromHexToRGB(tickHEX)+[1])
grayConfig=tuple(FromHexToRGB(grayHEX)+[1])

bpy.context.scene.world.horizon_color = horizonRGB

#Data elements
AddTypeAMaterial(bars,grayConfig)
AddTypeAMaterial(scatterDots,grayConfig)
AddTypeBMaterial(lineDots,dataElementsRGBA)
AddTypeBMaterial(lineLines,dataElementsRGBAL)

#Tick elements
AddTypeAMaterial(barTicks,grayConfig)
AddTypeAMaterial(scatterXticks,grayConfig)
AddTypeAMaterial(scatterYticks,grayConfig)
AddTypeBMaterial(lineXticks,tickElementRGBA)
AddTypeBMaterial(lineYticks,tickElementRGBA)