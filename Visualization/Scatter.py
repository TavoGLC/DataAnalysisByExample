
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
import random

###############################################################################
# Data
###############################################################################

nPoints=250

Xvalues=[(j/nPoints) + (random.random()/5) for j in range(nPoints)]
Yvalues=[(j/nPoints) + (random.random()/5) for j in range(nPoints)]

###############################################################################
# GeneralFunctions
###############################################################################

#Generates a list with the names of the objects
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

def InsertAndChangeText(TextLocation,TextLabel):
    
    bpy.ops.object.text_add(location=TextLocation,enter_editmode=True)
    for k in range(4):
        bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
    for char in TextLabel:
        bpy.ops.font.text_insert(text=char)
    bpy.ops.object.editmode_toggle()

#Wrapper function to create a list of shpere object names
def MakeDotNames(NumberOfElements):
    return MakeObjectNames("Sphere",NumberOfElements)

#Wrapper function to create a list of shpere object names
def MakeTextNames(NumberOfElements):
    return MakeObjectNames("Text",NumberOfElements)

###############################################################################
# Removing the original cube
###############################################################################

bpy.data.objects["Cube"].select=True
bpy.ops.object.delete()

###############################################################################
# Moving the camera
###############################################################################

bpy.data.objects["Camera"].location=(1.43,0.53,3.4)
bpy.data.objects["Camera"].rotation_euler=(0,0.2799,0)

###############################################################################
# Adding the axes 
###############################################################################

bpy.ops.mesh.primitive_cube_add(location=(-0.025,0.6,0))
bpy.ops.mesh.primitive_cube_add(location=(0.6,-0.025,0))

bpy.data.objects["Cube"].scale=(0.01,0.6,0.01)
bpy.data.objects["Cube.001"].scale=(0.6,0.01,0.01)

###############################################################################
# Adding the axes 
###############################################################################

counter=0
for k in range(3):
    InsertAndChangeText((counter,-0.25,0),str(counter))
    counter=counter+0.5

counter=0
for k in range(3):
    InsertAndChangeText((-0.3,counter,0),str(counter))
    counter=counter+0.5

TextNames=MakeTextNames(6)

for val in TextNames:
    bpy.data.objects[val].scale=(0.15,0.15,0.15)

###############################################################################
# Adding the data points 
###############################################################################

DotNames=MakeDotNames(nPoints)

for k in range(nPoints):
    bpy.ops.mesh.primitive_uv_sphere_add(location=(Xvalues[k],Yvalues[k],0))
    bpy.data.objects[DotNames[k]].scale=(0.01,0.01,0.01)

###############################################################################
# Adding the material 
###############################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.world.horizon_color = (1,1,1)

#Adding Material to the axis

for val in ['Cube','Cube.001']:

    currentMaterial = bpy.data.materials.new(name='TickMaterial'+val)
    currentMaterial.diffuse_color=(0,0,0)
    bpy.data.objects[val].data.materials.append(currentMaterial)

#Adding materials to the ticks
for k in range(6):

    currentMaterial = bpy.data.materials.new(name='TickMaterial'+str(k))
    currentMaterial.diffuse_color=(0,0,0)
    bpy.data.objects[TextNames[k]].data.materials.append(currentMaterial)

#Adding materials to the dots
for k in range(nPoints):

    currentMaterial = bpy.data.materials.new(name='Glass BSDF')
    currentMaterial.diffuse_color=(0.2,k/nPoints,k/nPoints)
    bpy.data.objects[DotNames[k]].data.materials.append(currentMaterial)

###############################################################################
# Adding the analytics
###############################################################################

import numpy as np 

A =np.transpose(np.vstack([Xvalues, np.ones(len(Xvalues))]))
slope,intercept = np.linalg.lstsq(A,Yvalues)[0]
Rsquared=np.corrcoef(Xvalues,Yvalues)[0,0]

Rsquared="R = " + str(round(Rsquared,3))

bpy.ops.mesh.primitive_cube_add(location=(0.5,(slope*0.5)+intercept,0))
bpy.data.objects["Cube.002"].scale=(0.01,0.6,0.01)
bpy.data.objects["Cube.002"].rotation_euler=(0,0,-np.arctan(slope))
currentMaterial = bpy.data.materials.new(name='TickMaterial'+'Cube.002')
currentMaterial.diffuse_color=(0.75,0.75,0.75)
bpy.data.objects['Cube.002'].data.materials.append(currentMaterial)

InsertAndChangeText((0.75,0.05,0.1),Rsquared)

bpy.data.objects['Text.006'].scale=(0.15,0.15,0.15)
currentMaterial = bpy.data.materials.new(name='TickMaterial'+str(k))
currentMaterial.diffuse_color=(0.75,0.75,0.75)
bpy.data.objects['Text.006'].data.materials.append(currentMaterial)
