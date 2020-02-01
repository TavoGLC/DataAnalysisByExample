"""

MIT License

Copyright (c) 2019 Octavio Gonzalez-Lugo 

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
#                Animating the protein.
######################################################################

import bpy 

######################################################################
#                 Intended Use
######################################################################

"""
The following code is intended to be used at the start of the program 
before making any changes in the scene. 
"""

######################################################################
#                 Data Preprocessing 
######################################################################


barHeight=[0.25,0.74,0.51,0.89,0.71]
barLabels=['October','November','December','January','February']
tickLabels=['0.0','0.25','0.5','0.75','1.0']

nbars=len(barHeight)
nticks=len(tickLabels)

BarNames=['Cube']
for k in range(1,nbars):
    BarNames.append('Cube'+'.00'+str(k))

LabelNames=['Text']
for k in range(1,nbars):
    LabelNames.append('Text'+'.00'+str(k))

TickNames=[]

for k in range(nbars,nbars+nticks):
    TickNames.append('Text.00'+str(k))

step=7/(nbars-1)
tickStep=4.5/(len(tickLabels)-1)

barLocation=[-3.5+(k*step) for k in range(nbars)]
tickLocation=[2.25-(k*tickStep) for k in range(nticks)]

######################################################################
#                Removing the cube from the new file.
######################################################################

bpy.data.objects['Cube'].select=True
bpy.ops.object.delete()

######################################################################
#                Moving the camera 
######################################################################

bpy.data.objects['Camera'].location = (-0.2,-4.0,12)
bpy.data.objects['Camera'].rotation_euler=(-0.0349066,-0.279253,1.5708)

######################################################################
#                Adding the grid 
######################################################################

bpy.ops.mesh.primitive_grid_add(radius=1,location=(0,0,0))
bpy.data.objects['Grid'].scale=(2.25,4.25,1)
bpy.data.objects['Grid'].modifiers.new('Wireframe',type='WIREFRAME')

######################################################################
#                Adding the bars 
######################################################################

for k in range(nbars):
    currentBarHeight = 2.25*barHeight[k]
    currentBarLocation=2.25-currentBarHeight
    bpy.ops.mesh.primitive_cube_add(radius=1,location=(currentBarLocation,barLocation[k],0.5))
    bpy.data.objects[BarNames[k]].scale=(currentBarHeight,0.25,0.25)

######################################################################
#                Adding the bar labels
######################################################################

for k in range(nbars):
    currentName=barLabels[k]
    bpy.ops.object.text_add(location=(2.8,barLocation[k]-0.25,0.5),enter_editmode=True)
    for j in range(4):
        bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
    for char in currentName:
        bpy.ops.font.text_insert(text=char)
    bpy.ops.object.editmode_toggle()

for k in range(nbars):
    currentLabelName=LabelNames[k]
    bpy.data.objects[currentLabelName].rotation_euler=(0,0,1.5708)
    bpy.data.objects[currentLabelName].scale=(0.34,0.34,0.34)

######################################################################
#                Adding the tick labels
######################################################################

for k in range(nticks):
    currentName=tickLabels[k]
    bpy.ops.object.text_add(location=(tickLocation[k],-4.9,0.5),enter_editmode=True)
    for j in range(4):
        bpy.ops.font.delete(type='PREVIOUS_OR_SELECTION')
    for char in currentName:
        bpy.ops.font.text_insert(text=char)
    bpy.ops.object.editmode_toggle()

for k in range(nticks):
    currentLabelName=TickNames[k]
    bpy.data.objects[currentLabelName].rotation_euler=(0,0,1.5708)
    bpy.data.objects[currentLabelName].scale=(0.34,0.34,0.34)

######################################################################
#                editing the plot
######################################################################

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.world.horizon_color = (1,1,1)

#Adding materials to the ticks
for k in range(nticks):

    currentMaterial = bpy.data.materials.new(name='TickMaterial'+str(k))
    currentMaterial.diffuse_color=(0,0,0)
    bpy.data.objects[TickNames[k]].data.materials.append(currentMaterial)

#Adding materials to the labels
for k in range(nbars):

    currentMaterial = bpy.data.materials.new(name='LabelMaterial'+str(k))
    currentMaterial.diffuse_color=(0,0,0)
    bpy.data.objects[LabelNames[k]].data.materials.append(currentMaterial)

#Adding materials to the bars
for k in range(nbars):

    currentMaterial = bpy.data.materials.new(name='Glass BSDF')
    currentMaterial.diffuse_color=(0.2,k/10,k/5+0.1)
    bpy.data.objects[BarNames[k]].data.materials.append(currentMaterial)
