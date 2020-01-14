#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
#                Loading the packages
######################################################################

import bpy
import random 

######################################################################
#                Functions
######################################################################

nalphas=254

def MakeBoneNames(nRes):

    """
    Creates a list with the names of all the bones to be animated.
    nRes : Number of Residues in the protein 
    """

    bonesLocal=[]

    for k in range(nRes-1):

        if k+1<10:
            nameLocal='Bone.00'+str(k+1)
        elif k+1>=10 and k+1<100:
            nameLocal='Bone.0'+str(k+1)
        else:
            nameLocal='Bone.'+str(k+1)
        
        bonesLocal.append(nameLocal)

    return bonesLocal

######################################################################
#                Animating the protein.
######################################################################

boneNames=MakeBoneNames(nAlphas)
rig = bpy.data.objects['Armature'] #Selects the current armature object

#Iterates trough the frames in the animation
for j in range(0,250,14):

    #Iterates trough all the bones in the model 
    for k in range(len(boneNames)):

        cbone=boneNames[k]
        bone = rig.pose.bones[cbone]
        bone.rotation_axis_angle = (0,0, 0, 0) #Original Position
        bone.keyframe_insert('rotation_axis_angle', frame=j)
        bone.rotation_axis_angle = (0.00015,0, 0,random.random()/5) #Random Rotation
        bone.keyframe_insert('rotation_axis_angle', frame=j+7)
        bone.rotation_axis_angle = (0,0, 0, 0) #Return to original position
        bone.keyframe_insert('rotation_axis_angle', frame=j+14)
        bpy.context.object.pose.bones[cbone].rotation_mode = 'AXIS_ANGLE'
