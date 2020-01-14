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
from bpy import context

######################################################################
#                Loading the data
######################################################################

Dir='pdbFile Location'

######################################################################
#                Functions
######################################################################

def GetPDBFile(pdbFileDir):
    
    """
    Opens a pdb file and returns each line in the file as text in a list

    pdbFileDir : Location of the file 
    """

    cDir = pdbFileDir
    
    with open(cDir) as file:
        
        Data=file.read().lower()
        Lines=Data.splitlines()
        
    return Lines

def GetAlphaAtomCoordinates(pdbData):

    """
    Returns a list with the coordinates of the alpha carbons of a protein

    pdbData : Output of the function GetPDBLines
    """
    
    nData=len(pdbData)
    container=[]
    
    for k in range(nData):
        splitLine=pdbData[k].split()
        
        if splitLine[0]=='atom' and splitLine[2]=='ca':  #Selects the alpha carbons in a protein 
            coordinates=splitLine[6:9]
            container.append([float(val) for val in coordinates])
    
    #Center of mas calculation to center the protein to (0,0,0)
    nAlpha=len(container)
    mX=sum([val[0] for val in container])/nAlpha
    mY=sum([val[1] for val in container])/nAlpha
    mZ=sum([val[2] for val in container])/nAlpha
    
    center=[mX,mY,mZ]
    centerCoordinates=[]
    
    #Centering the protein. 
    for coord in container:
        
        centered=[a-b for a,b in zip(coord,center)]
        centerCoordinates.append(centered)
    
    return centerCoordinates

pdbData=GetPDBFile(Dir)
AlphaCarbonCoordinates=GetAlphaAtomCoordinates(pdbData)

######################################################################
#                Rigging the protein. 
######################################################################

#Number of alpha carbons in the protein 
nAlphas=len(AlphaCarbonCoordinates)

#Adding the first bone
tailLocation=tuple(AlphaCarbonCoordinates[0])
bpy.ops.object.armature_add(enter_editmode=True,location=tailLocation)

#Extruding the remaining bones, one for each alpha carbon
for k in range(1,3):
    headLocation=tuple([a-b for a,b in zip(AlphaCarbonCoordinates[k],AlphaCarbonCoordinates[k-1])]) #Coordinates correction
    bpy.ops.armature.extrude_move(TRANSFORM_OT_translate={"value":headLocation})
