#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

###############################################################################
# Plotting functions
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Plotting functions
###############################################################################

GlobalDirectory=r"/home/tavoglc/localBuffer/"

###############################################################################
# Plotting functions
###############################################################################

def ImageStyle(Axes): 
    """
    Parameters
    ----------
    Axes : Matplotlib axes object
        Applies a general style to the matplotlib object

    Returns
    -------
    None.
    """    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(False)
    Axes.spines['left'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.set_xticks([])
    Axes.set_yticks([])

###############################################################################
# Plotting functions
###############################################################################

def MakeMaldebrontFractal(gridSize,maxIter):
    '''
    Parameters
    ----------
    gridSize : int
        number of divisions on the grid.
    maxIter : int
        max number of iterations taken to determine if a given point belogs or
        not to the Maldebront set.

    Returns
    -------
    Ns : 2D array
        relative colouring of the fractal.

    '''

    M=np.zeros((gridSize,gridSize))
    Ns=np.zeros((gridSize,gridSize))
    xvals=np.linspace(0.25,.5,gridSize)
    yvals=np.linspace(0.25,.5,gridSize)

    for u,x in enumerate(xvals):
        for v,y in enumerate(yvals):
            z=0
            c=complex(x,y)
            for i in range(maxIter):
                z=z*z+c
                if abs(z)>2:
                    M[u,v]=1
                    Ns[u,v]=i+1-np.log(np.log2(abs(z)))
                    break
                
    return Ns


def MakeJuliaSet(gridSize,maxIter):
    '''
    Parameters
    ----------
    gridSize : int
        number of divisions on the grid.
    maxIter : int
        max number of iterations taken to determine if a given point belogs or
        not to the Julia set.

    Returns
    -------
    Ns : 2D array
        relative colouring of the fractal.

    '''
        
    c=complex(-0.1,0.65)    
    Julia=np.zeros((gridSize,gridSize))
    Shades=np.zeros((gridSize,gridSize))    
    xvals=np.linspace(-1.5,1.5,gridSize)
    yvals=np.linspace(-1.5,1.5,gridSize)
    
    for k,x in enumerate(xvals):
        for j,y in enumerate(yvals):
            z=complex(x,y)
            for i in range(maxIter):
                z=z**2+c
                if abs(z)>(10):
                    Julia[k,j]=1
                    break
            shade=1-np.sqrt(i/maxIter)
            Shades[k,j]=shade
            
    return Shades
                    
###############################################################################
# Plotting functions
###############################################################################

for k in range(240):
    
    Ns=MakeJuliaSet(2000,k+1)
    imdir=GlobalDirectory+"fig"+str(k)+".png"
    plt.figure(figsize=(10,10))
    plt.imshow(Ns,cmap="gray")
    ax=plt.gca()
    ImageStyle(ax)
    plt.savefig(imdir)
    plt.close()
    






