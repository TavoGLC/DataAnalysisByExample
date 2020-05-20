#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 00:11:07 2020

@author: tavoglc
"""

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

###############################################################################
# Loading packages 
###############################################################################

import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import io 
from skimage.filters import threshold_otsu
from skimage.measure import label,regionprops
from skimage.color import label2rgb

###############################################################################
# Loading the data
###############################################################################

GlobalDirectory=r"/home/tavoglc/LocalData/LocalImages"

###############################################################################
# Ploting Functions
###############################################################################

def PlotStyle(Axes): 
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
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=13)
    Axes.yaxis.set_tick_params(labelsize=13)
    
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
    
def DisplayImage(Image,Size):
    """
    Parameters
    ----------
    Image : numpy array
        array with the pixel intensity data.
    Size : TYPE
        size for the image to be displayed.
    Returns
    -------
    None.

    """
    plt.figure(figsize=Size)
    if len(Image.shape)==2:
        plt.imshow(Image,cmap='gray')
    else:
        plt.imshow(Image)
    ax=plt.gca()
    ImageStyle(ax)
    
    
def DisplayImageHistograms(Image):
    """
    Parameters
    ----------
    Image : numpy array
        array with the pixel intensity data.

    Returns
    -------
    None.

    """
    
    fig,axes=plt.subplots(2,2,figsize=(12,12),sharex=True,sharey=True)
    Indexs=[(j,k) for j in range(2) for k in range(2)]
    
    axes[Indexs[0]].hist(np.ravel(Image),bins=50,density=True,color='gray',label='Complete Image')
    axes[Indexs[1]].hist(np.ravel(Image[:,:,0]),bins=50,density=True,color='red',label='Red Channel')
    axes[Indexs[2]].hist(np.ravel(Image[:,:,1]),bins=50,density=True,color='green',label='Green Channel')
    axes[Indexs[3]].hist(np.ravel(Image[:,:,2]),bins=50,density=True,color='blue',label='Blue Channel')
    
    for indx in Indexs:
        axes[indx].legend()
        PlotStyle(axes[indx])
        
def DisplayPropertiesPlot(Property,Title):
    """
    Parameters
    ----------
    Property : list,array
        list with the propertie values.
    Title : str
        Title for the graph, name of the propertie.

    Returns
    -------
    None.

    """
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    
    axes[0].plot(Property)
    axes[1].hist(Property)
    
    for indx in range(2):
        PlotStyle(axes[indx])
        
    fig.suptitle(Title,fontsize=16)
###############################################################################
# Loading the Data
###############################################################################

def GetImagesData(Dir):
    """
    Parameters
    ----------
    Dir : str
        dir withe the images.

    Returns
    -------
    completeDirs : list
        list of dirs od the images.

    """
    
    names=os.listdir(Dir)
    completeDirs=[Dir+'/'+val for val in names]
    
    return completeDirs

Images=GetImagesData(GlobalDirectory)
TestImage=io.imread(Images[19])
CurrentImage=TestImage[500:1500,500:1500,:]

DisplayImage(CurrentImage,(10,10))

DisplayImageHistograms(CurrentImage)

###############################################################################
# Image Thresholding
###############################################################################

def GetThresholdMask(Threshold,Image):
    """
    Parameters
    ----------
    Threshold : int
        Threshold value to be applied.
    Image : array
        array with the pixel intensity data.

    Returns
    -------
    threshImage : array
        mask of the threshold.

    """
    threshImage=np.zeros(Image.shape)
    binary=Image>Threshold
    
    for k in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            if binary[k,j]:
                threshImage[k,j]=1
                
    return threshImage

ThresholdValue=threshold_otsu(CurrentImage[:,:,1])
ThresholdImage=GetThresholdMask(ThresholdValue,CurrentImage[:,:,1])
DisplayImage(ThresholdImage,(10,10))

###############################################################################
# Labeling detections 
###############################################################################

ImageLabels=label(ThresholdImage,connectivity=1)
LabelDisplay=label2rgb(ImageLabels,image=CurrentImage,bg_label=0)
LabelsProps=regionprops(ImageLabels)

ExLabels=[val.eccentricity for val in LabelsProps]
ArLabels=[val.area for val in LabelsProps]

plt.figure(figsize=(7,5))
plt.hist(ExLabels,bins=50)
ax=plt.gca()
PlotStyle(ax)

plt.figure(figsize=(7,5))
plt.hist(ArLabels,bins=100)
ax=plt.gca()
PlotStyle(ax)


plt.figure(figsize=(10,10))
ax=plt.gca()
ax.imshow(LabelDisplay)
for region in LabelsProps:
    if region.eccentricity>=0.8 and region.area>=500 and region.area<5000:
        minr,minc,maxr,maxc=region.bbox
        fi=mpatches.Rectangle((minc,minr),maxc-minc,maxr-minr,fill=False,edgecolor='red',linewidth=1)
        ax.add_patch(fi)
ImageStyle(ax)

###############################################################################
# Detection metrics
###############################################################################

selectedLabels=[region for region in LabelsProps if region.eccentricity>=0.8 and region.area>=100 and region.area<5000]

selectedArea=[val.area for val in selectedLabels]
selectedEx=[val.eccentricity for val in selectedLabels]
selectedPer=[val.perimeter for val in selectedLabels]
selectedOr=[val.orientation for val in selectedLabels]

selectedPAR=[sal/val for sal,val in zip(selectedPer,selectedArea)]

DisplayPropertiesPlot(selectedArea,'Area')

DisplayPropertiesPlot(selectedEx,'Eccentricity')

DisplayPropertiesPlot(selectedPer,"Perimeter")

DisplayPropertiesPlot(selectedOr,"Orientation")

DisplayPropertiesPlot(selectedPAR,"Perimeter/Area")

plt.figure(figsize=(7,5))
plt.plot([val/sal for val,sal in zip(selectedPer,selectedArea)],selectedEx,'bo')
ax=plt.gca()
ax.set_xlabel("Perimeter/Area")
ax.set_ylabel('Eccentricity')
PlotStyle(ax)

###############################################################################
# Detection metrics
###############################################################################

plt.figure(figsize=(10,10))
ax=plt.gca()
ax.imshow(LabelDisplay)
for region in LabelsProps:
    if region.eccentricity>=0.875 and region.area>=500 and region.area<5000 and region.perimeter/region.area<=0.165:
        minr,minc,maxr,maxc=region.bbox
        fi=mpatches.Rectangle((minc,minr),maxc-minc,maxr-minr,fill=False,edgecolor='red',linewidth=1)
        ax.add_patch(fi)
ImageStyle(ax)
