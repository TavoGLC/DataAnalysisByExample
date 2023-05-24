#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 01:29:29 2023

@author: tavo
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

imdir = '/home/tavo/Documentos/cats/'
files = os.listdir(imdir)
files = [val for val in files if os.path.isfile(imdir+val)]

for val in files:
    
    fig,axs = plt.subplots(1,2,figsize=(20,10))
    
    img = cv2.imread(imdir+val)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 11)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19,19)
    
    axs[0].imshow(img)
    axs[0].axis("off")
    
    axs[1].imshow(edges,cmap="gray")
    axs[1].axis("off")

coloring = '/home/tavo/Documentos/cats/coloring/'

for val in files:
    
    img = cv2.imread(imdir+val)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 11)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19,19)
    
    plt.figure(figsize=(20,20))
    plt.imshow(edges,cmap="gray")
    plt.axis("off")
    plt.savefig(coloring+val,dpi=150)
    plt.close()
