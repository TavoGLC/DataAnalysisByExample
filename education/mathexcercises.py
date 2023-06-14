#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 21:12:31 2023

@author: tavo
"""

import numpy as np 
import matplotlib.pyplot as plt

###############################################################################
# Loading packages 
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

def BottomStyle(Axes): 
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
    Axes.spines['left'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.set_yticks([])

###############################################################################
# Loading packages 
###############################################################################

numbers = np.arange(1,10)

plt.figure(figsize=(8.27,11.7))

ax = plt.gca()

for k in range(19):
    for j in range(4):
        string = str(np.random.choice(numbers)) + ' + '  + str(np.random.choice(numbers)) + ' ='
        ax.text(0.05 + (j*0.25) ,0.95-(k*0.05),string,fontsize=18)
    
ImageStyle(ax)
plt.tight_layout()
plt.savefig("foo.png")

###############################################################################
# Loading packages 
###############################################################################

numbers = np.arange(1,10)
operations = ['+','-','*']
plt.figure(figsize=(8.27,11.7))

ax = plt.gca()

for k in range(19):
    for j in range(4):
        string = str(np.random.choice(numbers)) + ' ' + np.random.choice(operations) + ' ' + str(np.random.choice(numbers)) + ' ='
        ax.text(0.05 + (j*0.25) ,0.95-(k*0.05),string,fontsize=18)
    
ImageStyle(ax)
plt.tight_layout()
plt.savefig("foo2.png")

###############################################################################
# Loading packages 
###############################################################################

numbers = np.arange(1,10)
operations = ['+','-']

fig,axs = plt.subplots(10,1,figsize=(8.27,11.7))

for k,val in enumerate(axs):
    
    val.plot()
    first = np.random.choice(numbers)
    second = np.random.choice(numbers)
    operation = np.random.choice(operations)
    
    string = str(first) + ' ' + operation + ' ' + str(second) + ' ='
    
    if operation=='-':
        result = first - second
    else:
        result = first + second
    
    minv = min([result,first,second]) - 4
    maxv = max([result,first,second]) + 4
    
    rangeval = np.arange(minv,maxv)
    
    val.set_ylim([0,1])
    val.set_xlim([minv,maxv])
    val.set_xticks(rangeval)

    val.text(minv,0.75,string,fontsize=18)
    BottomStyle(val)
    
plt.tight_layout()
plt.savefig("foo3.png")


