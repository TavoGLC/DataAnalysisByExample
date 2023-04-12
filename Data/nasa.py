#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 01:40:41 2023

@author: tavo
"""
###############################################################################
# Loading packages 
###############################################################################

import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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

def MakeMap(data,ax):

    m = Basemap(projection='cyl',llcrnrlat=-65, urcrnrlat=85,
            llcrnrlon=-180, urcrnrlon=180,ax=ax)
    m.drawcoastlines(color='gray')
    m.drawcountries(color='gray')
    m.imshow(np.flip(data[5:155,:],axis=0),interpolation='gaussian')

###############################################################################
# Loading packages 
###############################################################################

def ProcessVariable(dset,name):
    
    container_x = []
    container_y = []
    
    for k,val in enumerate(dset):
        
        locald = np.array(val.variables[name])
        locations = np.argwhere(locald!=-9999)
        finaldata = np.hstack((locations,(k*np.ones((locations.shape[0],1)))))
        container_x.append(finaldata)
        container_y.append(locald[locald!=-9999])
        
    container_x = np.vstack(container_x)
    container_y = np.hstack(container_y)
    
    regr = RandomForestRegressor(n_estimators=4,max_depth=32,
                                 n_jobs=-3,
                                 random_state=0)
    regr.fit(container_x, container_y)
    
    grid = np.argwhere(np.zeros((180,360))==0)
    outdata = np.hstack((grid,(k*np.ones((grid.shape[0],1)))))
    ydata = regr.predict(outdata)
    
    ydata = ydata.reshape(180,360)
    lastdata = np.array(dset[-1].variables[name])
    locs = np.argwhere(lastdata==-9999)
    
    for sal in locs:    
       i,j = sal
       lastdata[i,j] = ydata[i,j]
        
    return (ydata+lastdata)/2

def ProcessData(dataList):
    
    dtacontainer = []
    
    dtacontainer.append(ProcessVariable(dataList,'SurfPres_Forecast_TqJ_A'))
    dtacontainer.append(ProcessVariable(dataList,'SurfAirTemp_TqJ_A'))
    dtacontainer.append(ProcessVariable(dataList,'RelHumSurf_TqJ_A'))
    dtacontainer.append(ProcessVariable(dataList,'CloudFrc_TqJ_A'))
    dtacontainer.append(ProcessVariable(dataList,'TotO3_TqJ_A'))
    dtacontainer.append(ProcessVariable(dataList,'ClrOLR_TqJ_A'))
        
    dtacontainer.append(np.divide(dtacontainer[1],dtacontainer[0]))
    
    return dtacontainer

###############################################################################
# Loading packages 
###############################################################################

dpi = 300

fig_size = (15,10)

img_dirs = ['/media/tavo/storage/dtaimg/0'+str(k+1) for k in range(7)]

dataorg = '/media/tavo/storage/dta_ts/'
datadir = dataorg + 'files/'

basedir = '/media/tavo/storage/dta/'
files = np.sort(os.listdir(basedir))

filePaths = [basedir+val for val in files]
dates = [fls[29:39].replace('.','/') for fls in filePaths]

filesDF = pd.DataFrame()

filesDF['date'] = pd.to_datetime(dates)
filesDF['paths'] = filePaths

idx = pd.period_range(filesDF['date'].min(), filesDF['date'].max())

finaldates = []
finalpaths = []
ii=0

for val in idx:
    if str(val)==str(filesDF['date'].iloc[ii])[0:10]:
        finaldates.append(str(val))
        finalpaths.append(filesDF['paths'].iloc[ii])
        ii=ii+1
    else:
        finaldates.append(str(val))
        finalpaths.append(filesDF['paths'].iloc[ii])
    
finalPathsDF = pd.DataFrame()
finalPathsDF['date'] = pd.to_datetime(finaldates)
finalPathsDF['paths'] = finalpaths

window_size = 90

window = []
for k in range(window_size):
    
    dta = Dataset(filePaths[k])
    window.append(dta)

###############################################################################
# Loading packages 
###############################################################################

filenames = []
dates = []

for k in range(window_size,finalPathsDF.shape[0]):
    
    window = []
    window_paths = finalPathsDF[k-window_size:k]
    
    for pt in window_paths['paths']:    
        dta = Dataset(pt)
        window.append(dta)
    
    localData = ProcessData(window)
    
    saveData = np.stack(localData)
    name = 'rollingmean_'+str(window_size)+'_'+ str(window_paths['date'].iloc[-1])
    np.save(datadir+name,saveData)
    filenames.append(name)
    
    for ii,sal in enumerate(img_dirs):
        
        plt.figure(figsize=fig_size)
        ax = plt.gca()
        MakeMap(localData[ii],ax)
        ax.margins(x=0)
        plt.box(False)
        plt.savefig(sal+'/img'+str(k-window_size)+'.png',dpi=dpi,bbox_inches='tight',pad_inches=0)
        plt.close()
        
    print(k)

finalPathsDF.to_csv(dataorg+'MetaData.csv')
