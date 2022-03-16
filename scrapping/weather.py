#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License
Copyright (c) 2022 Octavio Gonzalez-Lugo 

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
# Importing packages
###############################################################################

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from meteostat import Daily
from datetime import datetime
from meteostat import Stations

from math import radians 
from sklearn.neighbors import KNeighborsRegressor

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
# Loading locations data 
###############################################################################

GlobalDirectory=r"/media/tavoglc/Datasets/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
datadir = GlobalDirectory+'/mergedDF.csv'
dst = pd.read_csv(datadir)

uniqueLocations = np.array([list(x) for x in set(tuple(x) for x in np.array(dst[['geo_lat','geo_long','geo_alt']]))])

###############################################################################
# Downloading weather data 
###############################################################################

start = datetime(2021, 1, 1)
end = datetime(2021, 12, 31)

container = []
retrived = []
retrivedIndex = []
notRetrived = []
notRetrivedIndex = []

for k,val in enumerate(uniqueLocations):
    
    stations = Stations()
    stations = stations.nearby(val[0],val[1])
    station = stations.fetch(1)
    
    data = Daily(station, start, end)
    data = data.fetch()
    
    if len(data)>0:
    
        data['week'] = data.index.isocalendar().week
        ndst = data.groupby('week').mean()
        localdf = ndst[['tavg','prcp','pres']]
        disc = localdf.isna().sum().sum()
        
        if disc>40:
            notRetrived.append(val)
            notRetrivedIndex.append(k)
            print('Wait')
            time.sleep(4)
            print('Go')
        else:
            container.append(ndst[['tavg','prcp','pres']])
            retrived.append(val)
            retrivedIndex.append(k)
            print('Wait')
            time.sleep(4)
            print('Go')
            
    else:
        notRetrived.append(val)
        notRetrivedIndex.append(k)

###############################################################################
# Formating the data
###############################################################################

temperature = pd.DataFrame()
for k,val in zip(retrivedIndex,container):
    temperature[str(k)] = val['tavg']
    
temperature = temperature.fillna(temperature.mean())

precipitation = pd.DataFrame()
for k,val in zip(retrivedIndex,container):
    precipitation[str(k)] = val['prcp']
    
precipitation = precipitation.fillna(precipitation.mean())

pressure = pd.DataFrame()
for k,val in zip(retrivedIndex,container):
    pressure[str(k)] = val['pres']
    
pressure = pressure.fillna(pressure.mean())

###############################################################################
# Predicting missing data 
###############################################################################

retrivedLocations = np.array(retrived)[:,0:2]
retrivedLocations = np.array([[radians(val[0]),radians(val[1])] for val in retrivedLocations])

notRetrivedLocations = np.array(notRetrived)[:,0:2]
notRetrivedLocations = np.array([[radians(val[0]),radians(val[1])] for val in notRetrivedLocations])

tempPredictor = KNeighborsRegressor(n_neighbors=5,metric='haversine')
tempPredictor.fit(retrivedLocations,np.array(temperature).T)

prcpPredictor = KNeighborsRegressor(n_neighbors=5,metric='haversine')
prcpPredictor.fit(retrivedLocations,np.array(precipitation).T)

presPredictor = KNeighborsRegressor(n_neighbors=5,metric='haversine')
presPredictor.fit(retrivedLocations,np.array(pressure).T)

predictedTemp = tempPredictor.predict(notRetrivedLocations)
predictedPrcp = prcpPredictor.predict(notRetrivedLocations)
predictedPres = presPredictor.predict(notRetrivedLocations)

###############################################################################
# Adding it to the data frame
###############################################################################

for k,val in enumerate(notRetrivedIndex):
    temperature[str(val)] = predictedTemp[k,:]
    
for k,val in enumerate(notRetrivedIndex):
    precipitation[str(val)] = predictedPrcp[k,:]
    
for k,val in enumerate(notRetrivedIndex):
    pressure[str(val)] = predictedPres[k,:]
    
locationsDF = pd.DataFrame(uniqueLocations,columns=['geo_lat','geo_long','geo_alt'])

temperature.to_csv(GlobalDirectory+'/temperature2021.csv')
precipitation.to_csv(GlobalDirectory+'/precipitation2021.csv')
pressure.to_csv(GlobalDirectory+'/pressure2021.csv')
locationsDF.to_csv(GlobalDirectory+'/locations.csv')

###############################################################################
# Data visualization 
###############################################################################

locationToIndex = dict([(tuple(sal),val) for val,sal in zip(np.arange(uniqueLocations.shape[0]),uniqueLocations[:,0:2])])

plt.figure()
temperature.mean(axis=1).plot()

plt.figure()
precipitation.mean(axis=1).plot()

plt.figure()
pressure.mean(axis=1).plot()

for i in range(53):
    plt.figure()    
    plt.scatter(uniqueLocations[:,1],uniqueLocations[:,0],c=[precipitation[str(k)][i+1] for k in range(pressure.shape[1])],label='week - '+str(i+1))
    plt.ylim([24,50])
    plt.xlim([-130,-65])
    plt.legend(loc=3)
    ax = plt.gca()
    ImageStyle(ax)
    plt.savefig(GlobalDirectory+'/fig'+str(i)+'.png')

