#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 03:23:52 2023

@author: tavo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

###############################################################################
# Visualization functions
###############################################################################
fontsize = 16

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
    Axes.xaxis.set_tick_params(labelsize=fontsize)
    Axes.yaxis.set_tick_params(labelsize=fontsize)

###############################################################################
# Visualization functions
###############################################################################

def MaxPE(ytrue,ypred):
    cont = []
    for val,sal in zip(ytrue,ypred):
        cont.append(100*(val-sal)/val)
    return np.max(cont)

###############################################################################
# Visualization functions
###############################################################################
    
wldata = pd.read_csv('/media/tavo/storage/sunspots/solarcurrent.csv')
wldata = wldata[wldata['irradiance']>0]

wbins = [200,290,320,400,700,1000,2500]
wlnames = ['UVA','UVB','UVC','Vis','NIR','SWIR']

wdata = wldata.groupby(['date',pd.cut(wldata['wavelength'],wbins)])['irradiance'].mean().unstack()
wdata.columns = wlnames

wdata = wdata.reset_index()
wdata['date'] = pd.to_datetime(wdata['date'])

for val in wlnames:
    
    mean = wdata[val].mean()
    std = wdata[val].std()
    
    wdata[val] = [sal if np.abs((sal-mean)/std)<3 else mean for sal in wdata[val]]

wdata['dayofyear'] = wdata['date'].dt.dayofyear
wdata['week'] = wdata['date'].dt.isocalendar().week
wdata['dayofweek'] = wdata['date'].dt.dayofweek
wdata['year'] = wdata['date'].dt.year
wdata['month'] = wdata['date'].dt.month
wdata['quarter'] = wdata['date'].dt.quarter
wdata['dim'] = wdata['date'].dt.daysinmonth
wdata['weekend'] = [1 if val in [5,6] else 0 for val in wdata['week']]

wdata = wdata.dropna()

xlabs = ['dayofyear','week','dayofweek','year','month','quarter','dim','weekend']

for lab in xlabs:
    fig,axs = plt.subplots(1,len(wlnames),figsize=(20,7),sharey=True)
    grouped = wdata.groupby(lab)
    for jj,wl in enumerate(wlnames):
        dta = grouped[wl].mean()
        dta = (dta - dta.min())/(dta.max() - dta.min()) + 0.01
        if lab in xlabs[0:2]:
            dta.plot(ax=axs[jj])
        else:
            dta.plot.bar(ax=axs[jj])
        axs[jj].set_ylabel('Normalized Irradiance',fontsize=fontsize)
        axs[jj].text(0.01, 0.99, wl, size=16, color='black', ha='left', va='top', transform=axs[jj].transAxes)
        PlotStyle(axs[jj])

###############################################################################
# Visualization functions
###############################################################################

window = 7
xdata = wdata[xlabs].values
ydata = wdata[wlnames].values

xcont = []
ycont = []

for k in range(len(xdata)-window):
    
    xcont.append(xdata[k:k+window].ravel())
    ycont.append(ydata[k+window])
    
xcont = np.array(xcont)
ycont = np.array(ycont)

regr = RandomForestRegressor(n_estimators=300,random_state=354)
regr.fit(xcont,ycont)

predvals = regr.predict(xcont)

errors = [MaxPE(ycont[:,k],predvals[:,k]) for k in range(len(wlnames))]

plt.figure()
plt.bar(np.arange(len(wlnames)),errors)
plt.xticks(np.arange(len(wlnames)),wlnames)
plt.ylabel('Max % Error',fontsize=fontsize)
ax = plt.gca()
PlotStyle(ax)

###############################################################################
# Visualization functions
###############################################################################

fdata = pd.DataFrame()
fdata['date'] = pd.date_range(wdata['date'].min(), wdata['date'].max())
fdata['dayofyear'] = fdata['date'].dt.dayofyear
fdata['week'] = fdata['date'].dt.isocalendar().week
fdata['dayofweek'] = fdata['date'].dt.dayofweek
fdata['year'] = fdata['date'].dt.year
fdata['month'] = fdata['date'].dt.month
fdata['quarter'] = fdata['date'].dt.quarter
fdata['dim'] = fdata['date'].dt.daysinmonth
fdata['weekend'] = [1 if val in [5,6] else 0 for val in fdata['week']]

xfdata = fdata[xlabs].values
xfcont = []

for k in range(len(xfdata)-window):
    
    xfcont.append(xfdata[k:k+window].ravel())

xfcont = np.array(xfcont)
predvalsf = regr.predict(xfcont)

###############################################################################
# Visualization functions
###############################################################################

outdata = pd.DataFrame()
outdata['date'] = pd.date_range(wdata['date'].min()+pd.DateOffset(days=window), wdata['date'].max())
outdata['dayofyear'] = outdata['date'].dt.dayofyear
outdata['week'] = outdata['date'].dt.isocalendar().week
outdata['dayofweek'] = outdata['date'].dt.dayofweek
outdata['year'] = outdata['date'].dt.year
outdata['month'] = outdata['date'].dt.month
outdata['quarter'] = outdata['date'].dt.quarter
outdata['dim'] = outdata['date'].dt.daysinmonth
outdata['weekend'] = [1 if val in [5,6] else 0 for val in outdata['week']]


for k in range(len(wlnames)):
    outdata[wlnames[k]] = predvalsf[:,k]

for lab in xlabs:
    fig,axs = plt.subplots(1,len(wlnames),figsize=(20,7),sharey=True)
    grouped = outdata.groupby(lab)
    for jj,wl in enumerate(wlnames):
        dta = grouped[wl].mean()
        dta = (dta - dta.min())/(dta.max() - dta.min()) + 0.01
        if lab in xlabs[0:2]:
            dta.plot(ax=axs[jj])
        else:
            dta.plot.bar(ax=axs[jj])
        axs[jj].set_ylabel('Normalized Irradiance',fontsize=fontsize)
        axs[jj].text(0.01, 0.99, wl, size=16, color='black', ha='left', va='top', transform=axs[jj].transAxes)
        PlotStyle(axs[jj])

outdata = outdata.set_index('date')
outdata = outdata[wlnames]

outdata.to_csv('/media/tavo/storage/sunspots/solarinterpolated.csv')
