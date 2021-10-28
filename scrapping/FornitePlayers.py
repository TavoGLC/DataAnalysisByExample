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
# Importing packages
###############################################################################

import re
import time
import numpy as np
import pandas as pd
import requests as req

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

###############################################################################
# Plot functions
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

###############################################################################
# Scrapping utility functions
###############################################################################

#Wrapper function to acces the html data 
def GetPageData(PageUrl):

    headers = {'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"}
    resp = req.get(PageUrl,headers=headers)
    soup = BeautifulSoup(resp.text, 'lxml')
    
    return soup

###############################################################################
# Scrapping the initial data 
###############################################################################

location = r'https://fortnite-esports.fandom.com/wiki/North_America_West_Players'

IndexData = GetPageData(location)
content = IndexData.find('table')
tables = content.find("div",class_ = "tabs-content")
tableRows = tables.find_all('tr')

Ids = []
Countries = []

for val in tableRows:
    
    localId = val.find("td",class_="field_ID")
    country = val.find("td",class_="field_Country")
    
    idDisc = type(localId) != type(None)
    countryDisc = type(country) != type(None)
    
    if idDisc and countryDisc:
        if country.text == 'Mexico':
            Ids.append(localId.text)
            Countries.append(country.text)

###############################################################################
# Scrapping player data
###############################################################################

Ids2 = [''.join(val.split()) for val in Ids]

skeletonUrl = r'https://fortnite-esports.fandom.com/wiki/'

newPage = skeletonUrl +'Taquito'

def GetGamerData(Url):
    '''
    Scrapes and format all the data from a table 
    Parameters
    ----------
    Url : strinng
        custom url to scrape data.

    Returns
    -------
    formatedData : list
        list with all the data from an html table.

    '''
    
    gamer = GetPageData(Url)
    tournamentData = gamer.find("table", class_ ="wikitable sortable hoverable-rows")
    tableRows = tournamentData.find_all("tr")
    
    tableData =[]
    
    for val in tableRows:
        text = [sal.text for sal in val.find_all("td")]
        tableData.append(text)
        
    formatedData = [val[0:4] for val in tableData if len(val)!=0]
    time.sleep(20)
    
    return formatedData
        
###############################################################################
# checking correct data
###############################################################################

gamersData = []
gamersId = []
for nme in Ids2:
    
    try:    
        dta = GetGamerData(skeletonUrl+nme)
        gamersId.append(nme)
        gamersData.append(dta)
    except AttributeError:
        print(nme)
        

#gamersData = [GetGamerData(skeletonUrl+nme) for nme in Ids2[2::]]
    
###############################################################################
# Making a data frame from the player data
###############################################################################

#Wrapper function to change string to float
def FormatStringToFloat(Array):
    
    return [0 if len(val)==0 else float(val) for val in Array] 

#Wrapper function to change team description to float
def FormatTeam(Array):
    
    container = []
    for val in Array:
        
        if val=="Solo":
            container.append(1)
        elif val=="Duo":
            container.append(2)
        elif val == 'Squad (3)':
            container.append(3)
        elif val == 'Squad (4)':
            container.append(4)
        else:
            container.append(0)
    
    return container
        

def MakeGamerDF(GamerData,GamerId):
    """
    Formats the scraped data per player to a dataframe

    Parameters
    ----------
    GamerData : list
        Scraped table data.
    GamerId : string
        user name of the player.

    Returns
    -------
    localDF : pandas dataframe
        organized player data.

    """
    
    GamerData = np.array(GamerData)
    localDF = pd.DataFrame()
    Index = pd.to_datetime(GamerData[:,0], format='%Y-%m-%d')
    
    localDF['date'] = Index
    localDF['gamer'] = [GamerId for val in GamerData[:,0]]
    localDF['month'] = Index.month
    localDF['dow'] = Index.dayofweek
    localDF['team'] = FormatTeam(GamerData[:,1])
    localDF['ranking'] = GamerData[:,2].astype(float)
    localDF['pr_points'] = FormatStringToFloat(GamerData[:,3])
    localDF.set_index('date',inplace=True)
    
    return localDF

dframes = [MakeGamerDF(val,sal) for val,sal in zip(gamersData,gamersId)]

###############################################################################
# Visualizing whole player data
###############################################################################

#Wrapper function to acces all the data regardless of the player
def GetAllDataByCategory(category,DataFrames):
    
    arrContainer = np.array([])
    for dfms in DataFrames:
        
        arrContainer = np.append(arrContainer,dfms[category])
        
    return arrContainer

Rankings = GetAllDataByCategory("ranking",dframes)

plt.figure(figsize=(10,4))
plt.hist(Rankings,bins=50)
plt.xlabel('Tournament Place')
ax = plt.gca()
PlotStyle(ax)

Points = GetAllDataByCategory("pr_points",dframes)

plt.figure(figsize=(10,4))
plt.hist(Points,bins=50)
plt.xlabel('PR points')
ax = plt.gca()
PlotStyle(ax)

###############################################################################
# Selecting player data by category
###############################################################################

#Wrapper function to group data by player
def GetDataByGroup(Category,Group,DataFrames):
    
    container = []
    
    for dfr in DataFrames:
        
        data = dfr.groupby(Group)[Category].mean()
        Xdata = data.keys().to_list()
        Ydata = data.to_list()
        
        container.append([Xdata,Ydata])
        
    return container
        
###############################################################################
# Visualizing temporal player performance 
###############################################################################

MonthRankings = GetDataByGroup("ranking","month",dframes)
WeekRankings = GetDataByGroup("ranking","dow",dframes)

plt.figure(figsize=(7,7))
for val in MonthRankings:
    
    Xdata,Ydata = val
    plt.plot(Xdata,Ydata,'bo',alpha=0.5)

ax = plt.gca()
ax.set_ylabel("ranking")
ax.set_xticks(np.arange(1,13))
ax.set_xticklabels(["January","February","March","April","May","June","July","August","September","October","November","December"],rotation=80)
PlotStyle(ax)
    
    
plt.figure(figsize=(7,7))
for val in WeekRankings:
    
    Xdata,Ydata = val
    plt.plot(Xdata,Ydata,'bo',alpha=0.5)
    
ax = plt.gca()
ax.set_ylabel("ranking")
ax.set_xticks(np.arange(7))
ax.set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation=80)
PlotStyle(ax)

###############################################################################
# Visualizing temporal player performance 
###############################################################################
   
MonthPR = GetDataByGroup("pr_points","month",dframes)
WeekPR = GetDataByGroup("pr_points","dow",dframes)

plt.figure(figsize=(7,7))
for val in MonthPR:
    
    Xdata,Ydata = val
    plt.plot(Xdata,Ydata,'bo',alpha=0.5)

ax = plt.gca()
ax.set_ylabel("pr_points")
ax.set_xticks(np.arange(1,13))
ax.set_xticklabels(["January","February","March","April","May","June","July","August","September","October","November","December"],rotation=80)
PlotStyle(ax)
    
plt.figure(figsize=(7,7))
for val in WeekPR:
    
    Xdata,Ydata = val
    plt.plot(Xdata,Ydata,'bo',alpha=0.5)    

ax = plt.gca()
ax.set_ylabel("pr_points")
ax.set_xticks(np.arange(7))
ax.set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],rotation=80)
PlotStyle(ax)

###############################################################################
# Visualizing team player performance 
###############################################################################
   
MonthPR = GetDataByGroup("pr_points","team",dframes)

plt.figure(figsize=(7,7))
for val in MonthPR:
    
    Xdata,Ydata = val
    plt.plot(Xdata,Ydata,'bo',alpha=0.5)
    
ax = plt.gca()
ax.set_ylabel("pr_points")
ax.set_xticks(np.arange(1,5))
ax.set_xticklabels(["Solo","Duo","Squad(3)","Squad(4)"],rotation=80)
PlotStyle(ax)
    
    