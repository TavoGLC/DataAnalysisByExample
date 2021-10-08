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

import re
import time
import numpy as np
import pandas as pd
import requests as r

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

###############################################################################
# Getting the options
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
# Getting the options
###############################################################################

page = r'http://www.economia-sniim.gob.mx/SNIIM-AN/estadisticas/e_fyhAnuarioa.asp?'
headers = {'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"}

resp = r.get(page,headers=headers)
soup = BeautifulSoup(resp.text, 'lxml')

all_options = soup.find_all("option")

#Wrapper function to obtain the different avaliable options
def GetOption(string):
    
    start = string.find('=')
    end = string.find('>')
    
    return string[start+2:end-1]
    
options = [GetOption(str(val)) for val in all_options]
optionsDescriptions = [val.string for val in all_options]

###############################################################################
# formating the options
###############################################################################

marketop = []
marketdes = []
productop = []
productdes = []

for val,sal in zip(options,optionsDescriptions):
    
    try:
        float(val)
        marketop.append(val)
        marketdes.append(sal)
    except ValueError:
        productop.append(val)
        productdes.append(sal)
    
###############################################################################
# Scraping the data 
###############################################################################
    
skeletona = r'http://www.economia-sniim.gob.mx/SNIIM-AN/estadisticas/e_fyhAnuario1a.asp?cent='
skeletonb = '&prod='
skeletonc = '&ACCION=Aceptar'

aa = 'TTSA1'

#Creates a path to download the data
def GetWebData(OptionA,OptionB):
    
    pages = []
    
    for val in OptionA:
        
        url = skeletona + val +skeletonb + OptionB +skeletonc
        resp = r.get(url,headers=headers)
        soup = BeautifulSoup(resp.text, 'lxml')
        pages.append(soup)
        
        time.sleep(5)
    
    pageData = []
    for pag in pages:
        pageData.append(pag.find_all('td'))
        
    return pageData

###############################################################################
# Data Formating 
###############################################################################
  
#Returns all the text element from the html
def FindTextElements(elements):
    
    container=[]
    for val in elements:
        disc = val.text
        if len(disc)==0:
            container.append("Empty")
        else:
            container.append(disc)
            
    return container

#Takes only the price data 
def FormatTextElements(TextElements):
    
    position = [i for i,val in enumerate(TextElements) if val==' $Max'][0]
    remaningElements = TextElements[position::]
    container = []
     
    for k in range(0,len(remaningElements),19):
        container.append(remaningElements[k:k+19])
    
    return container

#Splits the data by year
def GroupByYear(TextData):
    
    container = []
    toTake = []
    currentYear = TextData[0][6]
    
    for val in TextData:
        
        if len(val)>6:
            
            nextYear = val[6]
            
            if currentYear == nextYear:
                toTake.append(val)
            else:
                container.append(toTake)
                currentYear = nextYear
                toTake = []
            
    container.append(toTake)
    
    return container

#Calculates the mean monthly value per year
def MakeYearBlockMean(YearBlock):
    
    container = []
    year = []
    for block in YearBlock:
        
        months = block[7::]
        loopContainer = []
        year.append(block[6])
        for month in months:
            try:
                loopContainer.append(float(month))
            except ValueError:
                loopContainer.append(np.nan)
        container.append(loopContainer)
        
    container = np.array(container)
    container = np.nanmean(container, axis=0)
    
    return year,container

#Wrapper function to iterate through the data from a single page 
def MonthlyPageData(Page):
    
    dta = GroupByYear(FormatTextElements(FindTextElements(Page)))
    monthlyMeans = [MakeYearBlockMean(val) for val in dta]
    
    container = []
    for val in monthlyMeans:
        if len(val[0])!=0:
            container.append([val[0][0],val[1]])
    
    return container[1::]

#Wrapper function to iterate through all the pages 
def FormatData(PagesData):
    
    index = []
    FormatedData = []
    
    for k,val in enumerate(PagesData):
        
        if len(val)>6:
            currentData = MonthlyPageData(val)
            index.append(k)
            FormatedData.append(currentData)
            
    return index,FormatedData
            

###############################################################################
# Data Formating 
###############################################################################

pageData = GetWebData(marketop,aa)
hh = FormatData(pageData)

#Wrapper function to create a data frame per year data
def MakeYearDataFrame(FormatedYearData,ColumnName):
    
    year, data = FormatedYearData
    
    startDate = year + '-01-01'
    endDate = str(int(year)+1) + '-01-01'
    date = pd.date_range(startDate,endDate,freq='BM')
    data = {'date': date, ColumnName: data}
    
    df = pd.DataFrame(data=data)
    
    return df 
    
#Wrapper function to create a data frame per page
def MakePageDataFrame(FormatedPageData,ColumnName):
    
    dfcontainer = []
    
    for val in FormatedPageData:
        
        df = MakeYearDataFrame(val,ColumnName)
        dfcontainer.append(df)
    
    gdf = pd.concat(dfcontainer)
    gdf.set_index("date",inplace=True)
    
    return gdf
  
###############################################################################
# Data merging 
###############################################################################
    
dataframes = [MakePageDataFrame(sal,marketdes[val]) for val,sal in zip(hh[0],hh[1])]    
date = pd.date_range('1998-01-01','2022-01-01',freq='BM')
df = pd.DataFrame({"date":date})

for val in dataframes:
    
    df = df.join(val,on="date")

nanSums = np.argsort(np.array(df.isnull().sum(axis=0)))
names = list(df)

lessNans = [names[val] for val in nanSums[1:11]]

plt.figure(figsize=(9,7))

for i,name in enumerate(lessNans):
    
    plt.plot(df[name],'-',color='grey',alpha=0.5) 
    
plt.plot(df[lessNans].mean(axis=1),'r-',alpha=0.75)    
ax = plt.gca()
PlotStyle(ax)

