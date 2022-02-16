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

import numpy as np
import pandas as pd
import requests as req
import seaborn as sns 
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup

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
    Axes.xaxis.set_tick_params(labelsize=12)
    Axes.yaxis.set_tick_params(labelsize=12)

###############################################################################
# Web scrapping
###############################################################################

#Wrapper function to acces the html data 
def GetPageData(PageUrl):

    headers = {'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"}
    resp = req.get(PageUrl,headers=headers)
    soup = BeautifulSoup(resp.text, 'lxml')
    
    return soup

def FormatTableData(PageData):
    '''
    Formats the table data inside an HTML page into a 
    numpy 2D array
    Parameters
    ----------
    PageData : bs4.BeautifulSoup
        Parsed HTML data.

    Returns
    -------
    headers : list
        list of strings with the table headers.
    tableData : array
        Data inside the table without the headers.

    '''
    
    headers = [val.text for val in PageData.find_all('th')]
    
    tableRows = PageData.find_all('tr')
    tableData = []
    
    for row in tableRows[1:len(tableRows)]:
        
        rowData = [val.text for val in row.find_all('td')]
        tableData.append(rowData)
        
    tableData = np.array(tableData)
    
    #Index of duplicated headers
    duplicates = [k for k,x in enumerate(headers) if headers.count(x) >= 2]
    
    #Check for duplicates in the headers
    if len(duplicates)>0:
        for val in duplicates:
            headers[val] = headers[val] + str(val)
    
    return headers, tableData

pageData = GetPageData('https://www.unitstatistics.com/age-of-mythology/')
headers,tableData = FormatTableData(pageData)

###############################################################################
# Data cleaning
###############################################################################

DataFrame = pd.DataFrame()

Numeric = [3,4,5,6,7,8,9,10,11,12,13,14,15,16]

for k,val in enumerate(headers):
    
    currentData = tableData[:,k]
    
    if k in Numeric:
        container = []
        for sal in currentData:
            if sal == '\xa0':
                container.append(0)
            else:
                container.append(float(''.join(e for e in sal if e!='%')))
    else:
        container = ['None' if xal =='\xa0' else xal for xal in currentData]
        
    
    DataFrame[val] = container

ToFullName = {'A':'Atlantean','G':'Greek','E':'Egyptian','N':'Nors'}

DataFrame['Civ'] = [ToFullName[val] for val in DataFrame['Civ']]

container = []
for val in DataFrame['Range']:
    
    if val=='None':
        container.append(0)
    else:
        disc = val.find('-')
        if disc==-1:
            container.append(int(val))
        else:
            container.append(int(val[disc+1:len(val)]))

DataFrame['Range'] = container 

###############################################################################
# Data visualization
###############################################################################

wrapperDF = DataFrame.groupby(['Civ','Type2'])['Unit'].count().to_frame().reset_index().pivot(index='Civ',columns='Type2',values='Unit')
plt.figure()
wrapperDF.T.plot(kind='bar')
ax = plt.gca()
PlotStyle(ax)

wrapperDF = DataFrame.groupby(['Civ','Type19'])['Unit'].count().to_frame().reset_index().pivot(index='Civ',columns='Type19',values='Unit')
plt.figure()
wrapperDF.T.plot(kind='bar')
ax = plt.gca()
PlotStyle(ax)

Numeric.append(20)

for val in Numeric:
    plt.figure()
    sns.barplot(data=DataFrame,x='Type2',y=headers[val],hue='Civ')
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xticks(rotation=45)
    ax = plt.gca()
    PlotStyle(ax)
    