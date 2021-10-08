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
import requests as req

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

#Wrapper function to acces the html data 
def GetPageData(PageUrl):

    headers = {'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"}
    resp = req.get(PageUrl,headers=headers,verify=False)
    soup = BeautifulSoup(resp.text, 'lxml')
    
    return soup

#Wrapper function to acces and format the scrapped data
def FormatProductData(PageData):
    
    entries = PageData.find_all("div",class_="product details product-item-details")
    
    itemNames = [val.find("a").text for val in entries]
    itemPrices = [val.find('span',class_="price").text for val in entries]
    itemNames = [' '.join(val.split()) for val in itemNames ]
    itemPrices = [float(val[1::]) for val in itemPrices ]
        
    return itemNames,itemPrices

def ListFlatten(ListOfLists):
    return [item for sublist in ListOfLists for item in sublist]

###############################################################################
# Getting the options
###############################################################################

pages = []
pages.append(r'https://tiendasneto.com.mx/productos')
skeleton = r"https://tiendasneto.com.mx/productos?p="

for k in range(2,27):
    pages.append(skeleton+str(k))
    
data = []
for val in pages:
    localData = GetPageData(val)
    time.sleep(20)
    data.append(FormatProductData(localData))
    
###############################################################################
# Getting the options
###############################################################################

Names = ListFlatten([val[0] for val in data])
Prices = ListFlatten([val[1] for val in data])

plt.figure(figsize=(10,5))
plt.hist(Prices,bins=50)
ax=plt.gca()
ax.set_xlabel("Price")
PlotStyle(ax)

###############################################################################
# Getting the options
###############################################################################

#Wrapper function to set the class of a given produt
#Iterates trough a list of classes until it finds the appropiate class
def AssignClass(Product,Classes):
    
    responce = 'None'
    for val in Classes:
        if re.match('(.*)' + val + '(.*)', Product):
            responce = val
            
    return responce


Categories = ['ACEITE','ACONDICIONADOR','AROMATIZANTE','AVENA', 
              'BLANQUEADOR','BOTANA','CEREAL','CERVEZA','DESECHABLE',
              'DESENGRASANTE', 'DESODORANTE', 'DETERGENTE','DULCE',
              'GALLETA','LECHE','MERMELADA','PAN','PAPEL','PERRO',
              'QUESO','REFRESCO','SALSA','YOGHURT']

ScrappedDF = pd.DataFrame()

ScrappedDF["name"] = Names
ScrappedDF["price"] = Prices
ScrappedDF["class"] = [AssignClass(val,Categories) for val in Names]

###############################################################################
# Getting the options
###############################################################################

plt.figure(figsize=(10,5))
ax = plt.gca()
ScrappedDF.boxplot(column = "price",by="class",grid=False,ax=ax)
ax.xaxis.set_tick_params(rotation=85)
ax.set_title('')
fig = ax.get_figure()
fig.suptitle('')
PlotStyle(ax)

###############################################################################
# Getting the options
###############################################################################

PurchaseCategories= ['ACEITE','ACONDICIONADOR','AROMATIZANTE','AVENA',
                     'BLANQUEADOR','BOTANA','CEREAL','DESECHABLE',
                     'DESENGRASANTE', 'DESODORANTE', 'DETERGENTE',
                     'DULCE','GALLETA','LECHE','MERMELADA','PAN',
                     'PAPEL','QUESO','REFRESCO','SALSA','YOGHURT',
                     'CERVEZA']

def MakeRandomPurchase(dataframe,categories):
    '''
    Simulation of a random single item purchase in the categories
    list
    
    Parameters
    ----------
    dataframe : pandas dataframe
        Data frame with the price and class information of the scrapped
        products.
    categories : list
        Contains the different categories for a single item purchase.

    Returns
    -------
    cost : float
        cost of the purchase.

    '''
    
    index = np.arange(dataframe.shape[0])
    np.random.shuffle(index)
    NameToLoc = {nme:k for nme,k in zip(categories,np.arange(len(categories)))}
    counter = np.zeros(len(categories))
    cost = 0
    
    for val in index:
        if np.sum(counter)==len(counter):
            break
        else:
            product = dataframe.iloc[val]
            productClass = product['class']
            InCategories = productClass in categories

            if InCategories:
                location = NameToLoc[productClass]
                if counter[location]==0:
                    cost = cost + product['price']
                    counter[location] = 1
            
    return cost
            
container = []

for k in range(2000):
    
    container.append(MakeRandomPurchase(ScrappedDF,PurchaseCategories))
    
plt.figure(figsize=(10,5))
plt.hist(container,bins=50)
ax = plt.gca()
PlotStyle(ax)
