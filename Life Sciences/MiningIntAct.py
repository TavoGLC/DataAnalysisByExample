# -*- coding: utf-8 -*-
"""

@author: TavoGLC
From:

-Mining biological databases: IntAct-

"""
####################################################################
#                   Importing packages and libraries 
####################################################################

import re
import csv
import time
import numpy as np

from bs4 import BeautifulSoup
from selenium import webdriver

####################################################################
#                            Data Directory 
####################################################################

GlobalDir='Global data directory'
DataDir=GlobalDir+'\\'+'Data'

TargetListDir=GlobalDir+'\\'+'TargetList.csv'

####################################################################
#                   Data saving functions
####################################################################

#Generates a directory to save a file, name will be the Uniprot identifier 
def GenCSVDir(ParentDirectory,Name):
    
    return ParentDirectory+'\\'+Name+'.csv'

#Save the interactions of a unique protein in a csv file 
def SaveCSVFile(Data,FileDirectory):
    
    cData=Data
    cFile=FileDirectory
    
    with open(cFile,'w') as output:
        
        LocalWriter=csv.writer(output)
        LocalWriter.writerow(cData)

####################################################################
#                   Importing packages and libraries 
####################################################################

#Makes an edit to merge the uniprot and EBI identifiers 
def EBIEdit(EbiID):
    
    cID=EbiID
    
    try:
        
        NewEBIID=cID[0:6]+'-'+cID[7:len(cID)]
        
    except IndexError:
        
        NewEBIID='none'
        
    return NewEBIID


#Find the number of pages in the IntAct Table
def FindNumberOfPages(ParsedHTML):
    
    cPage=ParsedHTML
    #Selection of the numeration in the header table 
    cStrings=cPage.find_all('span','ui-paginator-page','ui-corner-all')
    
    try:
        
        #It will work for a table with 10 or less result pages 
        cVal=int(str(cStrings[-1])[-8:-7])
    
        if cVal==0:
        
            nPages=int(str(cStrings[-1])[-9:-7])
    
        else:
        
            nPages=cVal
            
    except IndexError:
        
        
        nPages=0
        
    return nPages

#Returns the interactions of a protein 
def GetDataTable(ParsedHTML,localTarget):
    
    cParsed=ParsedHTML
    #Selects the text entries in the results table 
    cSelection=cParsed.select('.ui-dt-c') 
    Container=[]
    
    for line in cSelection:
        
        #Takes the text inside the table
        Container.append(line.text.strip()) 
    
    kLines=len(Container)
    IntContainer=[]
    
    EBIID='none'
    
    #Takes the text inside the table
    for line in Container:
    
        if re.match(localTarget+'(.*)',line):
        
            EBIID=line
        
        else:
        
            pass
    
    #Takes the text inside the table
    editID=EBIEdit(EBIID)
    
    #Takes the text inside the table
    if editID=='none':
        
        pass
    
    else:
        
        IntContainer.append(editID)
    
    for k in range(kLines):
        
        #Looks for the EBI identifier number 
        if re.match('(.*)EBI-(.*)',Container[k]):   
            
            #Selects only proteins expressed in humans 
            if Container[k+1]=='human (9606)':      
                
                IntContainer.append(Container[k])
                
            else:
                
                pass
            
        else:
            
            pass
        
    return IntContainer

#IntAct scraping function
def MakeEntryCall(UniprotID):
    
    localTarget=UniprotID
    
    #Browser mnipulation 
    localDriver=webdriver.Chrome(executable_path='ChromeDriverPath')
    localDriver.get("https://www.ebi.ac.uk/intact/")
    time.sleep(5)
    
    localDriver.find_element_by_id('queryTxt').send_keys(localTarget)    
    time.sleep(5)

    localDriver.find_element_by_id('quickSearchBtn').click()
    time.sleep(5)

    localDriver.find_element_by_link_text('Interactors').click()
    time.sleep(5)
    
    #Parsing results data 
    CurrentParsed = BeautifulSoup(localDriver.page_source, 'html.parser')
    cPages=FindNumberOfPages(CurrentParsed)
    cData=GetDataTable(CurrentParsed,localTarget)
    
    if cPages==0:
        
        time.sleep(5)
        localDriver.close()
       
    else:
        
        #Loop over the number of results 
        for k in range(cPages-1):
    
            #
            try:
                
                #Global xpath of the next element
                localDriver.find_element_by_xpath('/html/body/div[2]/div/div[2]/form[1]/div/div/div[2]/div[2]/div/div[1]/div[3]/table/thead/tr[1]/th/span[4]/span').click()
                LParsed=BeautifulSoup(localDriver.page_source, 'html.parser')
                lData=GetDataTable(LParsed,localTarget)    
                cData.extend(lData)
                time.sleep(5)
                
            except Exception:
                
                break

        localDriver.close()
        
    return cData

####################################################################
#                   Importing packages and libraries 
####################################################################

#Loading Uniprot identifiers 
targetList=np.genfromtxt(TargetListDir,dtype="|U6")

#Looping through the identifiers adn saving the data 
for target in targetList:

    data=MakeEntryCall(target)
    cDir=GenCSVDir(DataDir,target)
    SaveCSVFile(data,cDir)

