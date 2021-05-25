#MIT License
#Copyright (c) 2020 Octavio Gonzalez-Lugo 

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#@author: Octavio Gonzalez-Lugo

###############################################################################
# Loading packages 
###############################################################################

library(ggplot2)
library(tidygraph)
library(ggraph)
library(pracma)

###############################################################################
# Utility Functions
###############################################################################

MinMaxNormalization<-function(ColumnData){
  #Performs min-max normalization to the data. 
  MinVal=min(ColumnData)
  MaxVal=max(ColumnData)
  DataRange=MaxVal-MinVal
  nData=length(ColumnData)
  container=rep(0,nData)
  
  for( k in 1:nData){
    container[k]=(ColumnData[k]-MinVal)/DataRange
  }
  container
}

###############################################################################
# Plotting functions
###############################################################################

MakePlateModelPlot<-function(inputData,ColumNames,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-data.frame(inputData)
  colnames(ModelData)<-ColumnNames
  
  modelPlot<-ggplot(data=ModelData,aes(x=Time,y=TwoPlates,color="TwoPlates"))+geom_line()+
    geom_line(aes(y=FivePlates,color="FivePlates"))+
    geom_line(aes(y=TenPlates,color="TenPlates"))+
    geom_line(aes(y=TwentyPlates,color="TwentyPlates"))+
    geom_line(aes(y=TwentyFivePlates,color="TwentyFivePlates"))+
    geom_line(aes(y=FiftyPlates,color="FiftyPlates"))+
    scale_color_manual(values=c("TwoPlates"="black","FivePlates"="red","TenPlates"="blue","TwentyPlates"="green","TwentyFivePlates"="orange","FiftyPlates"="brown"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

MakeEDMModelPlot<-function(inputData,ColumNames,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-data.frame(inputData)
  colnames(ModelData)<-ColumnNames
  
  modelPlot<-ggplot(data=ModelData,aes(x=Time,y=Start,color="Start"))+geom_line()+
    geom_line(aes(y=Quarter,color="Quarter"))+
    geom_line(aes(y=Middle,color="Middle"))+
    geom_line(aes(y=End,color="End"))+
    scale_color_manual(values=c("Start"="black","Quarter"="red","Middle"="blue","End"="green"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

###############################################################################
# Plate model
###############################################################################

PlateFilter<-function(n,t,tau,tr){
  # Analytical solution of the stirred tank in series model, taken from 
  #Plate models in chromatography: Analysis and implications for scale up
  # A. Velayudhan and M.R Ladisch
  #Advances in biochemical engineering 1993 
  #n   -> number of theoretical plates
  #t   -> time
  #tau -> retention time 
  #tr  -> injection time 
  container<-rep(0,length(t))
  for (k in 1:length(t)){
    container[k]<-gammainc(n*t[k]/tr,n)[3]-gammainc(n*((t[k]-tau)/tr),n)[3]
  }
  container
}

GetPlateModelData<-function(ns,endTime,Tau,tr){
  #n   -> number of theoretical plates
  #t   -> time
  #tau -> retention time 
  #tr  -> injection time 
  
  time<-seq(0,endTime,by=0.01)
  filterPlateData<-matrix(nrow=length(time),ncol=length(ns)+1)
  filterPlateData[,1]<-time
  
  for (k in 1:length(ns)){
    normData<-PlateFilter(ns[k],time,Tau,tr)
    #normData<-MinMaxNormalization(normData)
    filterPlateData[,k+1]<-normData
  }
  filterPlateData
}

PlateNumbers<-c(2,5,10,20,25,50)
ColumnNames<-c("Time","TwoPlates","FivePlates","TenPlates","TwentyPlates","TwentyFivePlates","FiftyPlates")
PlateData<-GetPlateModelData(PlateNumbers,25,1.2,10)
MakePlateModelPlot(PlateData,ColumnNames,"Plate Model")

###############################################################################
# EDM model   
###############################################################################

EDM<-function(t,x,L,Flow,ep,Dapp,K,Area){
  #Analysis and Numerical Investigation of Dynamic Models for Liquid Chromatography
  # MS Math. Shumaila Javeed
  #t   -> time
  #L   -> length of the filter
  #Flow  -> volumetric flow trough the filter 
  #ep  -> filter porosity
  #Dapp -> Apparent difussion 
  #K    -> equilibrium constant
  #Area -> cross section area of the filter 
  
  Fr<-(1-ep)/ep
  u=Flow/(ep*Area)
  Pec<-L*u/Dapp
  a1<-sqrt(Pec/2)
  a2<-sqrt((L/u)*t*(1+K*Fr))
  a3<-(L/u)*(1+K*Fr)
  A1<-erfc(a1*(a3*x-t)/a2)
  A2<-exp(x*Pec)*erfc(a1*((a3*x+t)/a2))
  
  A3<-1/2*(A1+A2)
  A3
}

edmparts<-function(t,x,L,Flow,ep,Dapp,K,Area,tinj,cinit,cinj){
  #Wrapper function to simulate the filter loading
  #parameters same as EDM
  #tinj -> time of feeding the filter
  #cint -> time of feeding the filter
  #cinj -> time of feeding the filter
  out<-0
  if(t<tinj){
    out<-cinit+(cinj-cinit)*EDM(t,x,L,Flow,ep,Dapp,K,Area)
  }
  else
    out<-cinit+(cinj-cinit)*EDM(t,x,L,Flow,ep,Dapp,K,Area)-cinj*EDM(t-tinj,x,L,Flow,ep,Dapp,K,Area)
}

EDM1<-function(t,x,L,Flow,ep,Dapp,K,Area,tinj,cinit,cinj){
  #Wrapper function to vectorize edmparts 
  #parameters same as edmparts
  container<-rep(0,length(t))
  for (k in 1:length(t)){
    container[k]<-edmparts(t[k],x,L,Flow,ep,Dapp,K,Area,tinj,cinit,cinj)
  }
  container
}

GetEDMModelData<-function(endTime,L,Flow,ep,Dapp,K,Area,tinj,cinit,cinj){
  #Creates a data matrix from the EDM model 
  #parameters same as EDM1
  #returns a data matrix at different locations 
  
  time<-seq(0,endTime,by=0.01)
  zs<-seq(0.25,1,by=0.25)
  
  filterPlateData<-matrix(nrow=length(time),ncol=length(zs)+1)
  filterPlateData[,1]<-time
  
  for (k in 1:length(zs)){
    Data<-EDM1(time,zs[k],L,Flow,ep,Dapp,K,Area,tinj,cinit,cinj)
    filterPlateData[,k+1]<-Data
  }
  filterPlateData
}

modelEDMData<-GetEDMModelData(300,1,0.01,0.41,0.0005,0.005,0.1,5,0,1)
ColumnNames<-c("Time","Start","Quarter","Middle","End")

MakeEDMModelPlot(modelEDMData,ColumnNames,"High affinity")
