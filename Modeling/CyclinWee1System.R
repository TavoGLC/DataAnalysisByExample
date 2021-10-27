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

library(deSolve)
library(ggplot2)
library(tidygraph)
library(ggraph)

###############################################################################
#Solver function
###############################################################################

solveModel<- function(Model,InitialConditions,ModelParameters,ColumnNames){
  #Solves numerically an ODE system model,returns a formated dataframe
  #Model             -> function, Model to be solved
  #InitialConditions -> list, Initial conditions for the ODE system 
  #ModelParameters   -> list, Parameters of the ODE model 
  #ColumnNames       -> list, names of the columns for the dataframe 
  #MinMax            -> bool, controlls if a minmax normalization is applied to the data. 
  times <- seq(0, 100, by = 0.1)
  out <- ode(InitialConditions,times,Model,ModelParameters,method="rk4")
  ModelData<-data.frame(out)
  colnames(ModelData)<-ColumnNames
  ModelData
}

###############################################################################
# Plot Functions
###############################################################################

SignalingPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=Cdc2,color="Cdc2"))+geom_line()+
    geom_line(aes(y=Wee1,color="Wee1"))+
    scale_color_manual(values=c("Cdc2"="black","Wee1"="red"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}

PhasePlanePlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Cdc2,y=Wee1,color="Cdc2"))+geom_point()+
    scale_color_manual(values=c("Cdc2"="black"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}


###############################################################################
#Model CyclinWee1System 
###############################################################################

CyclinWee1System <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> Cdc2
  #          Y[2]     -> Wee1
  #
  #params -> Parameters of the ODE system model 
  #         
  #    
  
  with(as.list(c(Y,params)),{
    
    dx1dt = a1*(1-Y[1]) - (b1*Y[1]*(v*Y[2])**g1)/(K1+(v*Y[2])**g1)
    dy1dt = a2*(1-Y[2]) - (b2*Y[2]*Y[1]**g2)/(K2+Y[1]**g1)
    
    list(c(dx1dt,dy1dt))
  })
}

params <- c(a1=1,b1=200,g1=4,K1=30,v=1.9,a2=1,b2=10,g2=4,K2=1)
Y <- c(0.5,0.51)

columnNames<-c("Time","Cdc2","Wee1")
CyclinWee1SystemData<-solveModel(CyclinWee1System,Y,params,columnNames)
SignalingPlot(CyclinWee1SystemData,"Signaling (High Hill B)")
PhasePlanePlot(CyclinWee1SystemData,"Phase plane (High Hill B)")
  