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

ProgesteronePlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=cholesterol,color="cholesterol"))+geom_line()+
    geom_line(aes(y=progesterone,color="progesterone"))+
    scale_color_manual(values=c("cholesterol"="black","progesterone"="red"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}

ProgesteroneIntermediatePlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=cholesterol,color="cholesterol"))+geom_line()+
    geom_line(aes(y=progesterone,color="progesterone"))+
    geom_line(aes(y=pregnenolone,color="pregnenolone"))+
    scale_color_manual(values=c("cholesterol"="black","progesterone"="red","pregnenolone"="blue"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}

###############################################################################
#Model Progesterone
###############################################################################

ProgesteroneLinear <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> Cholesterol 
  #          Y[2]     -> Progesterone
  #
  #params -> Parameters of the ODE system model 
  #          k1       ->synthesis rate
  #          k2       ->decay rate
  #
  
  with(as.list(c(Y,params)),{
    
    dx1dt= -k1*Y[1] + k2*Y[2]
    dx2dt= k1*Y[1] - k2*Y[2]
    
    list(c(dx1dt,dx2dt))
  })
}

params <- c(k1=0.5,k2=0.1)
Y <- c(1,0)

columnNames<-c("Time","cholesterol","progesterone")
ProgesteroneData<-solveModel(ProgesteroneLinear,Y,params,columnNames)
ProgesteronePlot(ProgesteroneData,"Progesterone Synthesis")

###############################################################################
#Models Progesterone control
###############################################################################

ProgesteroneControlLinear <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> Cholesterol 
  #          Y[2]     -> Progesterone
  #
  #params -> Parameters of the ODE system model 
  #          k1       ->synthesis rate
  #          k2       ->decay rate
  #          kc       ->cholesterol intake rate
  #          kc       ->cholesterol intake
  #
  
  with(as.list(c(Y,params)),{
    
    dx1dt= -k1*Y[1] + k2*Y[2] + kc*u
    dx2dt= k1*Y[1] - k2*Y[2]
    
    list(c(dx1dt,dx2dt))
  })
}

params <- c(k1=0.05,k2=0.10,kc=0.1,u=0.025)
Y <- c(1,0)

columnNames<-c("Time","cholesterol","progesterone")
ProgesteroneControlData<-solveModel(ProgesteroneControlLinear,Y,params,columnNames)
ProgesteronePlot(ProgesteroneControlData,"LowProgesterone Synthesis Rate")

###############################################################################
#Models Progesterone Intermediate
###############################################################################

ProgesteroneIntermediateLinear <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> Cholesterol 
  #          Y[2]     -> pregnenolone 
  #          Y[3]     -> Progesterone
  #
  #params -> Parameters of the ODE system model 
  #          k1       ->Chiolesterol convertion rate
  #          k2       ->Pregenenolone decay rate
  #          k2       ->Pregnenolone convertion rate
  
  
  with(as.list(c(Y,params)),{
    
    dx1dt= -k1*Y[1] + k2*Y[2]
    dx2dt= k1*Y[1] - k2*Y[2]
    dx3dt= k2*Y[2] - k3*Y[3] 
    
    list(c(dx1dt,dx2dt,dx3dt))
  })
}

params <- c(k1=0.15,k2=0.50,k3=0.75)
Y <- c(1,0,0)
columnNames<-c("Time","cholesterol","pregnenolone","progesterone")
ProgesteroneIntermediate<-solveModel(ProgesteroneIntermediateLinear,Y,params,columnNames)
ProgesteroneIntermediatePlot(ProgesteroneIntermediate,"Low Cholesterol Convertion Rate")
