#MIT License
#Copyright (c) 2021 Octavio Gonzalez-Lugo 

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
  times <- seq(0, 50, by = 0.1)
  out <- ode(InitialConditions,times,Model,ModelParameters,method="rk4")
  ModelData<-data.frame(out)
  colnames(ModelData)<-ColumnNames
  ModelData
}

###############################################################################
# Plot Functions
###############################################################################

SiganllingPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=CytoplasmNFkB,color="CytoplasmNFkB"))+geom_line()+
    geom_line(aes(y=NuclearNFkB,color="NuclearNFkB"))+
    scale_color_manual(values=c("CytoplasmNFkB"="black","NuclearNFkB"="red"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}

###############################################################################
#Model Progesterone
###############################################################################

SignallingNfkB <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> cell receptors 
  #          Y[2]     -> gene regulation
  #
  #
  #params -> Parameters of the ODE system model 
  #         a,d       -> self regulation rates
  #         b,g       -> feedback regulation
  
  with(as.list(c(Y,params)),{
    
    dx1dt= s - a*Y[1] - b*Y[2]
    dx2dt= g*Y[1] - d*Y[2]
    
    list(c(dx1dt,dx2dt))
  })
}

params <- c(s=1,a=0.15,b=1.5,g=1.5,d=0.15)
Y <- c(1,1)

columnNames<-c("Time","CytoplasmNFkB","NuclearNFkB")
NfKbData<-solveModel(SignallingNfkB,Y,params,columnNames)
SiganllingPlot(NfKbData,"Signaling")

