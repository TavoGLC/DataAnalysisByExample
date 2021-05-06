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
# Utility Functions
###############################################################################

MinMaxNormalization<-function(ColumnData){
  
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

michaelisRate<-function(Vmax,Km,S){
  out<-(Vmax*S)/(Km+S)
  out
}

###############################################################################
# Plot Functions
###############################################################################

SimpleModelPlot<-function(modelData,PlotTitle){
  #Returns a ggplot
  #modelData -> dataframe with the integration results of a simple model
  #PlotTitle -> Title for the plot
  modelPlot<-ggplot(data=modelData,aes(x=Time,y=Waste,color="Waste"))+geom_line()+
    geom_line(aes(y=Compost,color="Compost"))+
    scale_color_manual(values=c("Waste"="black","Compost"="red"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

LIDELModelPlot<-function(modelData,PlotTitle){
  #Returns a ggplot
  #modelData -> dataframe with the integration results of a LIDEL model
  #PlotTitle -> Title for the plot
  modelPlot<-ggplot(data=modelData,aes(x=Time,y=SolubleCarbon,color="SolubleCarbon"))+geom_line()+
    geom_line(aes(y=SolubleNonLignin,color="SolubleNonLignin"))+
    geom_line(aes(y=SolubleLignin,color="SolubleLignin"))+
    geom_line(aes(y=Microbe,color="Microbe"))+
    geom_line(aes(y=MicrobeProducts,color="MicrobeProducts"))+
    geom_line(aes(y=DissolvedOrganicCarbon,color="DissolvedOrganicCarbon"))+
    geom_line(aes(y=CO2,color="CO2"))+
    geom_line(aes(y=CO2,color="CO2"))+
    geom_line(aes(y=(SolubleCarbon+SolubleNonLignin+SolubleLignin)/3,color="Total"))+
    scale_color_manual(values=c("SolubleCarbon"="black","SolubleNonLignin"="red","SolubleLignin"="blue","Microbe"="green","MicrobeProducts"="orange","DissolvedOrganicCarbon"="brown","CO2"="gray","Total"="Cyan"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

displayGraph<-function(graphElement,labels,PlotTitle){
  #Returns a gggraph
  #modelData -> dataframe with the integration results of a LIDEL model
  #labels -> labels for each node in the graph
  #PlotTitle -> Title for the plot
  net<-ggraph(graph,layout = 'fr', weights = weight) + 
    geom_edge_link() + 
    geom_node_point()+
    geom_node_text(aes(label=labels),size=2.5,nudge_x = 0.15 ,nudge_y = 0.15)+
    labs(title=PlotTitle,color=" ")
  show(net)
}

###############################################################################
#Solver function
###############################################################################

solveModel<- function(Model,InitialConditions,ModelParameters,ColumnNames,MinMax){
  #Solves numerically an ODE system model,returns a formated dataframe
  #Model             -> function, Model to be solved
  #InitialConditions -> list, Initial conditions for the ODE system 
  #ModelParameters   -> list, Parameters of the ODE model 
  #ColumnNames       -> list, names of the columns for the dataframe 
  #MinMax            -> bool, controlls if a minmax normalization is applied to the data. 
  times <- seq(0, 365, by = 0.1)
  out <- ode(InitialConditions,times,Model,ModelParameters,method="rk4")
  if (MinMax){
    dims<-dim(out)
    for (k in 2:dims[2]){
      out[,k]<-MinMaxNormalization(out[,k])
    }
  }
  ModelData<-data.frame(out)
  colnames(ModelData)<-ColumnNames
  ModelData
}


###############################################################################
# Linear kinetics model
###############################################################################
SimpleModel <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  #
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]-> Waste
  #          Y[2]-> Compost
  #
  #params -> Parameters of the ODE system model 
  #          Y0   -> Initial amount of organic waste
  #          k    -> Decomposition rate
  #    
  with(as.list(c(Y,params)),{
    
    dIdt=-Y[2]
    dCdt=Y0-k*Y[2]
    
    list(c(dIdt,dCdt))
  })
}

params <- c(Y0=100,k=0.008)
Y <- c(100,0.00)
columnNames<-c("Time","Waste","Compost")
compostData<-solveModel(SimpleModel,Y,params,columnNames,TRUE)

SimpleModelPlot(compostData,"Simple Model")

###############################################################################
# LIDEL model
###############################################################################

LIDELModel <- function(t,Y,params){
  
  #LIDEL model (Litter Decomposition and Leaching)
  #t      -> integration time value 
  #
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]-> SolubleCarbon, 
  #          Y[2]-> Soluble Non Lignin, 
  #          Y[3]-> Soluble Lignin, 
  #          Y[4]-> Microbe, 
  #          Y[5]-> Microbe Products, 
  #          Y[6]-> Dissolved Organic Carbon, 
  #          Y[7]-> CO2, 
  #
  #params -> Parameters of the ODE system model 
  #    
  with(as.list(c(Y,params)),{
    
    dY1=-muk*k1*Y[1]
    dY2=-muk*k2*Y[2]
    dY3=-k3*Y[3]
    dY4=-k4*Y[4]+mub*B2*(1-L1)*muk*k2*Y[2]+mub*B1*(1-L4)*muk*k1*Y[1]
    dY5=-k5*Y[5]+B3*(1-L2)*k4*Y[4]
    dY6=L1*muk*k2*Y[2]+L3*k3*Y[3]+L2*k4*Y[4]+L3*k5*Y[5]+L4*muk*k1*Y[1]
    dY7=((1-mub*B1)*(1-L4))*muk*k1*Y[1]+((1-B2*mub)*(1-L1))*muk*k2*Y[2]+(1-L3)*k3*Y[3]+
      ((1-B3)*(1-L2))*k4*Y[4]+(1-L3)*k5*Y[5]
    list(c(dY1,dY2,dY3,dY4,dY5,dY6,dY7))
  })
}

params <- c(muk=0.28,k1=0.24,k2=0.0079,k3=0.1,k4=0.60,k5=0.3,mub=0.1,B1=0.1,B2=0.81,B3=0.27,L1=0.1,L2=0.16,L3=0.01,L4=0.01)
Y <- c(25,25,25,0,0,0.001,0)
columnNames<-c("Time","SolubleCarbon","SolubleNonLignin","SolubleLignin","Microbe","MicrobeProducts","DissolvedOrganicCarbon","CO2")

compostData<-solveModel(LIDELModel,Y,params,columnNames,TRUE)

LIDELModelPlot(compostData,"LIDEL Model")

###############################################################################
# LIDEL model
###############################################################################

labels<-c("SolubleCarbon",
               "SolubleNonLignin",
               "SolubleLignin",
               "Microbe",
               "MicrobeProducts",
               "DissolvedOrganicCarbon",
               "CO2")

modelGraph<-data.frame(from =c(1,1,1,2,2,2,3,3,4,4,4,5,5),to = c(4,6,7,4,6,7,6,7,5,6,7,6,7),weight = rep(1,13))
graph <- as_tbl_graph(modelGraph)

displayGraph(graph,labels,"LIDEL Model Graph")

###############################################################################
# LIDEL model only hidrolisis
###############################################################################

LIDELModel <- function(t,Y,params){
  #Modified LIDEL model (Litter Decomposition and Leaching)
  #t      -> integration time value 
  #
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]-> SolubleCarbon, 
  #          Y[2]-> Soluble Non Lignin, 
  #          Y[3]-> Soluble Lignin, 
  #          Y[4]-> Microbe, 
  #          Y[5]-> Microbe Products, 
  #          Y[6]-> Dissolved Organic Carbon, 
  #          Y[7]-> CO2, 
  #
  #params -> Parameters of the ODE system model 
  #    
  with(as.list(c(Y,params)),{
    
    dY1=-muk*k1*Y[1]
    dY2=-muk*k2*Y[2]-michaelisRate(Vmax,Km,Y[2])*Y[4]
    dY3=-k3*Y[3]
    dY4=-k4*Y[4]+mub*B2*(1-L1)*muk*k2*Y[2]+mub*B1*(1-L4)*muk*k1*Y[1]
    dY5=-k5*Y[5]+B3*(1-L2)*k4*Y[4]+michaelisRate(Vmax,Km,Y[2])*Y[4]
    dY6=L1*muk*k2*Y[2]+L3*k3*Y[3]+L2*k4*Y[4]+L3*k5*Y[5]+L4*muk*k1*Y[1]
    dY7=((1-mub*B1)*(1-L4))*muk*k1*Y[1]+((1-B2*mub)*(1-L1))*muk*k2*Y[2]+(1-L3)*k3*Y[3]+
      ((1-B3)*(1-L2))*k4*Y[4]+(1-L3)*k5*Y[5]
    list(c(dY1,dY2,dY3,dY4,dY5,dY6,dY7))
  })
}

params <- c(muk=0.28,k1=0.24,k2=0.0079,k3=0.1,k4=0.60,k5=0.3,mub=0.1,B1=0.1,B2=0.81,B3=0.27,L1=0.1,L2=0.16,L3=0.01,L4=0.01,Vmax=15,Km=0.25)
Y <- c(25,25,25,0,0,0.001,0)
columnNames<-c("Time","SolubleCarbon","SolubleNonLignin","SolubleLignin","Microbe","MicrobeProducts","DissolvedOrganicCarbon","CO2")

compostData<-solveModel(LIDELModel,Y,params,columnNames,TRUE)

LIDELModelPlot(compostData,"Modified LIDEL Model")

