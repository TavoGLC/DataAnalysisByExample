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
library(gridExtra)
library(grid)
library(lattice)

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
  times <- seq(0, 1000, by = 0.1)
  out <- ode(InitialConditions,times,Model,ModelParameters,method="daspk")
  ModelData<-data.frame(out)
  colnames(ModelData)<-ColumnNames
  ModelData
}

###############################################################################
# Plot Functions
###############################################################################
columnNames<-c("Time","Mos","MEK","MEKPP","MAPK","MAPKPP")
SiganllingPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graph01<-ggplot(data=ModelData,aes(x=Time,y=Mos,color="Mos"))+geom_line()+
    scale_color_manual(values=c("Mos"="black"))+
    labs(title='Mos',color=" ")+
    theme(legend.position="none",axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  
  graph02<-ggplot(data=ModelData,aes(x=Time,y=MEK,color="MEK"))+geom_line()+
    scale_color_manual(values=c("MEK"="black"))+
    labs(title='MEK',color=" ")+
    theme(legend.position="none",axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  
  graph03<-ggplot(data=ModelData,aes(x=Time,y=MEKP,color="MEKP"))+geom_line()+
    scale_color_manual(values=c("MEKP"="black"))+
    labs(title='MEKP',color=" ")+
    theme(legend.position="none",axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  
  graph04<-ggplot(data=ModelData,aes(x=Time,y=MEKPP,color="MEKPP"))+geom_line()+
    scale_color_manual(values=c("MEKPP"="black"))+
    labs(title='MEKPP',color=" ")+
    theme(legend.position="none",axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  
  graph05<-ggplot(data=ModelData,aes(x=Time,y=MAPK,color="MAPK"))+geom_line()+
    scale_color_manual(values=c("MAPK"="black"))+
    labs(title='MAPK',color=" ")+
    theme(legend.position="none",axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  
  graph06<-ggplot(data=ModelData,aes(x=Time,y=MAPKP,color="MAPKP"))+geom_line()+
    scale_color_manual(values=c("MAPKP"="black"))+
    labs(title='MAPKP',color=" ")+
    theme(legend.position="none",axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  
  graph07<-ggplot(data=ModelData,aes(x=Time,y=MAPKPP,color="MAPKPP"))+geom_line()+
    scale_color_manual(values=c("MAPKPP"="black"))+
    labs(title='MAPKPP',color=" ")+
    theme(legend.position="none",axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())

    gridarrange<-rbind(c(1,1,2,3,4),c(1,1,5,6,7))
  graphContainer<-grid.arrange(graph01,graph02,graph03,graph04,graph05,graph06,graph07, layout_matrix=gridarrange)
  show(graphContainer)

}

###############################################################################
#Model Progesterone
###############################################################################

MapkSystem <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1],Y[2]-> Foward and reverse primer content
  #          Y[3],Y[4]-> Sample DNA foward and reverse strand
  #          Y[5],Y[6]-> Complex sampleDNA-primer foward and reverse 
  #          Y[7]     -> mRNA and protein content of gene 2
  #          Y[5]     -> mRNA and protein content of gene 2
  #
  #params -> Parameters of the ODE system model 
  #         
  #    
  
  with(as.list(c(Y,params)),{
    
    dxdt = -(V2*Y[1])/(K2+Y[1]) + V1 + V0*Y[1]*v*Y[7]
    
    dy1dt = (V6*Y[3])/(K6+Y[3]) - (V3*Y[1]*Y[2])/(K3+Y[2])
    dy2dt = (V3*Y[1]*Y[2])/(K3+Y[2]) + (V5*Y[4])/(K5+Y[4]) - (V4*Y[1]*Y[3])/(K4+Y[3]) + (V6*Y[3])/(K6+Y[3])
    dy3dt = (V4*Y[1]*Y[3])/(K4+Y[3]) - (V5*Y[4])/(K5+Y[4])
    
    dz1dt = (V10*Y[6])/(K10+Y[6]) - (V7*Y[4]*Y[5])/(K7+Y[5])
    dz2dt = (V7*Y[4]*Y[5])/(K7+Y[5]) + (V9*Y[7])/(K9+Y[7]) - (V8*Y[4]*Y[6])/(K8+Y[6]) - (V10*Y[6])/(K10+Y[6])
    dz3dt = (V8*Y[4]*Y[6])/(K8+Y[6]) - (V9*Y[7])/(K9+Y[7])
    
    list(c(dxdt,dy1dt,dy2dt,dy3dt,dz1dt,dz2dt,dz3dt))
  })
}

params <- c(V2=1.2,K2=200,V0=0.0015,v=1.5,V1=0.000002,V6=5,K6=1200,V3=0.064,K3=1200,V4=0.064,K4=1200,V5=5,K5=1200,V10=5,K10=300,V7=0.06,K7=300,V8=0.06,K8=300,V9=5,K9=300)
#Y <- c(2,1,0,0,10,0,0)
#Y <- c(2,1,25,0,10,20,0)
Y <- c(2,1,25,10,10,20,10)

columnNames<-c("Time","Mos","MEK","MEKP","MEKPP","MAPK","MAPKP","MAPKPP")
threeNodeData<-solveModel(MapkSystem,Y,params,columnNames)
SiganllingPlot(threeNodeData,"Three Nodes")

###############################################################################
#Models Progesterone control
###############################################################################
