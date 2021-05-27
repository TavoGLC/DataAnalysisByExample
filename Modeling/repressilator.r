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
# Plot Functions
###############################################################################

Make2NodeRepressilatorModelPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=mRNA_0,color="mRNA_0"))+geom_line()+
    geom_line(aes(y=Protein_0,color="Protein_0"))+
    geom_line(aes(y=mRNA_1,color="mRNA_1"))+
    geom_line(aes(y=Protein_1,color="Protein_1"))+
    scale_color_manual(values=c("mRNA_0"="black","Protein_0"="red","mRNA_1"="blue","Protein_1"="green"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}

Make3NodeRepressilatorModelPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=mRNA_0,color="mRNA_0"))+geom_line()+
    geom_line(aes(y=Protein_0,color="Protein_0"))+
    geom_line(aes(y=mRNA_1,color="mRNA_1"))+
    geom_line(aes(y=Protein_1,color="Protein_1"))+
    geom_line(aes(y=mRNA_2,color="mRNA_2"))+
    geom_line(aes(y=Protein_2,color="Protein_2"))+
    scale_color_manual(values=c("mRNA_0"="black","Protein_0"="red","mRNA_1"="blue","Protein_1"="green","mRNA_2"="orange","Protein_2"="brown"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}


displayGraph<-function(graphElement,labels,PlotTitle){
  #Returns a gggraph
  #modelData -> dataframe with the integration results of a LIDEL model
  #labels -> labels for each node in the graph
  #PlotTitle -> Title for the plot
  net<-ggraph(graph,layout = 'fr', weights = weight) + 
    geom_edge_link(arrow=arrow(angle=90,length=unit(0.1,"inches")),end_cap = circle(3, 'mm')) + 
    geom_node_point()+
    geom_node_text(aes(label=labels),size=2.5,nudge_x = 0.15 ,nudge_y = 0.15)+
    labs(title=PlotTitle,color=" ")
  show(net)
}

###############################################################################
# Graphs
###############################################################################

labels<-c("Gene A",
          "Gene B",
          "Gene C")

modelGraph<-data.frame(from =c(1,2,3),to = c(2,3,1),weight = rep(1,3))
graph <- as_tbl_graph(modelGraph)

displayGraph(graph,labels,"Repressilator Model")

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
  times <- seq(0, 100, by = 0.1)
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
#Helper functions
###############################################################################

mrna<-function(m,p,alpha,alpha0,n){
  dmdt<--m+alpha/(1+p**n)+alpha0
  dmdt
}

prt<-function(m,p,beta){
  dpdt<--beta*(p-m)
  dpdt
}

###############################################################################
#Models
###############################################################################

TwoNodeRepressilator <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  #
  #Y      -> list Values for the function to be evaluated 
  #          Y[1],Y[2]-> mRNA and protein content of gene 1
  #          Y[3],Y[4]-> mRNA and protein content of gene 2
  #
  #params -> Parameters of the ODE system model 
  #          alpha     -> protein copies per cell produced from a give promoter type
  #          alpha0    -> leakiness of the promoter 
  #          beta      -> ratio protein to mRNA decay
  #          n         -> Hill coefficient    
  
  with(as.list(c(Y,params)),{
    
    dr1dt=mrna(Y[1],Y[4],alpha,alpha0,n)
    dp1dt=prt(Y[1],Y[2],beta)
    dr2dt=mrna(Y[3],Y[2],alpha,alpha0,n)
    dp2dt=prt(Y[3],Y[4],beta)
    
    list(c(dr1dt,dp1dt,dr2dt,dp2dt))
  })
}

params <- c(alpha=15,beta=1,n=2.1,alpha0=0)
Y <- c(1,2,3,1)
columnNames<-c("Time","mRNA_0","Protein_0","mRNA_1","Protein_1")
twoNodeData<-solveModel(TwoNodeRepressilator,Y,params,columnNames,FALSE)
Make2NodeRepressilatorModelPlot(twoNodeData,"Two Nodes")

###############################################################################
#Models
###############################################################################

ThreeNodeRepressilator <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1],Y[2]-> mRNA and protein content of gene 1
  #          Y[3],Y[4]-> mRNA and protein content of gene 2
  #          Y[5],Y[6]-> mRNA and protein content of gene 2
  #
  #params -> Parameters of the ODE system model 
  #          alpha     -> protein copies per cell produced from a give promoter type
  #          alpha0    -> leakiness of the promoter 
  #          beta      -> ratio protein to mRNA decay
  #          n         -> Hill coefficient    
  
  #    
  with(as.list(c(Y,params)),{
    
    dr1dt=mrna(Y[1],Y[4],alpha,alpha0,n)
    dp1dt=prt(Y[1],Y[2],beta)
    dr2dt=mrna(Y[3],Y[6],alpha,alpha0,n)
    dp2dt=prt(Y[3],Y[4],beta)
    dr3dt=mrna(Y[5],Y[2],alpha,alpha0,n)
    dp3dt=prt(Y[5],Y[6],beta)
    
    list(c(dr1dt,dp1dt,dr2dt,dp2dt,dr3dt,dp3dt))
  })
}

params <- c(alpha=15,beta=1,n=2.1,alpha0=0)
Y <- c(1,2,3,1,2,1)
columnNames<-c("Time","mRNA_0","Protein_0","mRNA_1","Protein_1","mRNA_2","Protein_2")
threeNodeData<-solveModel(ThreeNodeRepressilator,Y,params,columnNames,FALSE)
Make3NodeRepressilatorModelPlot(threeNodeData,"Three Nodes")

