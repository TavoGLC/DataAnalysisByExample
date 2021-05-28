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
# Plot Functions
###############################################################################

MakePCRModelPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=ReversePrimer+FowardPrimer,color="Primers"))+geom_line()+
    geom_line(aes(y=ReverseStrand+FowardStrand,color="OriginalSample"))+
    geom_line(aes(y=FowardStrand_Primer+ReverseStrand_Primer,color="Amplicon"))+
    geom_line(aes(y=Reannealing,color="Reannealing"))+
    geom_line(aes(y=PrimerDimers,color="PrimerDimers"))+
    scale_color_manual(values=c("Primers"="black","OriginalSample"="red","Amplicon"="green","Reannealing"="blue","PrimerDimers"="Cyan"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
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
  times <- seq(0, 5, by = 0.01)
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
#Models
###############################################################################

PCRModel <- function(t,Y,params){
  
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
    
    dP1dt=-kah1*Y[1]*Y[3]+kdh1*Y[5]-kad*Y[1]*Y[2]+kdd*Y[8]
    dP2dt=-kah2*Y[2]*Y[4]+kdh2*Y[6]-kad*Y[1]*Y[2]+kdd*Y[8]
    dT1dt=-kah1*Y[1]*Y[3]+kdh1*Y[5]-kau*Y[3]*Y[4]+kdu*Y[7]
    dT2dt=-kah2*Y[2]*Y[4]+kdh2*Y[6]-kau*Y[3]*Y[4]+kdu*Y[7]
    dH1dt=kah1*Y[1]*Y[3]-kdh1*Y[5]
    dH2dt=kah2*Y[2]*Y[4]-kdh2*Y[6]
    dUdt=kau*Y[3]*Y[4]-kdu*Y[7]
    dDdt=kad*Y[1]*Y[2]-kdd*Y[8]
    
    list(c(dP1dt,dP2dt,dT1dt,dT2dt,dH1dt,dH2dt,dUdt,dDdt))
  })
}

params <- c(kah1=0.1,kah2=0.1,kdh1=10**-4,kdh2=10**-4,kad=1,kdd=0.1,kau=0.001,kdu=0)
Y <- c(100,100,50,50,0,0,0,0)

columnNames<-c("Time","ReversePrimer","FowardPrimer","ReverseStrand","FowardStrand","ReverseStrand_Primer","FowardStrand_Primer","Reannealing","PrimerDimers")
threeNodeData<-solveModel(PCRModel,Y,params,columnNames,FALSE)
MakePCRModelPlot(threeNodeData,"PCR Primers Dimers")

