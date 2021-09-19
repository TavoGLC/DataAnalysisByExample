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

LacOperonAPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=mRNA,color="mRNA"))+geom_line()+
    geom_line(aes(y=Bgalactosidasepermease,color="Bgalactosidasepermease"))+
    geom_line(aes(y=Bgalactosidase,color="Bgalactosidase"))+
    geom_line(aes(y=Lactose,color="Lactose"))+
    scale_color_manual(values=c("mRNA"="black","Bgalactosidasepermease"="red","Bgalactosidase"="blue","Lactose"="green"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}

LacOperonBPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=mRNA,color="mRNA"))+geom_line()+
    geom_line(aes(y=Bgalactosidasepermease,color="Bgalactosidasepermease"))+
    geom_line(aes(y=Bgalactosidase,color="Bgalactosidase"))+
    geom_line(aes(y=Lactose,color="Lactose"))+
    geom_line(aes(y=Allolactose,color="Allolactose"))+
    scale_color_manual(values=c("mRNA"="black","Bgalactosidasepermease"="red","Bgalactosidase"="blue","Lactose"="green","Allolactose"="orange"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}


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
  times <- seq(0, 10, by = 0.01)
  out <- ode(InitialConditions,times,Model,ModelParameters,method="rk4")
  ModelData<-data.frame(out)
  colnames(ModelData)<-ColumnNames
  ModelData
}

###############################################################################
#Models
###############################################################################

LacOperonA <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> mRNa concentration 
  #          Y[2]     -> Bgalactosidasepermease concentration 
  #          Y[3]     -> Bgalactosidase
  #          Y[4]     -> Lactose
  #
  #params -> Parameters of the ODE system model 
  #
  #

  with(as.list(c(Y,params)),{
    
    dy1dt=(1+k*Y[4]**p)/(1+Y[4]**p) - b1*Y[1]
    dy2dt=Y[1]-b2*Y[2]
    dy3dt=r3*Y[1]-b3*Y[3]
    dy4dt=S*Y[2]-Y[3]*Y[4]
    
    list(c(dy1dt,dy2dt,dy3dt,dy4dt))
  })
}

params <- c(p=1.5,k=15,b1=2.5,b2=6,r3=0.2,b3=0.5,S=0.75)
Y <- c(0,0.01,0,5)

columnNames<-c("Time","mRNA","Bgalactosidasepermease","Bgalactosidase","Lactose")
ModelAData<-solveModel(LacOperonA,Y,params,columnNames)
  LacOperonAPlot(ModelAData,"Simple Model")

###############################################################################
#Models
###############################################################################

LacOperonB <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> mRNA concentration
  #          Y[2]     -> Bgalactosidase concentration 
  #          Y[3]     -> Lactose concentration 
  #          Y[4]     -> Allolactose concentration 
  #          Y[5]     -> Bgalactosidasepermease concentration 
  #
  #params -> Parameters of the ODE system model 
  #         
  #    
  
  with(as.list(c(Y,params)),{
    
    dy1dt=am*(1+k1*(exp(mutm)*Y[4])**p)/(k+k1*(exp(mutm)*Y[4])**p) + G0 - gm*Y[1]
    dy2dt=ab*exp(-mutb)*Y[1] - gb*Y[2]
    dy3dt=al*(Y[5]*Le/(Kle+Le)) - bl1*(Y[5]*Y[3]/(Kl1+Y[3])) - bl2*(Y[2]*Y[3]/(Kl2+Y[3])) - gl*Y[3]
    dy4dt=aa*(Y[2]*Y[3]/(Kl+Y[3])) - ba*(Y[2]*Y[4]/(Ka+Y[4])) - ga*Y[4]
    dy5dt=ap*exp(-mutbp)*Y[1] - gp*Y[5]
    
    list(c(dy1dt,dy2dt,dy3dt,dy4dt,dy5dt))
  })
}

params <- c(am=2.5,k1=5,k=0.05,p=1.5,G0=0.001,gm=2.4,ab=0.3,gb=0.5,al=1.6,Le=0.1,Kle=1.4,bl1=0.4,Kl1=1,bl2=0.6,Kl2=0.66,gl=0.78,aa=0.7,Kl=0.6,ba=0.7,Ka=10.7,ga=0.5,ap=1,gp=1,mutm=0.1,mutb=0.1,mutbp=0.1)
Y <- c(0,0.01,5,0.001,0.001)

columnNames<-c("Time","mRNA","Bgalactosidase","Lactose","Allolactose","Bgalactosidasepermease")
ModelBData<-solveModel(LacOperonB,Y,params,columnNames)
LacOperonBPlot(ModelBData,"Low leakage")


