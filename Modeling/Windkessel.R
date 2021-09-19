#MIT License
#Copyright (c) 2021 Octavio Gonzalez-Lugo 

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to per mit persons to whom the Software is
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
  times <- seq(0, 10, by = 0.001)
  out <- ode(InitialConditions,times,Model,ModelParameters,method="bdf")
  ModelData<-data.frame(out)
  colnames(ModelData)<-ColumnNames
  ModelData
}

###############################################################################
# Plot Functions
###############################################################################

MakeWindkesselModelPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=Pressure,color="Pressure"))+geom_line()+
    scale_color_manual(values=c("Pressure"="black"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}

###############################################################################
# Plot Functions
###############################################################################

timeScale <- function(t,Tc,Ts){
  scaledTime <- t/Tc
  decimal <- scaledTime - floor(scaledTime)
  responce <- FALSE
  if (decimal <= (2/7)){
    responce <- TRUE
  }
  responce
}

inputLoad<-function(t,Tc,Ts){
  a <- t%%Tc
  i0 <- 400*sin(pi*(a/Ts))
  output <- 0
  if (timeScale(t,Tc,Ts)){
    output <- i0
  }
  output
}

inputLoadDer<-function(t,Tc,Ts){
  a<-t%%Tc
  i0 <- 400*cos(pi*(a/Ts))
  output<-0
  if (timeScale(t,Tc,Ts)){
    output <- i0
  }
  output
}

inputLoad2Der<-function(t,Tc,Ts){
  a<-t%%Tc
  i0 <- -400*sin(pi*(a/Ts))
  output<-0
  if (timeScale(t,Tc,Ts)){
    output <- i0
  }
  output
}
###############################################################################
#Volumetric Flow
###############################################################################

times <- seq(0, 5, by = 0.01)
flowt <- rep(0,length(times))
k <- 1

for (val in times){
  flowt[k]<-inputLoad(val,60/72,(2/5)*(60/72))
  k <- k+1
}

FlowData<-Container<-matrix(0,length(times),2)
FlowData[,1]<-times
FlowData[,2]<-flowt
FlowData<-data.frame(FlowData)
colnames(FlowData)<-c("Time","Flow")

graphContainer<-ggplot(data=FlowData,aes(x=Time,y=Flow,color="Flow"))+geom_line()+
  scale_color_manual(values=c("Flow"="blue"))+
  labs(title="Volumetric flow",color=" ")+
  theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
show(graphContainer)

###############################################################################
#Models
###############################################################################

WindkesselModel2Elements <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> Blood pressure
  #
  #params -> Parameters of the ODE system model 
  #         
  #    
  
  with(as.list(c(Y,params)),{
    
    dPdt=(1/C)*(inputLoad(t,Tc,Ts)-(Y[1]/R))
    
    list(c(dPdt))
  })
}

params <- c(C=1.0666,R=0.95000,Tc=60/72,Ts=(2/5)*(60/72))
Y <- c(80)

columnNames<-c("Time","Pressure")
threeNodeData<-solveModel(WindkesselModel2Elements,Y,params,columnNames)
MakeWindkesselModelPlot(threeNodeData,"Windkessel Model 2 Elements")

###############################################################################
#Models
###############################################################################

WindkesselModel3Elements <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> Blood preassure
  #
  #params -> Parameters of the ODE system model 
  #         
  #    
  
  with(as.list(c(Y,params)),{
    
    dPdt=(1/C)*((1+(R1/R2))*inputLoad(t,Tc,Ts)+(C*R1*inputLoadDer(t,Tc,Ts))-(Y[1]/R1))
  
    list(c(dPdt))
  })
}

params <- c(C=1.0666,R1=0.9,R2=0.05,Tc=60/72,Ts=(2/5)*(60/72))
Y <- c(80)

columnNames<-c("Time","Pressure")
threeNodeData<-solveModel(WindkesselModel3Elements,Y,params,columnNames)
MakeWindkesselModelPlot(threeNodeData,"Windkessel Model 3 Elements")

###############################################################################
#Models
###############################################################################

WindkesselModel4Elements <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> Blood preassure
  #
  #params -> Parameters of the ODE system model 
  #         
  #    
  
  
  with(as.list(c(Y,params)),{
    
    dPdt=(1/C)*((1+(R1/R2))*inputLoad(t,Tc,Ts)+((C*R1+L/R1)*inputLoadDer(t,Tc,Ts))+(C*L*inputLoad2Der(t,Tc,Ts))-(Y[1]/R1))
    
    list(c(dPdt))
  })
}

params <- c(C=1.0666,R1=0.9000,R2=0.05,L=25,Tc=60/72,Ts=(2/5)*(60/72))
Y <- c(80)

columnNames<-c("Time","Pressure")
threeNodeData<-solveModel(WindkesselModel4Elements,Y,params,columnNames)
MakeWindkesselModelPlot(threeNodeData,"Windkessel Model 4 Elements")

