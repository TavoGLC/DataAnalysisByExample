
#Created on Mon Jul 16 23:21:01 2018

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

###############################################################################
# General Functions
###############################################################################

sigmoidalRate<-function(MaxValue,MaxPopulation,CurrentPopulation){
  #Transforms a rate to a sigmoidal function 
  #MaxValue              -> max value to be returned 
  #CurrentPopulation     -> Current population, value to be evaluated in the sigmoidal function 
  #MaxPopulation         -> Normalization factor 
  popRatio=10*(CurrentPopulation/MaxPopulation)
  MaxValue*(1/(1+exp(-popRatio+5)))
}

hardThreshold<-function(MaxValue,MaxPopulation,CurrentPopulation){
  #Generate a picewise rate function 
  #MaxValue              -> max value to be returned 
  #CurrentPopulation     -> Current population 
  #MaxPopulation         -> Normalization factor 
  popRatio=CurrentPopulation/MaxPopulation
  if(popRatio<0.25){
    MaxValue
  }
  else{
    MaxValue*0.25
  }
}

solveModel<- function(Model,InitialConditions,ModelParameters,ColumnNames){
  #Solves numerically an ODE system model 
  #Model             -> function, Model to be solved
  #InitialConditions -> list, Initial conditions for the ODE system 
  #ModelParameters   -> list, Parameters of the ODE model 
  times <- seq(0, 25, by = 0.01)
  out <- ode(InitialConditions,times,Model,ModelParameters)
  ModelData<-data.frame(out)
  colnames(ModelData)<-ColumnNames
  ModelData
}

###############################################################################
# Visualization functions
###############################################################################

makeMergedCostPlot<-function(modelData,PlotTitle){
  #Returns a ggplot object 
  #modelData -> dataframe with the integration results of an epidemics model 
  modelPlot<-ggplot(data=modelData,aes(x=Time,y=Susceptible,color="Susceptible"))+geom_line()+
    geom_line(aes(y=Infected,color="Infected"))+
    geom_line(aes(y=Recovered,color="Recovered"))+
    geom_line(aes(y=Total,color="Total"))+
    geom_line(aes(y=Jobs,color="Jobs"))+
    geom_line(aes(y=Cost,color="Cost"))+
    geom_line(aes(y=Diseases,color="Diseases"))+
    geom_line(aes(y=Recovery,color="Recovery"))+
    scale_color_manual(values=c("Susceptible"="black","Infected"="red","Recovered"="blue","Total"="green","Jobs"="orange","Cost"="brown","Diseases"="gray","Recovery"="cyan"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}


###############################################################################
# Merged model.  
###############################################################################

mergedCostModel <- function(t,Y,params){
  #Modified SIR model, takes into account loss of immunity and demographic factors
  #Modification of the infection rate into a sigmoidal responce function 
  #of the infected population 
  #Modification of the recovery rate, hard threshold of resource avaliability 
  #function of the infected population
  #t      -> integration time value 
  #Y      -> list Values for the function to be evaluated 
  #params -> Parameters of the ODE system model 
  with(as.list(c(Y,params)),{
    
    dSdt=-sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]+de*Y[3]+nu*Y[6]-mu*Y[1]-be*Y[1]
    dIdt=sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]-hardThreshold(ga,Y[4],Y[2]+Y[4])*Y[2]-mu*Y[2]
    dRdt=hardThreshold(ga,Y[4],Y[2]+Y[4])*Y[2]-de*Y[3]-mu*Y[3]
    
    dIEdt=(be/2)*Y[1]-hardThreshold(ga,Y[6],Y[2]+Y[4])*Y[4]-mu*Y[4]
    dREdt=hardThreshold(ga,Y[6],Y[2]+Y[4])*Y[4]-mor*Y[4]-mu*Y[5]
    
    dNdt=dSdt+dIdt+dRdt+dIEdt+dREdt
    
    dJdt=-u1*Y[7]+u2*0.5*(Y[6]-(Y[2]+Y[4]))
    dCdt=c1*Y[2]-c2*0.5*(Y[6]-(Y[2]+Y[4]))
    
    list(c(dSdt,dIdt,dRdt,dIEdt,dREdt,dNdt,dJdt,dCdt))
  })
}

params <- c(be=0.8,ga=0.8,de=0.4,nu=4,mu=4,u1=0.05,u2=0.075,c1=0.025,c2=0.1,mor=0.001)
Y <- c(1000,1,0,50,0,1051,500,500)
columnNames<-c("Time","Susceptible","Infected","Recovered","Diseases","Recovery","Total","Jobs","Cost")

mergedData<-solveModel(mergedCostModel,Y,params,columnNames)
makeMergedCostPlot(mergedData,"Current Scenario")

###############################################################################
# Time Delay model.  
###############################################################################

timeDelayDefinition <- function(t,Y,params,tao){
  #Modified SIR model, takes into account loss of immunity and demographic factors
  #Modification of the infection rate into a sigmoidal responce function 
  #of the infected population 
  #Modification of the recovery rate, hard threshold of resource avaliability 
  #function of the infected population
  #t      -> integration time value 
  #Y      -> list Values for the function to be evaluated 
  #params -> Parameters of the ODE system model
  #tao    -> Time delay to enforce social restriction
  with(as.list(c(Y,params)),{
    
    if(t<tao){
      dSdt=-sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]+de*Y[3]+nu*Y[6]-mu*Y[1]-be*Y[1]
      dIdt=sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]-hardThreshold(ga,Y[4],Y[2]+Y[4])*Y[2]-mu*Y[2]
      dRdt=hardThreshold(ga,Y[4],Y[2]+Y[4])*Y[2]-de*Y[3]-mu*Y[3]
      dIEdt=(be/2)*Y[1]-hardThreshold(ga,Y[6],Y[2]+Y[4])*Y[4]-mu*Y[4]
      dREdt=hardThreshold(ga,Y[6],Y[2]+Y[4])*Y[4]-mor*Y[4]-mu*Y[5]
      
    }
    
    else{
      
      dSdt=-sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]+de*Y[3]+0*Y[6]-0*Y[1]-be*Y[1]
      dIdt=sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]-hardThreshold(ga,Y[4],Y[2]+Y[4])*Y[2]-0*Y[2]
      dRdt=hardThreshold(ga,Y[4],Y[2]+Y[4])*Y[2]-de*Y[3]-0*Y[3]
      dIEdt=(be/2)*Y[1]-hardThreshold(ga,Y[6],Y[2]+Y[4])*Y[4]-0*Y[4]
      dREdt=hardThreshold(ga,Y[6],Y[2]+Y[4])*Y[4]-mor*Y[4]-0*Y[5]
    }
    
    dNdt=dSdt+dIdt+dRdt+dIEdt+dREdt
    dJdt=-u1*Y[7]+u2*0.5*(Y[6]-(Y[2]+Y[4]))
    dCdt=c1*Y[2]-c2*0.5*(Y[6]-(Y[2]+Y[4]))
    
    list(c(dSdt,dIdt,dRdt,dIEdt,dREdt,dNdt,dJdt,dCdt))
  })
}

#Wraper function to solve the model
timeDelayModel<-function(t,Y,params){
  timeDelayDefinition(t,Y,params,1.0)
}

params <- c(be=0.8,ga=0.89,de=0.0,nu=4,mu=4,u1=0.05,u2=0.075,c1=0.025,c2=0.1,mor=0.0001)
Y <- c(1000,1,0,50,0,1051,500,500)
columnNames<-c("Time","Susceptible","Infected","Recovered","Diseases","Recovery","Total","Jobs","Cost")

mergedData<-solveModel(timeDelayModel,Y,params,columnNames)
makeMergedCostPlot(mergedData,"Current Scenario")
