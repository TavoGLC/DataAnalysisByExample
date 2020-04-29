
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
  if(popRatio<0.2){
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

makeInfectionPlot<-function(modelData){
  #Returns a ggplot object 
  #modelData -> dataframe with the integration results of an epidemics model 
  modelPlot<-ggplot(data=modelData,aes(x=Time,y=Susceptible,color="Susceptible"))+geom_line()+
    geom_line(aes(y=Infected,color="Infected"))+
    geom_line(aes(y=Recovered,color="Recovered"))+
    geom_line(aes(y=Total,color="Total"))+
    scale_color_manual(values=c("Susceptible"="black","Infected"="red","Recovered"="blue","Total"="green"))+
    labs(color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

makeMergedPlot<-function(modelData){
  #Returns a ggplot object 
  #modelData -> dataframe with the integration results of an epidemics model 
  modelPlot<-ggplot(data=modelData,aes(x=Time,y=Susceptible,color="Susceptible"))+geom_line()+
    geom_line(aes(y=Infected,color="Infected"))+
    geom_line(aes(y=Recovered,color="Recovered"))+
    geom_line(aes(y=Total,color="Total"))+
    geom_line(aes(y=Jobs,color="Jobs"))+
    scale_color_manual(values=c("Susceptible"="black","Infected"="red","Recovered"="blue","Total"="green","Jobs"="orange"))+
    labs(color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

makeMergedCostPlot<-function(modelData){
  #Returns a ggplot object 
  #modelData -> dataframe with the integration results of an epidemics model 
  modelPlot<-ggplot(data=modelData,aes(x=Time,y=Susceptible,color="Susceptible"))+geom_line()+
    geom_line(aes(y=Infected,color="Infected"))+
    geom_line(aes(y=Recovered,color="Recovered"))+
    geom_line(aes(y=Total,color="Total"))+
    geom_line(aes(y=Jobs,color="Jobs"))+
    geom_line(aes(y=Cost,color="Cost"))+
    scale_color_manual(values=c("Susceptible"="black","Infected"="red","Recovered"="blue","Total"="green","Jobs"="orange","Cost"="brown"))+
    labs(color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

###############################################################################
# Infection model.  
###############################################################################

SIRSDNLR <- function(t,Y,params){
  #Modified SIR model, takes into account loss of immunity and demographic factors
  #Modification of the infection rate into a sigmoidal responce function 
  #of the infected population 
  #Modification of the recovery rate, hard threshold of resource avaliability 
  #function of the infected population
  #t      -> integration time value 
  #Y      -> list Values for the function to be evaluated 
  #params -> Parameters of the ODE system model 
  with(as.list(c(Y,params)),{
    dSdt=-sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]+de*Y[3]+nu*Y[4]-mu*Y[1]
    dIdt=sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]-hardThreshold(ga,Y[4],Y[2])*Y[2]-mu*Y[2]
    dRdt=hardThreshold(ga,Y[4],Y[2])*Y[2]-de*Y[3]-mu*Y[3]
    dNdt=0
    list(c(dSdt,dIdt,dRdt,dNdt))
  })
}

params <- c(be=0.8,ga=0.85,de=0.425,nu=4.0,mu=4.0)
Y <- c(1000,1,0,1000)
columnNames<-c("Time","Susceptible","Infected","Recovered","Total")

SIRSDNLRData<-solveModel(SIRSDNLR,Y,params,columnNames)

#makeInfectionPlot(SIRSDNLRData)

###############################################################################
# Merged model.  
###############################################################################


mergedModelInfR <- function(t,Y,params){
  #Modified SIR model, takes into account loss of immunity and demographic factors
  #Modification of the infection rate into a sigmoidal responce function 
  #of the infected population 
  #Modification of the recovery rate, hard threshold of resource avaliability 
  #function of the infected population
  #t      -> integration time value 
  #Y      -> list Values for the function to be evaluated 
  #params -> Parameters of the ODE system model 
  with(as.list(c(Y,params)),{
    dSdt=-sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]+de*Y[3]+nu*Y[4]-mu*Y[1]
    dIdt=sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]-ga*Y[2]-mu*Y[2]
    dRdt=ga*Y[2]-de*Y[3]-mu*Y[3]
    dNdt=0
    dJdt=-u4*Y[5]+u5*0.5*(Y[1]+Y[3])
    list(c(dSdt,dIdt,dRdt,dNdt,dJdt))
  })
}

params <- c(be=0.8,ga=0.85,de=0,nu=0,mu=0,u4=0.05,u5=0.075)
Y <- c(1000,1,0,1000,500)
columnNames<-c("Time","Susceptible","Infected","Recovered","Total","Jobs")

mergedData<-solveModel(mergedModelInfR,Y,params,columnNames)
#makeMergedPlot(mergedData)

###############################################################################
# Merged model.  
###############################################################################

mergedModel <- function(t,Y,params){
  #Modified SIR model, takes into account loss of immunity and demographic factors
  #Modification of the infection rate into a sigmoidal responce function 
  #of the infected population 
  #Modification of the recovery rate, hard threshold of resource avaliability 
  #function of the infected population
  #t      -> integration time value 
  #Y      -> list Values for the function to be evaluated 
  #params -> Parameters of the ODE system model 
  with(as.list(c(Y,params)),{
    dSdt=-sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]+de*Y[3]+nu*Y[4]-mu*Y[1]
    dIdt=sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]-hardThreshold(ga,Y[4],Y[2])*Y[2]-mu*Y[2]
    dRdt=hardThreshold(ga,Y[4],Y[2])*Y[2]-de*Y[3]-mu*Y[3]
    dNdt=0
    dJdt=-u4*Y[5]+u5*0.5*(Y[1]+Y[3])
    list(c(dSdt,dIdt,dRdt,dNdt,dJdt))
  })
}

params <- c(be=0.8,ga=0.85,de=0.0,nu=0,mu=0,u4=0.05,u5=0.075)
Y <- c(1000,1,0,1000,500)
columnNames<-c("Time","Susceptible","Infected","Recovered","Total","Jobs")

mergedData<-solveModel(mergedModel,Y,params,columnNames)
#makeMergedPlot(mergedData)


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
    dSdt=-sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]+de*Y[3]+nu*Y[4]-mu*Y[1]
    dIdt=sigmoidalRate(be,Y[4],Y[2])*Y[2]*Y[1]-hardThreshold(ga,Y[4],Y[2])*Y[2]-mu*Y[2]
    dRdt=hardThreshold(ga,Y[4],Y[2])*Y[2]-de*Y[3]-mu*Y[3]
    dNdt=0
    dJdt=-u1*Y[5]+u2*0.5*(Y[1]+Y[3])
    dCdt=c1*Y[2]-c2*0.5*(Y[1]+Y[3])
    list(c(dSdt,dIdt,dRdt,dNdt,dJdt,dCdt))
  })
}

params <- c(be=0.8,ga=0.85,de=0,nu=0,mu=0,u1=0.05,u2=0.075,c1=0.025,c2=0.1)
Y <- c(1000,1,0,1000,500,500)
columnNames<-c("Time","Susceptible","Infected","Recovered","Total","Jobs","Cost")

mergedData<-solveModel(mergedCostModel,Y,params,columnNames)
makeMergedCostPlot(mergedData)
