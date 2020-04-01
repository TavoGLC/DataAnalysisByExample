"""
Created on Mon Jul 16 23:21:01 2018

MIT License
Copyright (c) 2020 Octavio Gonzalez-Lugo 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
@author: Octavio Gonzalez-Lugo

"""

###############################################################################
# Loading packages 
###############################################################################

library(deSolve)
library(ggplot2)

###############################################################################
# General Functions
###############################################################################

rectangleRule<-function(solution){
  0.01*sum(solution)
}

makeModelPlot<-function(modelData){
  modelPlot<-ggplot(data=modelData,aes(x=Time,y=Suceptible,color="Suceptible"))+geom_line()+
    geom_line(aes(y=Infected,color="Infected"))+
    geom_line(aes(y=Recovered,color="Recovered"))+
    geom_line(aes(y=Total,color="Total"))+
    scale_color_manual(values=c("Suceptible"="black","Infected"="red","Recovered"="blue","Total"="green"))+
    labs(color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

###############################################################################
# SIR model
###############################################################################

SIR <- function(t,Y,params){
  with(as.list(c(Y,params)),{
    dSdt=-be*Y[2]*Y[1]
    dIdt=be*Y[2]*Y[1]-ga*Y[2]
    dRdt=ga*Y[2]
    dNdt=0
    list(c(dSdt,dIdt,dRdt,dNdt))
  })
}

params <- c(be=0.05,ga=0.8)
Y <- c(1000,1,0,1000)
time <- seq(0, 4, by = 0.01)
out <- ode(Y,time,SIR,params)

SIRData<-data.frame(out)
columnNames<-c("Time","Suceptible","Infected","Recovered","Total")
colnames(SIRData)<-columnNames

#makeModelPlot(SIRData)

###############################################################################
# SIRS model
###############################################################################

SIRS <- function(t,Y,params){
  with(as.list(c(Y,params)),{
    dSdt=-be*Y[2]*Y[1]+de*Y[3]
    dIdt=be*Y[2]*Y[1]-ga*Y[2]
    dRdt=ga*Y[2]-de*Y[3]
    dNdt=0
    list(c(dSdt,dIdt,dRdt,dNdt))
  })
}

params <- c(be=0.05,ga=0.85,de=0.425)
Y <- c(1000,1,0,1000)
times <- seq(0, 4, by = 0.01)
out <- ode(Y,times,SIRS,params)

SIRSData<-data.frame(out)
columnNames<-c("Time","Suceptible","Infected","Recovered","Total")
colnames(SIRSData)<-columnNames

makeModelPlot(SIRSData)

###############################################################################
# SIRD model
###############################################################################

SIRD <- function(t,Y,params){
  with(as.list(c(Y,params)),{
    dSdt=-be*Y[2]*Y[1]+nu*Y[4]-mu*Y[1]
    dIdt=be*Y[2]*Y[1]-ga*Y[2]-mu*Y[2]
    dRdt=ga*Y[2]-mu*Y[3]
    dNdt=0
    list(c(dSdt,dIdt,dRdt,dNdt))
  })
}

params <- c(be=0.05,ga=0.85,mu=4,nu=4)
Y <- c(1000,1,0,1000)
times <- seq(0, 4, by = 0.01)
out <- ode(Y,times,SIRD,params)

SIRDData<-data.frame(out)
columnNames<-c("Time","Suceptible","Infected","Recovered","Total")
colnames(SIRDData)<-columnNames

#makeModelPlot(SIRDData)

SolveSIRD<-function(NuVal,MuVal){
  
  paramsP <- c(be=0.05,ga=0.85,mu=MuVal,nu=NuVal)
  Y <- c(1000,1,0,1000)
  timesL <- seq(0, 3, by = 0.01)
  outL <- ode(Y,timesL,SIRD,paramsP)
  SIRDDataL<-data.frame(outL)
  columnNames<-c("Time","Suceptible","Infected","Recovered","Total")
  colnames(SIRDDataL)<-columnNames
  SIRDDataL$Infected
}

paramRange<- seq(4,0,by=-0.1)
infectionDepletion<-rep(0,length(paramRange))
restriction<-rep(0,length(paramRange))

for (i in 1:length(paramRange)){
  cSolution<-SolveSIRD(paramRange[i],paramRange[i])
  infectionDepletion[i]<-rectangleRule(cSolution)
  restriction[i]<-(paramRange[1]-paramRange[i])/paramRange[1]
}

infectionDepletion<-(infectionDepletion[1]-infectionDepletion)/infectionDepletion[1]

depletion<-data.frame("Reduction"=infectionDepletion,"Restriction"=restriction)

Dplot<-ggplot(data=depletion,aes(x=Restriction,y=Reduction))+geom_line()
#show(Dplot)

###############################################################################
# SIRSV model
###############################################################################

SIRSV <- function(t,Y,params){
  with(as.list(c(Y,params)),{
    dSdt=-be*Y[2]*Y[1]+(mu-sig)*Y[4]-mu*Y[1]
    dIdt=be*Y[2]*Y[1]-ga*Y[2]-mu*Y[2]
    dRdt=ga*Y[2]-mu*Y[3]
    dNdt=0
    list(c(dSdt,dIdt,dRdt,dNdt))
  })
}

params <- c(be=0.05,ga=0.85,mu=4,sig=0)
Y <- c(1000,1,0,1000)
times <- seq(0, 4, by = 0.01)
out <- ode(Y,times,SIRSV,params)

SIRVData<-data.frame(out)
columnNames<-c("Time","Suceptible","Infected","Recovered","Total")
colnames(SIRVData)<-columnNames

#makeModelPlot(SIRVData)

SolveSIRV<-function(sigma){
  
  paramsL <- c(be=0.05,ga=0.85,mu=4,sig=sigma)
  YL <- c(1000,1,0,1000)
  timesL <- seq(0, 4, by = 0.01)
  outL <- ode(YL,timesL,SIRSV,paramsL)
  
  SIRVLData<-data.frame(outL)
  columnNames<-c("Time","Suceptible","Infected","Recovered","Total")
  colnames(SIRVLData)<-columnNames
  
  SIRVLData$Infected
}

paramRange<- seq(0,4,by=0.1)
infectionDepletion<-rep(0,length(paramRange))
fraction<-rep(0,length(paramRange))

for (i in 1:length(paramRange)){
  cSolution<-SolveSIRV(paramRange[i])
  infectionDepletion[i]<-rectangleRule(cSolution)
  fraction[i]<-(4-paramRange[i])/4
}

infectionDepletion<-(infectionDepletion[1]-infectionDepletion)/infectionDepletion[1]

vaccination<-data.frame("Infection"=infectionDepletion,"Vaccination"=fraction)

Vplot<-ggplot(data=vaccination,aes(x=Vaccination,y=Infection))+geom_line()
#show(Vplot)
