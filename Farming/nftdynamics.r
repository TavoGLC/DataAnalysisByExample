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

solveModel<- function(Model,InitialConditions,ModelParameters,ColumnNames){
  #Solves numerically an ODE system model,returns a formated dataframe
  #Model             -> function, Model to be solved
  #InitialConditions -> list, Initial conditions for the ODE system 
  #ModelParameters   -> list, Parameters of the ODE model 
  #ColumnNames       -> list, names of the columns for the dataframe 
  times <- seq(0, 12, by = 0.01)
  out <- ode(InitialConditions,times,Model,ModelParameters,method="rk4")
  ModelData<-data.frame(out)
  colnames(ModelData)<-ColumnNames
  ModelData
}

###############################################################################
# Visualization functions
###############################################################################

makeVolumePlot<-function(modelData,PlotTitle){
  #Returns a ggplot object 
  #modelData -> dataframe with the integration results of the volume model
  modelPlot<-ggplot(data=modelData,aes(x=Time,y=ReserveWater,color="ReserveWater"))+geom_line()+
    geom_line(aes(y=CirculatingWater,color="CirculatingWater"))+
    geom_line(aes(y=CropsUptake,color="CropsUptake"))+
    scale_color_manual(values=c("ReserveWater"="black","CirculatingWater"="red","CropsUptake"="blue"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

makeConcentrationPlot<-function(modelData,PlotTitle){
  #Returns a ggplot object 
  #modelData -> dataframe with the integration results of the volume model
  modelPlot<-ggplot(data=modelData,aes(x=Time,y=ReserveWater,color="ReserveWater"))+geom_line()+
    geom_line(aes(y=CirculatingWater,color="CirculatingWater"))+
    geom_line(aes(y=CropsUptake,color="CropsUptake"))+
    geom_line(aes(y=Concentration,color="Concentration"))+
    scale_color_manual(values=c("ReserveWater"="black","CirculatingWater"="red","CropsUptake"="blue","Concentration"="orange"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

makeCRPlot<-function(modelData,PlotTitle){
  #Returns a ggplot object 
  #modelData -> dataframe with the integration results of the volume model
  modelPlot<-ggplot(data=modelData,aes(x=ReserveWater,y=CirculatingWater,color="ReserveWater"))+geom_point()+
    scale_color_manual(values=c("ReserveWater"="black"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

makeCCPlot<-function(modelData,PlotTitle){
  #Returns a ggplot object 
  #modelData -> dataframe with the integration results of the volume model
  modelPlot<-ggplot(data=modelData,aes(x=CropsUptake,y=Concentration,color="Concentration"))+geom_point()+
    scale_color_manual(values=c("Concentration"="black"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(modelPlot)
}

###############################################################################
# Water Volume Evolution
###############################################################################

volumeModel <- function(t,Y,params){
  
  #Volume models, descrives the water use dynamics of an NFT hydroponic system
  #t      -> integration time value 
  #Y      -> list Values for the function to be evaluated 
  #          R1-> Reserve Water, 
  #          R2-> Runing water through the pipes, 
  #          R3-> Water taken by the crops
  #params -> Parameters of the ODE system model 
  #          Ycrit-> Minimum volume of water for the pump to work
  #          ke   -> entry rate to R2
  #          ks   -> entry rate to R1 from R2
  #          ku   -> crop water usage rate
  #          kt   -> crop transpiration rate
  
  with(as.list(c(Y,params)),{
    
    if (Y[1]>Ycrit){
      dR1dt=ke*Y[2]-ks*Y[1]
      dR2dt=-ke*Y[2]+ks*Y[1]-ku*Y[3]*Y[2]
      dR3dt=ku*Y[3]*Y[2]-kt*Y[3]
    }
    else{
      if (Y[2]>=0){
        dR1dt=ke*Y[2]
        dR2dt=-ke*Y[2]-ku*Y[3]*Y[2]
        dR3dt=ku*Y[3]*Y[2]-kt*Y[3]
      }
      else{
        dR1dt=0
        dR2dt=0
        dR3dt=-kt*Y[3]
      }
    }
    list(c(dR1dt,dR2dt,dR3dt))
  })
}

params <- c(ke=1,ks=4,ku=0.0025,kt=0.00025,Ycrit=10)
Y <- c(50,100,15)
volumeNames<-c("Time","ReserveWater","CirculatingWater","CropsUptake")

volumeData<-solveModel(volumeModel,Y,params,volumeNames)
#makeVolumePlot(volumeData,"High input rate")
#makeCRPlot(volumeData,"High input rate")

###############################################################################
# Water Volume Evolution
###############################################################################

concentrationModel <- function(t,Y,params){
  
  #Volume models, descrives the water use dynamics of an NFT hydroponic system
  #t      -> integration time value 
  #Y      -> list Values for the function to be evaluated 
  #          R1-> Reserve Water, 
  #          R2-> Runing water through the pipes, 
  #          R3-> Water taken by the crops
  #params -> Parameters of the ODE system model 
  #          Ycrit-> Minimum volume of water for the pump to work
  #          ke   -> entry rate to R2
  #          ks   -> entry rate to R1 from R2
  #          ku   -> crop water usage rate
  #          kt   -> crop transpiration rate
  #          C0   -> Grams of fertilizer added
  #          K    -> Fertilizer consumption rate
  
  with(as.list(c(Y,params)),{
    
    if (Y[1]>Ycrit){
      dR1dt=ke*Y[2]-ks*Y[1]
      dR2dt=-ke*Y[2]+ks*Y[1]-ku*Y[3]*Y[2]
      dR3dt=ku*Y[3]*Y[2]-kt*Y[3]
      dUdt=(C0/(Y[1]+Y[2]))-K*Y[3]
    }
    else{
      if (Y[2]>=0){
        dR1dt=ke*Y[2]
        dR2dt=-ke*Y[2]-ku*Y[3]*Y[2]
        dR3dt=ku*Y[3]*Y[2]-kt*Y[3]
        dUdt=(C0/(Y[1]+Y[2]))-K*Y[3]
      }
      else{
        dR1dt=0
        dR2dt=0
        dR3dt=-kt*Y[3]
        dUdt=0
      }
    }
    
    list(c(dR1dt,dR2dt,dR3dt,dUdt))
  })
}

params <- c(ke=1,ks=4,ku=0.0025,kt=0.00025,Ycrit=10,C0=100,K=0.005)
Y <- c(50,100,15,15)
concentrationNames<-c("Time","ReserveWater","CirculatingWater","CropsUptake","Concentration")

concentrationData<-solveModel(concentrationModel,Y,params,concentrationNames)
makeConcentrationPlot(concentrationData,"High Input Rate & Higher Water Uptake Rate")
#makeCCPlot(concentrationData,"High Input Rate & Higher Water Uptake Rate")
