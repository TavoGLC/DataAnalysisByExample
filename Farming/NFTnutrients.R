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
library(gridExtra)

###############################################################################
# Visualization functions
###############################################################################
concentrationNames<-c("Time","ReserveWater","CirculatingWater","CropsUptake","NO3",
                      "NH4","K","PO4","SO4","Ca","Mg")

makeConcentrationPlot<-function(modelData,PlotTitle){
  
  #Returns a ggplot object 
  #modelData -> dataframe with the integration results of the volume model
  colorSpecs<-c("ReserveWater"="black","CirculatingWater"="red","CropsUptake"="blue","NO3"="orange",
                "NH4"="yellowgreen","K"="violetred1" ,"PO4"="seagreen4",
                "SO4"="purple2","Ca"="orangered4","Mg"="navy")
  
  modelPlot<-ggplot(data=modelData,aes(x=Time,y=ReserveWater,color="ReserveWater"))+geom_line()+
    geom_line(aes(y=CirculatingWater,color="CirculatingWater"))+
    geom_line(aes(y=CropsUptake,color="CropsUptake"))+
    geom_line(aes(y=NO3,color="NO3"))+
    geom_line(aes(y=NH4,color="NH4"))+
    geom_line(aes(y=K,color="K"))+
    geom_line(aes(y=PO4,color="PO4"))+
    geom_line(aes(y=SO4,color="SO4"))+
    geom_line(aes(y=Ca,color="Ca"))+
    geom_line(aes(y=Mg,color="Mg"))+
    scale_color_manual(values=colorSpecs)+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  
  KPOPlot<-ggplot(data=modelData,aes(x=NH4,y=K,color="K-NH4"))+geom_point()+
    scale_color_manual(values=c("K-NH4"="orangered4"))+
    labs(color=" ")+
    theme(axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  
  NOCaPlot<-ggplot(data=modelData,aes(x=NO3,y=Ca,color="NO3-Ca"))+geom_point()+
    scale_color_manual(values=c("NO3-Ca"="seagreen4"))+
    labs(color=" ")+
    theme(axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  
  final<-grid.arrange(modelPlot,arrangeGrob(KPOPlot,NOCaPlot,ncol=2),nrow=2)
  
  show(final)
}

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
# Utility functions
###############################################################################

michaelisKinetics<-function(Vmax,Km,S){
  out<-(Vmax*S)/(Km+S)
  out
}

inhibitionKinetics<-function(Vmax,Km,Ki,S,Si){
  alpha<-Vmax/(1+(Si/Ki))
  out<-alpha*(S/(Km+S))
  out
}

###############################################################################
# Ion Concentration Model Evolution
###############################################################################

concentrationModel <- function(t,Y,params){
  
  #Volume models, descrives the water use dynamics of an NFT hydroponic system
  #t      -> integration time value 
  #
  #Y      -> list Values for the function to be evaluated 
  #          R1-> Reserve Water, 
  #          R2-> Runing water through the pipes, 
  #          R3-> Water taken by the crops
  #
  #params -> Parameters of the ODE system model 
  #          Ycrit-> Minimum volume of water for the pump to work
  #          ke   -> entry rate to R2
  #          ks   -> entry rate to R1 from R2
  #          ku   -> crop water usage rate
  #          kt   -> crop transpiration rate
  #          Vm1  -> Max uptake rate for NO3-
  #          Km1  -> Affinity constant for NO3-
  #          Vm2  -> Max uptake rate for NH4+
  #          Km2  -> Affinity constant for NH4+
  #          Vm3  -> Max uptake rate for K+
  #          Km3  -> Affinity constant for K+
  #          Vm4  -> Max uptake rate for PO4
  #          Km4  -> Affinity constant for PO4
  #          Vm5  -> Max uptake rate for SO4
  #          Km5  -> Affinity constant for SO4
  #          Vm6  -> Max uptake rate for Ca+
  #          Km6  -> Affinity constant for Ca+
  #          Vm7  -> Max uptake rate for Mg+
  #          Km7  -> Affinity constant for Mg+
  
  with(as.list(c(Y,params)),{
    
    if (Y[1]>Ycrit){
      dR1dt=ke*Y[2]-ks*Y[1]
      dR2dt=-ke*Y[2]+ks*Y[1]-ku*Y[3]*Y[2]
      dR3dt=ku*Y[3]*Y[2]-kt*Y[3]
      dS1dt=-michaelisKinetics(Vm1,Km1,Y[4]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS2dt=-michaelisKinetics(Vm2,Km2,Y[5]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS3dt=-michaelisKinetics(Vm3,Km3,Y[6]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS4dt=-michaelisKinetics(Vm4,Km4,Y[7]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS5dt=-michaelisKinetics(Vm5,Km5,Y[8]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS6dt=-michaelisKinetics(Vm6,Km6,Y[9]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS7dt=-michaelisKinetics(Vm7,Km7,Y[10]/(Y[1]+Y[2]))*(Y[1]+Y[2])
    }
    else{
      if (Y[2]>=0){
        dR1dt=ke*Y[2]
        dR2dt=-ke*Y[2]-ku*Y[3]*Y[2]
        dR3dt=ku*Y[3]*Y[2]-kt*Y[3]
        dS1dt=-michaelisKinetics(Vm1,Km1,Y[4]/(Y[1]+Y[2]))*(Y[1]+Y[2])
        dS2dt=-michaelisKinetics(Vm2,Km2,Y[5]/(Y[1]+Y[2]))*(Y[1]+Y[2])
        dS3dt=-michaelisKinetics(Vm3,Km3,Y[6]/(Y[1]+Y[2]))*(Y[1]+Y[2])
        dS4dt=-michaelisKinetics(Vm4,Km4,Y[7]/(Y[1]+Y[2]))*(Y[1]+Y[2])
        dS5dt=-michaelisKinetics(Vm5,Km5,Y[8]/(Y[1]+Y[2]))*(Y[1]+Y[2])
        dS6dt=-michaelisKinetics(Vm6,Km6,Y[9]/(Y[1]+Y[2]))*(Y[1]+Y[2])
        dS7dt=-michaelisKinetics(Vm7,Km7,Y[10]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      }
      else{
        dR1dt=0
        dR2dt=0
        dR3dt=-kt*Y[3]
        dS1dt=0
        dS2dt=0
        dS3dt=0
        dS4dt=0
        dS5dt=0
        dS6dt=0
        dS7dt=0
      }
    }
    
    list(c(dR1dt,dR2dt,dR3dt,dS1dt,dS2dt,dS3dt,dS4dt,dS5dt,dS6dt,dS7dt))
  })
}

params <- c(ke=1,ks=10,ku=0.0025,kt=0.025,Ycrit=10,K=0.005,
            Vm1=2.1,Km1=1.1,Vm2=1.1,Km2=2.5,Vm3=0.51,Km3=0.75,Vm4=2.1,Km4=2.1,
            Vm5=1.1,Km5=0.75,Vm6=0.81,Km6=0.71,Vm7=0.75,Km7=0.81)

Y <-  c(50,100,15,50,130,70,270,70,50,100)

#concentrationData<-solveModel(concentrationModel,Y,params,concentrationNames)
#makeConcentrationPlot(concentrationData,"High Discharge Rate")

###############################################################################
# Ion inhibition Model Evolution
###############################################################################

inhibitionModel <- function(t,Y,params){
  
  #Volume models, descrives the water use dynamics of an NFT hydroponic system
  #t      -> integration time value 
  #
  #Y      -> list Values for the function to be evaluated 
  #          R1-> Reserve Water, 
  #          R2-> Runing water through the pipes, 
  #          R3-> Water taken by the crops
  #
  #params -> Parameters of the ODE system model 
  #          Ycrit-> Minimum volume of water for the pump to work
  #          ke   -> entry rate to R2
  #          ks   -> entry rate to R1 from R2
  #          ku   -> crop water usage rate
  #          kt   -> crop transpiration rate
  #          Vm1  -> Max uptake rate for NO3-
  #          Km1  -> Affinity constant for NO3-
  #          Vm2  -> Max uptake rate for NH4+
  #          Km2  -> Affinity constant for NH4+
  #          Vm3  -> Max uptake rate for K+
  #          Km3  -> Affinity constant for K+
  #          Vm4  -> Max uptake rate for PO4
  #          Km4  -> Affinity constant for PO4
  #          Vm5  -> Max uptake rate for SO4
  #          Km5  -> Affinity constant for SO4
  #          Vm6  -> Max uptake rate for Ca+
  #          Km6  -> Affinity constant for Ca+
  #          Vm7  -> Max uptake rate for Mg+
  #          Km7  -> Affinity constant for Mg+
  
  with(as.list(c(Y,params)),{
    
    if (Y[1]>Ycrit){
      dR1dt=ke*Y[2]-ks*Y[1]
      dR2dt=-ke*Y[2]+ks*Y[1]-ku*Y[3]*Y[2]
      dR3dt=ku*Y[3]*Y[2]-kt*Y[3]
      dS1dt=-michaelisKinetics(Vm1,Km1,Y[4]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS2dt=-inhibitionKinetics(Vm2,Km2,Ki2,Y[5]/(Y[1]+Y[2]),Y[6]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS3dt=-inhibitionKinetics(Vm3,Km3,Ki3,Y[6]/(Y[1]+Y[2]),Y[5]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS4dt=-michaelisKinetics(Vm4,Km4,Y[7]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS5dt=-michaelisKinetics(Vm5,Km5,Y[8]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS6dt=-michaelisKinetics(Vm6,Km6,Y[9]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      dS7dt=-michaelisKinetics(Vm7,Km7,Y[10]/(Y[1]+Y[2]))*(Y[1]+Y[2])
    }
    else{
      if (Y[2]>=0){
        dR1dt=ke*Y[2]
        dR2dt=-ke*Y[2]-ku*Y[3]*Y[2]
        dR3dt=ku*Y[3]*Y[2]-kt*Y[3]
        dS1dt=-michaelisKinetics(Vm1,Km1,Y[4]/(Y[1]+Y[2]))*(Y[1]+Y[2])
        dS2dt=-inhibitionKinetics(Vm2,Km2,Ki2,Y[5]/(Y[1]+Y[2]),Y[6]/(Y[1]+Y[2]))
        dS3dt=-inhibitionKinetics(Vm3,Km3,Ki3,Y[6]/(Y[1]+Y[2]),Y[5]/(Y[1]+Y[2]))
        dS4dt=-michaelisKinetics(Vm4,Km4,Y[7]/(Y[1]+Y[2]))*(Y[1]+Y[2])
        dS5dt=-michaelisKinetics(Vm5,Km5,Y[8]/(Y[1]+Y[2]))*(Y[1]+Y[2])
        dS6dt=-michaelisKinetics(Vm6,Km6,Y[9]/(Y[1]+Y[2]))*(Y[1]+Y[2])
        dS7dt=-michaelisKinetics(Vm7,Km7,Y[10]/(Y[1]+Y[2]))*(Y[1]+Y[2])
      }
      else{
        dR1dt=0
        dR2dt=0
        dR3dt=-kt*Y[3]
        dS1dt=0
        dS2dt=0
        dS3dt=0
        dS4dt=0
        dS5dt=0
        dS6dt=0
        dS7dt=0
      }
    }
    
    list(c(dR1dt,dR2dt,dR3dt,dS1dt,dS2dt,dS3dt,dS4dt,dS5dt,dS6dt,dS7dt))
  })
}

params <- c(ke=1,ks=4,ku=0.0025,kt=0.025,Ycrit=10,K=0.005,
            Vm1=2.1,Km1=1.1,Vm2=1.1,Km2=2.5,Ki2=0.1,Vm3=0.51,Km3=0.75,Ki3=0.01,Vm4=2.1,Km4=2.1,
            Vm5=1.1,Km5=0.75,Vm6=0.81,Km6=0.71,Vm7=0.75,Km7=0.81)

Y <-  c(50,100,15,50,130,70,170,70,50,100)
concentrationData<-solveModel(inhibitionModel,Y,params,concentrationNames)
makeConcentrationPlot(concentrationData,"High K Inhibition")
