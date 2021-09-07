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
library(gridExtra)
library(grid)
library(lattice)

###############################################################################
# Plot Functions
###############################################################################

MakeActionPotentialPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphActionPotential<-ggplot(data=ModelData,aes(x=Time,y=ActionPotential,color="ActionPotential"))+geom_line()+
    labs(title="Action Potential")+
    scale_color_manual(values=c("ActionPotential"="red"))+
    theme(axis.title.y = element_blank(),axis.text.y = element_text(),
          axis.ticks.y = element_blank(),axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),legend.position = "none")
  
  graphPotassiumGating<-ggplot(data=ModelData,aes(x=Time,y=PotassiumGating,color="PotassiumGating"))+geom_line()+
    labs(title="Potassium Gating")+
    scale_color_manual(values=c("PotassiumGating"="blue"))+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),axis.title.x = element_blank(),
          legend.position = "none")
  
  graphSodiumActivation<-ggplot(data=ModelData,aes(x=Time,y=SodiumChannelActivation,color="SodiumChannelActivation"))+geom_line()+
    labs(title="Sodium Channel Activation")+
    scale_color_manual(values=c("SodiumChannelActivation"="orange"))+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),axis.title.x = element_blank(),
          legend.position = "none")
  
  graphSodiumInactivation<-ggplot(data=ModelData,aes(x=Time,y=SodiumChannelInactivation,color="SodiumChannelInactivation"))+geom_line()+
    labs(title="Sodium Channel Inactivation")+
    scale_color_manual(values=c("SodiumChannelInactivation"="cyan"))+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),legend.position = "none")
  
  gridarrange<-rbind(c(1,1,2),c(1,1,3),c(1,1,4))
  graphContainer<-grid.arrange(graphActionPotential,graphPotassiumGating,graphSodiumActivation,graphSodiumInactivation, layout_matrix=gridarrange)
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
  times <- seq(0, 25, by = 0.01)
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

AlphaN<-function(v){
  up<-0.01*(v+55)
  down<-1-exp(-(v+55)/10)
  an<-up/down
  an
}

BetaN<-function(v){
  bn<-0.125*exp(-(v+65)/80)
  bn
}

AlphaM<-function(v){
  up<-0.1*(v+40)
  down<-1-exp(-(v+40)/10)
  am<-up/down
  am
}

BetaM<-function(v){
  bm<-4*exp(-(v+65)/18)
  bm
}

AlphaH<-function(v){
  ah<-0.07*exp(-(v+65)/20)
  ah
}

BetaH<-function(v){
  down<-1+exp(-(v+35)/10)
  bh<-1/down
  bh
}

Im<-function(t,Impulse){
  responce<-0
  if(t<5 & t>3){
    responce<-Impulse
  }
  else{
    responce<-0
  }
  responce
}

ActionPotentialModel <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]     -> Potassium channel gating 
  #          Y[2]     -> Sodium channel activation 
  #          Y[3]     -> Sodium channel inactivation
  #          Y[4]     -> Action Potential
  #
  #params -> Parameters of the ODE system model 
  #         gk      -> Postassium (K) maximum conductances
  #         Vk      -> Postassium (K) Nernst reversal potentials
  #         gna     -> Sodium (Na) maximum conductances
  #         Vna     -> Sodium (Na) Nernst reversal potentials
  #         gl      -> Leak maximum conductances
  #         Vl      -> Leak Nernst reversal potentials
  #         Cm      -> membrane capacitance
  #         imp     -> External Current
  #    
  
  with(as.list(c(Y,params)),{
    
    dndt=AlphaN(Y[4])*(1-Y[1])-BetaN(Y[4])*Y[1]
    dmdt=AlphaM(Y[4])*(1-Y[2])-BetaM(Y[4])*Y[2]
    dhdt=AlphaH(Y[4])*(1-Y[3])-BetaH(Y[4])*Y[3]
  
    dvdt=(Im(t,imp)-gk*(Y[1]**4)*(Y[4]-Vk)-gna*(Y[2]**3)*Y[3]*(Y[4]-Vna)-gl*(Y[4]-Vl))/Cm
    
    list(c(dndt,dmdt,dhdt,dvdt))
  })
}

params <- c(gk=36,Vk=-77,gna=120,Vna=50,gl=0.3,Vl=-54.387,Cm=1,imp=35)
Y <- c(0.32,0.05,0.6,-65)

columnNames<-c("Time","PotassiumGating","SodiumChannelActivation","SodiumChannelInactivation","ActionPotential")
ActionPotentialData<-solveModel(ActionPotentialModel,Y,params,columnNames,FALSE)
MakeActionPotentialPlot(ActionPotentialData,"Action Potential")

###############################################################################
#All or nothing character of the action potential 
###############################################################################

k<-1
Impulses<-seq(0, 8, by = 0.1)
maxVoltages<-rep(0,length(Impulses))

for (val in Impulses){
  localParams<-c(gk=36,Vk=-77,gna=120,Vna=50,gl=0.3,Vl=-54.387,Cm=1,imp=val)
  localData<-solveModel(ActionPotentialModel,Y,localParams,columnNames,FALSE)
  maxVoltages[k]<-max(localData$ActionPotential)
  k<-k+1
}

IVData<-Container<-matrix(0,length(Impulses),2)
IVData[,1]<-Impulses
IVData[,2]<-maxVoltages
IVData<-data.frame(IVData)
colnames(IVData)<-c("Current","Voltage")


graphIV<-ggplot(data=IVData,aes(x=Current,y=Voltage,color="Voltage"))+geom_point(shape=8)+
  labs(title="Action Potential")+
  scale_color_manual(values=c("Voltage"="black"))+
  theme(axis.text.y = element_text(),axis.ticks.y = element_blank(),
        axis.text.x = element_text(),axis.ticks.x = element_blank(),
        legend.position = "none")

show(graphIV)

###############################################################################
#Membrane changes 
###############################################################################

MakeVoltagePlot<-function(ModelData,Title){
  graphActionPotential<-ggplot(data=ModelData,aes(x=Time,y=LowCapacitance,color="LowCapacitance"))+geom_line()+
    geom_line(aes(y=NormalCapacitance,color="NormalCapacitance"))+
    labs(title=Title,color="")+
    scale_color_manual(values=c("LowCapacitance"="red","NormalCapacitance"="black"))+
    theme(axis.title.y = element_blank(),axis.text.y = element_text(),
          axis.ticks.y = element_blank(),axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
  show(graphActionPotential)
}

makeComparisonDataFrame <- function(Capacitance,current){
  params <- c(gk=36,Vk=-77,gna=120,Vna=50,gl=0.3,Vl=-54.387,Cm=Capacitance[1],imp=current)
  Y <- c(0.32,0.05,0.6,-65)
  columnNames<-c("Time","PotassiumGating","SodiumChannelActivation","SodiumChannelInactivation","ActionPotential")
  lowCapacitance<-solveModel(ActionPotentialModel,Y,params,columnNames,FALSE)
  
  params <- c(gk=36,Vk=-77,gna=120,Vna=50,gl=0.3,Vl=-54.387,Cm=Capacitance[2],imp=current)
  Y <- c(0.32,0.05,0.6,-65)
  columnNames<-c("Time","PotassiumGating","SodiumChannelActivation","SodiumChannelInactivation","ActionPotential")
  normalCapacitance<-solveModel(ActionPotentialModel,Y,params,columnNames,FALSE)
  
  capacitanceDF <- data.frame(lowCapacitance$Time,lowCapacitance$ActionPotential,normalCapacitance$ActionPotential)
  colnames(capacitanceDF)<-c("Time","LowCapacitance","NormalCapacitance")
  capacitanceDF
}

comparisonCapacitance<- makeComparisonDataFrame(c(0.5,1),35)

MakeVoltagePlot(comparisonCapacitance,"Normal Activation Current")

comparisonImpulse<- makeComparisonDataFrame(c(0.5,1),3.5)

MakeVoltagePlot(comparisonImpulse,"Low Activation Current")

