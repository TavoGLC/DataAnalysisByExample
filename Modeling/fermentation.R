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

Monod<-function(S,Um,Ks){
  rate<-Um*S/(S+Ks)
  rate
}

###############################################################################
# Plot Functions
###############################################################################

MakeMonodFermentationPlot<-function(inputData,PlotTitle){
  #Returns a ggplot
  #inputData  -> model data
  #ColumNames -> Names for the columns in the data
  #PlotTitle  -> Title for the plot
  ModelData<-inputData
  
  graphContainer<-ggplot(data=ModelData,aes(x=Time,y=Biomass,color="Biomass"))+geom_line()+
    geom_line(aes(y=Substrate,color="Substrate"))+
    geom_line(aes(y=Product,color="Product"))+
    scale_color_manual(values=c("Biomass"="black","Substrate"="red","Product"="green"))+
    labs(title=PlotTitle,color=" ")+
    theme(axis.title.y = element_blank(),axis.text.y = element_blank(),axis.ticks.y = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
  show(graphContainer)
}

###############################################################################
#Models
###############################################################################

MonodFermentationModel <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]  -> Foward and reverse primer content
  #          Y[2]  -> Sample DNA foward and reverse strand
  #          Y[3]  -> Complex sampleDNA-primer foward and reverse 
  #
  #params -> Parameters of the ODE system model 
  #         Um      -> Max growth rate
  #         Ks      -> Substrate constant
  #         Yxs     -> Substrate yield coefficient
  #         k1      -> Product formation constant
  #         k2      -> Product formation constant
  #    
  
  with(as.list(c(Y,params)),{
    
    dXdt=Monod(Y[2],Um,Ks)*Y[1]
    dSdt=-(Monod(Y[2],Um,Ks)*Y[1])/Yxs
    dPdt=(k1+k2*Monod(Y[2],Um,Ks))*Y[1]
    
    list(c(dXdt,dSdt,dPdt))
  })
}

params <- c(Um=1.5,Ks=0.751,Yxs=0.85,k1=0.01,k2=0.1)
Y <- c(1,100,0.01)

columnNames<-c("Time","Biomass","Substrate","Product")
fermentationData<-solveModel(MonodFermentationModel,Y,params,columnNames,FALSE)
MakeMonodFermentationPlot(fermentationData,"Fermentation High Substrate")

params <- c(Um=1.5,Ks=0.751,Yxs=0.85,k1=0.01,k2=0.1)
Y <- c(1,0.5,0.01)

columnNames<-c("Time","Biomass","Substrate","Product")
fermentationData<-solveModel(MonodFermentationModel,Y,params,columnNames,FALSE)
MakeMonodFermentationPlot(fermentationData,"Fermentation Low Substrate")

###############################################################################
#Models
###############################################################################

MonodFermentationInhibitionModel <- function(t,Y,params){
  
  #Simple organic matter decomposition model 
  #t      -> integration time value 
  # 
  #Y      -> list Values for the function to be evaluated 
  #          Y[1]  -> Foward and reverse primer content
  #          Y[2]  -> Sample DNA foward and reverse strand
  #          Y[3]  -> Complex sampleDNA-primer foward and reverse 
  #
  #params -> Parameters of the ODE system model 
  #         Um      -> Max growth rate
  #         Ks      -> Substrate constant
  #         Yxs     -> Substrate yield coefficient
  #         k1      -> Product formation constant
  #         k2      -> Product formation constant
  #         cp      -> Product inhibitory concentration
  
  with(as.list(c(Y,params)),{
    
    dXdt=((1-Y[3]/cp))*Monod(Y[2],Um,Ks)*Y[1]
    dSdt=-(((1-Y[3]/cp))*Monod(Y[2],Um,Ks)*Y[1])/Yxs
    dPdt=(k1+k2*((1-Y[3]/cp))*Monod(Y[2],Um,Ks))*Y[1]
    
    list(c(dXdt,dSdt,dPdt))
  })
}

params <- c(Um=1.5,Ks=0.751,Yxs=0.85,k1=0.01,k2=0.1,cp=25)
Y <- c(1,100,0.01)

columnNames<-c("Time","Biomass","Substrate","Product")
fermentationData<-solveModel(MonodFermentationInhibitionModel,Y,params,columnNames,FALSE)
MakeMonodFermentationPlot(fermentationData,"High Inhibitory Concentration")

params <- c(Um=1.5,Ks=0.751,Yxs=0.85,k1=0.01,k2=0.1,cp=5)
Y <- c(1,100,0.01)

columnNames<-c("Time","Biomass","Substrate","Product")
fermentationData<-solveModel(MonodFermentationInhibitionModel,Y,params,columnNames,FALSE)
MakeMonodFermentationPlot(fermentationData,"Low Inhibitory Concentration")


