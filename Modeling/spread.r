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
# Automata definitions
###############################################################################

rulesPattern=matrix(0,16,4)

rulesPattern[1,]<-c(1,1,1,1)
rulesPattern[2,]<-c(1,0,1,1)
rulesPattern[3,]<-c(1,1,1,0)
rulesPattern[4,]<-c(1,0,1,0)
rulesPattern[5,]<-c(1,1,0,1)
rulesPattern[6,]<-c(1,0,0,1)
rulesPattern[7,]<-c(1,1,0,0)
rulesPattern[8,]<-c(1,0,0,0)
rulesPattern[9,]<-c(0,1,1,1)
rulesPattern[10,]<-c(0,0,1,1)
rulesPattern[11,]<-c(0,1,1,0)
rulesPattern[12,]<-c(0,0,1,0)
rulesPattern[13,]<-c(0,1,0,1)
rulesPattern[14,]<-c(0,0,0,1)
rulesPattern[15,]<-c(0,1,0,0)
rulesPattern[16,]<-c(0,0,0,0)


rulesValues=c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0)

###############################################################################
# Automata functions
###############################################################################

getPointNeighbours<-function(Xposition,Yposition,currentState){
  
  dimensions=dim(currentState)
  
  if(Yposition+1>dimensions[2] | Yposition+1==0){
    top<-0
  }
  else{
    top<-currentState[Xposition,Yposition+1]
  }
  
  if (Yposition-1>dimensions[2] | Yposition-1==0){
    bottom<-0
  }
  else{
    bottom<-currentState[Xposition,Yposition-1]
  }
  
  if(Xposition-1>dimensions[1] | Xposition-1==0){
    left<-0
  }
  else{
    left<-currentState[Xposition-1,Yposition]
  }
  
  if (Xposition+1>dimensions[1] | Xposition+1==0){
    right<-0
  }
  else{
    right<-currentState[Xposition+1,Yposition]
  }
  
  positions<-c(top,bottom,left,right)
}


findRuleValue<-function(value,RulesPatterns,RulesValues){
  
  #find the correspondig value for the current value
  #value
  #   -> list, current fragment to be evaluated
  #RulesPatterns
  #   -> matrix, contains the different patterns
  #RulesValues
  #   -> list, contains the output value for each pattern
  
  nRules<-length(RulesPatterns[,1])
  out<-2
  for(k in 1:nRules){
    if(isTRUE(all.equal(value,RulesPatterns[k,]))){
      out<-RulesValues[k]
      break
    }
  }
  out
}

###############################################################################
# Functions
###############################################################################

randomInitialState<-function(Xsize,Ysize,Threshold){
  
  #creates a random list of 0,1 values of variabble size
  #size             
  #   -> int, size of the list 
  Container<-matrix(0,Xsize,Ysize)
  for (k in 1:Ysize){
    localDisc<-runif(Xsize)
    localVec<-rep(0,Xsize)
    for (j in 1:Xsize){
      if (localDisc[j]>=Threshold){
        localVec[j]<-1
      }
    }
    Container[,k]<-localVec
  }
  Container
}

spacedGridState<-function(Xsize,Ysize,Space,Counter){
  Container<-matrix(0,Xsize,Ysize)
  currentX<-0
  currentY<-0
  rowState<-0
  for(k in 1:Counter){
    if(currentX>=Xsize | currentX+Space>=Xsize){
      if (rowState==0){
        currentX<-0
        rowState<-1
      }
      else{
        rowState<-0
        currentX<-as.integer(Space/2)
      }
      
      currentY<-currentY+Space
    }
    else{
      currentX<-currentX+Space
    }
    Container[currentX,currentY]<-1
  }
  Container
}

SocialDistancingState<-function(State){
  
  #creates a random list of 0,1 values of variabble size
  #size             
  #   -> int, size of the list 
  Step<-3
  Sizes<-dim(State)
  Ysize<-Sizes[2]
  Xsize<-Sizes[1]
  
  for (k in seq(1,Ysize,Step)){
    State[,k]<-rep(0,Xsize)
  }
  for (k in seq(1,Xsize,Step)){
    State[k,]<-rep(0,Ysize)
  }
  State
}

###############################################################################
# Functions
###############################################################################

iterateAutomata<-function(InitialState,Steps,RulesPatterns,RulesValues,distancing){
  
  #Wrapper function to evaluate the automata
  #Steps             
  #   -> int, Iterations for the rules to be applied
  #InitialState             
  #   -> list, Row o be evaluated
  #RulesPatterns
  #   -> matrix, contains the different patterns
  #RulesValues
  #   -> list, contains the output value for each pattern
  if(distancing==TRUE){
    automataContainer<-SocialDistancingState(InitialState)
  }
  else{
    automataContainer<-InitialState
  }
  
  png('automataStep0.png')
  image(automataContainer,axes=FALSE,col=gray((0:32)/32),main = paste("Spread fraction",toString(sum(automataContainer)/length(automataContainer))))
  dev.off()
  
  Sizes<-dim(automataContainer)
  Ysize<-Sizes[2]
  Xsize<-Sizes[1]
    
  for(k in 1:Steps){
    newState<-matrix(0,Xsize,Ysize)
    for(Ys in 1:Ysize){
      for(Xs in 1:Xsize){
        neighbours<-getPointNeighbours(Xs,Ys,automataContainer)
        newVal<-findRuleValue(neighbours,RulesPatterns,RulesValues)
        newState[Xs,Ys]<-newVal
      }
    }
    if(distancing==TRUE){
      newState<-SocialDistancingState(newState)
    }
    png(paste('automataStep',toString(k),'.png',sep=''))
    image(newState,axes=FALSE,col=gray((0:32)/32),main = paste("Spread fraction",toString(sum(newState)/length(newState))))
    dev.off()
    automataContainer<-newState
  }
  automataContainer
}

initState<-spacedGridState(200,250,8,780)
iterateAutomata(initState,30,rulesPattern,rulesValues,TRUE)
