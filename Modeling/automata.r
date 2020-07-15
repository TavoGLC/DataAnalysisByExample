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

rulesPattern=matrix(0,8,3)

rulesPattern[1,]<-c(0,0,0)
rulesPattern[2,]<-c(0,0,1)
rulesPattern[3,]<-c(0,1,0)
rulesPattern[4,]<-c(0,1,1)
rulesPattern[5,]<-c(1,0,0)
rulesPattern[6,]<-c(1,0,1)
rulesPattern[7,]<-c(1,1,0)
rulesPattern[8,]<-c(1,1,1)

rulesValues=c(1,1,1,1,1,1,1,0)

changeRuleValues<-function(Threshold){
  
  finalRule<-rep(0,8)
  disc<-runif(8)
  for(k in 1:8){
    if (disc[k]>=Threshold){
      finalRule[k]<-1
    }
  }
  finalRule
}

###############################################################################
# Functions
###############################################################################

randomInitialState<-function(size){
  
  #creates a random list of 0,1 values of variabble size
  #size             
  #   -> int, size of the list 
  
  container<-rep(0,size)
  disc<-runif(size)
  for (k in 1:size){
    if (disc[k]>=0.5){
      container[k]<-1
    }
  }
  container
}

inputManagement<-function(Fragment){
  
  #Format the input and correct the edege by adding zeros to the end
  #Fragment             
  #   -> list, list of length 3 witn the current pattern
  
  fragmentSize=length(Fragment)
  for(k in 1:fragmentSize){
    if (is.na(Fragment[k])){
      Fragment[k]=0
    }
  }
  Fragment
}

splitRow<-function(row){
  
  #split a list in fragments of splitSize size
  #row             
  #   -> list, list of variable size 
  
  splitSize<-3
  rowSize<-length(row)
  splitedRow<-matrix(0,rowSize,splitSize)
  for(k in 1:rowSize){
    end<-k+splitSize-1
    currentFragment<-row[k:end]
    splitedRow[k,]<-inputManagement(currentFragment)
  }
  splitedRow
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
  out<--1
  for(k in 1:nRules){
    if(isTRUE(all.equal(value,RulesPatterns[k,]))){
      out<-RulesValues[k]
      break
    }
  }
  out
}

applyRule<-function(Row,RulesPatterns,RulesValues){
  
  #Wrapper function to apply the automata rule to a row
  #Row             
  #   -> list, Row o be evaluated
  #Row             
  #   -> list, Row o be evaluated
  #RulesPatterns
  #   -> matrix, contains the different patterns
  #RulesValues
  #   -> list, contains the output value for each pattern
  
  splitedVals<-splitRow(Row)
  nvalues<-length(splitedVals[,1])
  container<-rep(0,nvalues)
  for(k in 1:nvalues){
    container[k]<-findRuleValue(splitedVals[k,],RulesPatterns,RulesValues)
  }
  container
}

iterateAutomata<-function(Steps,InitialState,RulesPatterns,RulesValues){
  
  #Wrapper function to evaluate the automata
  #Steps             
  #   -> int, Iterations for the rules to be applied
  #InitialState             
  #   -> list, Row o be evaluated
  #RulesPatterns
  #   -> matrix, contains the different patterns
  #RulesValues
  #   -> list, contains the output value for each pattern
  
  automataContainer<-matrix(0,Steps+1,length(InitialState))
  automataContainer[1,]<-InitialState
  for(k in 1:Steps){
    currentState<-automataContainer[k,]
    nextState<-applyRule(currentState,RulesPatterns,RulesValues)
    automataContainer[k+1,]<-nextState
  }
  automataContainer
}


iterateDynamicRuleAutomata<-function(Steps,StepsToChange,InitialState,RulesPatterns){
  
  #Wrapper function to evaluate the automata
  #Steps             
  #   -> int, Iterations for the rules to be applied
  #StepsToChange             
  #   -> int, Iterations for the rules to change
  #InitialState             
  #   -> list, Row o be evaluated
  #RulesPatterns
  #   -> matrix, contains the different patterns
  #RulesValues
  #   -> list, contains the output value for each pattern
  
  automataContainer<-matrix(0,Steps+1,length(InitialState))
  automataContainer[1,]<-InitialState
  currentRules<-changeRuleValues(0.5)
  
  for(k in 1:Steps){
    
    if(k%%StepsToChange==0){
      currentRules<-changeRuleValues(0.5)
    }
    currentState<-automataContainer[k,]
    nextState<-applyRule(currentState,RulesPatterns,currentRules)
    automataContainer[k+1,]<-nextState
  }
  automataContainer
}


###############################################################################
# Automata Visualization
###############################################################################

initialSt<-randomInitialState(300)
#automataIt<-iterateAutomata(600,initialSt,rulesPattern,rulesValues)

#image(automataIt,axes=FALSE,col=gray((0:32)/32))

automataItD<-iterateDynamicRuleAutomata(600,4,initialSt,rulesPattern)
image(automataItD,axes=FALSE,col=gray((0:32)/32))
