#Numpy array that contains the integration times 
SolverTime=np.linspace(0,2)

def ODE(C,t):
    
    return -kc*C

#Solution of the model 
ModelSolution=odeint(ODE,C0,SolverTime)
