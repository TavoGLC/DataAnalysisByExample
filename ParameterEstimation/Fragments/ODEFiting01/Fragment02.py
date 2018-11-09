#General function to solve the ODE model 
def GeneralSolver(t,k,C0):
    
    localK=k
    localC0=C0
    
    def ODEModel(C,t):
    
        return -localK*C

    sol=odeint(ODEModel,localC0,t)
    
    return sol[:,0]

#Solves the ODE model using the initial condition provided above
def ODESolution(t,k):
    
    return GeneralSolver(t,k,C0)
    
#Element wise sum of two iterables of the same size, name makes reference to the output rather than the process
def MakeNoisyData(Data,Noise):
    
    return [val+cal for val,cal in zip(Data,Noise)]
    
#Solving the ODE model 
t_vals=np.linspace(0,2,num=1000)
solution=ODESolution(t_vals,kc)

#Making some simulated data to perform regression 
WhiteNoise=[np.random.uniform(low=-1,high=1)/20 for val in solution]
WhiteSignal=MakeNoisyData(solution,WhiteNoise)
Kp=curve_fit(ODESolution,t_vals,WhiteSignal)[0][0]

#Parameter estimation 
fitSolution=ODESolution(t_vals,Kp)

