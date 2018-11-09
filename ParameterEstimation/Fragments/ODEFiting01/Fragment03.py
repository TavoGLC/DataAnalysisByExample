#Data size lenghts to test 
nums=[1000,500,100,50,25,10]

#Library of ODE solutions 
t_lib=[np.linspace(0,2,num=val) for val in nums]
sol_lib=[ODESolution(clib,kc) for clib in t_lib]

#Library of simulated data 
noises=[[np.random.uniform(low=-1,high=1)/20 for val in sol] for sol in sol_lib]
signal=[MakeNoisyData(sol,nos) for sol,nos in zip(sol_lib,noises)]

#Parameter estimation an performance evaluation 
params=[curve_fit(ODESolution,times,signals)[0][0] for times,signals in zip(t_lib,signal)] 
solutions=[ODESolution(times,kS) for times,kS in zip(t_lib,params)]
