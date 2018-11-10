#Generating noise data with mixed signals 
PeriodicNoise=[np.random.uniform(low=-1,high=1)/30+np.sin(val/np.pi)/30 for val in range(len(t_data))]
LinearNoise=[np.random.uniform(low=-1,high=1)/30-0.04*(val/30) for val in range(len(t_data))]
