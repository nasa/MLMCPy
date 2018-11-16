import numpy as np
import timeit

from spring_mass import SpringMassModel
from MLMCPy.input import RandomInput

'''
This script calculates a brute force MLMC solution given numbers of samples
on level 0 and level 1. This provides the reference solution for the test:
"test_hard_coded_spring_mass_random_input"
'''

# Step 1 - Define random variable for spring stiffness:
# Need to provide a sampleable function to create RandomInput instance in MLMCPy
np.random.seed(1)
def beta_distribution(shift, scale, alpha, beta, size):
    return shift + scale*np.random.beta(alpha, beta, size)


stiffness_distribution = RandomInput(distribution_function=beta_distribution,
                                     shift=1.0, scale=2.5, alpha=3., beta=2.)

# Step 2 - Initialize spring-mass models. Here using three levels with MLMC.
# defined by different time steps
model_level1 = SpringMassModel(mass=1.5, time_step=1.0, cost=1.0)
model_level2 = SpringMassModel(mass=1.5, time_step=0.1, cost=10.0)

# Step 3 - Initialize MLMC & predict max displacement to specified error.

#These numbers were generated for epsilon=0.1
N0 = 1113
N1 = 34

#Level 0 
outputs0 = np.zeros(N0)
inputs0 = stiffness_distribution.draw_samples(N0)

for i,sample in enumerate(inputs0):
    outputs0[i] = model_level1.evaluate([sample])

#Level 1:
inputs1 = stiffness_distribution.draw_samples(N1)
outputs1 = np.zeros(N1)

for i,sample in enumerate(inputs1):
    outputs1[i] = (model_level2.evaluate([sample]) -  
                   model_level1.evaluate([sample]))

#Combine levels for estimates
mean = np.mean(outputs0) + np.mean(outputs1)
print "Mean estimate: ", mean

var = np.var(outputs0)/N0 + np.var(outputs1)/N1
print "Estimator variance: ", var
