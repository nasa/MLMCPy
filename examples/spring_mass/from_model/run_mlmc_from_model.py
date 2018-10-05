import numpy as np

from spring_mass import SpringMassModel
from MLMCPy.input import RandomInput
from MLMCPy.mlmc import MLMCSimulator

'''
This script demonstrates MLMCPy for simulating a spring-mass system with a 
random spring stiffness to estimate the expected value of the maximum 
displacement using multi-level Monte Carlo. Here, we use Model and RandomInput
objects with functional forms as inputs to MLMCPy. See the
/examples/spring_mass/from_data/ for an example of using precomputed data
in files as inputs.
'''

#Step 1 - define random variable for spring stiffness:
#Need to provide a sampleable function to create RandomInput instance in MLMCPy
def beta_distribution(shift, scale, alpha, beta, size):

    return shift + scale*np.random.beta(alpha, beta, size)

stiffness_distribution = RandomInput(distribution_function=beta_distribution,
                                     shift=1.0, scale=2.5, alpha=3., beta=2.)

#Step 2 - initialize spring-mass models. Here using three levels with MLMC 
#defined by different time steps
model_level1 = SpringMassModel(mass=1.5, time_step=1.0, cost=.1)
model_level2 = SpringMassModel(mass=1.5, time_step=0.1, cost=1.0)
model_level3 = SpringMassModel(mass=1.5, time_step=0.01, cost=10.0)

models = [model_level1, model_level2, model_level3]

#Step 3 - initialize MLMC & predict max displacement to specified error 
mlmc_simulator = MLMCSimulator(stiffness_distribution, models)
[max_disp, num_evals, final_error] = mlmc_simulator.simulate(epsilon=1e-3,
                                                             verbose=True)
