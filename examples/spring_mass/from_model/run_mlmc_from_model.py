import numpy as np
import timeit

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


# Step 1 - Define random variable for spring stiffness:
# Need to provide a sampleable function to create RandomInput instance in MLMCPy
def beta_distribution(shift, scale, alpha, beta, size):

    return shift + scale*np.random.beta(alpha, beta, size)


stiffness_distribution = RandomInput(distribution_function=beta_distribution,
                                     shift=1.0, scale=2.5, alpha=3., beta=2.)


# Step 2 - Initialize spring-mass models. Here using three levels with MLMC.
# defined by different time steps
model_level1 = SpringMassModel(mass=1.5, time_step=1.0)
model_level2 = SpringMassModel(mass=1.5, time_step=0.1)
model_level3 = SpringMassModel(mass=1.5, time_step=0.01)

models = [model_level1, model_level2, model_level3]

# Step 3 - Initialize MLMC & predict max displacement to specified error.

mlmc_simulator = MLMCSimulator(stiffness_distribution, models)

start_mlmc = timeit.default_timer()

[estimates, sample_sizes, variances] = \
    mlmc_simulator.simulate(epsilon=1e-1,
                            initial_sample_size=100,
                            verbose=True)

mlmc_total_cost = timeit.default_timer() - start_mlmc

print
print 'MLMC estimate: %s' % estimates[0]
print 'MLMC precision: %s' % variances[0]
print 'MLMC total cost: %s' % mlmc_total_cost

# Step 4: Run standard Monte Carlo to achieve similar variance for comparison.
num_samples = 10000
input_samples = stiffness_distribution.draw_samples(num_samples)
output_samples = np.zeros(num_samples)

start_mc = timeit.default_timer()

for i, sample in enumerate(input_samples):
    output_samples[i] = model_level1.evaluate([sample])

mc_total_cost = timeit.default_timer() - start_mc

print
print "MC estimate: %s" % np.mean(output_samples)
print "MC precision: %s" % (np.var(output_samples) / float(num_samples))
print "MC total cost: %s" % mc_total_cost
