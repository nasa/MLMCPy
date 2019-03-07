import numpy as np
import matplotlib.pyplot as plt
import timeit

from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.input import RandomInput
from MLMCPy.model import CDFWrapperModel
from spring_mass_model import SpringMassModel
"""
This script demonstrates MLMCPy for simulating a spring-mass system with a
random spring stiffness. Here, we expand our usage of MLMC to produce CDFs using
the CDFWrapperModel in conjunction with the RandomInput and Model objects. See
/examples/spring_mass/from_model/run_mlmc_from_model.py for an example that
estimates the expected value of the maximum displacement using multi-level Monte
Carlo.
"""

# Step 1 - Define random variable for spring stiffness:
# Need to provide a sampleable function to create RandomInput instance in MLMCPy
def beta_distribution(shift, scale, alpha, beta, size):

    return shift + scale*np.random.beta(alpha, beta, size)

np.random.seed(1)
stiffness_distribution = RandomInput(distribution_function=beta_distribution,
                                     shift=1.0, scale=2.5, alpha=3., beta=2.,
                                     random_seed=1)

# Step 2 - Initialize spring-mass models for MLMC. Here using three levels
# with MLMC defined by different time steps:
model_level1 = SpringMassModel(mass=1.5, time_step=1.0)
model_level2 = SpringMassModel(mass=1.5, time_step=0.1)
model_level3 = SpringMassModel(mass=1.5, time_step=0.01)
models = [model_level1, model_level2, model_level3]

# Step 3 - Initialize the CDFWrapperModel object:
grid = np.linspace(8, 25, 100)
cdf_wrapper = CDFWrapperModel(grid)

# Step 4 - Initialize the MLMCSimulator and predict max displacement to
# specified precision:
precision = 2.5e-2
initial_sample = 100

print "Running MLMC..."
start_mlmc = timeit.default_timer()

mlmc_simulator = MLMCSimulator(stiffness_distribution, models, cdf_wrapper)

[estimates, sample_sizes, variances] = \
    mlmc_simulator.simulate(epsilon=precision,
                            initial_sample_sizes=initial_sample,
                            verbose=False)

mlmc_total_cost = timeit.default_timer() - start_mlmc


# Step 5 - Calculate Monte Carlo reference CDF
num_samples = 5000
model = SpringMassModel(mass=1.5, time_step=0.01)
input_samples = stiffness_distribution.draw_samples(num_samples)
output_samples_mc = np.zeros(num_samples)

print "Running Monte Carlo..."
start_mc = timeit.default_timer()

for i, sample in enumerate(input_samples):
    output_samples_mc[i] = model.evaluate([sample])

mc_total_cost = timeit.default_timer() - start_mc

print '\nMLMC number of samples: ', sample_sizes
print 'MLMC total cost: %s' % mlmc_total_cost
print "MC total cost: %s" % mc_total_cost
print "MLMC computational speedup: %s" %  (mc_total_cost / mlmc_total_cost)

# Step 6 - Compare CDFs

x_mc = np.sort(output_samples_mc)
cdf_mc = np.arange(len(x_mc))/float(len(x_mc))

plt.figure()
plt.plot(grid, estimates, 'r-', label="MLMC")
plt.plot(x_mc, cdf_mc, 'b-', label="MC")
plt.legend()
plt.show()
