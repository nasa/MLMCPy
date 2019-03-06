import timeit
import numpy as np

from spring_mass_model import SpringMassModel
from MLMCPy.input import RandomInput
from MLMCPy.mlmc import MLMCSimulator

"""
This script demonstrates the modularity of the MLMCPy Simulator by simulating a
spring-mass system with a random spring stiffness to estimate the expected value
of the maximum displacement using multi-level Monte Carlo. Here, we use Model
and RandomInput objects with functional forms as inputs to MLMCPy. See the
/examples/spring_mass/from_data/ for an example of using precomputed data
in files as inputs.
"""

# Step 1 - Define random variable for spring stiffness:
# Need to provide a sampleable function to create RandomInput instance in MLMCPy
def beta_distribution(shift, scale, alpha, beta, size):

    return shift + scale*np.random.beta(alpha, beta, size)

np.random.seed(1)
stiffness_distribution = RandomInput(distribution_function=beta_distribution,
                                     shift=1.0, scale=2.5, alpha=3., beta=2.,
                                     random_seed=1)

# Step 2 - Run standard Monte Carlo to generate a reference solution and target
# precision
num_samples = 5000
model = SpringMassModel(mass=1.5, time_step=0.01)
input_samples = stiffness_distribution.draw_samples(num_samples)
output_samples_mc = np.zeros(num_samples)

start_mc = timeit.default_timer()

for i, sample in enumerate(input_samples):
    output_samples_mc[i] = model.evaluate([sample])

mc_total_cost = timeit.default_timer() - start_mc

mean_mc = np.mean(output_samples_mc)
precision_mc = (np.var(output_samples_mc) / float(num_samples))
print "Target precision: ", precision_mc

# Step 3 - Initialize spring-mass models for MLMC. Here using three levels
# with MLMC defined by different time steps:
model_level1 = SpringMassModel(mass=1.5, time_step=1.0, cost=0.00034791)
model_level2 = SpringMassModel(mass=1.5, time_step=0.1, cost=0.00073748)
model_level3 = SpringMassModel(mass=1.5, time_step=0.01, cost=0.00086135)

models = [model_level1, model_level2, model_level3]

# Step 4 - Initialize MLMCSimulator:
mlmc_simulator = MLMCSimulator(stiffness_distribution, models)

# Step 5 - Calculate optimal sample size for each level:
initial_sample_size = 100
epsilon = np.sqrt(precision_mc)

# Optional - Compute cost and variances of model:
#start_mlmc = timeit.default_timer()

costs, variances = \
    mlmc_simulator.compute_costs_and_variances(initial_sample_size)

print 'Costs: ', costs
print 'Variances: ', variances

# Calculate the optimal sample size for each level from cost/variance/error:
sample_sizes = \
    mlmc_simulator.compute_optimal_sample_sizes(costs, variances, epsilon)

print
print 'Optimal sample sizes: ', np.array2string(sample_sizes)

# Optional - Call simulate now using the sample_sizes:

estimates, sample_count, variances = \
    mlmc_simulator.simulate(epsilon, sample_sizes=sample_sizes)

print
print 'Estimates: ', estimates
print 'Sample count: ', sample_count
print 'Variances: ', variances

# Step 6 - Run the model on each level the specified number of times in
# sample_sizes to calculate the output differences for levels greater than 1
def reshape_differences_per_level(outputs):
    outputs_array = np.asarray(outputs)
    for i in range(3):
        outputs_reshaped = \
            outputs_array[i].reshape(-1, 1)
        
        outputs_array[i] = outputs_reshaped
    
    return outputs_array

output_diffs_per_level = []

for level, model in enumerate(models):

    sample_size = sample_sizes[level]
    output_diffs = np.zeros((sample_size))
    stiffness_samples = stiffness_distribution.draw_samples(sample_size)

    for i, sample in enumerate(stiffness_samples):

        if level == 0:
            output_diffs[i] = model.evaluate([sample])
        else:
            output_diffs[i] = model.evaluate([sample]) - \
                                    models[level-1].evaluate([sample])

    output_diffs_per_level.append(output_diffs)

outputs = reshape_differences_per_level(output_diffs_per_level)

# Step 7 - Aggregate model outputs to compute estimators:
estimates, variances = \
    mlmc_simulator.compute_estimators(outputs)

#mlmc_total_cost = timeit.default_timer() - start_mlmc

# # Step 8 - Summarize results:
print
print 'MLMC estimate: %s' % estimates[0]
print 'MLMC precision: %s' % variances[0]
# print 'MLMC total cost: %s' % mlmc_total_cost

print
print "MC # samples: %s" % num_samples
print "MC estimate: %s" % mean_mc
print "MC precision: %s" % precision_mc
print "MC total cost: %s" % mc_total_cost
print
# print "MLMC computational speedup: %s" %  (mc_total_cost / mlmc_total_cost)
