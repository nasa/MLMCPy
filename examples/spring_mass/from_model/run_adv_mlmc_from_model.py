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

# Step 2 - Initialize spring-mass models for MLMC. Here using three levels
# with MLMC defined by different time steps:
model_level1 = SpringMassModel(mass=1.5, time_step=1.0)
model_level2 = SpringMassModel(mass=1.5, time_step=0.1)
model_level3 = SpringMassModel(mass=1.5, time_step=0.01)

models = [model_level1, model_level2, model_level3]

# Step 3 - Initialize MLMCSimulator:
mlmc_simulator = MLMCSimulator(stiffness_distribution, models)

# Step 4 - Calculate optimal sample size for each level:
initial_sample_size = 100
epsilon = 1e-2

# Optional - Compute cost and variances of model:
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

# Step 5 - Run the model on each level the specified number of times in
# sample_sizes to calculate the output differences for levels greater than 1

# output_diffs_per_level = []

# for level, model in enumerate(models):

#     sample_size = sample_sizes[level]
#     output_diffs = np.zeros((sample_size))
#     stiffness_samples = stiffness_distribution.draw_samples(sample_size)

#     for i, sample in enumerate(stiffness_samples):

#         if level == 0:
#             output_diffs[i] = model.evaluate([sample])
#         else:
#             output_diffs[i] = model.evaluate([sample]) - \
#                                     models[level-1].evaluate([sample])

#     output_diffs_per_level.append(output_diffs)

# Step 6 - Aggregate model outputs to compute estimators:
# estimates, variances = \
#     mlmc_simulator.compute_estimators(output_diffs_per_level)

# # Step 7 - Summarize results:

# print
# print 'MLMC estimate: %s' % estimates[0]
# print 'MLMC precision: %s' % variances[0]
# print 'MLMC total cost: %s' % mlmc_total_cost
