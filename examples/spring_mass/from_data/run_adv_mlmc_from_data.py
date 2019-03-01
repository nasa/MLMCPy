import numpy as np

from MLMCPy.input import InputFromData
from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.model import ModelFromData

"""
This script demonstrates the modularity of the MLMCPy Simulator by simulating a
spring-mass system with a random spring stiffness to estimate the expected value
of the maximum displacement using the multi-level Monte Carlo. Here we use the
ModelFromData to utilize precomputed data as inputs.
"""

# Step 1 - Define I/O files:
inputfile = \
    "data/spring_mass_1D_inputs.txt"
outputfile_level1 = \
    "data/spring_mass_1D_outputs_1.0.txt"
outputfile_level2 = \
    "data/spring_mass_1D_outputs_0.1.txt"
outputfile_level3 = \
    "data/spring_mass_1D_outputs_0.01.txt"

# Step 2 - Initialize random input from data:
data_input = InputFromData(inputfile)

# Step 3 - Initialize spring-mass models for MLMC. Using three levels with
# MLMC defined by different cost:
model_level1 = ModelFromData(inputfile, outputfile_level1, cost=1.0)
model_level2 = ModelFromData(inputfile, outputfile_level2, cost=10.0)
model_level3 = ModelFromData(inputfile, outputfile_level3, cost=100.0)

models = [model_level1, model_level2, model_level3]

# Step 3 - Initialize MLMCSimulator:
mlmc_simulator = MLMCSimulator(data_input, models)

# Step 4 - Calculate optimal sample size for each level:
initial_sample_size = 100
epsilon = 1e-1

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

# # Step 6 - Aggregate model outputs to compute estimators:
# estimates, variances = \
#     mlmc_simulator.compute_estimators(output_diffs_per_level)

# # # Step 7 - Summarize results:
# print
# print 'MLMC estimate: %s' % estimates[0]
# print 'MLMC precision: %s' % variances[0]
# # print 'MLMC total cost: %s' % mlmc_total_cost
