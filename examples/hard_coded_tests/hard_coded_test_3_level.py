import numpy as np
import os

from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.input import InputFromData
from MLMCPy.model import ModelFromData

my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../../tests/testing_data"
data_input = InputFromData(os.path.join(data_path, "spring_mass_1D_inputs.txt"),
                           shuffle_data=False)

input_filepath = os.path.join(data_path, "spring_mass_1D_inputs.txt")
output1_filepath = os.path.join(data_path, "spring_mass_1D_outputs_1.0.txt")
output2_filepath = os.path.join(data_path, "spring_mass_1D_outputs_0.1.txt")
output3_filepath = os.path.join(data_path, "spring_mass_1D_outputs_0.01.txt")

model1 = ModelFromData(input_filepath, output1_filepath, 1.)
model2 = ModelFromData(input_filepath, output2_filepath, 4.)
model3 = ModelFromData(input_filepath, output3_filepath, 16.)

models_from_data = [model1, model2, model3]

np.random.seed(1)
initial_sample_size = 200
epsilon = 1.

# Get output data for each layer.
level_0_data = np.zeros(initial_sample_size)
level_1_data = np.zeros(initial_sample_size)
level_2_data = np.zeros(initial_sample_size)

# Compute level 0 variance
input_samples = data_input.draw_samples(initial_sample_size)

for i, sample in enumerate(input_samples):
    level_0_data[i] = models_from_data[0].evaluate(sample)

level_0_variance = np.var(level_0_data)

# Compute level 0-1 discrepancy variance.
input_samples = data_input.draw_samples(initial_sample_size)

for i, sample in enumerate(input_samples):
    level_0_data[i] = models_from_data[0].evaluate(sample)

for i, sample in enumerate(input_samples):
    level_1_data[i] = models_from_data[1].evaluate(sample)

level_discrepancy_01 = level_1_data - level_0_data
discrepancy_variance_01 = np.var(level_discrepancy_01)

# Get new input samples for level 1-2 discrepancy.
input_samples = data_input.draw_samples(initial_sample_size)

for i, sample in enumerate(input_samples):
    level_1_data[i] = models_from_data[1].evaluate(sample)

for i, sample in enumerate(input_samples):
    level_2_data[i] = models_from_data[2].evaluate(sample)

# Compute level 1-2 discrepancy variance.
level_discrepancy_12 = level_2_data - level_1_data
discrepancy_variance_12 = np.var(level_discrepancy_12)

target_variance = epsilon ** 2

level_0_cost = 1
level_1_cost = 1 + 4
level_2_cost = 4 + 16

mu = (np.sqrt(level_0_variance * level_0_cost) +
      np.sqrt(discrepancy_variance_01 * level_1_cost) +
      np.sqrt(discrepancy_variance_12 * level_2_cost)) / target_variance

level_0_sample_size = mu * np.sqrt(level_0_variance / level_0_cost)
level_1_sample_size = mu * np.sqrt(discrepancy_variance_01 / level_1_cost)
level_2_sample_size = mu * np.sqrt(discrepancy_variance_12 / level_2_cost)

level_0_sample_size = int(np.ceil(level_0_sample_size))
level_1_sample_size = int(np.ceil(level_1_sample_size))
level_2_sample_size = int(np.ceil(level_2_sample_size))

# Draw samples based on computed sample sizes.
data_input.reset_sampling()
sample_0 = data_input.draw_samples(level_0_sample_size)
sample_1 = data_input.draw_samples(level_1_sample_size)
sample_2 = data_input.draw_samples(level_2_sample_size)

# Evaluate samples.
for i, sample in enumerate(sample_0):
    sample_0[i] = models_from_data[0].evaluate(sample)

for i, sample in enumerate(sample_1):
    sample_1[i] = models_from_data[1].evaluate(sample)

for i, sample in enumerate(sample_2):
    sample_2[i] = models_from_data[2].evaluate(sample)

hard_coded_variances = np.array([level_0_variance,
                                 discrepancy_variance_01,
                                 discrepancy_variance_12])

hard_coded_sample_sizes = np.array([level_0_sample_size,
                                    level_1_sample_size,
                                    level_2_sample_size])

hard_coded_estimate = np.mean(np.concatenate((sample_0,
                                              sample_1,
                                              sample_2), axis=0))

# Run Simulation for comparison to hard coded results.
# Note that this is NOT the proper way to use this class!
data_input.reset_sampling()
sim = MLMCSimulator(models=models_from_data, data=data_input)
sim_estimate, sim_sample_sizes, output_variances = \
    sim.simulate(epsilon=epsilon, initial_sample_size=initial_sample_size)
sim_costs, sim_variances = sim._compute_costs_and_variances()

assert np.array_equal(np.squeeze(sim_variances), hard_coded_variances)
assert np.array_equal(sim._sample_sizes, hard_coded_sample_sizes)
assert np.array_equal(sim_estimate[0], hard_coded_estimate)