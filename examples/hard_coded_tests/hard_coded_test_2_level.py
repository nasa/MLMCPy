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

input_samples = data_input.draw_samples(initial_sample_size)

for i, sample in enumerate(input_samples):
    level_0_data[i] = models_from_data[0].evaluate(sample)

level_0_variance = np.var(level_0_data)

# Must resample level 0 for level 0-1 discrepancy variance.
input_samples = data_input.draw_samples(initial_sample_size)
for i, sample in enumerate(input_samples):
    level_0_data[i] = models_from_data[0].evaluate(sample)

for i, sample in enumerate(input_samples):
    level_1_data[i] = models_from_data[1].evaluate(sample)

data_input.reset_sampling()

target_variance = epsilon ** 2

# Define discrepancy model and compute variance.
level_discrepancy = level_1_data - level_0_data
discrepancy_variance = np.var(level_discrepancy)

layer_0_cost = 1
layer_1_cost = 1 + 4

r = np.sqrt(discrepancy_variance / layer_1_cost *
            layer_0_cost / level_0_variance)

# Calculate sample sizes for each level.
s = (r * level_0_variance + discrepancy_variance) / (r * target_variance)
level_0_sample_size = int(np.ceil(s))
level_1_sample_size = int(np.ceil(r * s))

# Draw samples based on computed sample sizes.
data_input.reset_sampling()
sample_0 = data_input.draw_samples(level_0_sample_size)
sample_1 = data_input.draw_samples(level_1_sample_size)

# Evaluate samples.
for i, sample in enumerate(sample_0):
    sample_0[i] = models_from_data[0].evaluate(sample)

for i, sample in enumerate(sample_1):
    sample_1[i] = models_from_data[1].evaluate(sample)

# Package results for easy comparison with simulator results.
hard_coded_variances = np.array([level_0_variance, discrepancy_variance])
hard_coded_sample_sizes = np.array([level_0_sample_size, level_1_sample_size])
hard_coded_estimate = np.mean(np.concatenate((sample_0, sample_1), axis=0))

# Run Simulation for comparison to hard coded results.
# Note that this is NOT the proper way to use this class!
models = models_from_data[:2]

sim = MLMCSimulator(models=models, data=data_input)
sim_estimate, sim_sample_sizes, output_variances = \
    sim.simulate(epsilon=epsilon, initial_sample_size=initial_sample_size)
sim_costs, sim_variances = sim._compute_costs_and_variances()

assert np.array_equal(np.squeeze(sim_variances), hard_coded_variances)
assert np.array_equal(sim._sample_sizes, hard_coded_sample_sizes)
assert np.array_equal(sim_estimate[0], hard_coded_estimate)
