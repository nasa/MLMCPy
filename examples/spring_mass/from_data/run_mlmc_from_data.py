
from MLMCPy.input import InputFromData
from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.model import ModelFromData


# Define I/O files
inputfile = "data/spring_mass_1D_inputs.txt"
outputfile_level1 = "data/spring_mass_1D_outputs_1.0.txt"
outputfile_level2 = "data/spring_mass_1D_outputs_0.1.txt"
outputfile_level3 = "data/spring_mass_1D_outputs_0.01.txt"

# Initialize random input & model objects
data_input = InputFromData(inputfile)

model_level1 = ModelFromData(inputfile, outputfile_level1, cost=1.0)
model_level2 = ModelFromData(inputfile, outputfile_level2, cost=10.0)
model_level3 = ModelFromData(inputfile, outputfile_level3, cost=100.0)

models = [model_level1, model_level2, model_level3]

mlmc_simulator = MLMCSimulator(data_input, models)
[estimates, sample_sizes, variances] = mlmc_simulator.simulate(epsilon=1e-1)

print 'Estimate: %s' % estimates
print 'Sample sizes used: %s' % sample_sizes
print 'Variance: %s' % variances
