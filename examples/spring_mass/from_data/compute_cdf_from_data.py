import numpy as np
import matplotlib.pyplot as plt

from MLMCPy.input import InputFromData
from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.model import ModelFromData, CDFWrapperModel
"""
This script demonstrates MLMCPy for simulating a spring-mass system with a
random spring stiffness. Here, we expand our usage of MLMC to produce CDFs using
the CDFWrapperModel in conjunction with the InputFromData object. See
/examples/spring_mass/from_data/run_mlmc_from_data.py for an example that
estimates the expected value of the maximum displacement using multi-level Monte
Carlo.
"""

# Step 1 - Define I/O files:
inputfile = "data/spring_mass_1D_inputs.txt"
outputfile_level1 = "data/spring_mass_1D_outputs_1.0.txt"
outputfile_level2 = "data/spring_mass_1D_outputs_0.1.txt"
outputfile_level3 = "data/spring_mass_1D_outputs_0.01.txt"

# Step 2 - Initialize random input and model objects:
data_input = InputFromData(inputfile)

model_level1 = ModelFromData(inputfile, outputfile_level1, cost=1.0)
model_level2 = ModelFromData(inputfile, outputfile_level2, cost=10.0)
model_level3 = ModelFromData(inputfile, outputfile_level3, cost=100.0)
models = [model_level1, model_level2, model_level3]

# Step 3 - Initialize the CDFWrapperModel object:
grid = np.linspace(8, 25, 100)
cdf_wrapper = CDFWrapperModel(grid)

# Step 4 - Initialize the MLMCSimulator and predict max displacement to
# specified precision:
precision = 2.5e-2
initial_sample = 100

mlmc_simulator = MLMCSimulator(data_input, models, cdf_wrapper)

[estimates, sample_sizes, variances] = \
    mlmc_simulator.simulate(epsilon=precision,
                            initial_sample_sizes=initial_sample,
                            verbose=False)

# Step 5 - Summarize results:
print 'Sample sizes used: %s' % sample_sizes

# Step 6 - Plot CDFs:
outputfile_level3 = "data/spring_mass_1D_outputs_0.01.txt"
mc_level3 = np.genfromtxt(outputfile_level3)
x_mc = np.sort(mc_level3)
cdf_mc = np.arange(len(x_mc))/float(len(x_mc))

plt.figure()
plt.plot(grid, estimates, 'r-', label="MLMC")
plt.plot(x_mc, cdf_mc, 'b-', label="MC")
plt.legend()
plt.show()
