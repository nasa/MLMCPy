from MLMCPy.mlmc import MLMCSimulator

'''
This script demonstrates MLMCPy for simulating a spring-mass system with a
random spring stiffness to estimate the expected value of the maximum
displacement using multi-level Monte Carlo. Here, we use Model and RandomInput
objects with functional forms as inputs to MLMCPy. See the
/examples/spring_mass/from_data/ for an example of using precomputed data
in files as inputs.

Demonstrates the modular ("advanced") usage of MLMCPy where a user splits the
analysis into 3 steps/scripts. This is script #3 for computing MLMC estimators
using previously calculated model outputs by loading them from file
'''

model_outputs_per_level = \
    MLMCSimulator.load_model_outputs_for_each_level()

# Step 7 - Aggregate model outputs to compute estimators:
estimates, variances = \
    MLMCSimulator.compute_estimators(model_outputs_per_level)

# Step 8 - Summarize results:

print
print 'MLMC estimate: %s' % estimates
print 'MLMC precision: %s' % variances
