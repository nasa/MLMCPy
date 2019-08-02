from mpi4py import MPI
import numpy as np
import timeit

from spring_mass_model import SpringMassModel
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

# Step 0 - Set up MPI communicator
comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()


# Step 1 - Define random variable for spring stiffness:
# Need to provide a sampleable function to create RandomInput instance in MLMCPy
def beta_distribution(shift, scale, alpha, beta, size):

    return shift + scale*np.random.beta(alpha, beta, size)


np.random.seed(1)
stiffness_distribution = RandomInput(distribution_function=beta_distribution,
                                     shift=1.0, scale=2.5, alpha=3., beta=2.,
                                     random_seed=1)

# Step 2: Run standard Monte Carlo to generate a reference solution and target 
# precision
if rank == 0:
    num_samples = 5000
    model = SpringMassModel(mass=1.5, time_step=0.01)
    input_samples = stiffness_distribution.draw_samples(num_samples)
    output_samples_mc = np.zeros(num_samples)

    # only run MC on root process

    start_mc = timeit.default_timer()
    
    for i, sample in enumerate(input_samples):
        output_samples_mc[i] = model.evaluate([sample])
    
    mc_total_cost = timeit.default_timer() - start_mc

    mean_mc = np.mean(output_samples_mc)
    precision_mc = (np.var(output_samples_mc) / float(num_samples))
    print "Target precision: ", precision_mc
else:
    precision_mc = None

precision_mc = comm.bcast(precision_mc, root=0)

# Step 3 - Initialize spring-mass models for MLMC. Here using three levels 
# with MLMC defined by different time steps
model_level1 = SpringMassModel(mass=1.5, time_step=1.0)
model_level2 = SpringMassModel(mass=1.5, time_step=0.1)
model_level3 = SpringMassModel(mass=1.5, time_step=0.01)

models = [model_level1, model_level2, model_level3]

# Step 4 - Initialize MLMC & predict max displacement to specified precision
mlmc_simulator = MLMCSimulator(stiffness_distribution, models)

start_mlmc = timeit.default_timer()

[estimates, sample_sizes, variances] = \
    mlmc_simulator.simulate(epsilon=np.sqrt(precision_mc),
                            initial_sample_sizes=100,
                            verbose=True)

mlmc_cost = np.array([timeit.default_timer() - start_mlmc])

#Step 5 - summarize results:
mlmc_total_cost = np.zeros(1)
comm.Reduce([mlmc_cost, MPI.DOUBLE], [mlmc_total_cost, MPI.DOUBLE],
            op=MPI.MAX, root=0) # (take max of mlmc_cost across processors)

if rank == 0:
    # only print out results on root process

    print
    print 'MLMC estimate: %s' % estimates[0]
    print 'MLMC precision: %s' % variances[0]
    print 'MLMC total cost: %s' % mlmc_total_cost[0]
    
    print
    print "MC # samples: %s" % num_samples
    print "MC estimate: %s" % mean_mc
    print "MC precision: %s" % precision_mc
    print "MC total cost: %s" % mc_total_cost
    print
    print "MLMC computational speedup: %s" %  (mc_total_cost/mlmc_total_cost[0])
