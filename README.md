# MLMCPy - **M**ulti-**L**evel **M**onte **C**arlo with **Py**thon

<a href='https://coveralls.io/github/lukemorrill/MLMCPy?branch=master'><img src='https://coveralls.io/repos/github/lukemorrill/MLMCPy/badge.svg?branch=master' alt='Coverage Status' /></a>

## General
MLMCPy is an open source implementation of the Multi-Level Monte Carlo (MLMC) method in Python.
It was developed with ease of use in mind.

## Dependencies
MLMCPy is intended for use with Python 2.7.

Required packages:
- numpy
- scipy

Optional packages:
- mpi4py (for using with mpirun)
- pytest (for running unit tests)

## Tests
Well over one hundred tests are included to thoroughly test MLMCPy. 

## Example usage

```python
import numpy as np
import sys

from MLMCPy.input import RandomInput
from MLMCPy.mlmc import MLMCSimulator

# Add path for example SpringMassModel to sys path.
sys.path.append('./examples/spring_mass/from_model/spring_mass')
import SpringMassModel

'''
This script demonstrates MLMCPy for simulating a spring-mass system with a 
random spring stiffness to estimate the expected value of the maximum 
displacement using multi-level Monte Carlo. Here, we use Model and RandomInput
objects with functional forms as inputs to MLMCPy. See the
/examples/spring_mass/from_data/ for an example of using precomputed data
in files as inputs.
'''

# Step 1 - Define random variable for spring stiffness:
# Need to provide a sampleable function to create RandomInput instance in MLMCPy
def beta_distribution(shift, scale, alpha, beta, size):

    return shift + scale*np.random.beta(alpha, beta, size)

stiffness_distribution = RandomInput(distribution_function=beta_distribution,
                                     shift=1.0, scale=2.5, alpha=3., beta=2.)

# Step 2 - Initialize spring-mass models. Here using three levels with MLMC.
# defined by different time steps
model_level1 = SpringMassModel(mass=1.5, time_step=1.0)
model_level2 = SpringMassModel(mass=1.5, time_step=0.1)
model_level3 = SpringMassModel(mass=1.5, time_step=0.01)

models = [model_level1, model_level2, model_level3]

# Step 3 - Initialize MLMC & predict max displacement to specified error.

mlmc_simulator = MLMCSimulator(stiffness_distribution, models)

[estimates, sample_sizes, variances] = \
    mlmc_simulator.simulate(epsilon=1e-1,
                            initial_sample_sizes=100,
                            verbose=True)

```
-------------------------------------------------------------------------------

## Authors
Luke Morrill<br />Georgia Tech 

James Warner<br />UQ Center of Excellence<br />NASA Langley Research Center<br />james.e.warner@nasa.gov


This software was funded by and developed under the High Performance Computing
Incubator (HPCI) at NASA Langley Research Center. 

-------------------------------------------------------------------------------

## Notices
Copyright 2018 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
 
