MLMCPy - **M**ulti-**L**evel **M**onte **C**arlo with **Py**thon
===================================================================

<a href='https://travis-ci.com/nasa/MLMCPy'><img src='https://travis-ci.com/nasa/MLMCPy.svg?branch=master' alt='Build Status' /></a> <a href='https://coveralls.io/github/lukemorrill/MLMCPy?branch=master'><img src='https://coveralls.io/repos/github/lukemorrill/MLMCPy/badge.svg?branch=master' alt='Coverage Status' /></a>

General
--------

MLMCPy is an open source Python implementation of the Multi-Level Monte Carlo (MLMC) method for uncertainty propagation. Once a user defines their computational model and specifies the uncertainty in the model input parameters, MLMCPy can be used to estimate the expected value of a quantity of interest to within a specified precision. Support is available to perform the required model evaluations in parallel (if mpi4py is installed) and extensions of the MLMC method are provided to calculate more advanced statistics (e.g., covariance, CDFs). 

Dependencies
--------------

MLMCPy is intended for use with Python 2.7 and relies on the following packages:

* numpy
* scipy
* mpi4py (optional for running in parallel)
* pytest (optional for running unit tests)

Example Usage
---------------

```python
'''
Simple example of propagating uncertainty through a spring-mass model using MLMC. 
Estimates the expected value of the maximum displacement of the system when the spring 
stiffness is a random variable. See the /examples/spring_mass/from_model/ for more details.
'''

import numpy as np
import sys

from MLMCPy.input import RandomInput
from MLMCPy.mlmc import MLMCSimulator

# Add path for example SpringMassModel to sys path.
sys.path.append('./examples/spring_mass/from_model/spring_mass')
import SpringMassModel

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

# Step 3 - Initialize MLMC & predict max displacement to specified error (0.1).
mlmc_simulator = MLMCSimulator(stiffness_distribution, models)
[estimates, sample_sizes, variances] = mlmc_simulator.simulate(epsilon=1e-1)

```

Getting Started
----------------
MLMCPy can be installed via pip from [PyPI](https://pypi.org/project/MLMCPy/).

```
pip install mlmcpy
```

The best way to get started with MLMCPy is to take a look at the scripts in the examples/ directory. A simple example of propagating uncertainty through a spring mass system can be found in the ``examples/spring_mass/from_model`` directory. There is a second example that demonstrates the case where a user has access to input-output data from multiple levels of models (rather than a model they can directly evaluate) in the ``examples/spring_mass/from_data/`` directory. For more information, see the source code documentation in ``docs/MLMCPy_documentation.pdf`` (a work in progress).

Tests
------
The tests can be performed by running "py.test" from the tests/ directory to ensure a proper installation.

Developers
-----------

UQ Center of Excellence <br />
NASA Langley Research Center <br /> 
Hampton, Virginia <br /> 

This software was funded by and developed under the High Performance Computing Incubator (HPCI) at NASA Langley Research Center. <br /> 

Contributors: James Warner (james.e.warner@nasa.gov), Luke Morrill, Juan Barrientos


License
--------

Copyright 2018 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
 
