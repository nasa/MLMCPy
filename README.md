# MLMCPy - **M**ulti-**L**evel **M**onte **C**arlo with **Py**thon

Implementation of the Multi-Level Monte Carlo (MLMC) method in Python.

Example usage:

```python
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
estimates, sample_sizes, variances = mlmc_simulator.simulate(epsilon=.1)

print 'Estimate: %s' % estimates[0]
print 'Sample sizes used: %s' % sample_sizes
print 'Variance: %s' % variances[0]

```

-------------------------------------------------------------------------------
If you use MLMCPy for your research, please cite the technical report:

Warner, J. E. (2018). Multi-Level Monte Carlo with Python (MLMCPy). NASA/TM-2018-??????. 

The report can be found in the docs/references directory. Thanks!

-------------------------------------------------------------------------------

**Authors**: <br />
Luke Morrill <br />
Georgia Tech 

James Warner <br />
UQ Center of Excellence <br />
NASA Langley Research Center <br /> 
james.e.warner@nasa.gov

-------------------------------------------------------------------------------

Notices:
Copyright 2018 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
 