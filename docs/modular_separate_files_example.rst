
Example - Spring Mass System
=============================

This example provides a simple demonstration of MLMCPy modular functionality using multiple scripts. 
The goal is to estimate the maximum displacement of a spring-mass system with 
random stiffness using Multi-Level Monte Carlo (MLMC) and compare to standard 
Monte Carlo simulation. The example covers all steps for computing MLMC 
estimators using MLMCPy, including defining a random input parameter (spring 
stiffness) using a MLMCPy random input, creating a user-defined computational 
model (spring mass numerical integrator) that uses the standardized MLMCPy 
interface, and running MLMC with a hierarchy of these models (according to time
step size) to obtain an estimator for a quantity of interest (max. displacement)
within a prescribed precision. The full source code for this example can be 
found in the MLMCPy repositories: 
``/MLMCPy/examples/spring_mass/from_model/run_mlmc_spring_mass_step1.py``,
``/MLMCPy/examples/spring_mass/from_model/run_mlmc_spring_mass_step2.py``,
``/MLMCPy/examples/spring_mass/from_model/run_mlmc_spring_mass_step3.py``


.. _spring-mass:

.. figure:: images/spring_mass_diagram.png
    :align: center
    :width: 2in

Problem Specification
----------------------

The governing equation of motion for the system is given by

.. math:: m_s \ddot{z}  = -k_s z + m_s g
    :label: springmass

where :math:`m_s` is the mass, :math:`k_s` is the spring stiffness, :math:`g`
is the acceleration due to gravity, :math:`z` is the vertical displacement
of the mass, and :math:`\ddot{z}` is the acceleration of the mass. The
source of uncertainty in the system will be the spring stiffness, which is
modeled as a random variable of the following form:

.. math:: K_s = \gamma + \eta B
    :label: random-stiffness

where :math:`\gamma` and :math:`\eta` are shift and scale parameters,
respectively, and :math:`B = \text{Beta}(\alpha, \beta)` is a standard Beta
random variable with shape parameters :math:`\alpha` and :math:`\beta`. Let
these parameters take the following values: :math:`\gamma=1.0N/m`,
:math:`\eta = 2.5N/m`, :math:`\alpha=3.0`, and :math:`\beta=2.0`. The mass
is assumed to be deterministic, :math:`m_s = 1.5kg`, and the acceleration due
to gravity is :math:`g = 9.8 m^2/s`.

With uncertainty in an input parameter, the resulting displacement, :math:`Z`, is a random variable as well. The quantity of interest in this example with be the maximum displacement over a specified time window, :math:`Z_{max}=max_t(Z)`. The equation of motion in Equation (1) can be numerically integrated over the time window with a specified time step, and the maximum of the resulting displacement time series can be taken to obtain :math:`Z_{max}`. 

The goal of this example will be to estimate the expected value of the maximum displacement, :math:`E[Z_{max}]`, using the MLMC approach with MLMCPy and compare it to a Monte Carlo simulation solution. The MLMC expected value estimate of a random quantity, :math:`X`, is as follows:

.. math:: E[X] \approx E[X_0] + \sum_{l=1}^{L} E[X_l - X_{l-1}]
    :label: mlmc_estimate

where :math:`L` is the number of levels (and number of models of varying fidelity) used. Each expected value is approximated by it's Monte Carlo estimator:

.. math:: E[X_l] \approx \frac{1}{N_l} \sum_{i=1}^{N_l} X_l^{(i)}
    :label: mc_expected_value

The MLMC method prescribes the number of samples to be taken on each level, :math:`N_l`, based on a user-specified precision and the variance of the output on each level. 

For this example, three levels will be employed, where each level corresponds to a maximum displacement predicted using a spring mass simulator model with varying time step. First, the random spring stiffness is represented using a MLMCPy random input, then a spring mass model is created and three instantiations of it are made with different time steps, then MLMC is used to estimate the expected maximum displacement to a prescibed precision.

Step 1: Initialization; define the random input parameter 
-------------------------------------------

Note that because this example demonstrates the usage of modular MLMC over multiple files, modules and models will be imported/initialized in multiple scripts. 

Begin by importing the needed Python modules, including MLMCPy classes and the SpringMassModel class that defines the spring mass numerical integrator:

.. code-block:: python

    import numpy as np

    from spring_mass import SpringMassModel
    from MLMCPy.input import RandomInput
    from MLMCPy.mlmc import MLMCSimulator

Below is a snippet of the SpringMassModel class, the entire class can be found in the MLMCPy repo (``/MLMCPy/examples/spring_mass/from_model/spring_mass_model.py``):

.. code-block:: python

  from MLMCPy.model import Model

  class SpringMassModel(Model):
      """
      Defines Spring Mass model with 1 free param (stiffness of spring, k). The
      quantity of interest that is returned by the evaluate() function is the
      maximum displacement over the specified time interval
      """

      def __init__(self, mass=1.5, gravity=9.8, state0=None, time_step=None,
                 cost=None):

Note that user-defined models in MLMCPy must inherit from the MLMCPy abstract class ``Model`` and implement an  ``evaluate`` function that accepts and returns numpy arrays for inputs and outputs, respectively. Here, the ``time_step`` argument governs numerical integration and will define the three levels used for MLMC.

The first step in an analysis is to define the random variable representing the model inputs. Here, the spring stiffness :math:`K_s` is defined by a Beta random variable and created with MLMCPy as follows:

.. code-block:: python

    # Step 1 - Define random variable for spring stiffness:
    # Need to provide a sampleable function to create RandomInput instance in MLMCPy
    def beta_distribution(shift, scale, alpha, beta, size):

        return shift + scale*np.random.beta(alpha, beta, size)

    stiffness_distribution = RandomInput(distribution_function=beta_distribution,
                                    shift=1.0, scale=2.5, alpha=3., beta=2.)

The ``RandomInput`` class is initialized with a function that produces random samples and any parameters it requires. 
See the :ref:`input_module_docs` for more details about specifying random input parameters with MLMCPy.

Step 2: Initialize a hierarchy (3 levels) of models for MLMC
--------------------------------------------------------------

In order to apply the MLMC method (Equation (3)), multiple levels of models (defined by cost/accuracy) must be defined. The following code initializes three separate spring mass models defined by varying time step (the smaller the time step, the higher the cost and accuracy):

.. code-block:: python

  # Step 3 - Initialize spring-mass models for MLMC. Here using three levels 
  # with MLMC defined by different time steps:
  model_level1 = SpringMassModel(mass=1.5, time_step=1.0, cost=0.00034791)
  model_level2 = SpringMassModel(mass=1.5, time_step=0.1, cost=0.00073748)
  model_level3 = SpringMassModel(mass=1.5, time_step=0.01, cost=0.00086135)

  models = [model_level1, model_level2, model_level3]

Step 3: Initialize MLMC and calculate optimal sample sizes for each level
---------------------------------------------------------------

With a random input defined in Step 1 and multiple fidelity models defined in Step 2, MLMC can now be used to estimate the maximum displacement using the ``MLMCSimulator`` class. 
Here, the modular functions are utilized to calculate the optimal sample sizes per level. 

Note ``epsilon`` is taken from the example found in the MLMCPy repo
(``/MLMCPy/examples/spring_mass/from_model/adv_run_mlmc_from_model.py``):

.. code-block:: python

  # Step 4 - Calculate optimal sample size for each level:
  # Optional - compute cost and variances of model (or user knows these beforehand)
  initial_sample_size = 100
  epsilon = np.sqrt(0.00170890122096)

  costs, variances = \
      mlmc_simulator.compute_costs_and_variances(initial_sample_size)

  # Calculate optimal sample size for each level from cost/variance/error:
  sample_sizes = mlmc_simulator.compute_optimal_sample_sizes(costs, variances,
                                                             epsilon)

Note that this example demonstrates the use of the ``compute_costs_and_variances`` method, but if the costs and variances are known values, they can be plugged in directly to the ``compute_optimal_sample_sizes`` method.

Step 4: Store model inputs
---------------------------------------------------------------

With the ``sample_sizes`` defined in Step 3, MLMC can now be used to generate inputs for each level and then store them in a ``.txt`` file. 
Optionally, custom file names can be defined and given to ``store_model_inputs_to_run_for_each_level`` method. 

.. code-block:: python

  # Step 5 - Store inputs to be used in model evaluation step:
  mlmc_simulator.store_model_inputs_to_run_for_each_level(sample_sizes)

Note that if custom file names are not given to the ``store_model_inputs_to_run_for_each_level`` method, a standard file name of ``levelX_inputs.txt`` (where X is the level) will be generated.

Step 5: Initialize models; generate model outputs for each level
---------------------------------------------------------------

This step takes place in a separate file, begin by importing the needed Python modules and initialize the SpringMassModel class that defines the spring mass numerical integrator:

.. code-block:: python

    import numpy as np

    from spring_mass import SpringMassModel

A snippet of the SpringMassModel class can be found in Step 1.

Initialize the model hierarchy as found in Step 2:

.. code-block:: python

  # Step 3 - Initialize spring-mass models for MLMC. Here using three levels 
  # with MLMC defined by different time steps:
  model_level1 = SpringMassModel(mass=1.5, time_step=1.0, cost=0.00034791)
  model_level2 = SpringMassModel(mass=1.5, time_step=0.1, cost=0.00073748)
  model_level3 = SpringMassModel(mass=1.5, time_step=0.01, cost=0.00086135)

Note for this step of this example, a list of models is not necessary.

Using the files generated in Step 4, generate model outputs for each level and store them:

.. code-block:: python

  # Generate outputs for model on level 0:
  samples_level0 = np.genfromtxt("level0_inputs.txt")
  outputs_level0 = []

  for inputsample in samples_level0:
      outputs_level0.append(model_level0.evaluate([inputsample]))

  np.savetxt("level0_outputs.txt", np.array(outputs_level0))

  # Generate outputs for model on level 1:
  samples_level1 = np.genfromtxt("level1_inputs.txt")
  outputs_level1 = []

  for inputsample in samples_level1:
      outputs_level1.append(model_level1.evaluate([inputsample]))

  np.savetxt("level1_outputs.txt", np.array(outputs_level1))

  # Generate outputs for model on level 2:
  samples_level2 = np.genfromtxt("level2_inputs.txt")
  outputs_level2 = []

  for inputsample in samples_level2:
      outputs_level2.append(model_level2.evaluate([inputsample]))

  np.savetxt("level2_outputs.txt", np.array(outputs_level2))

Step 6: Load model outputs; aggregate model outputs to compute estimators
---------------------------------------------------------------

This step takes place in a separate script, begin by importing the MLMCSimulator class:

.. code-block:: python

  from MLMCPy.mlmc import MLMCSimulator

Use the ``load_model_outputs_for_each_level`` method to load the outputs generated in Step 5:

.. code-block:: python

  model_outputs_per_level = \
    MLMCSimulator.load_model_outputs_for_each_level()

Note that the file names used in Step 5 ``levelX_outputs.txt`` are a standard format. If custom file names are used, they must be passed to ``load_model_outputs_for_each_level`` as a list of file names.

The ``model_outputs_per_level`` are used to estimate the maximum displacement using the ``compute_estimators`` method.

.. code-block:: python

  # Step 7 - Aggregate model outputs to compute estimators:
  estimates, variances = \
      MLMCSimulator.compute_estimators(model_outputs_per_level)

Step 7: Summarize results
---------------------------------------------------------------

.. code-block:: python
  print 'MLMC estimate: %s' % estimates
  print 'MLMC precision: %s' % variances

====================     =====================
Description              MLMC Value           
====================     =====================
Estimate                 12.32531885505423    
Precision                0.0017062682486212561
====================     =====================