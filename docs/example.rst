
Example - Spring Mass System
=============================

This example will use MLMCPy to simulate a spring-mass system with random spring stiffness. The example covers modeling the random stiffness using a Beta random variable via RandomInput, generating a Model to obtain displacement realizations, then using MLMCSimulator to obtain an estimate for maximum displacement. The MLMCPy solution will then be compared to standard Monte Carlo simulation.

.. _spring-mass:

.. figure:: images/spring_mass_diagram.png
    :align: center
    :width: 2in

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

With uncertainty in an input parameter, the resulting displacement, :math:`Z`, is a random variable as well. The quantity of interest in this example with be the maximum displacement over a specified time window, :math:`Z_{max}=max_t(Z)`. It is assumed we have access to a computational model that numerically integrates the governing equation over this time window for a given sample of the random stiffness and returns the maximum displacement.

The goal of this example will be to approximate :math:`z_{max}`, using the MLMC approach with MLMCPy and compare it to a Monte Carlo simulation solution.

.. code-block:: python

    import numpy as np
    import timeit

    from spring_mass import SpringMassModel
    from MLMCPy.input import RandomInput
    from MLMCPy.mlmc import MLMCSimulator

    # Step 1 - Define random variable for spring stiffness:
    # Need to provide a sampleable function to create RandomInput instance.
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

    start_mlmc = timeit.default_timer()

    [estimates, sample_sizes, variances] = \
        mlmc_simulator.simulate(epsilon=1e-1,
                                initial_sample_sizes=100,
                                verbose=True)

    mlmc_total_cost = timeit.default_timer() - start_mlmc

    print
    print 'MLMC estimate: %s' % estimates[0]
    print 'MLMC precision: %s' % variances[0]
    print 'MLMC total cost: %s' % mlmc_total_cost

    # Step 4: Run standard Monte Carlo to achieve similar variance for comparison.
    num_samples = 1e6
    input_samples = stiffness_distribution.draw_samples(num_samples)
    output_samples = np.zeros(num_samples)

    start_mc = timeit.default_timer()

    for i, sample in enumerate(input_samples):
        output_samples[i] = model_level3.evaluate([sample])

    mc_total_cost = timeit.default_timer() - start_mc

    print
    print "MC estimate: %s" % np.mean(output_samples)
    print "MC precision: %s" % (np.var(output_samples) / float(num_samples))
    print "MC total cost: %s" % mc_total_cost


These are the results in a single-core environment, in which the number of samples was chosen based on an epsilon of 1e-1, resulting in 855 Monte Carlo runs on the highest cost model and 941, 69, and 0 samples from each model for Multi-Level Monte Carlo.

====================     =====================     =====================
Description              MLMC Value                MC Value
====================     =====================     =====================
Estimate                 12.2739151773             12.390705590117555
Error                    0.045171289               0.071619124
Precision                0.009916230329196151      0.010780941000560835
Total cost (seconds)     0.63                      1.14
====================     =====================     =====================

The expected value should converge to approximately 12.31908646652595, as determined by a 1e6 sample Monte Carlo simulation.

Note the significant discrepancy in cost between the two methods.
