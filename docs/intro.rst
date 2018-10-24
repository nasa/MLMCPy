
Introduction
=============


MLMCPy is an implementation of the Multi-Level Monte Carlo (MLMC) method in Python. It is a software package developed to enable user-friendly utilization of the Multi-Level Monte Carlo (MLMC) approach for uncertainty quantification.

MLMCPy's primary class, MLMCSimulator, is initialized with an instance of a descendant of the Input abstract base class and a list of instances of descendants the Model abstract base class. The Input class provides input data sampling that can be provided to the models, which produce outputs using increasingly high fidelity computation.

**DIAGRAM HERE**

Its simulate() function proceeds in two phases. First, it determines the number of samples that should be passed through each model. This is determined based on either the epsilon parameter, which specifies the target precision of the estimate, or by the target_cost parameter, which specifies the desired cost (amount of time spent) performing the simulation.

Once the number of samples to be taken at each level has been determined, the simulator enters its second phase, in which the model