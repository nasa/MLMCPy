
Example - Spring Mass System
=============================

This example provides a simple demonstration of MLMCPy modular functionality. 
The goal is to estimate the maximum displacement of a spring-mass system with 
random stiffness using Multi-Level Monte Carlo (MLMC) and compare to standard 
Monte Carlo simulation. The example covers all steps for computing MLMC 
estimators using MLMCPy, including defining a random input parameter (spring 
stiffness) using a MLMCPy random input, creating a user-defined computational 
model (spring mass numerical integrator) that uses the standardized MLMCPy 
interface, and running MLMC with a hierarchy of these models (according to time
step size) to obtain an estimator for a quantity of interest (max. displacement)
within a prescribed precision. The full source code for this example can be 
found in the MLMCPy repository: 
``/MLMCPy/examples/spring_mass/from_model/run_mlmc_from_model.py``

.. _spring-mass:

.. figure:: images/spring_mass_diagram.png
    :align: center
    :width: 2in