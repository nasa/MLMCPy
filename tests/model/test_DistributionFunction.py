import pytest
import numpy as np
import os

from MLMCPy.model.DistributionFunction import DistributionFunction
from MLMCPy.input.RandomInput import RandomInput
from MLMCPy.mlmc.MLMCSimulator import MLMCSimulator
from spring_mass import SpringMassModel

# Access spring mass data:
my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../testing_data"


@pytest.fixture
def beta_distribution_input():

    np.random.seed(1)

    def beta_distribution(shift, scale, alpha, beta, size):
        return shift + scale * np.random.beta(alpha, beta, size)

    return RandomInput(distribution_function=beta_distribution,
                       shift=1.0, scale=2.5, alpha=3., beta=2.)


@pytest.fixture
def spring_models():

    model_level1 = SpringMassModel(mass=1.5, time_step=1.0, cost=1.0)
    model_level2 = SpringMassModel(mass=1.5, time_step=0.1, cost=10.0)
    model_level3 = SpringMassModel(mass=1.5, time_step=0.01, cost=100.0)

    return [model_level1, model_level2, model_level3]


@pytest.fixture
def grid():

    return np.arange(0., 1., .1)


@pytest.fixture
def distribution_functions(spring_models):

    distribution_function_list = list()
    for spring_model in spring_models:
        distribution_function_list.append(DistributionFunction(spring_model,
                                                               grid))

    return distribution_function_list


@pytest.fixture
def simulator(distribution_functions, beta_distribution_input):

    return MLMCSimulator(models=distribution_functions,
                         data=beta_distribution_input)


def test_evaluate_returns_expected_results(beta_distribution_input,
                                           simulator, grid):

    cfs, sample_counts, variances = \
        simulator.simulate(epsilon=1., initial_sample_size=100)

