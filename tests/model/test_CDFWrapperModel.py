import pytest
import numpy as np
import os

from MLMCPy.model import CDFWrapperModel
from MLMCPy.input import RandomInput
from MLMCPy.mlmc import MLMCSimulator
from tests.testing_tools import SpringMassModel
from tests.testing_tools import TestingModel
from tests.testing_tools import TestingInput

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
def spring_model():
    return SpringMassModel(mass=1.5, time_step=1.0, cost=1.0)


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
        distribution_function_list.append(CDFWrapperModel(spring_model,
                                                          grid))

    return distribution_function_list


def test_init_fails_on_bad_parameters(spring_model, grid):

    with pytest.raises(TypeError):
        CDFWrapperModel("Model", grid)

    with pytest.raises(TypeError):
        CDFWrapperModel(spring_model, "Grid")

    with pytest.raises(ValueError):
        CDFWrapperModel(spring_model, np.zeros(1))

    with pytest.raises(ValueError):
        CDFWrapperModel(spring_model, np.array([[1, 2], [3, 4]]))

    with pytest.raises(TypeError):
        CDFWrapperModel(spring_model, grid, "Super smooth")


@pytest.mark.parametrize('sample', [0, -1, .5, 2.])
def test_single_indicator(grid, sample):

    inner_model = TestingModel('repeat')
    distribution_function = CDFWrapperModel(inner_model, grid)

    output = distribution_function.evaluate(sample)

    expected_output_sum = np.count_nonzero(sample <= grid)

    assert np.array_equal(np.sum(output), expected_output_sum)


def test_evaluate_returns_expected_results(grid):

    inner_model = TestingModel('repeat')

    num_samples = 10

    distribution_function = CDFWrapperModel(inner_model, grid)

    outputs = np.zeros((num_samples, grid.size))
    samples = np.arange(0., 1., 1. / num_samples)

    for i, sample in enumerate(samples):
        outputs[i] = distribution_function.evaluate(sample)

    estimate = np.sum(outputs, axis=0) / num_samples

    expected_result = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    assert np.array_equal(estimate, expected_result)


def test_sim_evaluate_returns_expected_results():

    grid = np.arange(0, 100)

    inner_model = TestingModel('repeat')
    distribution_function1 = CDFWrapperModel(inner_model, grid)
    distribution_function2 = CDFWrapperModel(inner_model, grid)
    models = [distribution_function1, distribution_function2]

    data = np.arange(0, 100)
    data_input = TestingInput(data)

    simulator = MLMCSimulator(data_input, models)
    cdf, sample_counts, variances = \
        simulator.simulate(epsilon=.05, initial_sample_size=100)

    # Verify that cdf is monotonic.
    for i in range(1, len(cdf)):

        if cdf[i] < cdf[i-1]:
            assert False

    # Verify that area under function sums to one.
    assert np.isclose(np.sum(cdf[1:-1] - cdf[:-2]), 1., atol=.05)


def test_monte_carlo_sim_evaluate_returns_expected_results():

    grid = np.arange(0, 100)

    inner_model = TestingModel('repeat')
    distribution_function = CDFWrapperModel(inner_model, grid)
    models = [distribution_function]

    data = np.arange(0, 100)
    data_input = TestingInput(data)

    simulator = MLMCSimulator(data_input, models)
    cdf, sample_counts, variances = \
        simulator.simulate(epsilon=.05, initial_sample_size=100)

    # Verify that cdf is monotonic.
    for i in range(1, len(cdf)):

        if cdf[i] < cdf[i-1]:
            assert False

    # Verify that area under function sums to one.
    assert np.isclose(np.sum(cdf[1:-1] - cdf[:-2]), 1., atol=.05)
