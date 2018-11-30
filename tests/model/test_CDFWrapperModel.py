import pytest
import numpy as np
import os

from MLMCPy.model import CDFWrapperModel
from MLMCPy.input import RandomInput
from MLMCPy.input import InputFromData
from MLMCPy.model import ModelFromData
from MLMCPy.mlmc import MLMCSimulator
from tests.testing_scripts import SpringMassModel
from tests.testing_scripts import ModelForTesting
from tests.testing_scripts import InputForTesting

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

    inner_model = ModelForTesting('repeat')
    distribution_function = CDFWrapperModel(inner_model, grid)

    output = distribution_function.evaluate(sample)

    expected_output_sum = np.count_nonzero(sample <= grid)

    assert np.array_equal(np.sum(output), expected_output_sum)


def test_evaluate_returns_expected_results(grid):

    inner_model = ModelForTesting('repeat')

    num_samples = 10

    distribution_function = CDFWrapperModel(inner_model, grid)

    outputs = np.zeros((num_samples, grid.size))
    samples = np.arange(0., 1., 1. / num_samples)

    for i, sample in enumerate(samples):
        outputs[i] = distribution_function.evaluate(sample)

    estimate = np.sum(outputs, axis=0) / num_samples

    expected_result = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    assert np.array_equal(estimate, expected_result)


@pytest.mark.parametrize('num_data_points', [100, 250, 1000])
def test_sim_evaluate_returns_expected_results(num_data_points):

    grid = np.arange(0, 100)

    inner_model = ModelForTesting('repeat')
    distribution_function1 = CDFWrapperModel(inner_model, grid)
    distribution_function2 = CDFWrapperModel(inner_model, grid)
    models = [distribution_function1, distribution_function2]

    data = np.arange(0, num_data_points)
    data_input = InputForTesting(data)

    sim = MLMCSimulator(data_input, models)
    cdf, sample_counts, variances = \
        sim.simulate(epsilon=.05, initial_sample_size=500)

    # Verify that no negative values exist.
    assert np.all(cdf >= 0.)

    # Verify that cdf is monotonic.
    for i in range(1, len(cdf)):

        if cdf[i] < cdf[i-1]:
            assert False

    # Verify that area under function sums to one.
    cdf_sum = np.sum(cdf[1:-1] - cdf[:-2])
    assert np.isclose(cdf_sum, 1., atol=.05)


@pytest.mark.parametrize('num_data_points', [100, 250, 1000])
def test_compare_cdf_sim_cached_uncached_results(num_data_points):

    # Set up CDF model.
    grid = np.arange(0, 100)

    inner_model = ModelForTesting('repeat')
    distribution_function1 = CDFWrapperModel(inner_model, grid)
    distribution_function2 = CDFWrapperModel(inner_model, grid)
    models = [distribution_function1, distribution_function2]

    # Hard code costs so that we can expect consistent sample size results.
    distribution_function1.cost = 1
    distribution_function2.cost = 10

    data = np.arange(0, num_data_points)
    data_input = InputForTesting(data)

    # Create and run uncached simulation.
    uncached_sim = MLMCSimulator(data_input, models)
    uncached_sim._caching_enabled = False
    uncached_cdf, uncached_sample_counts, uncached_variances = \
        uncached_sim.simulate(epsilon=.05, initial_sample_size=500)

    # Create and run cached simulation.
    data_input = InputForTesting(data)

    cached_sim = MLMCSimulator(data_input, models)
    cached_cdf, cached_sample_counts, cached_variances = \
        cached_sim.simulate(epsilon=.05, initial_sample_size=500)

    # Test equality of all outputs.
    assert np.all(cached_sample_counts == uncached_sample_counts)
    assert np.all(cached_variances == uncached_variances)
    assert np.all(cached_cdf == uncached_cdf)


@pytest.mark.parametrize('initial_sample_size', [20, 100, 250, 1000])
def test_sim_cdf_from_data(initial_sample_size):

    # Define I/O files
    inputfile = os.path.join(data_path, "spring_mass_1D_inputs.txt")

    outputfile_level1 = os.path.join(data_path,
                                     "spring_mass_1D_outputs_1.0.txt")
    outputfile_level2 = os.path.join(data_path,
                                     "spring_mass_1D_outputs_0.1.txt")
    outputfile_level3 = os.path.join(data_path,
                                     "spring_mass_1D_outputs_0.01.txt")

    # Initialize random input & model objects
    data_input = InputFromData(inputfile, shuffle_data=False)

    model_level1 = ModelFromData(inputfile, outputfile_level1, cost=1.0)
    model_level2 = ModelFromData(inputfile, outputfile_level2, cost=10.0)
    model_level3 = ModelFromData(inputfile, outputfile_level3, cost=100.0)

    grid = np.linspace(8, 25, 100)

    cdfw_level1 = CDFWrapperModel(model_level1, grid)
    cdfw_level2 = CDFWrapperModel(model_level2, grid)
    cdfw_level3 = CDFWrapperModel(model_level3, grid)

    cdfws = [cdfw_level1, cdfw_level2, cdfw_level3]

    mlmc_simulator = MLMCSimulator(data_input, cdfws)
    mlmc_simulator._caching_enabled = False
    [cdf, sample_sizes, variances] = \
        mlmc_simulator.simulate(epsilon=2.5e-2,
                                initial_sample_size=initial_sample_size)

    # Verify that no negative values exist.
    assert np.all(cdf >= 0.)

    # Verify that cdf is monotonic.
    for i in range(1, len(cdf)):

        if cdf[i] < cdf[i-1]:
            assert False

    # Verify that area under function sums to one.
    cdf_sum = np.sum(cdf[1:-1] - cdf[:-2])
    assert np.isclose(cdf_sum, 1., atol=.05)
