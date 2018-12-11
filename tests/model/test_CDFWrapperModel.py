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
def data_input():
    """
    Creates an InputFromData object that produces samples from a file
    containing spring mass input data.
    """
    return InputFromData(os.path.join(data_path, "spring_mass_1D_inputs.txt"),
                         shuffle_data=False)


@pytest.fixture
def models_from_data():
    """
    Creates a list of three ModelFromData objects of increasing fidelity.
    :return:
    """
    input_filepath = os.path.join(data_path, "spring_mass_1D_inputs.txt")
    output1_filepath = os.path.join(data_path, "spring_mass_1D_outputs_1.0.txt")
    output2_filepath = os.path.join(data_path, "spring_mass_1D_outputs_0.1.txt")
    output3_filepath = os.path.join(data_path,
                                    "spring_mass_1D_outputs_0.01.txt")

    model1 = ModelFromData(input_filepath, output1_filepath, cost=1.)
    model2 = ModelFromData(input_filepath, output2_filepath, cost=4.)
    model3 = ModelFromData(input_filepath, output3_filepath, cost=16.)

    return [model1, model2, model3]


@pytest.fixture
def beta_distribution_input():
    """
    Creates a RandomInput object that produces samples from a
    beta distribution.
    """
    np.random.seed(1)

    def beta_distribution(shift, scale, alpha, beta, size):
        return shift + scale * np.random.beta(alpha, beta, size)

    return RandomInput(distribution_function=beta_distribution,
                       shift=1.0, scale=2.5, alpha=3., beta=2.)


@pytest.fixture
def spring_model():
    """
    Creates a SpringMassModel object.
    """
    return SpringMassModel(mass=1.5, time_step=1.0, cost=1.0)


@pytest.fixture
def spring_models():
    """
    Creates a list of three SpringMassModel objects.
    """
    model_level1 = SpringMassModel(mass=1.5, time_step=1.0, cost=1.0)
    model_level2 = SpringMassModel(mass=1.5, time_step=0.1, cost=10.0)
    model_level3 = SpringMassModel(mass=1.5, time_step=0.01, cost=100.0)

    return [model_level1, model_level2, model_level3]


@pytest.fixture
def grid_0_1():
    """
    Creates a one dimensional ndarray with values ranging from 0 to .9 in
    increments of .1.
    """
    return np.arange(0., 1., .1)


def test_init_fails_on_bad_parameters(spring_model, grid_0_1):
    """
    Ensures that an expected exception type occurs when instantiating
    the model with unacceptable parameters.
    """
    with pytest.raises(TypeError):
        CDFWrapperModel("Model", grid_0_1)

    with pytest.raises(TypeError):
        CDFWrapperModel(spring_model, "Grid")

    with pytest.raises(ValueError):
        CDFWrapperModel(spring_model, np.zeros(1))

    with pytest.raises(ValueError):
        CDFWrapperModel(spring_model, np.array([[1, 2], [3, 4]]))

    with pytest.raises(TypeError):
        CDFWrapperModel(spring_model, grid_0_1, "Super smooth")


@pytest.mark.parametrize('sample', [0, -1, .5, 2.])
def test_simple_indicator(grid_0_1, sample):
    """
    Checks for the expected result when evaluating the CDF Wrapper on a very
    simple model that simply repeats its inputs.
    """
    inner_model = ModelForTesting('repeat')
    cdfw = CDFWrapperModel(inner_model, grid_0_1)

    output = cdfw.evaluate(sample)

    expected_output_sum = np.count_nonzero(sample <= grid_0_1)

    # Ensure sum of value matches.
    assert np.array_equal(np.sum(output), expected_output_sum)

    # Ensure all values are either 0 or 1.
    assert np.all(np.logical_or(output == 0, output == 1))


@pytest.mark.parametrize('num_samples', [1, 2, 5, 500])
def test_evaluate_returns_expected_results_from_data(data_input, num_samples,
                                                     models_from_data):
    """
    Tests CDF Wrapper output with spring model data.
    """
    grid = np.linspace(8, 25, 100)

    cdfw = CDFWrapperModel(models_from_data[0], grid)

    outputs = np.zeros((num_samples, grid.size))
    input_samples = data_input.draw_samples(num_samples)

    for i, sample in enumerate(input_samples):
        outputs[i] = cdfw.evaluate(sample)

    cdf = np.sum(outputs, axis=0) / num_samples

    # Ensure all indicators are either 0 or 1.
    assert np.all(np.logical_or(outputs == 0, outputs == 1))

    # Verify that no negative values exist.
    assert np.all(cdf >= 0.)

    # Verify that cdf is monotonic.
    assert np.all(cdf[1: -1] >= cdf[0: -2])

    # Verify that area under function sums to one.
    cdf_sum = np.sum(cdf[1:-1] - cdf[:-2])
    assert np.isclose(cdf_sum, 1., atol=.01)


@pytest.mark.parametrize('model_num', [0, 1, 2])
def test_single_cdf_wrapper_output_consistency(data_input, models_from_data,
                                               model_num):
    """
    Runs CDF Wrapper evaluate function for same samples twice to ensure
    output is consistent.
    """
    grid_size = 100
    num_samples = 400
    grid = np.linspace(8, 25, grid_size)

    cdfw = CDFWrapperModel(models_from_data[model_num], grid)

    input_samples = data_input.draw_samples(num_samples)
    outputs1 = np.zeros((num_samples, grid_size))
    outputs2 = np.zeros_like(outputs1)

    for i, sample in enumerate(input_samples):
        outputs1[i] = cdfw.evaluate(sample)

    for i, sample in enumerate(input_samples):
        outputs2[i] = cdfw.evaluate(sample)

    assert np.array_equal(outputs1, outputs2)


@pytest.mark.parametrize('initial_sample_size', [100, 250, 500])
def test_sim_result_consistency(data_input, models_from_data,
                                initial_sample_size):
    """
    Runs MLMC Simulator with CDF Wrapper models twice and checks results
    to ensure outputs are consistent.
    """
    grid_size = 100
    grid = np.linspace(8, 25, grid_size)

    cdfw_level1 = CDFWrapperModel(models_from_data[0], grid)
    cdfw_level2 = CDFWrapperModel(models_from_data[1], grid)
    cdfw_level3 = CDFWrapperModel(models_from_data[2], grid)

    cdfws = [cdfw_level1, cdfw_level2, cdfw_level3]

    # Run simulation to generate cache.
    sim = MLMCSimulator(data_input, cdfws)

    cdf1, ss1, v1 = sim.simulate(epsilon=.05,
                                 initial_sample_size=initial_sample_size)

    cdf2, ss2, v2 = sim.simulate(epsilon=.05,
                                 initial_sample_size=initial_sample_size)

    assert np.all(np.isclose(ss1, ss2))
    assert np.all(np.isclose(v1, v2))
    assert np.all(np.isclose(cdf1, cdf2))


@pytest.mark.parametrize('initial_sample_size', [250, 300, 350, 400])
def test_output_caching_cdf_wrapper(initial_sample_size, data_input,
                                    models_from_data):
    """
    Runs simulator's _evaluate_sample() with and without caching enabled
    to ensure consistency of outputs when running a cdf wrapper model. Also
    tests the estimate and variances with and without caching.
    """
    grid_size = 100
    grid = np.linspace(8, 25, grid_size)

    cdfw_level1 = CDFWrapperModel(models_from_data[0], grid)
    cdfw_level2 = CDFWrapperModel(models_from_data[1], grid)
    cdfw_level3 = CDFWrapperModel(models_from_data[2], grid)

    cdfws = [cdfw_level1, cdfw_level2, cdfw_level3]

    # Run simulation to generate cache.
    cached_sim = MLMCSimulator(data_input, cdfws)
    cached_cdf, cached_sample_sizes, cached_variances = \
        cached_sim.simulate(epsilon=.05,
                            initial_sample_size=initial_sample_size)

    cached_inputs_caching = np.copy(cached_sim._cached_inputs)
    cached_outputs_caching = np.copy(cached_sim._cached_outputs)

    # Collect same data with caching disabled.
    uncached_sim = MLMCSimulator(data_input, cdfws)
    uncached_sim._caching_enabled = False
    uncached_cdf, uncached_sample_sizes, uncached_variances = \
        uncached_sim.simulate(epsilon=.05,
                              initial_sample_size=initial_sample_size)

    cached_inputs_no_caching = np.copy(uncached_sim._cached_inputs)
    cached_outputs_no_caching = np.copy(uncached_sim._cached_outputs)

    assert np.array_equal(cached_inputs_caching, cached_inputs_no_caching)
    assert np.array_equal(cached_outputs_caching, cached_outputs_no_caching)

    # Now compare estimator and output variances.
    # If caching is working properly, they should match.
    assert np.array_equal(cached_sample_sizes, uncached_sample_sizes)
    assert np.array_equal(cached_cdf, uncached_cdf)
    assert np.array_equal(cached_variances, uncached_variances)


@pytest.mark.parametrize('grid_size', [10, 25, 50])
@pytest.mark.parametrize('initial_sample_size', [10, 250, 500])
def test_sim_evaluate_returns_expected_results_from_data(initial_sample_size,
                                                         data_input, grid_size,
                                                         models_from_data):
    """
    Tests MLMC Simulator with CDF Wrapper models and Spring Mass Data to
    ensure outputs are valid. Uses a wide variety of initial sample sizes and
    grid sizes.
    """
    grid = np.linspace(8, 25, grid_size)

    cdfw_level1 = CDFWrapperModel(models_from_data[0], grid)
    cdfw_level2 = CDFWrapperModel(models_from_data[1], grid)
    cdfw_level3 = CDFWrapperModel(models_from_data[2], grid)

    cdfws = [cdfw_level1, cdfw_level2, cdfw_level3]

    mlmc_simulator = MLMCSimulator(data_input, cdfws)
    mlmc_simulator._caching_enabled = False
    cdf, sample_sizes, variances = \
        mlmc_simulator.simulate(epsilon=2.5e-2,
                                initial_sample_size=initial_sample_size)

    # Verify that no negative values exist.
    assert np.all(cdf >= 0.)

    # Verify that cdf is monotonic. Allows for some noise by counting number
    # of sequential values that increase and comparing to number of grid points.
    assert np.count_nonzero(cdf[1: -1] >= cdf[0: -2]) > grid_size * .75

    # Verify that area under function sums to one.
    cdf_sum = np.sum(cdf[1:-1] - cdf[:-2])
    assert np.isclose(cdf_sum, 1., atol=.05)
