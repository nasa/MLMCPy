import pytest
import numpy as np
import os

from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.model import ModelFromData
from MLMCPy.input import RandomInput
from MLMCPy.input import InputFromData

from spring_mass import SpringMassModel

# Create list of paths for each data file.
# Used to parametrize tests.
my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../testing_data"


@pytest.fixture
def random_input():

    return RandomInput()


@pytest.fixture
def data_input():

    return InputFromData(os.path.join(data_path, "spring_mass_1D_inputs.txt"))


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
def models_from_data():

    input_filepath = os.path.join(data_path, "spring_mass_1D_inputs.txt")
    output1_filepath = os.path.join(data_path, "spring_mass_1D_outputs_1.0.txt")
    output2_filepath = os.path.join(data_path, "spring_mass_1D_outputs_0.1.txt")
    output3_filepath = os.path.join(data_path, "spring_mass_1D_outputs_0.01.txt")

    model1 = ModelFromData(input_filepath, output1_filepath, 1.)
    model2 = ModelFromData(input_filepath, output2_filepath, 4.)
    model3 = ModelFromData(input_filepath, output3_filepath, 16.)

    return [model1, model2, model3]


def test_model_from_data(data_input, models_from_data):

    sim = MLMCSimulator(models=models_from_data, data=data_input)
    sim.simulate(.1, initial_sample_size=1000)


def test_spring_model(beta_distribution_input, spring_models):

    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)
    sim.simulate(.1, initial_sample_size=1000)


def test_simulate_exception_for_invalid_parameters(data_input,
                                                   models_from_data):

    test_mlmc = MLMCSimulator(models=models_from_data, data=data_input)

    with pytest.raises(ValueError):
        test_mlmc.simulate(epsilon=-.1, initial_sample_size=20)

    with pytest.raises(TypeError):
        test_mlmc.simulate(epsilon='one', initial_sample_size=20)

    with pytest.raises(TypeError):
        test_mlmc.simulate(epsilon=.1, initial_sample_size='five')


def test_simulate_expected_output_types(data_input, models_from_data):

    test_mlmc = MLMCSimulator(models=models_from_data, data=data_input)

    result, sample_counts, variances = test_mlmc.simulate(epsilon=.1,
                                                          initial_sample_size=20)

    assert isinstance(result, float)
    assert isinstance(sample_counts, np.ndarray)
    assert isinstance(variances, np.ndarray)


def test_compute_optimal_sample_sizes_expected_outputs(data_input,
                                                       models_from_data):

    # Set up simulator with values that should produce predictable results
    # from computation of optimal sample sizes.
    test_mlmc = MLMCSimulator(models=models_from_data[:2], data=data_input)

    test_mlmc._epsilons = np.array([.1])

    variances = np.array([[4.], [1.]])
    costs = np.array([1., 4.])

    test_mlmc._compute_optimal_sample_sizes(variances, costs)

    # Check results.
    sample_sizes = test_mlmc._sample_sizes

    assert np.array_equal(sample_sizes, [800, 200])


def test_compute_optimal_sample_sizes_expected_outputs_2_qoi(data_input,
                                                          models_from_data):

    # Set up simulator with values that should produce predictable results
    # from computation of optimal sample sizes.
    test_mlmc = MLMCSimulator(models=models_from_data[:2], data=data_input)

    test_mlmc._epsilons = np.array([.1, .01])

    variances = np.array([[4., 4.], [1, 1.]])
    costs = np.array([1., 4.])

    test_mlmc._compute_optimal_sample_sizes(variances, costs)

    # Check results.
    sample_sizes = test_mlmc._sample_sizes

    assert np.array_equal(sample_sizes, [80000, 20000])


def test_compute_optimal_sample_sizes_expected_outputs_3_qoi(data_input,
                                                             models_from_data):

    # Set up simulator with values that should produce predictable results
    # from computation of optimal sample sizes.
    test_mlmc = MLMCSimulator(models=models_from_data[:2], data=data_input)

    test_mlmc._epsilons = np.array([.1, 1., .01])

    variances = np.array([[4., 4., 4.], [1, 1., 1.]])
    costs = np.array([1., 4.])

    test_mlmc._compute_optimal_sample_sizes(variances, costs)

    # Check results.
    sample_sizes = test_mlmc._sample_sizes

    assert np.array_equal(sample_sizes, [80000, 20000])


def test_calculate_initial_variances(beta_distribution_input, spring_models):

    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)

    initial_sample_size = 100

    costs, outputs = sim._compute_costs_and_outputs(initial_sample_size)
    variances = sim._compute_variances(outputs)

    true_variances = np.array([[8.245224951411819],
                               [0.0857219498864355],
                               [7.916295509470576e-06]])

    assert np.isclose(true_variances, variances)

