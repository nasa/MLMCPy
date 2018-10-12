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

    return InputFromData(os.path.join(data_path, "spring_mass_1D_inputs.txt"),
                         shuffle_data=False)


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
    sim.simulate(1., initial_sample_size=20)


def test_spring_model(beta_distribution_input, spring_models):

    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)
    sim.simulate(1., initial_sample_size=20)


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

    result, sample_counts, variances = test_mlmc.simulate(epsilon=1.,
                                                          initial_sample_size=20)

    assert isinstance(result, np.ndarray)
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

    sim._initial_sample_size = 100

    sim._compute_costs()
    variances = sim._compute_variances()

    true_variances = np.array([[8.245224951411819],
                               [0.0857219498864355],
                               [7.916295509470576e-06]])

    assert np.isclose(true_variances, variances).all()


def test_calculate_costs_and_variances_for_springmass_from_data(data_input,
                                                              models_from_data):

    sim = MLMCSimulator(models=models_from_data, data=data_input)
    
    sim._initial_sample_size = 100
    costs = sim._compute_costs()
    variances = sim._compute_variances()

    true_variances = np.array([[9.262628271266264],
                               [0.07939834631411287],
                               [5.437083709623372e-06]])

    true_costs = np.array([1.0, 5.0, 20.0])

    assert np.isclose(true_costs, costs).all()
    assert np.isclose(true_variances, variances).all()


@pytest.mark.parametrize("num_levels", [2, 3])
def test_calculate_estimate_for_springmass_random_input(beta_distribution_input,
                                                        spring_models,
                                                        num_levels):

    np.random.seed(1)
    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    sim = MLMCSimulator(models=spring_models[:num_levels],
                        data=beta_distribution_input)

    estimate, sample_sizes, variances = sim.simulate(1., 100)

    assert np.isclose(estimate, mc_20000_output_sample_mean, .5)


@pytest.mark.parametrize("cache_size", [20, 200, 2000])
def test_output_caching(data_input, models_from_data, cache_size):

    sim = MLMCSimulator(models=models_from_data, data=data_input)

    # Run simulation with caching.
    estimate1, sample_sizes, variances1 = sim.simulate(1., cache_size)

    # Set initial_sample_size to 0 and run simulation again so that it will
    # not use cached values.
    sim._initial_sample_size = 0
    estimate2, sample_sizes, variances2 = sim._run_simulation()

    # Now compare final estimator and output variances.
    # If caching is working properly, they should match.
    assert np.array_equal(estimate1, estimate2)
    assert np.array_equal(variances1, variances2)


def test_monte_carlo(data_input, models_from_data):

    np.random.seed(1)

    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    # Passing in one model into MLMCSimulator should make it run in monte
    # carlo simulation mode.
    models = [models_from_data[0]]

    sim = MLMCSimulator(models=models, data=data_input)
    estimate, sample_sizes, variances = sim.simulate(1., 50)

    assert np.isclose(estimate, mc_20000_output_sample_mean, .25)


def test_geoff_test_2_level(data_input, models_from_data):

    np.random.seed(1)
    initial_sample_size = 200
    epsilon = 1.

    # Get output data for each layer.
    level_0_data = np.zeros(initial_sample_size)
    input_samples = data_input.draw_samples(initial_sample_size)

    for i, sample in enumerate(input_samples):
        level_0_data[i] = models_from_data[0].evaluate(sample)[0]

    level_1_data = np.zeros(initial_sample_size)
    input_samples = data_input.draw_samples(initial_sample_size)

    for i, sample in enumerate(input_samples):
        level_1_data[i] = models_from_data[1].evaluate(sample)[0]

    data_input.reset_sampling()

    target_variance = epsilon ** 2

    # Define discrepancy model.
    level_discrepancy = level_1_data - level_0_data

    # These assume we have a pretty good estimate of var.
    level_0_variance = np.var(level_0_data)
    discrepancy_variance = np.var(level_discrepancy)

    layer_0_cost = 1
    layer_1_cost = 1 + 10

    r = np.sqrt(discrepancy_variance / layer_1_cost *
                layer_0_cost / level_0_variance)

    # Calculate sample sizes for each level.
    s = (r * level_0_variance + discrepancy_variance) / (r * target_variance)
    layer_0_sample_size = int(np.ceil(s))
    layer_1_sample_size = int(np.ceil(r * s))

    subset_size = min(len(level_0_data),
                      layer_0_sample_size + layer_1_sample_size)

    subset = np.random.choice(np.arange(len(level_0_data)),
                              subset_size,
                              replace=False)

    sample_x0 = level_0_data[subset[:layer_0_sample_size]]
    sample_x10 = level_discrepancy[subset[:layer_1_sample_size]]

    # Package results for easy comparison with simulator results.
    geoff_variances = np.array([level_0_variance, discrepancy_variance])
    geoff_sample_sizes = np.array([layer_0_sample_size, layer_1_sample_size])
    geoff_estimate = np.mean(sample_x0) + np.mean(sample_x10)

    # Run Simulation for comparison to Geoff's results.
    models = models_from_data[:2]
    models[0].cost = 1
    models[1].cost = 10

    sim = MLMCSimulator(models=models, data=data_input)
    sim_estimate, sim_sample_sizes, output_variances = \
        sim.simulate(epsilon=epsilon, initial_sample_size=initial_sample_size)
    sim_variances = np.squeeze(sim._compute_variances())

    assert np.array_equal(sim_variances, geoff_variances)
    assert np.array_equal(sim._sample_sizes, geoff_sample_sizes)
    assert np.isclose(sim_estimate[0], geoff_estimate, atol=.5)
