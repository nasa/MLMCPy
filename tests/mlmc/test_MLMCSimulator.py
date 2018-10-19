import pytest
import numpy as np
import timeit
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

    with pytest.raises(TypeError):
        test_mlmc.simulate(epsilon=.1, initial_sample_size=5, target_cost='3')

    with pytest.raises(ValueError):
        test_mlmc.simulate(epsilon=.1, initial_sample_size=5, target_cost=-1)


def test_simulate_expected_output_types(data_input, models_from_data):

    test_mlmc = MLMCSimulator(models=models_from_data, data=data_input)

    result, sample_count, variances = test_mlmc.simulate(epsilon=1.,
                                                         initial_sample_size=20)

    assert isinstance(result, np.ndarray)
    assert isinstance(sample_count, np.ndarray)
    assert isinstance(variances, np.ndarray)


def test_compute_optimal_sample_sizes_expected_outputs(data_input,
                                                       models_from_data):

    # Set up simulator with values that should produce predictable results
    # from computation of optimal sample sizes.
    test_mlmc = MLMCSimulator(models=models_from_data[:2], data=data_input)

    test_mlmc._epsilons = np.array([.1])

    variances = np.array([[4.], [1.]])
    costs = np.array([1., 4.])

    test_mlmc._compute_optimal_sample_sizes(costs, variances)

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

    test_mlmc._compute_optimal_sample_sizes(costs, variances)

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

    test_mlmc._compute_optimal_sample_sizes(costs, variances)

    # Check results.
    sample_sizes = test_mlmc._sample_sizes

    assert np.array_equal(sample_sizes, [80000, 20000])


def test_calculate_initial_variances(beta_distribution_input, spring_models):

    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)

    sim._initial_sample_size = 100

    costs, variances = sim._compute_costs_and_variances()

    true_variances = np.array([[8.245224951411819],
                               [0.0857219498864355],
                               [7.916295509470576e-06]])

    assert np.isclose(true_variances, variances, rtol=.1).all()


def test_calculate_costs_and_variances_for_springmass_from_data(data_input,
                                                              models_from_data):

    sim = MLMCSimulator(models=models_from_data, data=data_input)
    
    sim._initial_sample_size = 100
    costs, variances = sim._compute_costs_and_variances()

    true_variances = np.array([[9.262628271266264],
                               [0.07939834631411287],
                               [5.437083709623372e-06]])

    true_costs = np.array([1.0, 5.0, 20.0])

    assert np.isclose(true_costs, costs).all()
    assert np.isclose(true_variances, variances, rtol=.1).all()


@pytest.mark.parametrize("num_levels", [2, 3])
def test_calculate_estimate_for_springmass_random_input(beta_distribution_input,
                                                        spring_models,
                                                        num_levels):

    np.random.seed(1)
    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    sim = MLMCSimulator(models=spring_models[:num_levels],
                        data=beta_distribution_input)

    estimate, sample_sizes, variances = sim.simulate(.1, 100)

    assert np.isclose(estimate[0], mc_20000_output_sample_mean, rtol=.5)


@pytest.mark.parametrize("epsilon", [1., .5, .1])
def test_final_variances_less_than_epsilon_squared(beta_distribution_input,
                                                   spring_models,
                                                   epsilon):

    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)
    estimate, sample_sizes, variances = sim.simulate(epsilon, 200)

    assert variances[0] < epsilon ** 2


@pytest.mark.parametrize('cost', [100, 10, 1])
def test_fixed_cost(beta_distribution_input, spring_models, cost):

    # Ensure costs are evaluated by simulator via timeit.
    for model in spring_models:
        delattr(model, 'cost')

    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)

    start_time = timeit.default_timer()
    sim.simulate(.1, 100, target_cost=cost)
    compute_time = timeit.default_timer() - start_time

    assert np.isclose(compute_time, cost, rtol=.1)


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

    assert np.isclose(estimate, mc_20000_output_sample_mean, atol=.25)


def test_hard_coded_test_2_level(data_input, models_from_data):

    np.random.seed(1)
    initial_sample_size = 200
    epsilon = 1.

    # Get output data for each layer.
    level_0_data = np.zeros(initial_sample_size)
    level_1_data = np.zeros(initial_sample_size)

    input_samples = data_input.draw_samples(initial_sample_size)

    for i, sample in enumerate(input_samples):
        level_0_data[i] = models_from_data[0].evaluate(sample)[0]

    level_0_variance = np.var(level_0_data)

    # Must resample level 0 for level 0-1 discrepancy variance.
    input_samples = data_input.draw_samples(initial_sample_size)
    for i, sample in enumerate(input_samples):
        level_0_data[i] = models_from_data[0].evaluate(sample)[0]

    for i, sample in enumerate(input_samples):
        level_1_data[i] = models_from_data[1].evaluate(sample)[0]

    data_input.reset_sampling()

    target_variance = epsilon ** 2

    # Define discrepancy model and compute variance.
    level_discrepancy = level_1_data - level_0_data
    discrepancy_variance = np.var(level_discrepancy)

    layer_0_cost = 1
    layer_1_cost = 1 + 4

    r = np.sqrt(discrepancy_variance / layer_1_cost *
                layer_0_cost / level_0_variance)

    # Calculate sample sizes for each level.
    s = (r * level_0_variance + discrepancy_variance) / (r * target_variance)
    level_0_sample_size = int(np.ceil(s))
    level_1_sample_size = int(np.ceil(r * s))

    # Draw samples based on computed sample sizes.
    data_input.reset_sampling()
    sample_0 = data_input.draw_samples(level_0_sample_size)
    sample_1 = data_input.draw_samples(level_1_sample_size)

    # Evaluate samples.
    for i, sample in enumerate(sample_0):
        sample_0[i] = models_from_data[0].evaluate(sample)

    for i, sample in enumerate(sample_1):
        sample_1[i] = models_from_data[1].evaluate(sample)

    # Package results for easy comparison with simulator results.
    hard_coded_variances = np.array([level_0_variance, discrepancy_variance])
    hard_coded_sample_sizes = np.array([level_0_sample_size, level_1_sample_size])
    #hard_coded_estimate = (np.mean(sample_0) + np.mean(sample_1)) / 2.
    hard_coded_estimate = np.mean(np.concatenate((sample_0, sample_1), axis=0))

    # Run Simulation for comparison to hard coded results.
    models = models_from_data[:2]

    sim = MLMCSimulator(models=models, data=data_input)
    sim_estimate, sim_sample_sizes, output_variances = \
        sim.simulate(epsilon=epsilon, initial_sample_size=initial_sample_size)
    sim_costs, sim_variances = sim._compute_costs_and_variances()

    assert np.array_equal(np.squeeze(sim_variances), hard_coded_variances)
    assert np.array_equal(sim._sample_sizes, hard_coded_sample_sizes)
    assert np.array_equal(sim_estimate[0], hard_coded_estimate)


def test_hard_coded_test_3_level(data_input, models_from_data):

    np.random.seed(1)
    initial_sample_size = 200
    epsilon = 1.

    # Get output data for each layer.
    level_0_data = np.zeros(initial_sample_size)
    level_1_data = np.zeros(initial_sample_size)
    level_2_data = np.zeros(initial_sample_size)

    # Compute level 0 variance
    input_samples = data_input.draw_samples(initial_sample_size)

    for i, sample in enumerate(input_samples):
        level_0_data[i] = models_from_data[0].evaluate(sample)[0]

    level_0_variance = np.var(level_0_data)

    # Compute level 0-1 discrepancy variance.
    input_samples = data_input.draw_samples(initial_sample_size)

    for i, sample in enumerate(input_samples):
        level_0_data[i] = models_from_data[0].evaluate(sample)[0]

    for i, sample in enumerate(input_samples):
        level_1_data[i] = models_from_data[1].evaluate(sample)[0]

    level_discrepancy_01 = level_1_data - level_0_data
    discrepancy_variance_01 = np.var(level_discrepancy_01)

    # Get new input samples for level 1-2 discrepancy.
    input_samples = data_input.draw_samples(initial_sample_size)

    for i, sample in enumerate(input_samples):
        level_1_data[i] = models_from_data[1].evaluate(sample)[0]

    for i, sample in enumerate(input_samples):
        level_2_data[i] = models_from_data[2].evaluate(sample)[0]

    # Compute level 1-2 discrepancy variance.
    level_discrepancy_12 = level_2_data - level_1_data
    discrepancy_variance_12 = np.var(level_discrepancy_12)

    target_variance = epsilon ** 2

    level_0_cost = 1
    level_1_cost = 1 + 4
    level_2_cost = 4 + 16

    mu = (np.sqrt(level_0_variance * level_0_cost) +
            np.sqrt(discrepancy_variance_01 * level_1_cost) +
            np.sqrt(discrepancy_variance_12 * level_2_cost)) / target_variance

    level_0_sample_size = mu * np.sqrt(level_0_variance / level_0_cost)
    level_1_sample_size = mu * np.sqrt(discrepancy_variance_01 / level_1_cost)
    level_2_sample_size = mu * np.sqrt(discrepancy_variance_12 / level_2_cost)

    level_0_sample_size = int(np.ceil(level_0_sample_size))
    level_1_sample_size = int(np.ceil(level_1_sample_size))
    level_2_sample_size = int(np.ceil(level_2_sample_size))

    # Draw samples based on computed sample sizes.
    data_input.reset_sampling()
    sample_0 = data_input.draw_samples(level_0_sample_size)
    sample_1 = data_input.draw_samples(level_1_sample_size)
    sample_2 = data_input.draw_samples(level_2_sample_size)

    # Evaluate samples.
    for i, sample in enumerate(sample_0):
        sample_0[i] = models_from_data[0].evaluate(sample)

    for i, sample in enumerate(sample_1):
        sample_1[i] = models_from_data[1].evaluate(sample)

    for i, sample in enumerate(sample_2):
        sample_2[i] = models_from_data[2].evaluate(sample)

    hard_coded_variances = np.array([level_0_variance,
                                    discrepancy_variance_01,
                                    discrepancy_variance_12])

    hard_coded_sample_sizes = np.array([level_0_sample_size,
                                        level_1_sample_size,
                                        level_2_sample_size])

    # hard_coded_estimate = (np.mean(sample_0) +
    #                     np.mean(sample_1) +
    #                     np.mean(sample_2)) / 3.

    hard_coded_estimate = np.mean(np.concatenate((sample_0,
                                                  sample_1,
                                                  sample_2), axis=0))

    # Run Simulation for comparison to hard coded results.
    data_input.reset_sampling()
    sim = MLMCSimulator(models=models_from_data, data=data_input)
    sim_estimate, sim_sample_sizes, output_variances = \
        sim.simulate(epsilon=epsilon, initial_sample_size=initial_sample_size)
    sim_costs, sim_variances = sim._compute_costs_and_variances()

    assert np.array_equal(np.squeeze(sim_variances), hard_coded_variances)
    assert np.array_equal(sim._sample_sizes, hard_coded_sample_sizes)
    assert np.array_equal(sim_estimate[0], hard_coded_estimate)
