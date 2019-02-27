import pytest
import numpy as np
import timeit
import imp
import os
import sys
import warnings

# Needed when running mpiexec. Be sure to run from tests directory.
if 'PYTHONPATH' not in os.environ:

    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.model import ModelFromData
from MLMCPy.input import RandomInput
from MLMCPy.input import InputFromData

from tests.testing_scripts.spring_mass import SpringMassModel

# Create list of paths for each data file.
# Used to parametrize tests.
my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../testing_data"


@pytest.fixture
def random_input():
    """
    Creates a RandomInput object that produces samples from a
    uniform distribution.
    """
    return RandomInput()


@pytest.fixture
def data_input():
    """
    Creates an InputFromData object that produces samples from a file
    containing spring mass input data.
    """
    return InputFromData(os.path.join(data_path, "spring_mass_1D_inputs.txt"),
                         shuffle_data=False)


@pytest.fixture
def data_input_2d():
    """
    Creates an InputFromData object that produces samples from a file
    containing two dimensional data.
    """
    return InputFromData(os.path.join(data_path, "2D_test_data.csv"),
                         shuffle_data=False)


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
def spring_models():
    """
    Creates a list of three SpringMassModel objects of increasing fidelity.
    """
    model_level1 = SpringMassModel(mass=1.5, time_step=1.0, cost=1.0)
    model_level2 = SpringMassModel(mass=1.5, time_step=0.1, cost=10.0)
    model_level3 = SpringMassModel(mass=1.5, time_step=0.01, cost=100.0)

    return [model_level1, model_level2, model_level3]


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

    model1 = ModelFromData(input_filepath, output1_filepath, 1.)
    model2 = ModelFromData(input_filepath, output2_filepath, 4.)
    model3 = ModelFromData(input_filepath, output3_filepath, 16.)

    return [model1, model2, model3]


@pytest.fixture
def models_from_2d_data():
    """
    Creates a list of three ModelFromData objects with a small amount of
    two dimensional data.
    """
    input_filepath = os.path.join(data_path, "2D_test_data.csv")
    output1_filepath = os.path.join(data_path, "2D_test_data_output.csv")
    output2_filepath = os.path.join(data_path, "2D_test_data_output.csv")
    output3_filepath = os.path.join(data_path, "2D_test_data_output.csv")

    model1 = ModelFromData(input_filepath, output1_filepath, 1.)
    model2 = ModelFromData(input_filepath, output2_filepath, 4.)
    model3 = ModelFromData(input_filepath, output3_filepath, 16.)

    return [model1, model2, model3]


@pytest.fixture
def filename_2d_5_column_data():
    """
    Creates a string containing the path to a file with a large number of rows
    of data with five columns.
    """
    return os.path.join(data_path, "2D_test_data_long.csv")


@pytest.fixture
def filename_2d_3_column_data():
    """
    Creates a string containing the path to a file with a large number of rows
    of data with three columns.
    """
    return os.path.join(data_path, "2D_test_data_output_3_col.csv")


@pytest.fixture
def comm():
    """
    Creates a MPI.COMM_WORLD object for working with multi-process information.
    """
    try:

        imp.find_module('mpi4py')

        from mpi4py import MPI
        return MPI.COMM_WORLD

    except ImportError:

        class FakeCOMM:

            def __init__(self):
                self.size = 1
                self.rank = 0

            @staticmethod
            def allgather(thing):

                return np.array([thing])

        return FakeCOMM()


def test_model_from_data(data_input, models_from_data):
    """
    Executes  simulate() with models and inputs created from files
    to ensure there are no exceptions while performing basic functionality.
    """
    sim = MLMCSimulator(models=models_from_data, random_input=data_input)
    sim.simulate(1., initial_sample_sizes=20)


def test_model_with_random_input(beta_distribution_input, spring_models):
    """
    Executes simulate() with models and inputs created from random
    distributions to ensure there are no exceptions while performing basic
    functionality.
    """
    sim = MLMCSimulator(models=spring_models, random_input=beta_distribution_input)
    sim.simulate(1., initial_sample_sizes=20)


def test_for_verbose_exceptions(data_input, models_from_data):
    """
    Executes simulate() with verbose enabled to ensure that there are
    no exceptions while in verbose mode.
    """
    # Redirect the verbose out to null.
    stdout = sys.stdout
    with open(os.devnull, 'w') as f:
        sys.stdout = f

        sim = MLMCSimulator(models=models_from_data, random_input=data_input)
        sim.simulate(1., initial_sample_sizes=20, verbose=True)

    # Put stdout back in place.
    sys.stdout = stdout


def test_init_exception_for_invalid_parameters(beta_distribution_input,
                                               spring_models):
    """
    Ensures the parameters passed to MLMCSimulator are valid, or will throw the
    implemented exceptions.
    """
    with pytest.raises(TypeError):
        MLMCSimulator('Not Input', spring_models)
    
    with pytest.raises(TypeError):
        MLMCSimulator(beta_distribution_input, spring_models[0])

    with pytest.raises(TypeError):
        MLMCSimulator(beta_distribution_input, ['Random', 'List', 'Testing'])


def test_simulate_exception_for_invalid_parameters(data_input,
                                                   models_from_data):
    """
    Ensures that expected exceptions occur when running simulate() with invalid
    parameters.
    """
    test_mlmc = MLMCSimulator(models=models_from_data, random_input=data_input)

    with pytest.raises(ValueError):
        test_mlmc.simulate(epsilon=-.1, initial_sample_sizes=20)

    with pytest.raises(TypeError):
        test_mlmc.simulate(epsilon='one', initial_sample_sizes=20)

    with pytest.raises(ValueError):
        test_mlmc.simulate(epsilon=[.1,.2], initial_sample_sizes=20)

    with pytest.raises(TypeError):
        test_mlmc.simulate(epsilon=.1, initial_sample_sizes='five')

    with pytest.raises(TypeError):
        test_mlmc.simulate(epsilon=.1, initial_sample_sizes=5, target_cost='3')

    with pytest.raises(ValueError):
        test_mlmc.simulate(epsilon=.1, initial_sample_sizes=5, target_cost=-1)

    with pytest.raises(ValueError):
        test_mlmc.simulate(epsilon=.1, initial_sample_sizes=1)

    with pytest.raises(ValueError):
        test_mlmc.simulate(epsilon=.1, initial_sample_sizes=[5, 4, 3, 2])


def test_simulate_expected_output_types(data_input, models_from_data):
    """
    Tests the data types returned by simulate().
    """
    test_mlmc = MLMCSimulator(models=models_from_data, random_input=data_input)

    result, sample_count, variances = \
        test_mlmc.simulate(epsilon=1., initial_sample_sizes=20)

    assert isinstance(result, np.ndarray)
    assert isinstance(sample_count, np.ndarray)
    assert isinstance(variances, np.ndarray)


@pytest.mark.parametrize("num_qoi, variances, epsilons",
                         [[1, [[4.], [1.]], [.1]],
                          [2, [[4., 4.], [1, 1.]], [.1, .01]],
                          [3, [[4., 4., 4.], [1, 1., 1.]], [.1, 1., .01]]])
def test_optimal_sample_sizes_expected_outputs(num_qoi, variances, epsilons,
                                               data_input, models_from_data):
    """
    Tests samples sizes produced by simulator's compute_optimal_sample_sizes()
    against expected computed sample sizes for various sets of parameters.
    """
    test_mlmc = \
        MLMCSimulator(models=models_from_data[:2], random_input=data_input)

    data_input._data = np.broadcast_to(data_input._data,
                                       (data_input._data.shape[0], num_qoi))

    test_mlmc._epsilons = epsilons
    costs = np.array([1., 4.])

    test_mlmc.compute_optimal_sample_sizes(costs, np.array(variances))

    # Check results.
    sample_sizes = test_mlmc._sample_sizes

    if num_qoi == 1:
        expected_sample_size = [800, 200]
    else:
        expected_sample_size = [80000, 20000]

    assert np.all(np.isclose(sample_sizes, expected_sample_size, atol=1))


def test_costs_and_initial_variances_spring_models(beta_distribution_input,
                                                   spring_models):
    """
    Tests costs and variances computed by simulator's
    compute_costs_and_variances() against expected values based on a
    beta distribution.
    """
    sim = MLMCSimulator(models=spring_models, 
                        random_input=beta_distribution_input)

    np.random.seed(1)

    sim._initial_sample_sizes = np.array([100,100,100])
    costs, variances = sim.compute_costs_and_variances()

    

    true_variances = np.array([[8.245224951411819],
                               [0.0857219498864355],
                               [7.916295509470576e-06]])

    true_costs = np.array([1., 11., 110.])

    assert np.all(np.isclose(true_costs, costs))
    assert np.all(np.isclose(true_variances, variances, rtol=.1))


def test_costs_and_initial_variances_models_from_data(data_input,
                                                      models_from_data):
    """
    Tests costs and variances computed by simulator's
    compute_costs_and_variances() against expected values based on data loaded
    from files.
    """
    np.random.seed(1)
    sim = MLMCSimulator(models=models_from_data, random_input=data_input)

    sim._initial_sample_sizes = np.array([100,100,100])
    costs, variances = sim.compute_costs_and_variances()

    true_variances = np.array([[9.262628271266264],
                               [0.07939834631411287],
                               [5.437083709623372e-06]])

    true_costs = np.array([1.0, 5.0, 20.0])

    assert np.all(np.isclose(true_costs, costs))
    assert np.all(np.isclose(true_variances, variances, rtol=.1))


def test_modular_costs_and_initial_variances_from_model(beta_distribution_input,
                                                        spring_models):
    """
    Tests costs and variances computed by simulator's modular
    compute_costs_and_variances() against expected values based on a
    beta distribution.
    """
    sim = MLMCSimulator

    sim = MLMCSimulator(models=spring_models, 
                        random_input=beta_distribution_input)

    np.random.seed(1)

    initial_sample_sizes = np.array([100,100,100])
    costs, variances = sim.compute_costs_and_variances(initial_sample_sizes)

    

    true_variances = np.array([[8.245224951411819],
                               [0.0857219498864355],
                               [7.916295509470576e-06]])

    true_costs = np.array([1., 11., 110.])

    assert np.all(np.isclose(true_costs, costs))
    assert np.all(np.isclose(true_variances, variances, rtol=.1))
    

def test_modular_costs_and_initial_variances_from_data(data_input, 
                                                       models_from_data):
    """
    Tests modular costs and variances computed by simulator's
    compute_costs_and_variances() against expected values based on data loaded
    from files.
    """
    np.random.seed(1)
    sim = MLMCSimulator(models=models_from_data, random_input=data_input)

    sample_sizes = np.array([100,100,100])
    costs, variances = sim.compute_costs_and_variances(sample_sizes)

    true_variances = np.array([[9.262628271266264],
                               [0.07939834631411287],
                               [5.437083709623372e-06]])

    true_costs = np.array([1.0, 5.0, 20.0])

    assert np.all(np.isclose(true_costs, costs))
    assert np.all(np.isclose(true_variances, variances, rtol=.1))


def test_modular_compute_optimal_sample_sizes_models(beta_distribution_input,
                                                     spring_models):
    """
    Tests optimal sample sizes computed by simulator's modular
    compute_optimal_sample_sizes() against expected values based on a
    beta distribution.
    """
    sim = MLMCSimulator(models=spring_models, 
                        random_input=beta_distribution_input)

    np.random.seed(1)

    initial_sample_sizes = np.array([100,100,100])
    costs, variances = sim.compute_costs_and_variances(initial_sample_sizes)
    epsilon = np.sqrt(0.00170890122096)

    optimal_sample_sizes = sim.compute_optimal_sample_sizes(costs,
                                                            variances,
                                                            epsilon)

    true_optimal_sizes = np.array([6506, 200, 0])

    assert np.all(np.array_equal(true_optimal_sizes, optimal_sample_sizes))


def test_modular_compute_estimators_expected_outputs(beta_distribution_input,
                                                    spring_models):
    """
    Ensures consistent outputs when using compute_estimators().
    """
    sim = MLMCSimulator(models=spring_models,
                        random_input=beta_distribution_input)
    
    sim.simulate(epsilon=np.sqrt(.00174325), sample_sizes=[100, 10, 5])

    outputs = [(5,6,7),(8,9,10),(4,3,2)]

    estimates, variances = sim.compute_estimators(outputs)
    true_estimate = 14.8735050632
    true_variance = 0.29611765335

    assert np.isclose(true_estimate, estimates[0])
    assert np.isclose(true_variance, variances[0])


def test_modular_compute_estimators_parameter(beta_distribution_input,
                                              spring_models):
    """
    Ensures the reshape/type check function within compute_estimators()
    appropriately reshapes an object into a np.ndarray.
    """                                          
    sim = MLMCSimulator(models=spring_models,
                        random_input=beta_distribution_input)
    
    output_list = [(1,2,3),(4,5,6),(7,8,9)]
    outputs_array = sim._check_compute_estimators_parameter(output_list)
    assert isinstance(outputs_array, np.ndarray)

    output_tuple = ((1,2,3),(4,5,6),(7,8,9))
    outputs_array = sim._check_compute_estimators_parameter(output_tuple)
    assert isinstance(outputs_array, np.ndarray)

    output_nparr = np.array((1,2,3))
    outputs_array = sim._check_compute_estimators_parameter(output_nparr)
    assert isinstance(outputs_array, np.ndarray)

    with pytest.raises(TypeError):
        sim._check_compute_estimators_parameter('Not a Valid Parameter')


def test_calculate_estimate_for_springmass_random_input(beta_distribution_input,
                                                        spring_models):
    """
    Tests simulator estimate against expected value for beta distribution.
    """
    np.random.seed(1)

    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    sim = MLMCSimulator(models=spring_models,
                        random_input=beta_distribution_input)

    estimate, sample_sizes, variances = sim.simulate(0.1, 100)

    assert np.isclose(estimate[0], mc_20000_output_sample_mean, atol=.25)


def test_monte_carlo_estimate_value(data_input, models_from_data):
    """
    Tests simulator estimate against expected value for spring mass file data.
    """
    np.random.seed(1)

    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    # Passing in one model into MLMCSimulator should make it run in monte
    # carlo simulation mode.
    models = [models_from_data[0]]

    sim = MLMCSimulator(models=models, random_input=data_input)
    estimate, sample_sizes, variances = sim.simulate(.05, 50)

    assert np.isclose(estimate, mc_20000_output_sample_mean, atol=.25)


def test_hard_coded_springmass_random_input(beta_distribution_input,
                                            spring_models, comm):
    """
    Tests simulator estimate and variances against precomputed values.
    """
    np.random.seed(1)

    mlmc_hard_coded_mean = [12.274674424393805]
    mlmc_hard_coded_variance = [0.01078008]

    sim = MLMCSimulator(models=spring_models,
                        random_input=beta_distribution_input)

    all_sample_sizes = np.array([1113, 34, 0])
    get_cpu_samples = np.vectorize(sim._determine_num_cpu_samples)
    sim._cpu_sample_sizes = get_cpu_samples(all_sample_sizes)
    sim._determine_input_output_size()

    sim._caching_enabled = False
    sim._sample_sizes = all_sample_sizes

    np.random.seed(1)
    estimate, cpu_sample_sizes, variances = sim._run_simulation()

    assert np.all(np.isclose(estimate, mlmc_hard_coded_mean))
    assert np.all(np.isclose(variances, mlmc_hard_coded_variance))


def test_estimate_and_variance_improved_by_lower_epsilon(data_input,
                                                         models_from_data):
    """
    Runs simulate with decreasing epsilons and ensures that the resulting
    estimates are increasingly accurate and that the variances decrease.
    """
    np.random.seed(1)

    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    sim = MLMCSimulator(models=models_from_data,
                        random_input=data_input)

    estimates = np.zeros(3)
    variances = np.zeros_like(estimates)
    for i, epsilon in enumerate([1., .5, .1]):

        estimates[i], sample_sizes, variances[i] = \
            sim.simulate(epsilon=epsilon, initial_sample_sizes=50)

    error = np.abs(estimates - mc_20000_output_sample_mean)
    assert error[0] > error[1] > error[2]

    assert variances[0] > variances[1] > variances[2]


def test_always_at_least_one_sample_taken(data_input, models_from_data):

    sim = MLMCSimulator(models=models_from_data, random_input=data_input)

    estimates, sample_sizes, variances = sim.simulate(epsilon=5.,
                                                      initial_sample_sizes=100)

    assert np.sum(sample_sizes) > 0


def test_estimate_and_variance_improved_by_higher_target_cost(data_input,
                                                              models_from_data):
    """
    Runs simulator with increasing target costs and ensures that the resulting
    estimates are increasingly accurate and variances decrease.
    """
    np.random.seed(1)

    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    sim = MLMCSimulator(models=models_from_data, random_input=data_input)

    estimates = np.zeros(3)
    variances = np.zeros_like(estimates)
    sample_sizes = np.zeros((3, 3))
    for i, target_cost in enumerate([5, 25, 500]):

        estimates[i], sample_sizes[i], variances[i] = \
            sim.simulate(epsilon=.5,
                         initial_sample_sizes=100,
                         target_cost=target_cost)

    error = np.abs(estimates - mc_20000_output_sample_mean)
    assert error[0] > error[1] > error[2]

    assert np.sum(sample_sizes[0]) < np.sum(sample_sizes[1])
    assert np.sum(sample_sizes[1]) < np.sum(sample_sizes[2])

    assert variances[0] > variances[1] > variances[2]


@pytest.mark.parametrize("epsilon", [1., .5, .1, .05])
def test_final_variances_less_than_epsilon_goal(data_input,
                                                models_from_data,
                                                epsilon):
    """
    Ensures that square root of variances produced by simulator are lower than
    the specified epsilon parameter.
    """
    sim = MLMCSimulator(models=models_from_data, random_input=data_input)

    estimate, sample_sizes, variances = \
        sim.simulate(epsilon=epsilon,
                     initial_sample_sizes=50)

    assert np.sqrt(variances[0]) < epsilon
    assert not np.isclose(variances[0], 0.)


@pytest.mark.parametrize('cpu_sample_sizes', [[1, 0, 0], [1, 0, 1], [1, 1, 0],
                         [1, 1, 1], [1, 2, 1], [10, 5, 2]])
def test_outputs_for_small_sample_sizes(data_input, models_from_data,
                                        cpu_sample_sizes, comm):
    """
    Test various combinations of small sample sizes to ensure stability of
    simulator under these conditions as well as accuracy of estimate and
    variances.
    """
    output1_filepath = os.path.join(data_path, "spring_mass_1D_outputs_1.0.txt")
    output2_filepath = os.path.join(data_path, "spring_mass_1D_outputs_0.1.txt")
    output3_filepath = os.path.join(data_path,
                                    "spring_mass_1D_outputs_0.01.txt")

    outputs = list()
    outputs.append(np.genfromtxt(output1_filepath)[comm.rank::comm.size])
    outputs.append(np.genfromtxt(output2_filepath)[comm.rank::comm.size])
    outputs.append(np.genfromtxt(output3_filepath)[comm.rank::comm.size])

    all_sample_sizes = np.array(cpu_sample_sizes) * comm.size

    sim = MLMCSimulator(models=models_from_data, random_input=data_input)

    sim._caching_enabled = False
    sim._cpu_sample_sizes = np.array(cpu_sample_sizes)
    sim._sample_sizes = np.copy(all_sample_sizes)
    sim._determine_input_output_size()
    sim_estimate, ss, sim_variance = sim._run_simulation()

    # Acquire samples in same sequence simulator would.
    samples = []
    sample_index = 0
    for i, s in enumerate(cpu_sample_sizes):

        output = outputs[i][sample_index:sample_index + s]

        if i > 0:
            lower_output = outputs[i-1][sample_index:sample_index + s]
        else:
            lower_output = np.zeros_like(output)

        diff = output - lower_output
        all_diff = np.concatenate(comm.allgather(diff))

        samples.append(all_diff)
        sample_index += s

    # Compute mean and variances.
    sample_mean = 0.
    sample_variance = 0.
    for i, sample in enumerate(samples):

        if all_sample_sizes[i] > 0:
            sample_mean += np.sum(sample, axis=0) / all_sample_sizes[i]
            sample_variance += np.var(sample, axis=0) / all_sample_sizes[i]

    # Test sample computations vs simulator.
    assert np.isclose(sim_estimate, sample_mean, atol=10e-15)
    assert np.isclose(sim_variance, sample_variance, atol=10e-15)


@pytest.mark.parametrize("cache_size", [10, 7, 200])
def test_output_caching(data_input, models_from_data, cache_size):
    """
    Runs simulator's _evaluate_sample() with and without caching enabled
    to ensure consistency of outputs. Also tests the estimate and variances
    with and without caching.
    """
    sim = MLMCSimulator(models=models_from_data, random_input=data_input)

    # Run simulation to generate cache.
    estimate1, sample_sizes, variances1 = sim.simulate(1., cache_size)

    # Collect output from _evaluate_sample with caching enabled.
    num_levels = len(models_from_data)
    max_samples = np.max(sim._sample_sizes)

    outputs_with_caching = np.zeros((num_levels, max_samples, 1))
    outputs_without_caching = np.zeros_like(outputs_with_caching)

    data_input.reset_sampling()

    for level in range(num_levels):

        num_samples = sim._sample_sizes[level]

        if num_samples == 0:
            continue

        samples = sim._draw_samples(num_samples)

        for i, sample in enumerate(samples):

            outputs_with_caching[level, i] = \
                sim._evaluate_sample(sample, level)

    # Collect same data with caching disabled.
    sim._caching_enabled = False
    sim._data.reset_sampling()

    for level in range(num_levels):
        num_samples = sim._sample_sizes[level]

        if num_samples == 0:
            continue

        samples = sim._draw_samples(num_samples)
        for i, sample in enumerate(samples):

            outputs_without_caching[level, i] = \
                sim._evaluate_sample(sample, level)

    assert np.all(np.isclose(outputs_without_caching, outputs_with_caching))

    estimate2, sample_sizes, variances2 = sim._run_simulation()

    # Now compare estimator and output variances.
    # If caching is working properly, they should match.
    assert np.array_equal(estimate1, estimate2)
    assert np.array_equal(variances1, variances2)


def test_input_output_with_differing_column_count(filename_2d_5_column_data,
                                                  filename_2d_3_column_data):
    """
    Ensures that simulator handles input and output data with differing numbers
    of columns.
    """
    model1 = ModelFromData(filename_2d_5_column_data,
                           filename_2d_3_column_data,
                           1.)

    model2 = ModelFromData(filename_2d_5_column_data,
                           filename_2d_3_column_data,
                           4.)

    data_input = InputFromData(filename_2d_5_column_data)

    sim = MLMCSimulator(models=[model1, model2], random_input=data_input)
    sim.simulate(100., 10)


def test_fail_if_model_outputs_do_not_match_shapes(filename_2d_5_column_data,
                                                   filename_2d_3_column_data):
    """
    Ensures simulator throws an exception if inputs and outputs with differing
    numbers of samples are provided.
    """
    model1 = ModelFromData(filename_2d_5_column_data,
                           filename_2d_5_column_data,
                           1.)

    model2 = ModelFromData(filename_2d_5_column_data,
                           filename_2d_3_column_data,
                           4.)

    data_input = InputFromData(filename_2d_5_column_data)

    with pytest.raises(ValueError):
        MLMCSimulator(models=[model1, model2], random_input=data_input)


def test_hard_coded_test_2_level(data_input, models_from_data):
    """
    Test simulator cost, initial variance, and sample size computations against
    precomputed values with two models.
    """
    # Get simulation results.
    np.random.seed(1)
    models = models_from_data[:2]

    sim = MLMCSimulator(models=models, random_input=data_input)
    sim_estimate, sim_sample_sizes, output_variances = \
        sim.simulate(epsilon=1., initial_sample_sizes=200)
    sim_costs, sim_variances = sim.compute_costs_and_variances()

    # Results from hard coded testing with same parameters.
    hard_coded_variances = np.array([[7.659619446414387],
                                     [0.07288894751770203]])

    hard_coded_sample_sizes = np.array([9, 0])
    hard_coded_estimate = np.array([11.639166038233583])

    assert np.all(np.isclose(sim_variances, hard_coded_variances))
    assert np.all(np.isclose(sim_estimate, hard_coded_estimate))
    assert np.all(np.isclose(sim._sample_sizes, hard_coded_sample_sizes))


def test_hard_coded_test_3_level(data_input, models_from_data):
    """
    Test simulator cost, initial variance, and sample size computations against
    precomputed values with three models.
    """
    # Get simulation results.
    sim = MLMCSimulator(models=models_from_data, random_input=data_input)
    sim_estimate, sim_sample_sizes, output_variances = \
        sim.simulate(epsilon=1., initial_sample_sizes=200)
    sim_costs, sim_variances = sim.compute_costs_and_variances()

    # Results from hard coded testing with same parameters.
    hard_coded_variances = np.array([[7.659619446414387],
                                     [0.07288894751770203],
                                     [7.363159154583542e-06]])

    hard_coded_sample_sizes = np.array([9, 0, 0])
    hard_coded_estimate = np.array([11.639166038233583])

    assert np.all(np.isclose(sim_variances, hard_coded_variances))
    assert np.all(np.isclose(sim_estimate, hard_coded_estimate))
    assert np.all(np.isclose(sim._sample_sizes, hard_coded_sample_sizes))


def test_graceful_handling_of_insufficient_samples(data_input_2d, comm,
                                                   models_from_2d_data):
    """
    Ensure that the simulator does not throw an exception when insufficient
    samples are provided.
    """
    # Warnings will be triggered; avoid displaying them during testing.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # We only have five rows of data, so ignore cpus of rank > 4.
        # An intentional exception would be thrown by the simulator.
        if comm.rank > 4:
            return

        # Test when sampling with too large initial sample size.
        sim = MLMCSimulator(models=models_from_2d_data, random_input=data_input_2d)
        sim.simulate(epsilon=1., initial_sample_sizes=10)

        # Test when sampling with too large computed sample sizes.
        sim = MLMCSimulator(models=models_from_2d_data, random_input=data_input_2d)
        sim.simulate(epsilon=.01, initial_sample_sizes=5)


def test_multiple_run_consistency(data_input, models_from_data):
    """
    Ensure that simulator can be run multiple times without exceptions and
    returns consistent results.
    """
    sim = MLMCSimulator(models=models_from_data, random_input=data_input)
    estimate1, sample_sizes1, variances1 = \
        sim.simulate(epsilon=1., initial_sample_sizes=100)

    sim = MLMCSimulator(models=models_from_data, random_input=data_input)
    estimate2, sample_sizes2, variances2 = \
        sim.simulate(epsilon=1., initial_sample_sizes=100)

    estimate3, sample_sizes3, variances3 = \
        sim.simulate(epsilon=1., initial_sample_sizes=100)

    assert np.all(np.isclose(estimate1, estimate2))
    assert np.all(np.isclose(estimate2, estimate3))

    assert np.all(np.isclose(sample_sizes1, sample_sizes2))
    assert np.all(np.isclose(sample_sizes2, sample_sizes3))

    assert np.all(np.isclose(variances1, variances2))
    assert np.all(np.isclose(variances2, variances3))


@pytest.mark.parametrize('target_cost', [3, 1, .1, .01])
def test_fixed_cost(beta_distribution_input, spring_models, target_cost):
    """
    Ensure that when running the simulator with a specified target_cost that
    the costs and sample sizes are consistent with the target cost and that
    the actual simulation run time is reasonably consistent as well.
    """
    np.random.seed(1)

    # Ensure costs are evaluated by simulator via timeit.
    for model in spring_models:
        delattr(model, 'cost')

    sim = MLMCSimulator(models=spring_models,
                        random_input=beta_distribution_input)

    # Multiply sample sizes times costs and take the sum; verify that this is
    # close to the target cost.
    sim._initial_sample_sizes = np.array([100, 100, 100])
    sim._target_cost = float(target_cost)

    sim._determine_input_output_size()
    costs, variances = sim.compute_costs_and_variances()
    sim.compute_optimal_sample_sizes(costs, variances)

    expected_cost = np.dot(costs, sim._cpu_sample_sizes)

    assert expected_cost <= target_cost and expected_cost * .9 < target_cost

    # Disable caching to ensure accuracy of compute time measurement.
    sim._caching_enabled = False

    # Ignore divide by zero warning caused by 0 initial_sample_size.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        start_time = timeit.default_timer()
        sim._run_simulation()
        compute_time = timeit.default_timer() - start_time

    # We should be less than or at least very close to the target.
    assert compute_time < target_cost * 1.2


@pytest.mark.parametrize('num_cpus', [1, 2, 3, 4, 7, 12])
def test_multi_cpu_sample_splitting(data_input, models_from_data, num_cpus):
    """
    Tests simulator's _determine_num_cpu_samples() by ensuring that all samples
    will be used and that the difference in number of samples between processes
    is never greater than one.
    """
    total_samples = 100

    sample_sizes = np.zeros(num_cpus)

    sim = MLMCSimulator(models=models_from_data, random_input=data_input)

    for cpu_rank in range(num_cpus):

        sim._num_cpus = num_cpus
        sim._cpu_rank = cpu_rank

        sample_sizes[cpu_rank] = sim._determine_num_cpu_samples(total_samples)

    # Test that all samples will be utilized.
    assert np.sum(sample_sizes) == total_samples

    # Test that there is never more than a difference of one sample
    # between processes.
    assert np.max(sample_sizes) - np.min(sample_sizes) <= 1


def test_gather_arrays(data_input, models_from_data, comm):
    """
    Tests simulator's _gather_arrays() to ensure that it produces expected
    results for axis=0 and axis=1 parameters.
    """
    sim = MLMCSimulator(random_input=data_input, models=models_from_data)

    # Axis 0 test.
    test = np.ones((2, 2)) * comm.rank

    expected_result = np.zeros((2, 2))

    for i in range(1, comm.size):
        new_block = np.ones((2, 2)) * i
        expected_result = np.concatenate((expected_result, new_block), axis=0)

    test_result = sim._gather_arrays(test, axis=0)

    assert np.array_equal(expected_result, test_result)

    # Axis 1 test.
    test2 = np.ones((2, 2)) * comm.rank

    expected_result2 = np.zeros((2, 2))

    for i in range(1, comm.size):
        new_block = np.ones((2, 2)) * i
        expected_result2 = np.concatenate((expected_result2, new_block), axis=1)

    test2_result = sim._gather_arrays(test2, axis=1)

    assert np.array_equal(expected_result2, test2_result)

    # Test for cross-sync failure issue that could occur if some processes
    # run samples for a particular level while others don't.
    if comm.rank % 2 == 0:
        sim._cpu_sample_sizes = np.array([2, 1, 0])
    else:
        sim._cpu_sample_sizes = np.array([2, 0, 0])

    sim._sample_sizes = sim._sample_sizes * comm.size

    # An exception will occur here if the problem is present.
    sim._run_simulation()


@pytest.mark.parametrize('num_samples', [2, 3, 5, 7, 11, 23, 101])
def test_multiple_cpu_compute_costs_and_variances(data_input, num_samples,
                                                  models_from_data):
    """
    Tests simulator's computation of costs and initial variances in an MPI
    environment for various numbers of initial sample sizes vs single cpu case.
    Also tests to ensure consistency of sampling.
    """
    sim = MLMCSimulator(random_input=data_input, models=models_from_data)

    num_cpu_samples = sim._determine_num_cpu_samples(num_samples)

    cache = np.zeros((3, num_cpu_samples, 1))
    test_input_samples = np.zeros_like(cache)

    # Get samples/outputs for MPI case.
    for level in range(3):

        test_input_samples[level] = sim._draw_samples(num_samples)
        lower_level_outputs = np.zeros((num_cpu_samples, 1))

        for i, sample in enumerate(test_input_samples[level]):

            cache[level, i] = models_from_data[level].evaluate(sample)

            if level > 0:
                lower_level_outputs[i] = \
                    models_from_data[level - 1].evaluate(sample)

        cache[level] -= lower_level_outputs

    gathered_test_input_samples = \
        sim._gather_arrays(test_input_samples, axis=1)

    # Get outputs across all CPUs before computing variances.
    gathered_test_outputs = sim._gather_arrays(cache, axis=1)

    expected_outputs = np.zeros((3, num_samples, 1))
    expected_input_samples = np.zeros((3, num_samples, 1))
    data_input.reset_sampling()

    # Get samples/outputs for single processor for comparison.
    for level in range(3):

        expected_input_samples[level] = \
            data_input.draw_samples(num_samples)
        lower_level_outputs = np.zeros((num_samples, 1))

        for i, sample in enumerate(expected_input_samples[level]):

            expected_outputs[level, i] = \
                models_from_data[level].evaluate(sample)

            if level > 0:
                lower_level_outputs[i] = \
                    models_from_data[level-1].evaluate(sample)

        expected_outputs[level] -= lower_level_outputs

    assert gathered_test_input_samples.shape == expected_input_samples.shape
    assert np.array_equal(gathered_test_input_samples, expected_input_samples)

    assert gathered_test_outputs.shape == expected_outputs.shape
    assert np.array_equal(gathered_test_outputs, expected_outputs)


def test_multiple_cpu_simulation(data_input, models_from_data, comm):
    """
    Compares outputs of simulator in single cpu vs MPI environments to ensure
    consistency.
    """
    # Set up baseline simulation like single processor run.
    data_filename = os.path.join(data_path, "spring_mass_1D_inputs.txt")
    full_data_input = InputFromData(data_filename)
    full_data_input._data = np.genfromtxt(data_filename)
    full_data_input._data = \
        full_data_input._data.reshape(full_data_input._data.shape[0], -1)

    base_sim = MLMCSimulator(models=models_from_data, random_input=full_data_input)
    base_sim._num_cpus = 1
    base_sim._cpu_rank = 0

    base_estimate, base_sample_sizes, base_variances = \
        base_sim.simulate(.1, 200)

    full_data_input.reset_sampling()
    base_costs, base_initial_variances = base_sim.compute_costs_and_variances()

    sim = MLMCSimulator(models=models_from_data, random_input=data_input)
    estimates, sample_sizes, variances = sim.simulate(.1, 200)

    data_input.reset_sampling()
    sim_costs, initial_variances = sim.compute_costs_and_variances()

    assert np.all(np.isclose(base_initial_variances, initial_variances))
    assert np.all(np.isclose(base_costs, sim_costs))

    all_estimates = comm.allgather(estimates)
    all_sample_sizes = comm.allgather(sample_sizes)
    all_variances = comm.allgather(variances)

    assert np.all(estimates[0] == estimates)
    assert np.all(variances[0] == variances)

    for estimate in all_estimates:
        assert np.all(np.isclose(estimate, base_estimate))

    for variance in all_variances:
        assert np.all(np.isclose(variance, base_variances))

    for i, sample_size in enumerate(all_sample_sizes):
        assert np.array_equal(base_sample_sizes, sample_size)

def test_simulate_with_set_sample_sizes(data_input, models_from_data):
    """
    Tests running MLMC by specifying the number of samples to run on each
    level rather than computing it. Takes precomputed reference solution from
    spring-mass data example
    """
    test_mlmc = MLMCSimulator(models=models_from_data, random_input=data_input)

    sample_sizes = [7007, 290, 1]

    result, sample_count, variances = \
        test_mlmc.simulate(epsilon=1., sample_sizes=sample_sizes)

    assert np.array_equal(sample_sizes, sample_count)
    assert np.isclose(result[0], 12.31220864)
 
@pytest.mark.parametrize('sample_sizes', [[5], [2,2,2,2], [-1,5,5]])
def test_simulate_with_bad_sample_sizes_input(data_input, models_from_data,
                                              sample_sizes):
    """
    Tests running MLMC by specifying the number of samples but providing 
    bad values for the sample_sizes input. Makes sure exceptions are raised.
    """

    test_mlmc = MLMCSimulator(models=models_from_data, random_input=data_input)

    with pytest.raises(ValueError):
        test_mlmc.simulate(epsilon=1., sample_sizes=sample_sizes)


@pytest.mark.parametrize('sample_sizes', ['foo', set([1,2,3])])
def test_simulate_with_bad_type_sample_sizes_input(data_input, models_from_data,
                                              sample_sizes):
    """
    Tests running MLMC by specifying the number of samples but providing 
    wrong type for sample_sizes input
    """

    test_mlmc = MLMCSimulator(models=models_from_data, random_input=data_input)

    with pytest.raises(TypeError):
        test_mlmc.simulate(epsilon=1., sample_sizes=sample_sizes)


def test_simulate_with_scalar_sample_sizes(data_input, models_from_data):
    """
    Tests running MLMC by specifying the number of samples to run on each
    level. Tests providing just a scalar value that mlmc handles this
    """
    test_mlmc = MLMCSimulator(models=models_from_data, random_input=data_input)

    sample_sizes = 5

    result, sample_count, variances = \
        test_mlmc.simulate(epsilon=1., sample_sizes=sample_sizes)

    assert np.array_equal(sample_count, np.array([5,5,5]))


