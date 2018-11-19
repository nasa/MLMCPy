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
def data_input_no_mpi_slice():

    return InputFromData(os.path.join(data_path, "spring_mass_1D_inputs.txt"),
                         shuffle_data=False, mpi_slice=False)


@pytest.fixture
def data_input_2d():

    return InputFromData(os.path.join(data_path, "2D_test_data.csv"),
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
    output3_filepath = os.path.join(data_path,
                                    "spring_mass_1D_outputs_0.01.txt")

    model1 = ModelFromData(input_filepath, output1_filepath, 1.)
    model2 = ModelFromData(input_filepath, output2_filepath, 4.)
    model3 = ModelFromData(input_filepath, output3_filepath, 16.)

    return [model1, model2, model3]


@pytest.fixture
def models_from_2d_data():

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

    return os.path.join(data_path, "2D_test_data_long.csv")


@pytest.fixture
def comm():
    imp.find_module('mpi4py')

    from mpi4py import MPI
    return MPI.COMM_WORLD


@pytest.fixture
def filename_2d_3_column_data():

    return os.path.join(data_path, "2D_test_data_output_3_col.csv")


def test_model_from_data(data_input, models_from_data):

    sim = MLMCSimulator(models=models_from_data, data=data_input)
    sim.simulate(1., initial_sample_size=20)


def test_spring_model(beta_distribution_input, spring_models):

    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)
    sim.simulate(1., initial_sample_size=20)


def test_for_verbose_exceptions(beta_distribution_input, spring_models):

    # Redirect the verbose out to null.
    stdout = sys.stdout
    with open(os.devnull, 'w') as f:
        sys.stdout = f

        sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)
        sim.simulate(1., initial_sample_size=20, verbose=True)

    # Put stdout back in place.
    sys.stdout = stdout


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


@pytest.mark.parametrize("num_qoi, variances, epsilons",
                         [[1, [[4.], [1.]], [.1]],
                         [2, [[4., 4.], [1, 1.]], [.1, .01]],
                         [3, [[4., 4., 4.], [1, 1., 1.]], [.1, 1., .01]]])
def test_optimal_sample_sizes_expected_outputs(num_qoi, variances, epsilons,
                                               data_input, models_from_data):

    # Set up simulator with values that should produce predictable results
    # from computation of optimal sample sizes.
    test_mlmc = MLMCSimulator(models=models_from_data[:2], data=data_input)

    data_input._data = np.broadcast_to(data_input._data,
                                       (data_input._data.shape[0], num_qoi))

    test_mlmc._epsilons = epsilons
    costs = np.array([1., 4.])

    test_mlmc._compute_optimal_sample_sizes(costs, np.array(variances))

    # Check results.
    sample_sizes = test_mlmc._all_sample_sizes

    if num_qoi == 1:
        expected_sample_size = [800, 200]
    else:
        expected_sample_size = [80000, 20000]

    assert np.all(np.isclose(sample_sizes, expected_sample_size, atol=1))


def test_costs_and_initial_variances_spring_models(beta_distribution_input,
                                                   spring_models):

    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)

    np.random.seed(1)
    sim._initial_sample_size = sim._determine_num_cpu_samples(100)

    costs, variances = sim._compute_costs_and_variances()

    true_variances = np.array([[8.245224951411819],
                               [0.0857219498864355],
                               [7.916295509470576e-06]])

    true_costs = np.array([1., 11., 110.])

    assert np.all(np.isclose(true_costs, costs))
    assert np.all(np.isclose(true_variances, variances, rtol=.1))


def test_costs_and_initial_variances_models_from_data(data_input,
                                                      models_from_data):

    np.random.seed(1)
    sim = MLMCSimulator(models=models_from_data, data=data_input)
    
    sim._initial_sample_size = sim._determine_num_cpu_samples(100)
    costs, variances = sim._compute_costs_and_variances()

    true_variances = np.array([[9.262628271266264],
                               [0.07939834631411287],
                               [5.437083709623372e-06]])

    true_costs = np.array([1.0, 5.0, 20.0])

    assert np.all(np.isclose(true_costs, costs))
    assert np.all(np.isclose(true_variances, variances, rtol=.1))


def test_calculate_estimate_for_springmass_random_input(beta_distribution_input,
                                                        spring_models):

    np.random.seed(1)

    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    sim = MLMCSimulator(models=spring_models,
                        data=beta_distribution_input)

    estimate, sample_sizes, variances = sim.simulate(0.1, 100)

    assert np.isclose(estimate[0], mc_20000_output_sample_mean, atol=.25)


def test_hard_coded_springmass_random_input(beta_distribution_input,
                                            spring_models, comm):

    np.random.seed(1)

    # Result from 20,000 sample monte carlo spring mass simulation.
    mlmc_hard_coded_mean = [12.274674424393805]
    mlmc_hard_coded_variance = [0.01078008]

    sim = MLMCSimulator(models=spring_models,
                        data=beta_distribution_input)

    all_sample_sizes = np.array([1113, 34, 0])
    sample_sizes = []
    for i, sample_size in enumerate(all_sample_sizes):
        sample_sizes.append(sim._determine_num_cpu_samples(sample_size))
    sample_sizes = np.array(sample_sizes)

    sim._initial_sample_size = 0
    sim._sample_sizes = sample_sizes
    sim._all_sample_sizes = all_sample_sizes
    np.random.seed(1)
    estimate, sample_sizes, variances = sim._run_simulation()

    assert np.all(np.isclose(estimate, mlmc_hard_coded_mean))
    assert np.all(np.isclose(variances, mlmc_hard_coded_variance))


def test_estimate_and_variance_improved_by_lower_epsilon(data_input,
                                                         models_from_data):

    np.random.seed(1)

    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    sim = MLMCSimulator(models=models_from_data,
                        data=data_input)

    estimates = np.zeros(3)
    variances = np.zeros_like(estimates)
    for i, epsilon in enumerate([1., .5, .1]):

        estimates[i], sample_sizes, variances[i] = \
            sim.simulate(epsilon=epsilon,
                         initial_sample_size=50)

    error = np.abs(estimates - mc_20000_output_sample_mean)
    assert error[0] > error[1] > error[2]

    assert variances[0] > variances[1] > variances[2]


def test_estimate_and_variance_improved_by_higher_target_cost(data_input,
                                                              models_from_data):

    np.random.seed(1)

    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    sim = MLMCSimulator(models=models_from_data,
                        data=data_input)

    estimates = np.zeros(3)
    variances = np.zeros_like(estimates)
    for i, target_cost in enumerate([5, 25, 500]):

        estimates[i], sample_sizes, variances[i] = \
            sim.simulate(epsilon=.5,
                         initial_sample_size=100,
                         target_cost=target_cost)

    error = np.abs(estimates - mc_20000_output_sample_mean)
    assert error[0] > error[1] > error[2]

    assert variances[0] > variances[1] > variances[2]


@pytest.mark.parametrize("epsilon", [1., .5, .1, .05])
def test_final_variances_less_than_epsilon_goal(data_input,
                                                models_from_data,
                                                epsilon):

    sim = MLMCSimulator(models=models_from_data, data=data_input)

    estimate, sample_sizes, variances = \
        sim.simulate(epsilon=epsilon,
                     initial_sample_size=50)

    assert np.sqrt(variances[0]) < epsilon
    assert not np.isclose(variances[0], 0.)


@pytest.mark.parametrize('sample_sizes', [[1, 0, 0], [1, 0, 1], [1, 1, 0],
                         [1, 1, 1], [1, 2, 1], [10, 5, 2]])
def test_outputs_for_small_sample_sizes(data_input, models_from_data,
                                        sample_sizes, comm):

    output1_filepath = os.path.join(data_path, "spring_mass_1D_outputs_1.0.txt")
    output2_filepath = os.path.join(data_path, "spring_mass_1D_outputs_0.1.txt")
    output3_filepath = os.path.join(data_path,
                                    "spring_mass_1D_outputs_0.01.txt")

    outputs = list()
    outputs.append(np.genfromtxt(output1_filepath)[comm.rank::comm.size])
    outputs.append(np.genfromtxt(output2_filepath)[comm.rank::comm.size])
    outputs.append(np.genfromtxt(output3_filepath)[comm.rank::comm.size])

    sim = MLMCSimulator(models=models_from_data, data=data_input)

    sim._sample_sizes = sample_sizes
    sim._all_sample_sizes = sample_sizes * comm.size
    est, ss, sim_variance = sim._run_simulation()

    # Acquire samples in same sequence simulator would.
    samples = []
    sample_index = 0
    for i, s in enumerate(sample_sizes):

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

        if sample_sizes[i] > 0:
            sample_mean += np.sum(sample) / sample_sizes[i]
            sample_variance += np.var(sample) / sample_sizes[i]

    # Test sample computations vs simulator.
    assert np.isclose(est, sample_mean, atol=10e-15)
    assert np.isclose(sim_variance, sample_variance, atol=10e-15)


@pytest.mark.parametrize("cache_size", [20, 200, 2000])
def test_output_caching(data_input, models_from_data, cache_size):

    sim = MLMCSimulator(models=models_from_data, data=data_input)

    # Run simulation with caching.
    estimate1, sample_sizes, variances1 = sim.simulate(1., cache_size)

    num_levels = len(models_from_data)
    max_samples = np.max(sim._sample_sizes)

    outputs_with_caching = np.zeros((num_levels, max_samples, 1))
    outputs_without_caching = np.zeros_like(outputs_with_caching)

    data_input.reset_sampling()

    for level in range(num_levels):
        for num_samples in sim._sample_sizes:

            if num_samples == 0:
                continue

            samples = data_input.draw_samples(num_samples)

            for i, sample in enumerate(samples):

                outputs_with_caching[level, i] = \
                    sim._evaluate_sample(i, sample, level)

    # Set initial_sample_size to 0 so that it will not use cached values.
    sim._initial_sample_size = 0
    data_input.reset_sampling()

    for level in range(num_levels):
        for num_samples in sim._sample_sizes:

            if num_samples == 0:
                continue

            samples = data_input.draw_samples(num_samples)

            for i, sample in enumerate(samples):

                outputs_without_caching[level, i] = \
                    sim._evaluate_sample(i, sample, level)

    assert np.all(np.isclose(outputs_without_caching, outputs_with_caching))

    # Ignore divide by zero warning caused by 0 initial_sample_size.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        estimate2, sample_sizes, variances2 = sim._run_simulation()

    # Now compare final estimator and output variances.
    # If caching is working properly, they should match.
    assert np.array_equal(estimate1, estimate2)
    assert np.array_equal(variances1, variances2)


def test_input_output_with_differing_column_count(filename_2d_5_column_data,
                                                  filename_2d_3_column_data):

    model1 = ModelFromData(filename_2d_5_column_data,
                           filename_2d_3_column_data,
                           1.)

    model2 = ModelFromData(filename_2d_5_column_data,
                           filename_2d_3_column_data,
                           4.)

    data_input = InputFromData(filename_2d_5_column_data)

    sim = MLMCSimulator(models=[model1, model2], data=data_input)
    sim.simulate(100., 10)


def test_fail_if_model_outputs_do_not_match_shapes(filename_2d_5_column_data,
                                                   filename_2d_3_column_data):

    model1 = ModelFromData(filename_2d_5_column_data,
                           filename_2d_5_column_data,
                           1.)

    model2 = ModelFromData(filename_2d_5_column_data,
                           filename_2d_3_column_data,
                           4.)

    data_input = InputFromData(filename_2d_5_column_data)

    with pytest.raises(ValueError):
        MLMCSimulator(models=[model1, model2], data=data_input)


def test_monte_carlo_estimate_value(data_input, models_from_data):

    np.random.seed(1)

    # Result from 20,000 sample monte carlo spring mass simulation.
    mc_20000_output_sample_mean = 12.3186216602

    # Passing in one model into MLMCSimulator should make it run in monte
    # carlo simulation mode.
    models = [models_from_data[0]]

    sim = MLMCSimulator(models=models, data=data_input)
    estimate, sample_sizes, variances = sim.simulate(.05, 50)

    assert np.isclose(estimate, mc_20000_output_sample_mean, atol=.25)


def test_mc_output_shapes_match_mlmc(data_input, models_from_data):

    first_model = [models_from_data[0]]

    mc_sim = MLMCSimulator(models=first_model, data=data_input)
    mc_estimate, mc_sample_sizes, mc_variances = mc_sim.simulate(1., 50)

    mlmc_sim = MLMCSimulator(models=models_from_data, data=data_input)
    mlmc_estimate, mlmc_sample_sizes, mlmc_variances = mlmc_sim.simulate(1., 50)

    assert mc_estimate.shape == mlmc_estimate.shape
    assert mc_variances.shape == mlmc_variances.shape
    assert mc_sample_sizes.shape != mlmc_sample_sizes.shape


def test_hard_coded_test_2_level(data_input, models_from_data):

    # Get simulation results.
    np.random.seed(1)
    models = models_from_data[:2]

    sim = MLMCSimulator(models=models, data=data_input)
    sim_estimate, sim_sample_sizes, output_variances = \
        sim.simulate(epsilon=1., initial_sample_size=200)
    sim_costs, sim_variances = sim._compute_costs_and_variances()

    # Results from hard coded testing with same parameters.
    hard_coded_variances = np.array([[7.659619446414387],
                                     [0.07288894751770203]])

    hard_coded_sample_sizes = np.array([9, 0])
    hard_coded_estimate = np.array([11.639166038233583])

    assert np.all(np.isclose(sim_variances, hard_coded_variances))
    assert np.all(np.isclose(sim_estimate, hard_coded_estimate))
    assert np.all(np.isclose(sim._all_sample_sizes, hard_coded_sample_sizes))


def test_hard_coded_test_3_level(data_input, models_from_data):

    # Get simulation results.
    sim = MLMCSimulator(models=models_from_data, data=data_input)
    sim_estimate, sim_sample_sizes, output_variances = \
        sim.simulate(epsilon=1., initial_sample_size=200)
    sim_costs, sim_variances = sim._compute_costs_and_variances()

    # Results from hard coded testing with same parameters.
    hard_coded_variances = np.array([[7.659619446414387],
                                     [0.07288894751770203],
                                     [7.363159154583542e-06]])

    hard_coded_sample_sizes = np.array([9, 0, 0])
    hard_coded_estimate = np.array([11.639166038233583])

    assert np.all(np.isclose(sim_variances, hard_coded_variances))
    assert np.all(np.isclose(sim_estimate, hard_coded_estimate))
    assert np.all(np.isclose(sim._all_sample_sizes, hard_coded_sample_sizes))


def test_graceful_handling_of_insufficient_samples(data_input_2d,
                                                   models_from_2d_data):

    # Warnings will be triggered; avoid displaying them during testing.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Test when sampling with too large initial sample size.
        sim = MLMCSimulator(models=models_from_2d_data, data=data_input_2d)
        sim.simulate(epsilon=1., initial_sample_size=10)

        # Test when sampling with too large computed sample sizes.
        sim = MLMCSimulator(models=models_from_2d_data, data=data_input_2d)
        sim.simulate(epsilon=.01, initial_sample_size=5)


def test_can_run_simulation_multiple_times_without_exception(data_input,
                                                             models_from_data):

    sim = MLMCSimulator(models=models_from_data, data=data_input)
    sim.simulate(epsilon=1., initial_sample_size=10)

    sim = MLMCSimulator(models=models_from_data, data=data_input)
    sim.simulate(epsilon=1., initial_sample_size=10)
    sim.simulate(epsilon=2., initial_sample_size=20)


@pytest.mark.parametrize('target_cost', [3, 1, .1, .001])
def test_fixed_cost(beta_distribution_input, spring_models, target_cost):

    np.random.seed(1)

    # Ensure costs are evaluated by simulator via timeit.
    for model in spring_models:
        delattr(model, 'cost')

    sim = MLMCSimulator(models=spring_models,
                        data=beta_distribution_input)

    # Multiply sample sizes times costs and take the sum; verify that this is
    # close to the target cost.
    sim._initial_sample_size = sim._determine_num_cpu_samples(100)
    sim._target_cost = float(target_cost)

    costs, variances = sim._compute_costs_and_variances()
    sim._compute_optimal_sample_sizes(costs, variances)
    sample_sizes = sim._sample_sizes

    expected_cost = np.sum(costs * sample_sizes)

    assert expected_cost <= target_cost

    # Disable caching to ensure accuracy of compute time measurement.
    sim._initial_sample_size = 0

    # Ignore divide by zero warning caused by 0 initial_sample_size.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        start_time = timeit.default_timer()
        sim._run_simulation()
        compute_time = timeit.default_timer() - start_time

    # We should be less than or at least very close to the target.
    # TODO: Try to reduce this overrun with profiling/refactoring.
    assert compute_time < target_cost * 1.2


@pytest.mark.parametrize('num_cpus', [1, 2, 3, 4, 7, 12])
def test_multi_cpu_sample_splitting(data_input, models_from_data, num_cpus):

    total_samples = 100

    sample_sizes = np.zeros(num_cpus)

    sim = MLMCSimulator(models=models_from_data, data=data_input)

    for cpu_rank in range(num_cpus):

        sim._num_cpus = num_cpus
        sim._cpu_rank = cpu_rank

        sample_sizes[cpu_rank] = sim._determine_num_cpu_samples(total_samples)

    # Test that all samples will be utilized.
    assert np.sum(sample_sizes) == total_samples

    # Test that there is never more than a difference of one sample
    # between processes.
    assert np.max(sample_sizes) - np.min(sample_sizes) <= 1


def test_gather_arrays_over_all_cpus(data_input,
                                                  data_input_no_mpi_slice,
                                                  models_from_data):

    # Basic test that does not require axis swapping for reordering.
    sim = MLMCSimulator(data=data_input, models=models_from_data)

    expected_result = data_input_no_mpi_slice._data

    test_result = sim._gather_arrays_over_all_cpus(data_input._data)

    assert np.array_equal(expected_result, test_result)

    # Advanced test that requires axis swapping for reordering.
    test2 = np.ones((2, 10)) * sim._cpu_rank

    expected_result2 = np.repeat([np.arange(sim._num_cpus)], 2, axis=0)
    expected_result2 = np.tile(expected_result2, 10).astype(float)

    test2_result = sim._gather_arrays_over_all_cpus(test2, axis=1)

    assert np.array_equal(expected_result2, test2_result)

    # Test for cross-sync failure issue that could occur if some processes
    # run samples for a particular level while others don't.
    if sim._cpu_rank % 2 == 0:
        sim._sample_sizes = np.array([2, 1, 0])
    else:
        sim._sample_sizes = np.array([2, 0, 0])

    sim._all_sample_sizes = sim._sample_sizes * sim._num_cpus

    # An exception will occur if the problem is present.
    sim._run_simulation()


def test_multiple_cpu_compute_costs_and_variances(data_input,
                                                  data_input_no_mpi_slice,
                                                  models_from_data):

    sim = MLMCSimulator(data=data_input, models=models_from_data)

    num_samples = sim._determine_num_cpu_samples(100)

    cache = np.zeros((3, num_samples, 1))
    test_input_samples = np.zeros_like(cache)

    for level in range(3):

        test_input_samples[level] = sim._data.draw_samples(num_samples)
        lower_level_outputs = np.zeros((num_samples, 1))

        for i, sample in enumerate(test_input_samples[level]):

            cache[level, i] = models_from_data[level].evaluate(sample)

            if level > 0:
                lower_level_outputs[i] = models_from_data[level - 1].evaluate(sample)

        cache[level] -= lower_level_outputs

    gathered_test_inputs = sim._gather_arrays_over_all_cpus(test_input_samples, axis=1)

    # Get outputs across all CPUs before computing variances.
    gathered_test_outputs = sim._gather_arrays_over_all_cpus(cache, axis=1)

    expected_outputs = np.zeros_like(cache)
    expected_input_samples = np.zeros((3, 100, 1))

    for level in range(3):

        expected_input_samples[level] = data_input_no_mpi_slice.draw_samples(100)
        lower_level_outputs = np.zeros((num_samples, 1))

        for i, sample in enumerate(expected_input_samples[level]):

            expected_outputs[level, i] = models_from_data[level].evaluate(sample)

            if level > 0:
                lower_level_outputs[i] = models_from_data[level-1].evaluate(sample)

        expected_outputs -= lower_level_outputs

    assert np.array_equal(gathered_test_inputs, expected_input_samples)
    print np.sum(np.abs(gathered_test_outputs - expected_outputs))
    assert np.array_equal(gathered_test_outputs, expected_outputs)


def test_multiple_cpu_simulation(data_input, models_from_data, comm):

    # Set up baseline simulation like single processor run.
    data_filename = os.path.join(data_path, "spring_mass_1D_inputs.txt")
    full_data_input = InputFromData(data_filename)
    full_data_input._data = np.genfromtxt(data_filename)
    full_data_input._data = \
        full_data_input._data.reshape(full_data_input._data.shape[0], -1)

    base_sim = MLMCSimulator(models=models_from_data, data=full_data_input)
    base_sim._num_cpus = 1
    base_sim._cpu_rank = 0

    base_estimate, base_sample_sizes, base_variances = \
        base_sim.simulate(.1, 200)

    full_data_input.reset_sampling()
    base_costs, base_initial_variances = base_sim._compute_costs_and_variances()

    sim = MLMCSimulator(models=models_from_data, data=data_input)
    estimates, sample_sizes, variances = sim.simulate(.1, 200)

    data_input.reset_sampling()
    sim_costs, initial_variances = sim._compute_costs_and_variances()

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
