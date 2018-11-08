import pytest
import numpy as np
import timeit
import imp
import os
import warnings

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
def filename_2d_3_column_data():

    return os.path.join(data_path, "2D_test_data_output_3_col.csv")


@pytest.fixture
def mpi_info():
    try:
        imp.find_module('mpi4py')

        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        return comm.size, comm.rank

    except ImportError:

        return 1, 0


def test_model_from_data(data_input, models_from_data):

    sim = MLMCSimulator(models=models_from_data, data=data_input)
    sim.simulate(1., initial_sample_size=20)


def test_spring_model(beta_distribution_input, spring_models):

    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)
    sim.simulate(1., initial_sample_size=20)


def test_for_verbose_exceptions(beta_distribution_input, spring_models):

    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)
    sim.simulate(1., initial_sample_size=20, verbose=True)



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

    assert np.all(np.isclose(sample_sizes, [800, 200], atol=1))


def test_optimal_sample_sizes_expected_outputs_2_qoi(data_input,
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

    np.random.seed(1)
    sim._initial_sample_size = 100 // sim._num_cpus

    costs, variances = sim._compute_costs_and_variances()

    true_variances = np.array([[8.245224951411819],
                               [0.0857219498864355],
                               [7.916295509470576e-06]])

    assert np.isclose(true_variances, variances, rtol=.05).all()


def test_costs_and_variances_for_springmass_from_data(data_input,
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


@pytest.mark.parametrize("cache_size", [20, 200, 2000])
def test_output_caching(data_input, models_from_data, cache_size):

    sim = MLMCSimulator(models=models_from_data, data=data_input)

    # Run simulation with caching.
    estimate1, sample_sizes, variances1 = sim.simulate(1., cache_size)

    # Set initial_sample_size to 0 and run simulation again so that it will
    # not use cached values.
    sim._initial_sample_size = 0

    # Ignore divide by zero warning cause by 0 initial_sample_size.
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


@pytest.mark.parametrize('num_cpus', [1, 2, 3, 4, 7, 12])
def test_multi_cpu_sample_sizing(data_input, models_from_data, num_cpus):

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

    assert np.array_equal(sim_variances, hard_coded_variances)
    assert np.array_equal(sim._sample_sizes, hard_coded_sample_sizes)
    assert np.array_equal(sim_estimate, hard_coded_estimate)


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

    assert np.array_equal(sim_variances, hard_coded_variances)
    assert np.array_equal(sim._sample_sizes, hard_coded_sample_sizes)
    assert np.array_equal(sim_estimate, hard_coded_estimate)


def test_graceful_handling_of_insufficient_samples(data_input_2d,
                                                   models_from_2d_data):

    # Test when sampling with too large initial sample size.
    sim = MLMCSimulator(models=models_from_2d_data, data=data_input_2d)
    sim.simulate(epsilon=1., initial_sample_size=10)

    # Test when sampling with too large computed sample sizes.
    sim = MLMCSimulator(models=models_from_2d_data, data=data_input_2d)
    sim.simulate(epsilon=.01, initial_sample_size=10)


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
    sim._initial_sample_size = 100 // sim._num_cpus
    sim._target_cost = float(target_cost)

    costs, variances = sim._compute_costs_and_variances()
    sim._compute_optimal_sample_sizes(costs, variances)
    sample_sizes = sim._sample_sizes

    expected_cost = np.sum(costs * sample_sizes)

    assert expected_cost <= target_cost

    # Disable caching to ensure accuracy of compute time measurement.
    sim._initial_sample_size = 0

    # Ignore divide by zero warning cause by 0 initial_sample_size.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        start_time = timeit.default_timer()
        sim._run_simulation()
        compute_time = timeit.default_timer() - start_time

    # We should be less than or at least very close to the target.
    assert compute_time < target_cost * 1.01


def test_mpi_random_input_unique_per_cpu(mpi_info, beta_distribution_input,
                                         spring_models):

    # This is an MPI only test, so pass if we're running in single cpu mode.
    if mpi_info == (1, 0):
        return

    # Allow simulator to set up sampling and draw a sample.
    sim = MLMCSimulator(models=spring_models, data=beta_distribution_input)
    sim_data = sim._data.draw_samples(10)

    # Share data across cpus and compare to ensure each cpu has unique samples.
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    all_sim_data = comm.allgather(sim_data)

    assert len(all_sim_data) == 2

    for cpu_data in all_sim_data:
        assert not np.array_equal(all_sim_data[0], cpu_data)
