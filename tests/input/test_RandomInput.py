import os
import sys
import imp
import pytest
import numpy as np

# Needed when running mpiexec. Be sure to run from tests directory.
if 'PYTHONPATH' not in os.environ:

    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

from MLMCPy.input import RandomInput


@pytest.fixture
def uniform_distribution_input():

    return RandomInput(np.random.uniform)


@pytest.fixture
def beta_distribution_input():

    np.random.seed(1)

    def beta_distribution(shift, scale, alpha, beta, size):
        return shift + scale * np.random.beta(alpha, beta, size)

    return RandomInput(distribution_function=beta_distribution,
                       shift=1.0, scale=2.5, alpha=3., beta=2.)


@pytest.fixture
def invalid_distribution_function():

    def invalid_function():
        return [1, 2, 3]

    return invalid_function


@pytest.fixture
def comm():
    imp.find_module('mpi4py')

    from mpi4py import MPI
    return MPI.COMM_WORLD


def test_init_invalid_input():

    with pytest.raises(TypeError):
        RandomInput(1)


def test_draw_samples_expected_output(uniform_distribution_input):

    for num_samples in range(1, 10):

        sample = uniform_distribution_input.draw_samples(num_samples)
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (num_samples, 1)


def test_exception_invalid_distribution_function(invalid_distribution_function):

    with pytest.raises(TypeError):
        invalid_sampler = RandomInput(invalid_distribution_function)
        invalid_sampler.draw_samples(5)


def test_extra_distribution_function_parameters():

    normal_sampler = RandomInput(np.random.normal, loc=1., scale=2.0)
    sample = normal_sampler.draw_samples(5)

    assert isinstance(sample, np.ndarray)
    assert sample.shape == (5, 1)


@pytest.mark.parametrize('argument', [1., 0, 'a'])
def test_draw_samples_invalid_arguments(uniform_distribution_input, argument):

    with pytest.raises(Exception):
        uniform_distribution_input.draw_samples(argument)


def test_distribution_exception_if_size_parameter_not_accepted():

    def invalid_distribution_function():
        return np.zeros(5)

    invalid_input = \
        RandomInput(distribution_function=invalid_distribution_function)

    with pytest.raises(TypeError):
        invalid_input.draw_samples(10)


def test_multi_cpu_sampling_uniform(uniform_distribution_input, comm):

    # Get per process samples, then aggregate them to compare to single cpu
    # sampling.
    np.random.seed(1)
    this_test_sample = uniform_distribution_input.draw_samples(10)

    test_sample_list = comm.allgather(this_test_sample)

    test_samples = np.zeros((10 * comm.size, 1))
    for i, sample in enumerate(test_sample_list):
        test_samples[i::comm.size] = sample

    # Get samples that would be returned in single cpu environment.
    uniform_distribution_input._num_cpus = 1
    uniform_distribution_input._cpu_rank = 0

    np.random.seed(1)
    baseline_samples = uniform_distribution_input.draw_samples(10 * comm.size)

    assert np.array_equal(test_samples, baseline_samples)


def test_multi_cpu_sampling_beta(beta_distribution_input, comm):

    # Get per process samples, then aggregate them to compare to single cpu
    # sampling.
    np.random.seed(1)
    this_test_sample = beta_distribution_input.draw_samples(10)

    test_sample_list = comm.allgather(this_test_sample)

    test_samples = np.zeros((10 * comm.size, 1))
    for i, sample in enumerate(test_sample_list):
        test_samples[i::comm.size] = sample

    # Get samples that would be returned in single cpu environment.
    beta_distribution_input._num_cpus = 1
    beta_distribution_input._cpu_rank = 0

    np.random.seed(1)
    baseline_samples = beta_distribution_input.draw_samples(10 * comm.size)

    assert np.array_equal(test_samples, baseline_samples)


