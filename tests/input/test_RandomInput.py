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
    """
    Creates a RandomInput object that produces samples from a
    uniform distribution.
    """
    return RandomInput(np.random.uniform)


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
def comm():
    """
    Creates a MPI.COMM_WORLD object for working with multi-process information.
    """
    imp.find_module('mpi4py')

    from mpi4py import MPI
    return MPI.COMM_WORLD


def test_init_invalid_input():
    """
    Ensure an exception is raised if an object that is not a function is
    provided.
    """
    with pytest.raises(TypeError):
        RandomInput(1)


def test_draw_samples_expected_output(uniform_distribution_input):
    """
    Ensure outputs from draw_samples matches expected type and shape.
    """
    for num_samples in range(1, 10):

        sample = uniform_distribution_input.draw_samples(num_samples)
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (num_samples, 1)


def test_exception_invalid_distribution_function():
    """
    Ensure an exception is thrown if the provided function does not return
    the expected data type.
    """
    def invalid_distribution_function():
        return [1, 2, 3]

    with pytest.raises(TypeError):
        invalid_sampler = RandomInput(invalid_distribution_function)
        invalid_sampler.draw_samples(5)


def test_extra_distribution_function_parameters():
    """
    Test ability to specify optional distribution function parameters.
    """
    np.random.seed(1)

    normal_sampler = RandomInput(np.random.normal, loc=1.)
    sample = normal_sampler.draw_samples(100)

    assert isinstance(sample, np.ndarray)
    assert sample.shape == (100, 1)
    assert np.abs(np.mean(sample) - 1.) < .2


@pytest.mark.parametrize('argument', [1., 0, 'a'])
def test_draw_samples_invalid_arguments(uniform_distribution_input, argument):
    """
    Ensure an exception occurs if invalid parameters are passed to draw_samples.
    """
    with pytest.raises(Exception):
        uniform_distribution_input.draw_samples(argument)


def test_distribution_exception_if_size_parameter_not_accepted():
    """
    Ensure an exception occurs if the distribution function does not accept
    a size parameter.
    """
    def invalid_distribution_function():
        return np.zeros(5)

    invalid_input = \
        RandomInput(distribution_function=invalid_distribution_function)

    with pytest.raises(TypeError):
        invalid_input.draw_samples(10)


