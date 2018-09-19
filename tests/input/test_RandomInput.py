import pytest
import numpy as np

from MLMCPy.input import RandomInput


@pytest.fixture
def uniform_distribution_input():

    return RandomInput(np.random.uniform)


@pytest.fixture
def invalid_distribution_function():

    def invalid_function():
        return [1, 2, 3]

    return invalid_function


def test_init_invalid_input():

    with pytest.raises(TypeError):
        RandomInput(1)


def test_draw_samples_expected_output(uniform_distribution_input):

    sample = uniform_distribution_input.draw_samples(5)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == (5,)


def test_exception_invalid_distribution_function(invalid_distribution_function):

    with pytest.raises(TypeError):
        invalid_sampler = RandomInput(invalid_distribution_function)
        invalid_sampler.draw_samples(5)


def test_extra_distribution_function_parameters():

    normal_sampler = RandomInput(np.random.normal, loc=1., scale=2.0)
    sample = normal_sampler.draw_samples(5)

    assert isinstance(sample, np.ndarray)
    assert sample.shape == (5,)


def test_draw_samples_invalid_arguments(uniform_distribution_input):

    with pytest.raises(TypeError):
        uniform_distribution_input.draw_samples(1.)

    with pytest.raises(ValueError):
        uniform_distribution_input.draw_samples(0)

