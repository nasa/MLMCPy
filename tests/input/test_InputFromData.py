import pytest
import os
import numpy as np

from MLMCPy.input import InputFromData

# Access spring mass data:
my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../../examples/spring_mass/from_data/data"


@pytest.fixture
def spring_data_filename():
    input_data_file = os.path.join(data_path, "spring_mass_1D_inputs.txt")
    return input_data_file


@pytest.fixture
def spring_data_sampler(spring_data_filename):
    return InputFromData(spring_data_filename)


@pytest.fixture
def all_sampled_data(spring_data_sampler):

    all_sampler_samples = spring_data_sampler.draw_samples(5)
    sampler_sample = spring_data_sampler.draw_samples(5)

    # stack all samples into one ndarray.
    while sampler_sample is not None:

        all_sampler_samples = np.hstack((all_sampler_samples, sampler_sample))
        sampler_sample = spring_data_sampler.draw_samples(5)

    return all_sampler_samples


def test_init_fails_on_invalid_input_file():

    with pytest.raises(IOError):
        InputFromData("not_a_real_file.txt")


def test_init_does_not_fail_on_valid_file(spring_data_filename):

    InputFromData(spring_data_filename)


def test_draw_samples_returns_expected_output(spring_data_sampler):

    sample = spring_data_sampler.draw_samples(5)

    assert isinstance(sample, np.ndarray)
    assert sample.shape[0] == 5


def test_draw_samples_pulls_all_input_data(all_sampled_data,
                                           spring_data_filename):

    spring_data = np.genfromtxt(spring_data_filename)

    assert all_sampled_data.shape == spring_data.shape


def test_sample_data_is_scrambled(all_sampled_data, spring_data_filename):

    spring_data = np.genfromtxt(spring_data_filename)

    assert not np.array_equal(all_sampled_data, spring_data)
    assert np.isclose(np.sum(all_sampled_data), np.sum(spring_data))


def test_draw_samples_invalid_parameters_fails(spring_data_sampler):

    with pytest.raises(TypeError):
        spring_data_sampler.draw_samples("five")

    with pytest.raises(ValueError):
        spring_data_sampler.draw_samples(0)
