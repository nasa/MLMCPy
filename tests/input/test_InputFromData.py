import pytest
import os
import numpy as np

from MLMCPy.input import InputFromData

# Create list of paths for each data file.
# Used to parametrize tests.
my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../testing_data"

data_file_names = ["spring_mass_1D_inputs.txt", "2D_test_data.csv"]

data_file_paths = []
for data_file in data_file_names:
    data_file_paths.append(os.path.join(data_path, data_file))


# Pull all scrambled sample data from a file as one ndarray.
def get_full_data_set(filename):

    data_sampler = InputFromData(filename)

    full_data_set = data_sampler.draw_samples(500)
    sampler_sample = data_sampler.draw_samples(500)

    # Number of dimensions will determine whether we should use hstack
    # or vstack to build the data set.
    is_one_dimensional_data = len(full_data_set.shape) == 1

    # stack all samples into one ndarray (full_sample).
    while sampler_sample is not None:

        if is_one_dimensional_data:
            full_data_set = np.hstack((full_data_set, sampler_sample))
        else:
            full_data_set = np.vstack((full_data_set, sampler_sample))

        sampler_sample = data_sampler.draw_samples(500)

    return full_data_set


def test_init_fails_on_invalid_input_file():

    with pytest.raises(IOError):
        InputFromData("not_a_real_file.txt")


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_init_does_not_fail_on_valid_input_file(data_filename):

    InputFromData(data_filename)


@pytest.mark.parametrize("delimiter, filename",
                         [(",", "2D_test_data_comma_delimited.csv"),
                          (";", "2D_test_data_semicolon_delimited.csv"),
                          (1, "2D_test_data_length_delimited.csv")],
                         ids=["comma", "semicolon", "length"])
def test_can_load_alternatively_delimited_files(delimiter, filename):

    file_path = os.path.join(data_path, filename)
    sampler = InputFromData(file_path, delimiter=delimiter)
    sample = sampler.draw_samples(5)

    assert int(np.sum(sample)) == 125


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_draw_samples_returns_expected_output(data_filename):

    data_sampler = InputFromData(data_filename)
    sample = data_sampler.draw_samples(5)

    assert isinstance(sample, np.ndarray)
    assert sample.shape[0] == 5


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_draw_samples_pulls_all_input_data(data_filename):

    all_sampled_data = get_full_data_set(data_filename)

    file_data = np.genfromtxt(data_filename)

    assert all_sampled_data.shape == file_data.shape


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_sample_data_is_scrambled(data_filename):

    all_sampled_data = get_full_data_set(data_filename)
    file_data = np.genfromtxt(data_filename)

    assert not np.array_equal(all_sampled_data, file_data)
    assert np.isclose(np.sum(all_sampled_data), np.sum(file_data))


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_draw_samples_invalid_parameters_fails(data_filename):

    data_sampler = InputFromData(data_filename)

    with pytest.raises(TypeError):
        data_sampler.draw_samples("five")

    with pytest.raises(ValueError):
        data_sampler.draw_samples(0)
