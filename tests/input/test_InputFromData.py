import pytest
import os
import sys
import numpy as np
import imp

# Needed when running mpiexec. Be sure to run from tests directory.
if 'PYTHONPATH' not in os.environ:

    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

from MLMCPy.input import InputFromData

# Create list of paths for each data file.
# Used to parametrize tests.
my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../testing_data"

data_file_names = ["spring_mass_1D_inputs.txt", "2D_test_data.csv"]

# Number of lines in each data file.
data_file_lengths = {'spring_mass_1D_inputs.txt': 10000,
                     '2D_test_data.csv': 5}

data_file_paths = []
for data_file in data_file_names:
    data_file_paths.append(os.path.join(data_path, data_file))


# Sample all scrambled sample data from a file as one ndarray.
def sample_entire_data_set(file_path):

    filename = os.path.basename(file_path)
    file_length = data_file_lengths[filename]

    data_sampler = InputFromData(file_path)
    full_data_set = data_sampler.draw_samples(file_length)

    return full_data_set


def get_data_file_size(file_path):

    file_name = os.path.basename(file_path)
    return data_file_lengths[file_name]


@pytest.fixture
def spring_data_input():

    return InputFromData(os.path.join(data_path, "spring_mass_1D_inputs.txt"),
                         shuffle_data=False)


@pytest.fixture
def data_filename_2d():

    return os.path.join(data_path, "2D_test_data.csv")


@pytest.fixture
def bad_data_file():

    data_file_with_bad_data = os.path.join(data_path, "bad_data.txt")
    return data_file_with_bad_data


def test_init_fails_on_invalid_input_file():

    with pytest.raises(IOError):
        InputFromData("not_a_real_file.txt")


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_init_does_not_fail_on_valid_input_file(data_filename):

    InputFromData(data_filename)


@pytest.fixture
def comm():
    imp.find_module('mpi4py')

    from mpi4py import MPI
    return MPI.COMM_WORLD


@pytest.mark.parametrize("delimiter, filename",
                         [(",", "2D_test_data_comma_delimited.csv"),
                          (";", "2D_test_data_semicolon_delimited.csv"),
                          (1, "2D_test_data_length_delimited.csv")],
                         ids=["comma", "semicolon", "length"])
def test_can_load_alternatively_delimited_files(delimiter, filename):

    file_path = os.path.join(data_path, filename)
    sampler = InputFromData(file_path, delimiter=delimiter)
    sample = sampler.draw_samples(5)

    assert np.sum(sample) == 125.


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_draw_samples_returns_expected_output(data_filename):

    data_sampler = InputFromData(data_filename)

    for num_samples in range(1, 4):

        sample = data_sampler.draw_samples(num_samples)
        data_sampler.reset_sampling()

        # Returns correct data type.
        assert isinstance(sample, np.ndarray)

        # Returns correct shape of data.
        assert len(sample.shape) == 2

        # Returns requested number of samples.
        assert sample.shape[0] == num_samples


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_draw_samples_pulls_all_input_data(data_filename):

    all_sampled_data = sample_entire_data_set(data_filename)

    file_data = np.genfromtxt(data_filename)
    file_data = file_data.reshape(file_data.shape[0], -1)

    assert all_sampled_data.shape[0] == file_data.shape[0]


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_sample_data_is_scrambled(data_filename):

    all_file_data = sample_entire_data_set(data_filename)
    file_length = all_file_data.shape[0]

    data_sampler = InputFromData(data_filename)
    sample_data = data_sampler.draw_samples(file_length)

    assert not np.array_equal(all_file_data, sample_data)
    assert np.isclose(np.sum(all_file_data), np.sum(sample_data))


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_draw_samples_invalid_parameters_fails(data_filename):

    data_sampler = InputFromData(data_filename)

    with pytest.raises(TypeError):
        data_sampler.draw_samples("five")

    with pytest.raises(ValueError):
        data_sampler.draw_samples(0)


def test_fail_on_nan_data(bad_data_file):

    with pytest.raises(ValueError):
        InputFromData(bad_data_file)


@pytest.mark.parametrize("rows_to_skip", [1, 2, 3])
def test_skip_rows(data_filename_2d, rows_to_skip):

    normal_input = InputFromData(data_filename_2d)
    normal_row_count = normal_input._data.shape[0]

    skipped_row_input = InputFromData(data_filename_2d,
                                      skip_header=rows_to_skip)

    skipped_row_count = skipped_row_input._data.shape[0]

    assert normal_row_count - rows_to_skip == skipped_row_count


def test_draw_sample_warning_issued_for_insufficient_data(data_filename_2d):

    small_input = InputFromData(data_filename_2d)

    with pytest.warns(UserWarning):
        small_input.draw_samples(1000)
