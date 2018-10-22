import pytest
import os
import imp
import numpy as np

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


# Pull all scrambled sample data from a file as one ndarray.
def get_full_data_set(file_path):

    filename = os.path.basename(file_path)
    file_length = data_file_lengths[filename]

    data_sampler = InputFromData(file_path)
    full_data_set = data_sampler.draw_samples(file_length)

    return full_data_set


@pytest.fixture
def data_filename_2d():

    return os.path.join(data_path, "2D_test_data.csv")


@pytest.fixture
def bad_data_file():

    data_file_with_bad_data = os.path.join(data_path, "bad_data.txt")
    return data_file_with_bad_data


@pytest.fixture
def mpi_info():
    try:
        imp.find_module('mpi4py')

        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        return comm.size, comm.rank

    except ImportError:

        return 1, 0


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

    all_sampled_data = get_full_data_set(data_filename)

    file_data = np.genfromtxt(data_filename)
    file_data = file_data.reshape(file_data.shape[0], -1)

    assert all_sampled_data.shape == file_data.shape


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_sample_data_is_scrambled(data_filename):

    all_sampled_data = get_full_data_set(data_filename)

    file_data = np.genfromtxt(data_filename)
    file_data = file_data.reshape(file_data.shape[0], -1)

    assert not np.array_equal(all_sampled_data, file_data)
    assert np.isclose(np.sum(all_sampled_data), np.sum(file_data))


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
                                      skip_header=rows_to_skip,)

    skipped_row_count = skipped_row_input._data.shape[0]

    assert normal_row_count - rows_to_skip == skipped_row_count


@pytest.mark.parametrize("data_filename", data_file_paths, ids=data_file_names)
def test_mpi_input_sample_sliced(mpi_info, data_filename):

    # This is an MPI only test, so pass if we're running in single cpu mode.
    if mpi_info == (1, 0):
        return

    num_cpus, cpu_rank = mpi_info

    # Get expected slice.
    data_input = InputFromData(data_filename)
    all_data = get_full_data_set(data_filename)

    slice_size = all_data.shape[0] // num_cpus

    slice_start_index = slice_size * cpu_rank
    sliced_data = all_data[slice_start_index: slice_start_index + slice_size]

    input_data = data_input._data

    # Ensure InputFromData sliced the data in the same way.
    assert np.array_equal(sliced_data, input_data)
