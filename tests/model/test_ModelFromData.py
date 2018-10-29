import pytest
import numpy as np
import os
import timeit

from MLMCPy.model import ModelFromData

# Access spring mass data:
my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../testing_data"


@pytest.fixture
def input_data_file():
    input_data_file = os.path.join(data_path, "spring_mass_1D_inputs.txt")
    return input_data_file


@pytest.fixture
def output_data_file():
    output_data_file = os.path.join(data_path, "spring_mass_1D_outputs_0.1.txt")
    return output_data_file


@pytest.fixture
def bad_data_file():
    data_file = os.path.join(data_path, "bad_data.txt")
    return data_file


@pytest.fixture
def input_data_file_2d():
    input_data_file = os.path.join(data_path, "2D_test_data.csv")
    return input_data_file


@pytest.fixture
def output_data_file_2d():
    output_data_file = os.path.join(data_path, "2D_test_data_output.csv")
    return output_data_file


@pytest.fixture
def input_data_file_with_duplicates():
    output_data_file = os.path.join(data_path, "2D_test_data_duplication.csv")
    return output_data_file


@pytest.mark.parametrize("index", [2, 4, 7, 13, 17])
def test_evaluate_1d_data(input_data_file, output_data_file, index):

    # Initialize model from spring-mass example data files:
    data_model = ModelFromData(input_data_file, output_data_file, 1.)

    input_data = np.genfromtxt(input_data_file)
    output_data = np.genfromtxt(output_data_file)

    # Model expects arrays as inputs/outputs.
    model_output = data_model.evaluate(input_data[index])

    true_output = output_data[index]
    assert np.all(np.isclose(model_output, true_output))


@pytest.mark.parametrize("index", [0, 2, 3])
def test_evaluate_2d_data(input_data_file_2d, output_data_file_2d, index):

    # Initialize model from spring-mass example data files:
    data_model = ModelFromData(input_data_file_2d, output_data_file_2d, 1.)

    input_data = np.genfromtxt(input_data_file_2d)
    output_data = np.genfromtxt(output_data_file_2d)

    # Model expects arrays as inputs/outputs
    model_output = data_model.evaluate(input_data[index])

    true_output = output_data[index]
    assert np.all(np.isclose(model_output, true_output))


@pytest.mark.parametrize("cost", [1, [1, 2], "one", np.zeros(7)])
def test_init_fails_on_invalid_cost(input_data_file, output_data_file, cost):

    with pytest.raises(Exception):
        ModelFromData(input_data_file, output_data_file, cost)


def test_evaluate_fails_on_invalid_input(input_data_file, output_data_file):

    data_model = ModelFromData(input_data_file, output_data_file, 1.)

    bogus_input = "five"

    with pytest.raises(TypeError):
        data_model.evaluate(bogus_input)

    bogus_input = [[1, 2], [3, 4]]

    with pytest.raises(ValueError):
        data_model.evaluate(bogus_input)


def test_fails_on_duplicate_input_data(input_data_file_with_duplicates,
                                       output_data_file):

    with pytest.raises(ValueError):

        data_model = ModelFromData(input_data_file_with_duplicates,
                                   output_data_file, 1.)
        data_model.evaluate([1, 2, 3, 4, 5])


def test_init_fails_on_incompatible_data(input_data_file, output_data_file_2d):

    with pytest.raises(ValueError):
        ModelFromData(input_data_file, output_data_file_2d, 1.)


def test_fail_on_nan_data(bad_data_file, input_data_file, output_data_file):

    with pytest.raises(ValueError):
        ModelFromData(bad_data_file, output_data_file, 1.)

    with pytest.raises(ValueError):
        ModelFromData(input_data_file, bad_data_file, 1.)


@pytest.mark.parametrize("rows_to_skip", [1, 2, 3])
def test_skip_rows(input_data_file_2d, output_data_file_2d, rows_to_skip):

    normal_model = ModelFromData(input_data_file_2d, output_data_file_2d, 1.)

    normal_row_count = normal_model._inputs.shape[0]

    skipped_row_model = ModelFromData(input_data_file_2d,
                                      output_data_file_2d,
                                      skip_header=rows_to_skip,
                                      cost=1.)

    skipped_row_count = skipped_row_model._inputs.shape[0]

    assert normal_row_count - rows_to_skip == skipped_row_count


@pytest.mark.parametrize("cost", [.01, .05, .1])
def test_evaluate_with_cost_delay(cost, input_data_file, output_data_file):

    model = ModelFromData(input_data_file, output_data_file, cost=cost,
                          wait_cost_duration=True)

    sample = model._inputs[0]

    start_time = timeit.default_timer()
    model.evaluate(sample)
    evaluation_time = timeit.default_timer() - start_time

    # Ensure evaluation time was close to specified cost.
    assert np.abs(evaluation_time - cost) < .01
