import pytest
import numpy as np
import os

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
def input_data_file_2d():
    input_data_file = os.path.join(data_path, "2D_test_data.csv")
    return input_data_file


@pytest.fixture
def output_data_file_2d():
    output_data_file = os.path.join(data_path, "2D_test_data_output.csv")
    return output_data_file


@pytest.mark.parametrize("index", [6, 368, 599, 612, 987])
def test_model_from_data_evaluate(input_data_file, output_data_file, index):

    # Initialize model from spring-mass example data files:
    data_model = ModelFromData(input_data_file, output_data_file)

    input_data = np.genfromtxt(input_data_file)
    output_data = np.genfromtxt(output_data_file)

    # Model expects arrays as inputs/outputs
    model_output = data_model.evaluate([input_data[index]])

    true_output = output_data[index]
    assert np.all(np.isclose(model_output, true_output))


@pytest.mark.parametrize("index", [0, 2, 3])
def test_model_from_data_evaluate_2d(input_data_file_2d, output_data_file_2d,
                                     index):

    # Initialize model from spring-mass example data files:
    data_model = ModelFromData(input_data_file_2d, output_data_file_2d)

    input_data = np.genfromtxt(input_data_file_2d)
    output_data = np.genfromtxt(output_data_file_2d)

    # Model expects arrays as inputs/outputs
    model_output = data_model.evaluate([input_data[index]])

    true_output = output_data[index]
    assert np.all(np.isclose(model_output, true_output))


def test_model_from_data_invalid_input(input_data_file, output_data_file):

    data_model = ModelFromData(input_data_file, output_data_file, 1.)

    bogus_input = -999.0

    with pytest.raises(ValueError):
        data_model.evaluate([bogus_input])

    bogus_input = [[1, 2], [3, 4]]

    with pytest.raises(ValueError):
        data_model.evaluate([bogus_input])


def test_init_fails_on_incompatible_data(input_data_file, output_data_file_2d):

    with pytest.raises(ValueError):
        ModelFromData(input_data_file, output_data_file_2d, 1.)
