import pytest
import numpy as np
import os

from MLMCPy.model import ModelFromData

#Access spring mass data:
my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../../examples/spring_mass/from_data/data"


@pytest.fixture
def input_data_file():
    input_data_file = os.path.join(data_path, "spring_mass_1D_inputs.txt")
    return input_data_file

@pytest.fixture
def output_data_file():
    output_data_file = os.path.join(data_path, "spring_mass_1D_outputs_0.1.txt")
    return output_data_file


@pytest.mark.parametrize("index", [(6), (368), (599), (612), (987)])

def test_model_from_data_evaluate(input_data_file, output_data_file, index):

    #Initialize model from spring-mass example data files:
    data_model = ModelFromData(input_data_file, output_data_file)

    input_data = np.genfromtxt(input_data_file)
    output_data = np.genfromtxt(output_data_file)

    #Model expects arrays as inputs/outputs
    model_output = data_model.evaluate([input_data[index]])

    true_output = output_data[index]
    assert np.isclose(model_output[0], true_output)


def test_model_from_data_invalid_input(input_data_file, output_data_file):

    data_model = ModelFromData(input_data_file, output_data_file)

    bogus_input = -999.0

    with pytest.raises(ValueError):
        output = data_model.evaluate([bogus_input])


