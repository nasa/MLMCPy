import pytest
import os
import numpy as np

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

