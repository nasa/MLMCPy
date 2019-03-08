import pytest
import numpy as np
import timeit
import imp
import os
import sys
import warnings

# Needed when running mpiexec. Be sure to run from tests directory.
if 'PYTHONPATH' not in os.environ:

    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.model import ModelFromData
from MLMCPy.input import RandomInput
from MLMCPy.input import InputFromData
from MLMCPy.input import Input
from tests.testing_scripts.spring_mass import SpringMassModel

# Create list of paths for each data file.
# Used to parametrize tests.
my_path = os.path.dirname(os.path.abspath(__file__))
data_path = my_path + "/../testing_data"

@pytest.fixture
def random_input():
    """
    Creates a RandomInput object that produces samples from a
    uniform distribution.
    """
    return RandomInput()


@pytest.fixture
def data_input():
    """
    Creates an InputFromData object that produces samples from a file
    containing spring mass input data.
    """
    return InputFromData(os.path.join(data_path, "spring_mass_1D_inputs.txt"),
                         shuffle_data=False)


@pytest.fixture
def data_input_2d():
    """
    Creates an InputFromData object that produces samples from a file
    containing two dimensional data.
    """
    return InputFromData(os.path.join(data_path, "2D_test_data.csv"),
                         shuffle_data=False)


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
def dummy_arange_random_input():
    """
    Creates a random input object that just returns np.arange(size) 
    for testing
    """

    def get_arange(size):
        return np.arange(size)

    return RandomInput(distribution_function=get_arange)


@pytest.fixture
def spring_models():
    """
    Creates a list of three SpringMassModel objects of increasing fidelity.
    """
    model_level1 = SpringMassModel(mass=1.5, time_step=1.0, cost=1.0)
    model_level2 = SpringMassModel(mass=1.5, time_step=0.1, cost=10.0)
    model_level3 = SpringMassModel(mass=1.5, time_step=0.01, cost=100.0)

    return [model_level1, model_level2, model_level3]


@pytest.fixture
def models_from_data():
    """
    Creates a list of three ModelFromData objects of increasing fidelity.
    :return:
    """
    input_filepath = os.path.join(data_path, "spring_mass_1D_inputs.txt")
    output1_filepath = os.path.join(data_path, "spring_mass_1D_outputs_1.0.txt")
    output2_filepath = os.path.join(data_path, "spring_mass_1D_outputs_0.1.txt")
    output3_filepath = os.path.join(data_path,
                                    "spring_mass_1D_outputs_0.01.txt")

    model1 = ModelFromData(input_filepath, output1_filepath, 1.)
    model2 = ModelFromData(input_filepath, output2_filepath, 4.)
    model3 = ModelFromData(input_filepath, output3_filepath, 16.)

    return [model1, model2, model3]


@pytest.fixture
def models_from_2d_data():
    """
    Creates a list of three ModelFromData objects with a small amount of
    two dimensional data.
    """
    input_filepath = os.path.join(data_path, "2D_test_data.csv")
    output1_filepath = os.path.join(data_path, "2D_test_data_output.csv")
    output2_filepath = os.path.join(data_path, "2D_test_data_output.csv")
    output3_filepath = os.path.join(data_path, "2D_test_data_output.csv")

    model1 = ModelFromData(input_filepath, output1_filepath, 1.)
    model2 = ModelFromData(input_filepath, output2_filepath, 4.)
    model3 = ModelFromData(input_filepath, output3_filepath, 16.)

    return [model1, model2, model3]


@pytest.fixture
def filename_2d_5_column_data():
    """
    Creates a string containing the path to a file with a large number of rows
    of data with five columns.
    """
    return os.path.join(data_path, "2D_test_data_long.csv")


@pytest.fixture
def filename_2d_3_column_data():
    """
    Creates a string containing the path to a file with a large number of rows
    of data with three columns.
    """
    return os.path.join(data_path, "2D_test_data_output_3_col.csv")


@pytest.fixture
def comm():
    """
    Creates a MPI.COMM_WORLD object for working with multi-process information.
    """
    try:

        imp.find_module('mpi4py')

        from mpi4py import MPI
        return MPI.COMM_WORLD

    except ImportError:

        class FakeCOMM:

            def __init__(self):
                self.size = 1
                self.rank = 0

            @staticmethod
            def allgather(thing):

                return np.array([thing])

        return FakeCOMM()


@pytest.fixture
def spring_mlmc_simulator(beta_distribution_input, spring_models):
    np.random.seed(1)
    sim = MLMCSimulator(models=spring_models,
                        random_input=beta_distribution_input)
    
    return sim

@pytest.fixture
def dummy_arange_simulator(dummy_arange_random_input, spring_models):
    """
    Dummy simulator to test get model inputs modular function where the 
    random input object simply returns numpy.arange for testing
    """
    sim = MLMCSimulator(models=spring_models,
                        random_input=dummy_arange_random_input)
    return sim
    