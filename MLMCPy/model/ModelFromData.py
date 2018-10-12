import numpy as np
import numbers
import os

from Model import Model


class ModelFromData(Model):
    """
    Used to acquire output data corresponding to given input data, all
    of which is acquired via data stored in files.
    """
    def __init__(self, input_filename, output_filename, cost, delimiter=None):
        """
        :param input_filename: path to file containing input data.
        :type string
        :param output_filename: path to file containing output data.
        :type string
        :param delimiter: delimiter used to separate data in data files.
        :type string, int (fixed width data), list of ints
        """
        self.__check_parameters(output_filename, input_filename, cost)

        self._inputs = np.genfromtxt(input_filename, delimiter=delimiter)
        self._outputs = np.genfromtxt(output_filename, delimiter=delimiter)

        self.cost = cost

        if self._inputs.shape[0] != self._outputs.shape[0]:
            raise ValueError("input and output data must have same length.")

    def evaluate(self, input_data):
        """
        Returns outputs corresponding to provided input_data. input_data will
        be searched for within the stored input data and indices of matches
        will be used to extract and return output data.

        :param input_data: scalar or vector to be searched for in input data.
        :return: ndarray of matched output_data.
        """
        # input_data should be an ndarray.
        # Automatically convert a list or numeric type into ndarray.
        if not isinstance(input_data, np.ndarray):
            if isinstance(input_data, list):
                input_data = np.array(input_data)
            elif isinstance(input_data, numbers.Number):
                input_data = np.array([input_data])
            else:
                raise TypeError("input_data must be a list or ndarray.")

        # input_data should not have more than one dimension.
        if len(input_data.shape) > 1:
            raise ValueError("input_data should be zero or one dimensional.")

        # Get outputs that matched the input data.
        output_data = self._outputs[input_data[:, ...] == self._inputs[:, ...]]

        if len(output_data) == 0:
            raise ValueError("Input data not found in model.")

        # Check for duplication in input_data based on shape of matched output.
        if isinstance(output_data[0], np.ndarray) and len(output_data[0]) > 1:
            raise ValueError("Input data contains duplicate information.")

        return output_data

    @staticmethod
    def __check_parameters(output_filename, input_filename, cost):

        if not isinstance(output_filename, str):
            raise TypeError("output_filename must be a string.")

        if not os.path.isfile(output_filename):
            raise IOError("output_file is not a valid file.")

        if not isinstance(input_filename, str):
            raise ValueError("input_filename must be a string.")

        if not os.path.isfile(input_filename):
            raise IOError("input_filename is not a valid file.")

        if not (isinstance(cost, float) or isinstance(cost, np.float)):
            raise TypeError("costs must be a list or ndarray.")
