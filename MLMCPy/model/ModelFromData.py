import numpy as np
import os

from Model import Model


class ModelFromData(Model):

    def __init__(self, input_filename, output_filename, delimiter=" "):
        """
        Used to acquire output data corresponding to given input data.

        :param input_filename: path to file containing input data.
        :type string
        :param output_filename: path to file containing output data.
        :type string
        :param delimiter: delimiter used to separate data in data files.
        :type string, int (fixed width data), list of ints
        """
        self.__check_parameters(output_filename, input_filename)

        self.inputs = np.genfromtxt(input_filename, delimiter=delimiter)
        self.outputs = np.genfromtxt(output_filename, delimiter=delimiter)

        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("input and output data must have same length.")

    def evaluate(self, input_data):
        """
        Returns outputs corresponding to provided input_data. input_data will
        be searched for within the stored input data and indices of matches
        will be used to extract and return output data.

        :param input_data: scalar or vector to be searched for in input data.
        :return: ndarray of matched output_data.
        """
        # Get matching members or rows, depending on dimensionality.
        if len(self.inputs.shape) == 1:
            matches = self.inputs == input_data
        else:
            matches = np.all(self.inputs == input_data, axis=1)

        # Get outputs that matched the input data.
        output_data = self.outputs[np.argwhere(matches)]

        if len(output_data) == 0:
            raise ValueError("Input data not found in model.")

        return np.squeeze(output_data)

    def __check_parameters(self, output_filename, input_filename):

        if not isinstance(output_filename, str):
            raise TypeError("output_filename must be a string.")

        if not os.path.isfile(output_filename):
            raise IOError("output_file is not a valid file.")

        if not isinstance(input_filename, str):
            raise ValueError("input_filename must be a string.")

        if not os.path.isfile(input_filename):
            raise IOError("input_filename is not a valid file.")
