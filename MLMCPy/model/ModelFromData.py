import numpy as np
import os

from Model import Model


class ModelFromData(Model):

    def __init__(self, input_filename, output_filename, cost):

        self.__check_parameters(output_filename, input_filename, cost)

        self.inputs = np.genfromtxt(input_filename)
        self.outputs = np.genfromtxt(output_filename)

        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("input and output data row counts must match.")

    def evaluate(self, input_data):

        output_data = self.outputs[np.argwhere(self.inputs == input_data)]

        if len(output_data) == 0:
            raise ValueError("Input data not found in model.")

        return output_data

    def __check_parameters(self, output_filename, input_filename, cost):

        if not isinstance(output_filename, str):
            raise TypeError("output_filename must be a string " +
                             "containing a valid file path.")

        if not os.path.isfile(output_filename):
            raise IOError("output_file is not a valid file.")

        if not isinstance(input_filename, str):
            raise ValueError("input_filename must be a string " +
                             "containing a valid file path.")

        if not os.path.isfile(input_filename):
            raise IOError("input_filename is not a valid file.")

        if not isinstance(cost, float):
            raise TypeError("cost must be a float.")

        if cost < 0.:
            raise ValueError("cost must be greater than 0.")
