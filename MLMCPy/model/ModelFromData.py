import numpy as np
import numbers
import os
import time

from Model import Model


class ModelFromData(Model):
    """
    Used to produce outputs from inputs based on data provided in text files.
    """
    def __init__(self, input_filename, output_filename, cost, delimiter=None,
                 skip_header=0, wait_cost_duration=False):
        """
        :param input_filename: Path to file containing input data.
        :type input_filename: string
        :param output_filename: Path to file containing output data.
        :type output_filename: string
        :param cost: The average cost of computing a sample output. If multiple
            quantities of interest are provided in the data, an ndarray of costs
            can specify cost for each quantity of interest.
        :type cost: float or ndarray
        :param delimiter: Delimiter used to separate data in data files, or
            size of each entry in the case of fixed width data.
        :type delimiter: string, int, list(int)
        :param wait_cost_duration: Whether to sleep for the duration of the
            cost in order to simulate real time model evaluation.
        :type wait_cost_duration: bool
        """
        self.__check_parameters(output_filename, input_filename, cost)

        self._inputs = np.genfromtxt(input_filename,
                                     delimiter=delimiter,
                                     skip_header=skip_header)

        self._outputs = np.genfromtxt(output_filename,
                                      delimiter=delimiter,
                                      skip_header=skip_header)

        self._wait_full_cost_duration_on_evaluate = wait_cost_duration

        # Data should not contain NaN.
        if np.isnan(self._inputs).any():
            raise ValueError("Input data file contains invalid (NaN) entries.")

        if np.isnan(self._outputs).any():
            raise ValueError("Output data file contains invalid (NaN) entries.")

        self.cost = cost

        if self._inputs.shape[0] != self._outputs.shape[0]:
            raise ValueError("input and output data must have same length.")

        # If multiple costs are provided, ensure the number of costs matches
        # the number of quantities of interest.
        if isinstance(cost, np.ndarray) and cost.size > 1:

            if cost.size != self._inputs.shape[-1]:
                raise ValueError("Size of array of costs must match number of" +
                                 " quantities of interest in sample data.")

    def evaluate(self, input_data):
        """
        Returns outputs corresponding to provided input_data. input_data will
        be searched for within the stored input data and the index of the match
        will be used to extract and return output data.

        :param input_data: Scalar or vector to be searched for in input data.
        :type input_data: ndarray
        :return: A ndarray of matched output_data.
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
        matches = np.equal(input_data, self._inputs)

        # If we are matching by row instead of element, we want indices
        # of matching rows.
        if len(matches.shape) > 1:
            matches = matches.all(-1)

        output_data = self._outputs[matches]

        if len(output_data) == 0:
            raise ValueError("Input data not found in model.")

        # Check for duplication in input_data based on number of matches.
        if np.sum(matches.astype(int)) > 1:
            raise ValueError("Input data contains duplicate information.")

        # Simulate cost if specified.
        if self._wait_full_cost_duration_on_evaluate:
            time.sleep(self.cost)

        return np.squeeze(output_data)

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

        if not (isinstance(cost, float) or isinstance(cost, np.ndarray)):
            raise TypeError("cost must be of type float or ndarray.")
