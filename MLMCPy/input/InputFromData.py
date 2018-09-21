import numpy as np
import os


class InputFromData:

    def __init__(self, input_filename, delimiter=" "):
        """
        Used to draw random samples from a data file.

        :param input_filename path of file containing data to be sampled.
        :type string
        """
        if not os.path.isfile(input_filename):
            raise IOError("input_filename must refer to a file.")

        self.data = np.genfromtxt(input_filename, delimiter=delimiter)
        np.random.shuffle(self.data)
        self.index = 0

    def draw_samples(self, num_samples):
        """
        Returns an array of samples from the previously loaded file data.
        :param num_samples: Number of samples to be returned.
        :type int
        :return: ndarray of samples
        """

        if not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer.")

        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        # If we have already returned all samples, return None.
        if self.index > self.data.shape[0]:
            return None

        # Otherwise return the requested sample and increment the index.
        sample = self.data[self.index: self.index+num_samples]
        self.index += num_samples

        return sample
