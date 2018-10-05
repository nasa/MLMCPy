import numpy as np
import os

from Input import Input


class InputFromData(Input):

    def __init__(self, input_filename, delimiter=" ", shuffle_data=True):
        """
        Used to draw random samples from a data file.

        :param input_filename path of file containing data to be sampled.
        :type string
        :param shuffle_data: whether or not to randomly shuffle data during
                             initialization
        :type shuffle_data: Bool
        """
        if not os.path.isfile(input_filename):
            raise IOError("input_filename must refer to a file.")

        self._data = np.genfromtxt(input_filename, delimiter=delimiter)
        if shuffle_data:
            np.random.shuffle(self._data)
        self._index = 0

    def draw_samples(self, num_samples):
        """
        Returns an array of samples from the previously loaded file data.
        :param num_samples: Number of samples to be returned.
        :type int
        :return: 2d ndarray of samples, each row being one sample.
                 For one dimensional input data, this will have
                 shape (num_samples, 1)
        """

        if not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer.")

        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        # Otherwise return the requested sample and increment the index.
        sample = self._data[self._index: self._index + num_samples]
        self._index += num_samples

        # Output should be shape (num_samples, sample_size), so reshape
        # one dimensional data to a 2d array with one column.
        sample = sample.reshape(sample.shape[0], -1)

        sample_size = sample.shape[0]
        if num_samples > sample_size:

            error_message = "Only %s of the %s requested samples are " + \
                            "available.\nMore sample data is required."
            raise ValueError(error_message % (sample_size, num_samples))

        return sample

    def reset_sampling(self):
        """
        Used to shuffle and restart sampling from beginning of data set.
        """
        np.random.shuffle(self._data)
        self._index = 0
