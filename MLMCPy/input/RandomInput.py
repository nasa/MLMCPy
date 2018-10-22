import numpy as np
import imp

from Input import Input


class RandomInput(Input):
    """
    Used to draw samples from a specified distribution, with a uniform
    distribution as the default. Any distribution function provided
    must accept a "size" parameter that determines the sample size.

    :param distribution_function returns a sample of a distribution
        with the sample sized determined by a "size" parameter.
    :type function
    :param distribution_function_args any arguments required by the distribution
        function, with the exception of "size", which will be provided to the
        function when draw_samples is called.
    """
    def __init__(self, distribution_function=np.random.uniform,
                 **distribution_function_args):

        if not callable(distribution_function):
            raise TypeError('distribution_function must be a function.')

        self._distribution = distribution_function
        self._args = distribution_function_args

        # Set random seed based on cpu rank.
        self.__detect_parallelization()

    def draw_samples(self, num_samples):
        """
        Returns num_samples samples from a distribution in the form of a
        numpy array.

        :param num_samples: Size of array to return.
        :type int
        :return: ndarray of distribution sample.
        """
        if not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer.")

        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        self._args['size'] = num_samples
        sample = self._distribution(**self._args)

        # Output should be shape (num_samples, sample_size), so reshape
        # one dimensional data to a 2d array with one column.
        samples = sample.reshape(sample.shape[0], -1)

        return samples

    def reset_sampling(self):
        pass

    @staticmethod
    def __detect_parallelization():

        try:
            imp.find_module('mpi4py')

            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            cpu_rank = comm.rank

        except ImportError:
            cpu_rank = 0

        finally:
            np.random.seed(cpu_rank)
