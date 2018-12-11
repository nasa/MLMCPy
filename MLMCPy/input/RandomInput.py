import numpy as np

from Input import Input


class RandomInput(Input):
    """
    Used to draw samples from a specified distribution . Any distribution
    function provided must accept a "size" parameter that determines the
    sample size.
    """
    def __init__(self, distribution_function, random_seed=None,
                 **distribution_function_args):
        """
        :param distribution_function: Returns a sample of a distribution
            with the sample sized determined by a "size" parameter. Typically,
            a numpy function such as numpy.random.uniform() is used.
        :type distribution_function: function
        :param distribution_function_args: Any arguments required by the
            distribution function, with the exception of "size", which will be
            provided to the function when draw_samples is called.
        """

        if not callable(distribution_function):
            raise TypeError('distribution_function must be a function.')

        self._distribution = distribution_function
        self._args = distribution_function_args

        self._num_cpus = 1
        self._cpu_rank = 0

        self._random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def draw_samples(self, num_samples):
        """
        Returns num_samples samples from a distribution in the form of a
        numpy array.

        :param num_samples: Total number of samples to take across all CPUs.
        :type num_samples: int
        :return: A ndarray of distribution sample. If multiple CPUs are
            available, will return a subset of sample determined by number
            of CPUs.
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

        # Take subsample of data in MPI environments.
        if self._num_cpus > 1 and self._mpi_slice:

            # Determine subsample sizes for all cpus.
            subsample_size = samples.shape[0] // self._num_cpus
            remainder = samples.shape[0] - subsample_size * self._num_cpus
            subsample_sizes = np.ones(self._num_cpus+1).astype(int) * \
                subsample_size

            subsample_sizes[:remainder+1] += 1
            subsample_sizes[0] = 0

            # Determine starting index of subsample.
            subsample_index = int(np.sum(subsample_sizes[:self._cpu_rank+1]))

            # Take subsample.
            samples = samples[subsample_index:
                              subsample_index +
                              subsample_sizes[self._cpu_rank+1],
                              :]

        return samples

    def reset_sampling(self):

        if self._random_seed is not None:
            np.random.seed(self._random_seed)
