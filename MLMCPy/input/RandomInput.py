import numpy as np


class RandomInput:
    """
    Used to draw samples from a specified distribution, with a uniform
    distribution as the default. Any distribution function provided
    must accept a "size" parameter that determines the sample size.

    :param distribution_function function that returns a sample of a distribution
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

        self.distribution = distribution_function
        self.args = distribution_function_args

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

        # Pass in num_samples as size argument to distribution function.
        self.args['size'] = num_samples
        return self.distribution(**self.args)
