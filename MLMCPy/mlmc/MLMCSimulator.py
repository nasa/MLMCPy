import numpy as np
import timeit

from MLMCPy.input import Input
from MLMCPy.model import Model


class MLMCSimulator:

    def __init__(self, data, models):
        """
        Take a data object that provides input samples and a list of models
        of increasing fidelity.

        :param data: Provides a data sampling function.
        :type: Input
        :param models: Each model Produces outputs from sample data input.
        :type: list of Model objects
        """
        self.__check_init_parameters(data, models)

        self.data = data
        self.models = models

        self.num_levels = len(self.models)
        self.sample_sizes = np.zeros(self.num_levels, dtype=np.int)

        self.verbose = False

    def simulate(self, epsilon, initial_sample_size=1000, costs=None,
                 verbose=False):

        """
        Perform MLMC simulation.
        Computes number of samples per level before running simulations
        to determine estimate.

        :param epsilon: Desired accuracy to be achieved.
        :type: float, list of floats, or ndarray.
        :param initial_sample_size: number of samples to use during
               configuration phase.
        :type: int
        :param costs:
        :type:
        :returns (value, list of sample count at each level, error)
        :param verbose: Whether to print useful diagnostic information.
        :type: bool
        """
        self.verbose = verbose

        self.__check_simulate_parameters(epsilon,
                                         initial_sample_size,
                                         costs)

        # Compute optimal sample sizes for each level, as well as alpha value.
        self.__setup_simulation(epsilon, initial_sample_size)

        # Run models and return estimate.
        return self.__run_simulation()

    def __run_simulation(self):
        """
        Compute estimate by extracting number of samples from each level
        determined in the setup phase.
        :return:
        """
        output_sums = np.zeros(self.num_levels)
        num_samples_taken = np.zeros(self.num_levels)

        # Update sample outputs.
        for level in range(0, self.num_levels):

            # Sample input data.
            sample = self.data.draw_samples(self.sample_sizes[level])
            self.data.reset_sampling()
            num_samples_taken[level] = len(sample)

            # If the requested number of samples is not reached, report to the
            # user if verbose is enabled.
            if self.verbose and len(sample) < self.sample_sizes[level]:
                print "WARNING: Only %s samples were provided at level %s " + \
                      "instead of the requested %s." % \
                      (len(sample), self.sample_sizes[level], level)

            # Compute the model output.
            output = self.models[level].evaluate(sample)

            # Compute sum of outputs.
            output_sums[level] = np.sum(output)

        # Compute error.
        alpha = max(0, self.precomputed_alpha)
        means = np.abs(output_sums / num_samples_taken)

        # TODO: Verify this with Dr Warner (Page 25).
        # Recompute alpha if original value below zero.
        # Use linear regression to determine new value.
        if self.precomputed_alpha <= 0:

            A = np.ones((2, 2))
            A[:, 0] = range(1, 3)

            x = np.linalg.solve(A, np.log2(means[1:]))

            alpha = max(0.5, -x[0])

        error = means[-1] / (2.0 ** alpha - 1.0)

        # Evaluate the multilevel estimator.
        p = sum(output_sums / self.num_levels)

        return p, num_samples_taken, error

    def __setup_simulation(self, epsilon, initial_sample_size, costs=None):
        """
        Computes variance and cost at each level in order to estimate optimal
        number of samples at each level.
        :param epsilon:
        :param initial_sample_size:
        :return:
        """
        if costs is None:  # TODO: user provided costs
            costs = np.zeros(self.num_levels)

        variances = np.zeros(self.num_levels)
        outputs = np.zeros((self.num_levels, initial_sample_size))
        sums = np.zeros(self.num_levels)

        # Compute cost and variance at each level by evaluating model once
        # for each level.
        for level in range(self.num_levels):

            sample = self.data.draw_samples(initial_sample_size)
            self.data.reset_sampling()

            if sample is None:
                raise ValueError("There were not enough samples to " +
                                 "complete the setup phase! Reduce " +
                                 "initial_sample_size or provide a larger " +
                                 "data set!")

            # Get cost by timing model evaluation of sample at this level.
            start_time = timeit.default_timer()
            outputs[level] = self.models[level].evaluate(sample)
            costs[level] = timeit.default_timer() - start_time

            sums[level] = np.sum(outputs[level]) / len(sample)

            # Present kurtosis analysis if verbose enabled.
            if self.verbose and level > 0:
                self.__compute_kurtosis(outputs[level], len(sample))

            # Compute variance.
            if level == 0:
                variances[level] = np.var(outputs[level])
            else:
                variances[level] = np.var(outputs[level] - outputs[level - 1])

        # Fix zero variances.
        variances[np.where(variances == 0)] = 1.e-30

        # Compute optimal sample size at each level.
        self.__compute_optimal_sample_sizes(epsilon, variances, costs)

        self.__compute_alpha(outputs)

    def __compute_optimal_sample_sizes(self, epsilon, variances, costs):

        sum_sqrt_var_times_costs = np.sum(np.sqrt(variances * costs))
        for level in range(self.num_levels):
            self.sample_sizes[level] = 2. * epsilon ** -2 * \
                                    np.sqrt(variances[level] / costs[level]) /\
                                    sum_sqrt_var_times_costs

    def __compute_alpha(self, results):

        L1 = int(np.ceil(0.4 * self.num_levels))
        L2 = self.num_levels + 1

        polyfit_x = range(L1 + 1, L2 + 1)
        polyfit_y = np.log2(np.abs(results[0, L1: L2]))

        poly_a = np.polyfit(polyfit_x, polyfit_y, 1)

        self.precomputed_alpha = -poly_a[0]

    @staticmethod
    def __compute_kurtosis(output, sample_size):

        # TODO: Finish this.
        sum_squares = np.sum(np.square(output)) / sample_size
        sum_cubes = np.sum(np.power(output, 3)) / sample_size
        sum_quads = np.sum(np.power(output, 4)) / sample_size

        # kurt = (sum_quads - 4 * sum_cubes * sums[level] + \
        #         6 * sum_cubes * sums[level] - \
        #         3 * sums[level] * sums[level] ** 3) /

    @staticmethod
    def __check_init_parameters(data, models):

        if not isinstance(data, Input):
            TypeError("data must inherit from Input class.")

        if not isinstance(models, list):
            TypeError("models must be a list of models.")

        for model in models:
            if not isinstance(model, Model):
                TypeError("models must be a list of models.")

    @staticmethod
    def __check_simulate_parameters(epsilon, starting_sample_size, costs):

        if not (isinstance(epsilon, float) or isinstance(epsilon, np.float)):
            raise TypeError("epsilon must be a float.")

        if epsilon < 0.:
            raise ValueError("epsilon must be greater than zero.")

        if not isinstance(starting_sample_size, int):
            raise TypeError("starting_sample_size must be an integer.")

        if starting_sample_size < 1:
            raise ValueError("starting_sample_size must be greater than zero.")

        if costs is not None:

            if not (isinstance(costs, list) or isinstance(costs, np.ndarray)):
                raise TypeError("costs must be a list or ndarray.")

            if not np.all(np.array(costs) > 0):
                raise ValueError("costs must all be greater than zero.")
