import numpy as np
import timeit
import sys

from MLMCPy.input import Input
from MLMCPy.model import Model


class MLMCSimulator:
    """
    Description
    """
    def __init__(self, data, models):
        """
        Requires a data object that provides input samples and a list of models
        of increasing fidelity.

        :param data: Provides a data sampling function.
        :type: Input
        :param models: Each model Produces outputs from sample data input.
        :type: list of Model objects
        """
        self.__check_init_parameters(data, models)

        self._data = data
        self._models = models
        self._num_levels = len(self._models)

        # Sample size to be taken at each level.
        self._sample_sizes = np.zeros(self._num_levels, dtype=np.int)

        # Desired level of precision.
        self._epsilons = np.zeros(1)

        # Number of quantities of interest.
        self._data_width = 1

        # Enabled diagnostic text output.
        self._verbose = False

    def simulate(self, epsilon, initial_sample_size=1000, verbose=False):
        """
        Perform MLMC simulation.
        Computes number of samples per level before running simulations
        to determine estimate.

        :param epsilon: Desired accuracy to be achieved.
        :type: float, list of floats, or ndarray.
        :param initial_sample_size: number of samples to use during
               configuration phase.
        :type: int
        :returns (value, list of sample count at each level, error)
        :param verbose: Whether to print useful diagnostic information.
        :type: bool
        """
        self._verbose = verbose

        self.__check_simulate_parameters(initial_sample_size)

        # Compute optimal sample sizes for each level, as well as alpha value.
        self._setup_simulation(epsilon, initial_sample_size)

        # Run models and return estimate.
        return self._run_simulation()

    def _setup_simulation(self, epsilon, initial_sample_size):
        """
        Computes variance and cost at each level in order to estimate optimal
        number of samples at each level.
        :param epsilon:
        :param initial_sample_size:
        :return:
        """
        # Run models with initial sample sizes to compute costs, outputs.
        costs, outputs = self._compute_costs_and_outputs(initial_sample_size)

        # Compute variances.
        variances = self._compute_variances(outputs)

        # Present kurtosis analysis if verbose enabled.
        if self._verbose:
            self._compute_kurtosis(outputs)

        # Epsilon should be array that matches output width.
        self._epsilons = self._process_epsilon(epsilon)

        # Compute optimal sample size at each level.
        self._compute_optimal_sample_sizes(variances, costs)

    def _run_simulation(self):
        """
        Compute estimate by extracting number of samples from each level
        determined in the setup phase.
        :return:
        """
        # Compute sample outputs.
        output_means = np.zeros((self._num_levels, self._data_width))
        output_variances = np.zeros((self._num_levels, self._data_width))
        for level in range(self._num_levels):

            if self._verbose:
                step = str(level+1) + " of " + str(self._num_levels)
                print "\nRunning simulation level " + step

            # Sample input data.
            sample_size = self._sample_sizes[level]
            samples = self._data.draw_samples(sample_size)
            self._data.reset_sampling()

            # Produce the model output.
            output = np.zeros(sample_size)
            for i, sample in enumerate(samples):

                if self._verbose:
                    progress = str((float(i) / sample_size) * 100)[:5]
                    sys.stdout.write("\r" + progress + "%")

                output[i] = self._models[level].evaluate(sample)

            # Compute mean of outputs.
            output_means[level] = np.mean(output)
            output_variances[level] = np.var(output)

        # Compute sum of variances across levels for each quantity of interest
        # and compare results to corresponding epsilon values.
        if self._verbose:

            for i in range(self._data_width):

                variance_sum = np.sum(output_variances[:, i])
                epsilon_squared = self._epsilons[i]
                passed = variance_sum < epsilon_squared

                print 'QOI #%s: var: %s, eps^2: %s, success: %s' % \
                      (i, variance_sum, epsilon_squared, passed)

        # Evaluate the multilevel estimator.
        p = np.sum(output_means / self._sample_sizes)

        return p, self._sample_sizes, output_variances

    def _compute_optimal_sample_sizes(self, variances, costs):

        if self._verbose:
            sys.stdout.write("Computing optimal sample sizes: ")

        # Compute mu.
        mu = np.zeros(self._epsilons.shape)
        eps_n2 = np.power(self._epsilons, -2)

        for i in range(len(self._epsilons)):
            mu[i] = eps_n2[i] * np.sum(np.sqrt(variances[:, i] * costs))

        # Compute sample sizes.
        for level in range(self._num_levels):

            nl = mu * np.sqrt(variances[level] / costs[level])
            self._sample_sizes[level] = np.max(np.ceil(nl))

        if self._verbose:
            print np.array2string(self._sample_sizes)

    def _compute_costs_and_outputs(self, initial_sample_size):

        if self._verbose:
            sys.stdout.write("Determining costs: ")

        costs = np.ones(self._num_levels)

        # If the models have cost precomputed, use them to compute interlayer
        # costs.
        costs_precomputed = False
        if hasattr(self._models[0], 'cost'):

            costs_precomputed = True
            costs[0] = self._models[0].cost
            for i in range(1, len(self._models)):

                costs[i] = self._models[i].cost + self._models[i-1].cost

        # Compute cost between each level.
        samples = self._data.draw_samples(initial_sample_size)
        self._data.reset_sampling()

        self._data_width = samples[0].shape[0]
        outputs = np.zeros((self._num_levels,
                            initial_sample_size,
                            self._data_width))

        # Process samples at each level.
        compute_times = []
        for level in range(self._num_levels):

            start_time = timeit.default_timer()
            for i, sample in enumerate(samples):
                outputs[level, i] = self._models[level].evaluate(sample)

            compute_times.append(timeit.default_timer() - start_time)

        # Compute costs based on compute time differences between levels.
        if not costs_precomputed:
            for i in range(1, len(compute_times)):
                costs[i] = compute_times[i] - compute_times[i - 1]

        if self._verbose:
            print np.array2string(costs)

        return costs, outputs

    def _compute_variances(self, outputs):

        if self._verbose:
            print "Determining variances: "

        variances = np.zeros((self._num_levels, self._data_width))

        variances[0] = np.var(outputs[0])
        for level in range(1, self._num_levels):
            for i in range(self._data_width):
                variances[level, i] = np.var(outputs[level, :, i] -
                                             outputs[level - 1, :, i])

        # Fix zero variances.
        variances[np.where(variances == 0)] = 1.e-30

        if self._verbose:
            print np.array2string(variances)

        return variances

    @staticmethod
    def _compute_kurtosis(output):

        # TODO: Finish this.
        # sum_squares = np.sum(np.square(output)) / sample_size
        # sum_cubes = np.sum(np.power(output, 3)) / sample_size
        # sum_quads = np.sum(np.power(output, 4)) / sample_size

        # kurt = (sum_quads - 4 * sum_cubes * sums[level] + \
        #         6 * sum_cubes * sums[level] - \
        #         3 * sums[level] * sums[level] ** 3) /
        pass

    def _process_epsilon(self, epsilon):
        """
        Produce an ndarray of epsilon values from float or list of epsilons.
        :param epsilon: float, list of floats, or ndarray.
        :return: ndarray of epsilons of size (self.num_levels).
        """
        if isinstance(epsilon, list):
            epsilon = np.array(epsilon)

        if isinstance(epsilon, float):
            epsilon = np.ones(self._data_width) * epsilon

        if not isinstance(epsilon, np.ndarray):
            raise TypeError("Epsilon must be a float, list of floats, " +
                            "or an ndarray.")

        if np.any(epsilon <= 0.):
            raise ValueError("Epsilon values must be greater than 0.")

        if len(epsilon) != self._data_width:
            raise ValueError("Number of epsilons must match number of levels.")

        return epsilon

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
    def __check_simulate_parameters(starting_sample_size):

        if not isinstance(starting_sample_size, int):
            raise TypeError("starting_sample_size must be an integer.")

        if starting_sample_size < 1:
            raise ValueError("starting_sample_size must be greater than zero.")
