import numpy as np
import timeit
import sys

from MLMCPy.input import Input
from MLMCPy.model import Model


class MLMCSimulator:
    """
    Computes an estimate based on the MultiLevel Monte Carlo algorithm.
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

        # Used to compute sample sizes based on a fixed cost.
        self._target_cost = None

        # Sample size used in setup.
        self._initial_sample_size = 0

        # Desired level of precision.
        self._epsilons = np.zeros(1)

        # Number of elements in model output.
        self._output_size = 1

        # Enabled diagnostic text output.
        self._verbose = False

    def simulate(self, epsilon, initial_sample_size=1000, target_cost=None,
                 verbose=False):
        """
        Perform MLMC simulation.
        Computes number of samples per level before running simulations
        to determine estimate.

        :param epsilon: Desired accuracy to be achieved for each quantity of
            interest.
        :type: float, list of floats, or ndarray.
        :param initial_sample_size: Sample size used when computing sample sizes
            for each level in simulation.
        :type: int
        :param target_cost: Target cost to run simulation.
        :type: float or int
        :param verbose: Whether to print useful diagnostic information.
        :type: bool
        :returns (value, list of sample count at each level, error)
        """
        self._verbose = verbose

        self.__check_simulate_parameters(initial_sample_size, target_cost)
        self._target_cost = target_cost

        # If only one model was provided, run standard monte carlo.
        if self._num_levels == 1:
            return self._run_monte_carlo(initial_sample_size)

        # Compute optimal sample sizes for each level, as well as alpha value.
        self._setup_simulation(epsilon, initial_sample_size)

        # Run models and return estimate.
        return self._run_simulation()

    def _setup_simulation(self, epsilon, initial_sample_size):
        """
        Computes variance and cost at each level in order to estimate optimal
        number of samples at each level.

        :param epsilon: Epsilon values for each quantity of interest.
        :param initial_sample_size: Sample size used when computing sample sizes
            for each level in simulation.
        """
        self._initial_sample_size = initial_sample_size

        # Run models with initial sample sizes to compute costs, outputs.
        costs, variances = self._compute_costs_and_variances()

        # Epsilon should be array that matches output width.
        self._epsilons = self._process_epsilon(epsilon)

        # Compute optimal sample size at each level.
        self._compute_optimal_sample_sizes(costs, variances)

    def _run_simulation(self):
        """
        Compute estimate by extracting number of samples from each level
        determined in the setup phase.

        :return: tuple containing three ndarrays:
            estimates: Estimates for each quantity of interest
            sample_sizes: The sample sizes used at each level.
            variances: Variance of model outputs at each level.
        """
        # Restart sampling from beginning.
        self._data.reset_sampling()

        output_means = np.zeros((self._num_levels, self._output_size))
        output_variances = np.zeros((self._num_levels, self._output_size))

        # Compute sample outputs.
        for level in range(self._num_levels):

            samples = self._data.draw_samples(self._sample_sizes[level])
            output = np.zeros((self._sample_sizes[level], self._output_size))

            for i, sample in enumerate(samples):
                output[i] = self._evaluate_sample(i, sample, level)

            output_means[level] = np.mean(output, axis=0)
            output_variances[level] = np.var(output, axis=0)

        estimates, variances = self._compute_summary_data(output_means,
                                                          output_variances)

        return estimates, self._sample_sizes, variances

    def _compute_summary_data(self, output_means, output_variances):
        """
        Compute means and variances of output data.

        :param output_means: ndarray of model output means.
        :param output_variances: ndarray of model output variances.
        :return:
        """
        # Compute total variance for each quantity of interest.
        sample_sizes_2d = self._sample_sizes[:, np.newaxis]
        variances = np.sum(output_variances / sample_sizes_2d, axis=0)

        # Compare variance for each quantity of interest to epsilon values.
        if self._verbose:

            print

            for i, variance in enumerate(variances):

                epsilon_squared = self._epsilons[i] ** 2
                passed = variance < epsilon_squared

                print 'QOI #%s: variance: %s, epsilon^2: %s, success: %s' % \
                      (i, variance, epsilon_squared, passed)

        # Evaluate the multilevel estimator.
        means = np.mean(output_means, axis=0)

        return means, variances

    def _evaluate_sample(self, i, sample, level):
        """
        Evaluate output of an input sample, either by running the model or
        retrieving the output from the cache.

        :param i: sample index
        :param sample: sample value
        :param level: model level
        :return: result of evaluation
        """

        if self._verbose:
            progress = str((float(i) / self._sample_sizes[level]) * 100)[:5]
            sys.stdout.write("\rLevel %s progress: %s%%" % level, progress)

        # If we have the output for this sample cached, use it.
        # Otherwise, compute the output via the model.

        # Absolute index of current sample.
        sample_index = np.sum(self._sample_sizes[:level]) + i

        # Level in cache that a sample with above index would be at.
        # This must match the current level.
        cached_level = sample_index // self._initial_sample_size

        # Index within cached level for sample output.
        cached_index = sample_index - level * self._initial_sample_size

        # Level and index within cache must be correct for the
        # appropriate cached value to be found.
        can_use_cache = cached_index < self._initial_sample_size and \
                        cached_level == level

        if can_use_cache:
            return self._cache[level][cached_index]
        else:
            return self._models[level].evaluate(sample)

    def _compute_costs_and_variances(self):
        """
        Compute costs and variances across levels.

        :return: tuple of ndarrays:
            1d ndarray of costs
            2d ndarray of variances
        """
        if self._verbose:
            sys.stdout.write("Determining costs and variances: ")

        self._determine_output_size()

        # Cache model outputs computed here so that they can be reused
        # in the simulation.
        self._cache = np.zeros((self._num_levels,
                                self._initial_sample_size,
                                self._output_size))

        # Process samples in model. Gather compute times for each level.
        # Variance is computed from difference between outputs of adjacent
        # layers evaluated from the same samples.
        compute_times = np.zeros(self._num_levels)
        variances = np.zeros((self._num_levels, self._output_size))

        for level in range(self._num_levels):

            input_samples = self._data.draw_samples(self._initial_sample_size)
            sublevel_outputs = np.zeros((self._initial_sample_size,
                                        self._output_size))

            start_time = timeit.default_timer()
            for i, sample in enumerate(input_samples):

                self._cache[level, i] = self._models[level].evaluate(sample)

                if level > 0:
                    sublevel_outputs[i] = self._models[level-1].evaluate(sample)

            compute_times[level] = timeit.default_timer() - start_time

            variances[level] = np.var(self._cache[level] - sublevel_outputs)

        return self._compute_costs(compute_times), variances

    def _compute_costs(self, compute_times):
        """
        Set costs for each level, either from precomputed values from each
        model or based on computation times provided by compute_times.

        :param compute_times: ndarray of computation times for computing
        model at each layer and preceding layer.
        """
        costs = np.ones(self._num_levels)

        # If the models have costs precomputed, use them to compute costs
        # between each level.
        costs_precomputed = False
        if hasattr(self._models[0], 'cost'):

            costs_precomputed = True
            for i, model in enumerate(self._models):
                costs[i] = model.cost

            # Costs at level > 0 should be summed with previous level.
            costs[1:] = costs[1:] + costs[:-1]

        # Compute costs based on compute time differences between levels.
        if not costs_precomputed:
            costs[0] = compute_times[0]
            costs[1:] = compute_times[1:] + compute_times[0:-1]

        if self._verbose:
            print np.array2string(costs)

        return costs

    def _determine_output_size(self):
        """
        Runs model on a small test sample to determine shape of output.
        """
        test_sample = self._data.draw_samples(1)[0]
        self._data.reset_sampling()

        test_output = self._models[0].evaluate(test_sample)
        self._output_size = test_output.shape[0]

    def _compute_optimal_sample_sizes(self, costs, variances):
        """
        Compute the sample size for each level to be used in simulation.

        :param variances: 2d ndarray of variances
        :param costs: 1d ndarray of costs
        """
        if self._verbose:
            sys.stdout.write("Computing optimal sample sizes: ")

        # Need 2d version of costs in order to vectorize the operations.
        costs_2d = costs[:, np.newaxis]

        # Compute mu.
        sum_sqrt_vc = np.sum(np.sqrt(variances * costs_2d), axis=0)

        if self._target_cost is None:
            mu = np.power(self._epsilons, -2) * sum_sqrt_vc
        else:
            mu = self._target_cost / sum_sqrt_vc

        # Compute sample sizes.
        sqrt_v_over_c = np.sqrt(variances / costs_2d)
        self._sample_sizes = np.amax(np.ceil(mu * sqrt_v_over_c), axis=1).\
            astype(int)

        if self._verbose:
            print np.array2string(self._sample_sizes)

    def _run_monte_carlo(self, num_samples):
        """
        Runs a standard monte carlo simulation. Used when only one model
        is provided.

        :param num_samples: Number of samples to take.
        :return: tuple containing three ndarrays with one element each:
            estimates: Estimates for each quantity of interest
            sample_sizes: The sample sizes used at each level.
            variances: Variance of model outputs at each level.
        """
        if self._verbose:
            print 'Only one model provided; running standard monte carlo.'

        samples = self._data.draw_samples(num_samples)
        outputs = np.zeros(num_samples)

        for i, sample in enumerate(samples):
            outputs[i] = self._models[0].evaluate(sample)

        # Return values should have same signature as regular MLMC simulation.
        estimates = np.array([np.mean(outputs)])
        sample_sizes = np.array([num_samples])
        variances = np.array([np.var(outputs)])

        return estimates, sample_sizes, variances

    def _process_epsilon(self, epsilon):
        """
        Produce an ndarray of epsilon values from scalar or vector of epsilons.
        If a vector, length should match the number of quantities of interest.

        :param epsilon: float, list of floats, or ndarray.
        :return: ndarray of epsilons of size (self.num_levels).
        """
        if isinstance(epsilon, list):
            epsilon = np.array(epsilon)

        if isinstance(epsilon, float):
            epsilon = np.ones(self._output_size) * epsilon

        if not isinstance(epsilon, np.ndarray):
            raise TypeError("Epsilon must be a float, list of floats, " +
                            "or an ndarray.")

        if np.any(epsilon <= 0.):
            raise ValueError("Epsilon values must be greater than 0.")

        if len(epsilon) != self._output_size:
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
    def __check_simulate_parameters(starting_sample_size, maximum_cost):

        if not isinstance(starting_sample_size, int):
            raise TypeError("starting_sample_size must be an integer.")

        if starting_sample_size < 1:
            raise ValueError("starting_sample_size must be greater than zero.")

        if maximum_cost is not None:

            if not (isinstance(maximum_cost, float) or
                    isinstance(maximum_cost, int)):

                raise TypeError('maximum cost must be an int or float.')

            if maximum_cost <= 0:
                raise ValueError("maximum cost must be greater than zero.")
