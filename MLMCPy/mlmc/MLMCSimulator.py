import imp
from datetime import timedelta
import timeit
import numpy as np

from MLMCPy.input import Input
from MLMCPy.model import Model


class MLMCSimulator(object):
    """
    Computes an estimate based on the Multi-Level Monte Carlo algorithm.
    """
    def __init__(self, random_input, models):
        """
        Requires a data object that provides input samples and a list of models
        of increasing fidelity.

        :param random_input: Provides a data sampling function.
        :type data: Input
        :param models: Each model Produces outputs from sample data input.
        :type models: list(Model)
        """
        # Detect whether we have access to multiple CPUs.
        self.__detect_parallelization()

        self.__check_init_parameters(random_input, models)

        self._data = random_input
        self._models = models
        self._num_levels = len(self._models)

        # Sample size to be taken at each level.
        self._sample_sizes = np.zeros(self._num_levels, dtype=np.int)

        # Used to compute sample sizes based on a fixed cost.
        self._target_cost = None

        # Sample sizes used in setup.
        self._initial_sample_sizes = np.empty(0, dtype=int)

        # Desired level of precision.
        self._epsilons = np.zeros(1)

        # Number of elements in model output.
        self._output_size = 1
        self._input_size = 1

        self._estimates = None
        self._variances = None

        self._cached_inputs = np.empty(0)
        self._cached_outputs = np.empty(0)

        # Whether to allow use of model output caching.
        self._caching_enabled = True

        # Enabled diagnostic text output.
        self._verbose = False

    def simulate(self, epsilon, initial_sample_sizes=100, target_cost=None,
                 sample_sizes=None, verbose=False):
        """
        Perform MLMC simulation.
        Computes number of samples per level before running simulations
        to determine estimates.
        Can be specified based on target precision to achieve (epsilon),
        total target cost (in seconds), or on number of sample to run on each
        level directly.

        :param epsilon: Desired accuracy to be achieved for each quantity of
            interest.
        :type epsilon: float, list of floats, or ndarray.
        :param initial_sample_sizes: Sample sizes used when computing cost
            and variance for each model in simulation.
        :type initial_sample_sizes: ndarray, int, list
        :param target_cost: Target cost to run simulation (optional).
            If specified, overrides any epsilon value provided.
        :type target_cost: float or int
        :param sample_sizes: Number of samples to compute at each level
        :type sample_sizes: ndarray
        :param verbose: Whether to print useful diagnostic information.
        :type verbose: bool
        :param only_collect_sample_sizes: indicates whether to bypass simulation
            phase and simply return prescribed number of samples for each model.
            Return value is changed to one dimensional ndarray.
        :type only_collect_sample_sizes: bool
        :return: Tuple of ndarrays
            (estimates, sample count per level, variances)
        """
        self._verbose = verbose and self._cpu_rank == 0

        self.__check_simulate_parameters(target_cost)

        self._process_target_cost(target_cost)

        self._determine_input_output_size()

        self._setup_simulation(epsilon, initial_sample_sizes, sample_sizes)

        # Run models and return estimate, sample sizes, and variances.
        return self._run_simulation()

    def compute_costs_and_variances(self, user_sample_size=None):
        """
        Computes costs and variances across levels.

        :return: tuple of ndarrays:
            1d ndarray of costs
            2d ndarray of variances
        """
        if self._verbose:
            print "Determining costs: "

        if user_sample_size is not None:
            user_samples = self._verify_sample_sizes(user_sample_size)
            self._initialize_cache(user_samples)
        else:
            self._initialize_cache()

        # Evaluate samples in model. Gather compute times for each level.
        # Variance is computed from difference between outputs of adjacent
        # layers evaluated from the same samples.
        compute_times = np.zeros(self._num_levels)

        for level in range(self._num_levels):
            if user_sample_size is not None:
                input_samples = self._draw_setup_samples(level, user_samples)
            else:
                input_samples = self._draw_setup_samples(level)

            start_time = timeit.default_timer()
            self._compute_setup_outputs(input_samples, level)
            compute_times[level] = timeit.default_timer() - start_time

        # Get outputs across all CPUs before computing variances.
        all_outputs = self._gather_arrays(self._cached_outputs, axis=1)

        variances = np.var(all_outputs, axis=1)
        costs = self._compute_costs(compute_times)

        if self._verbose:
            print 'Initial sample variances: \n%s' % variances

        return costs, variances

    def compute_optimal_sample_sizes(self, costs, variances, user_epsilon=None):
        """
        Computes the sample size for each level to be used in simulation.

        :param variances: 2d ndarray of variances
        :param costs: 1d ndarray of costs
        """
        if self._verbose:
            print "Computing optimal sample sizes: "

        # Need 2d version of costs in order to vectorize the operations.
        costs = costs[:, np.newaxis]

        if user_epsilon is not None:
            self._process_epsilon(user_epsilon)

        mu = self._compute_mu(costs, variances)

        # Compute sample sizes.
        sqrt_v_over_c = np.sqrt(variances / costs)
        sample_sizes = np.amax(np.trunc(mu * sqrt_v_over_c), axis=1)

        self._process_sample_sizes(sample_sizes, costs)

        if self._verbose:

            print np.array2string(self._sample_sizes)

            estimated_runtime = np.dot(self._sample_sizes, np.squeeze(costs))

            self._show_time_estimate(estimated_runtime)

        if user_epsilon is not None:
            return self._sample_sizes

        return None

    def compute_estimators(self, outputs):
        """Computes the estimators using the output differences per level.

        :param outputs: The differences per level.
        :type outputs: ndarray, list
        :return: Returns the estimates and variances as an ndarray.
        """
        self._check_compute_estimators_parameter(outputs)
        
        estimates = 0
        variances = 0
        num_samples = 0

        for level in range(self._num_levels):
            num_samples = float(len(outputs[level]))

            estimates += \
                np.sum(outputs[level], axis=0) / num_samples
            variances += \
                np.var(outputs[level], axis=0) / num_samples

        return estimates, variances

    def _check_compute_estimators_parameter(self, outputs):
        """
        Checks the parameter given to compute_estimators(), and ensures that it
        is a np.ndarray.
        """
        if not isinstance(outputs, np.ndarray):
            raise TypeError("outputs must be a np.ndarray")      

    def _setup_simulation(self, epsilon, initial_sample_sizes, sample_sizes):
        """
        Performs any necessary manipulation of epsilon and initial_sample_sizes.
        Computes variance and cost at each level in order to estimate optimal
        number of samples at each level.

        :param epsilon: Epsilon values for each quantity of interest.
        :param initial_sample_sizes: Sample sizes used when computing costs
            and variance for each model in simulation.
        """
        if sample_sizes is None:
            self._process_epsilon(epsilon)
            self._initial_sample_sizes = \
                self._verify_sample_sizes(initial_sample_sizes)

            costs, variances = self.compute_costs_and_variances()
            self.compute_optimal_sample_sizes(costs, variances)

        else:
            self._target_cost = None
            self._caching_enabled = False
            sample_sizes = self._verify_sample_sizes(sample_sizes, False)
            self._process_sample_sizes(sample_sizes, None)

    def _initialize_cache(self, user_sample_size=None):
        """
        Sets up the cache for retaining model outputs evaluated in the setup
        phase for reuse in the simulation phase.
        """
        # Determine number of samples to be taken on this processor.
        get_cpu_sample_sizes = np.vectorize(self._determine_num_cpu_samples)

        if user_sample_size is not None:
            self._cpu_initial_sample_sizes = \
                get_cpu_sample_sizes(user_sample_size)
        else:
            self._cpu_initial_sample_sizes = \
                get_cpu_sample_sizes(self._initial_sample_sizes)

        max_cpu_sample_size = int(np.max(self._cpu_initial_sample_sizes))

        # Cache model outputs computed here so that they can be reused
        # in the simulation.
        self._cached_inputs = np.zeros((self._num_levels,
                                        max_cpu_sample_size,
                                        self._input_size))
        self._cached_outputs = np.zeros((self._num_levels,
                                         max_cpu_sample_size,
                                         self._output_size))

    def _draw_setup_samples(self, level, user_samples=None):
        """
        Draw samples based on initial sample size at specified level.
        Store samples in _cached_inputs.
        :param level: int level
        """
        if user_samples is not None:
            num_samples = user_samples[level]
        else:
            num_samples = self._initial_sample_sizes[level]
        input_samples = self._draw_samples(num_samples)

        # To cache these samples, we have to account for the possibility
        # of the data source running out of samples so that we can
        # broadcast into the cache successfully.
        self._cached_inputs[level, :input_samples.shape[0], :] = input_samples

        return input_samples

    def _compute_setup_outputs(self, input_samples, level):
        """
        Evaluate model outputs for a given level. If level > 0, subtract outputs
        at level below specified level. Store results in _cached_outputs.
        :param input_samples: samples to evaluate in model.
        :param level: int level of model
        """
        num_cpu_samples = self._cpu_initial_sample_sizes[level]
        lower_level_outputs = np.zeros((num_cpu_samples, self._output_size))
        for i, sample in enumerate(input_samples):

            self._cached_outputs[level, i] = \
                self._models[level].evaluate(sample)

            if level > 0:
                lower_level_outputs[i] = \
                    self._models[level - 1].evaluate(sample)

        self._cached_outputs[level] -= lower_level_outputs

    def _compute_costs(self, compute_times):
        """
        Set costs for each level, either from precomputed values from each
        model or based on computation times provided by compute_times.

        :param compute_times: ndarray of computation times for computing
        model at each layer and preceding layer.
        """
        # If the models have costs predetermined, use them to compute costs
        # between each level.
        if self._models_have_costs():
            costs = self._get_costs_from_models()
        else:
            # Compute costs based on compute time differences between levels.
            costs = compute_times / self._cpu_initial_sample_sizes \
                    * self._num_cpus

        costs = self._mean_over_all_cpus(costs)

        if self._verbose:
            print np.array2string(costs)

        return costs

    def _models_have_costs(self):
        """
        :return: bool indicating whether the models all have a cost attribute.
        """
        model_cost_defined = True
        for model in self._models:

            model_cost_defined = model_cost_defined and hasattr(model, 'cost')

            if not model_cost_defined:
                return False

            model_cost_defined = model_cost_defined and model.cost is not None

        return model_cost_defined

    def _get_costs_from_models(self):
        """
        Collect cost value from each model.
        :return: ndarray of costs.
        """
        costs = np.ones(self._num_levels)
        for i, model in enumerate(self._models):
            costs[i] = model.cost

        # Costs at level > 0 should be summed with previous level.
        costs[1:] = costs[1:] + costs[:-1]

        return costs

    def _compute_mu(self, costs, variances):
        """
        Computes the mu value used to compute sample sizes.

        :param costs: 2d ndarray of costs
        :param variances: ndarray of variances
        :return: ndarray of mu value for each QoI.
        """
        sum_sqrt_vc = np.sum(np.sqrt(variances * costs), axis=0)

        if self._target_cost is None:
            mu = np.power(self._epsilons, -2) * sum_sqrt_vc
        else:
            mu = self._target_cost * float(self._num_cpus) / sum_sqrt_vc

        return mu

    def _process_sample_sizes(self, sample_sizes, costs):
        """
        Make any necessary adjustments to computed sample sizes, including
        adjustments for staying under target cost and distributing among
        processors.
        """
        self._sample_sizes = sample_sizes

        # Manually tweak sample sizes to get predicted cost closer to target.
        if self._target_cost is not None:
            self._fit_samples_sizes_to_target_cost(np.squeeze(costs))

        # Set sample sizes to ints.
        self._sample_sizes = self._sample_sizes.astype(int)

        # If target cost is less than cost of least expensive model, run it
        # once so we are at least doing something in the simulation.
        if np.sum(self._sample_sizes) == 0.:
            self._sample_sizes[0] = 1

        # Divide sampling evenly across CPUs.
        split_samples = np.vectorize(self._determine_num_cpu_samples)
        self._cpu_sample_sizes = split_samples(self._sample_sizes)

    def _fit_samples_sizes_to_target_cost(self, costs):
        """
        Adjust sample sizes to be as close to the target cost as possible.
        """
        # Find difference between projected total cost and target.
        total_cost = np.dot(costs, self._sample_sizes)
        difference = self._target_cost - total_cost
        # If the difference is greater than the lowest cost model, adjust
        # the sample sizes.
        if abs(difference) > costs[0]:

            # Start with highest cost model and add samples in order to fill
            # the cost gap as much as possible.
            for i in range(len(costs) - 1, -1, -1):
                if costs[i] < abs(difference):

                    # Compute number of samples we can fill the gap with at
                    # current level.
                    delta = np.trunc(difference / costs[i])
                    self._sample_sizes[i] += delta

                    if self._sample_sizes[i] < 0:
                        self._sample_sizes[i] = 0

                    # Update difference from target cost.
                    total_cost = np.sum(costs * self._sample_sizes)
                    difference = self._target_cost - total_cost

    def _run_simulation(self):
        """
        Compute estimate by extracting number of samples from each level
        determined in the setup phase.

        :return: tuple containing three ndarrays:
            estimates: Estimates for each quantity of interest.
            sample_sizes: The sample sizes used at each level.
            variances: Variance of model outputs at each level.
        """
        # Sampling needs to be restarted from beginning due to sampling
        # having been performed in setup phase.
        self._data.reset_sampling()

        start_time = timeit.default_timer()
        estimates, variances = self._run_simulation_loop()
        run_time = timeit.default_timer() - start_time

        if self._verbose:
            self._show_summary_data(estimates, variances, run_time)

        return estimates, self._sample_sizes, variances

    def _run_simulation_loop(self):
        """
        Main simulation loop where sample sizes determined in setup phase are
        drawn from the input data and run through the models. Values for
        computing the estimates and variances are accumulated at each level.

        :return: tuple containing two ndarrays:
            estimates: Estimates for each quantity of interest.
            variances: Variance of model outputs at each level.
        """
        for level in range(self._num_levels):

            if self._sample_sizes[level] == 0:
                continue

            samples = self._get_sim_loop_samples(level)
            output_differences = self._get_sim_loop_outputs(samples, level)
            self._update_sim_loop_values(output_differences, level)

        return self._estimates, self._variances

    def _get_sim_loop_samples(self, level):
        """
        Acquires input samples for designated level.

        :param level: int of level for which samples are to be acquired.
        :return: ndarray of input samples.
        """
        samples = self._draw_samples(self._sample_sizes[level])
        num_samples = samples.shape[0]

        # Update sample sizes in case we've run short on samples.
        self._cpu_sample_sizes[level] = num_samples

        return samples

    def _get_sim_loop_outputs(self, samples, level):
        """
        Get the output differences for given level and samples.

        :param samples: ndarray of input samples.
        :param level: int level of model to run.

        :return: ndarray of output differences between samples from
            designated level and level below (if applicable).
        """
        num_samples = samples.shape[0]

        if num_samples == 0:
            return np.zeros((1, self._output_size))

        output_differences = np.zeros((num_samples, self._output_size))

        for i, sample in enumerate(samples):
            output_differences[i] = self._evaluate_sample(sample, level)

        return output_differences

    def _update_sim_loop_values(self, outputs, level):
        """
        Update running totals for estimates and variances based on the output
        differences at a particular level.

        :param outputs: ndarray of output differences.
        :param level: int of level at which differences were computed.
        """
        cpu_samples = self._cpu_sample_sizes[level]

        all_output_differences = self._gather_arrays(outputs, axis=0)

        self._sample_sizes[level] = self._sum_over_all_cpus(cpu_samples)
        num_samples = float(self._sample_sizes[level])

        self._estimates += np.sum(all_output_differences, axis=0) / num_samples
        self._variances += np.var(all_output_differences, axis=0) / num_samples

    def _evaluate_sample(self, sample, level):
        """
        Evaluate output of an input sample, either by running the model or
        retrieving the output from the cache. For levels > 0, returns
        difference between current level and lower level outputs.

        :param sample: sample value
        :param level: model level
        :return: result of evaluation
        """
        sample_indices = np.empty(0)
        if self._caching_enabled:
            sample_indices = np.argwhere(sample == self._cached_inputs[level])

        if len(sample_indices) == 1:
            output = self._cached_outputs[level, sample_indices[0]][0]
        else:
            output = self._models[level].evaluate(sample)

            # If we are at a level greater than 0, compute outputs for lower
            # level and subtract them from this level's outputs.
            if level > 0:
                output -= self._models[level-1].evaluate(sample)

        return output

    def _show_summary_data(self, estimates, variances, run_time):
        """
        Shows summary of simulation.

        :param estimates: ndarray of estimates for each QoI.
        :param variances: ndarray of variances for each QoI.
        """
        # Compare variance for each quantity of interest to epsilon values.
        print
        print 'Total run time: %s' % str(run_time)
        print

        epsilons_squared = np.square(self._epsilons)
        for i, variance in enumerate(variances):

            passed = variance < epsilons_squared[i]
            estimate = estimates[i]

            print 'QOI #%s: estimate: %s, variance: %s, ' \
                  'epsilon^2: %s, met target precision: %s' % \
                  (i, float(estimate), float(variance),
                   float(epsilons_squared[i]), passed)

    def _determine_input_output_size(self):
        """
        Runs first model on a small test sample to determine
        shapes of input and output.
        """
        self._data.reset_sampling()
        test_sample = self._draw_samples(self._num_cpus)

        if test_sample.shape[0] == 0:
            message = "The environment has more CPUs than data samples! " + \
                "Please provide more data or specify fewer CPUs."

            raise ValueError(message)

        test_sample = test_sample[0]
        self._data.reset_sampling()

        test_output = self._models[0].evaluate(test_sample)

        self._input_size = test_sample.size
        self._output_size = test_output.size

        self._estimates = np.zeros(self._output_size)
        self._variances = np.zeros_like(self._estimates)

    def _process_epsilon(self, epsilon):
        """
        Produce an ndarray of epsilon values from scalar or vector of epsilons.
        If a vector, length should match the number of quantities of interest.

        :param epsilon: float, list of floats, or ndarray.
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

        self._epsilons = epsilon

    def _verify_sample_sizes(self, sample_sizes, initial_samples=True):
        """
        Produce an array of sample sizes, ensuring that its length
        matches the number of models.
        :param sample_sizes: scalar or vector of sample sizes

        returns verified/adjusted sample sizes array
        """
        if isinstance(sample_sizes, np.ndarray):
            verified_sample_sizes = sample_sizes
        elif isinstance(sample_sizes, list):
            verified_sample_sizes = np.array(sample_sizes)
        else:
            if not isinstance(sample_sizes, int) and \
                    not isinstance(sample_sizes, float):

                raise TypeError("Initial sample sizes must be numeric.")

            verified_sample_sizes = \
                np.ones(self._num_levels).astype(int) * \
                int(sample_sizes)

        if verified_sample_sizes.size != self._num_levels:
            raise ValueError("Number of initial sample sizes must match " +
                             "number of models.")

        if not np.all(verified_sample_sizes > 1) and initial_samples:
            raise ValueError("Each initial sample size must be at least 2.")

        return verified_sample_sizes

    def _process_target_cost(self, target_cost):

        if target_cost is not None:
            self._target_cost = float(target_cost)

    @staticmethod
    def __check_init_parameters(data, models):
        """
        Inspect parameters given to init method.
        :param data: Input object provided to init().
        :param models: Model object provided to init().
        """
        if not isinstance(data, Input):
            raise TypeError("data must inherit from Input class.")

        if not isinstance(models, list):
            raise TypeError("models must be a list of models.")

        # Reset sampling in case input data is used more than once.
        data.reset_sampling()

        # Ensure all models have the same output dimensions.
        output_sizes = []
        test_sample = data.draw_samples(1)[0]
        data.reset_sampling()

        for model in models:
            if not isinstance(model, Model):
                raise TypeError("models must be a list of models.")

            test_output = model.evaluate(test_sample)
            output_sizes.append(test_output.size)

        output_sizes = np.array(output_sizes)
        if not np.all(output_sizes == output_sizes[0]):
            raise ValueError("All models must return the same output " +
                             "dimensions.")

    @staticmethod
    def __check_simulate_parameters(target_cost):
        """
        Inspect parameters to simulate method.
        :param target_cost: float or int specifying desired simulation cost.
        """
        if target_cost is not None:

            if not isinstance(target_cost, (int, float)):

                raise TypeError('maximum cost must be an int or float.')

            if target_cost <= 0:
                raise ValueError("maximum cost must be greater than zero.")

    def _draw_samples(self, num_samples):
        """
        Draw samples from data source.
        :param num_samples: Total number of samples to draw over all CPUs.
        :return: ndarray of samples sliced according to number of CPUs.
        """
        samples = self._data.draw_samples(num_samples)
        if self._num_cpus == 1:
            return samples

        sample_size = samples.shape[0]

        # Determine subsample sizes for all CPUs.
        subsample_size = sample_size // self._num_cpus
        remainder = sample_size - subsample_size * self._num_cpus
        subsample_sizes = np.ones(self._num_cpus + 1).astype(int)*subsample_size

        # Adjust for sampling that does not divide evenly among CPUs.
        subsample_sizes[:remainder + 1] += 1
        subsample_sizes[0] = 0

        # Determine starting index of subsample.
        subsample_index = int(np.sum(subsample_sizes[:self._cpu_rank + 1]))

        # Take subsample.
        samples = samples[subsample_index:
                          subsample_index + subsample_sizes[self._cpu_rank + 1],
                          :]

        return samples

    def __detect_parallelization(self):
        """
        Detects whether multiple processors are available and sets
        self.number_CPUs and self.cpu_rank accordingly.
        """
        try:
            imp.find_module('mpi4py')

            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            self._num_cpus = comm.size
            self._cpu_rank = comm.rank
            self._comm = comm

        except ImportError:

            self._num_cpus = 1
            self._cpu_rank = 0

    def _mean_over_all_cpus(self, this_cpu_values, axis=0):
        """
        Finds the mean of ndarray of values across CPUs and returns result.
        :param this_cpu_values: ndarray of any shape.
        :return: ndarray of same shape as values with mean from all cpus.
        """
        if self._num_cpus == 1:
            return this_cpu_values

        all_values = self._comm.allgather(this_cpu_values)

        return np.mean(all_values, axis)

    def _sum_over_all_cpus(self, this_cpu_values, axis=0):
        """
        Collect arrays from all CPUs and perform summation over specified
        axis.
        :param this_cpu_values: ndarray provided for current CPU.
        :param axis: int axis to perform summation over.
        :return: ndarray of summation result
        """
        if self._num_cpus == 1:
            return this_cpu_values

        all_values = self._comm.allgather(this_cpu_values)

        return np.sum(all_values, axis)

    def _gather_arrays(self, this_cpu_array, axis=0):
        """
        Collects an array from all processes and combines them so that single
        processor ordering is preserved.
        :param this_cpu_array: Arrays to be combined.
        :param axis: Axis to concatenate the arrays on.
        :return: ndarray
        """
        if self._num_cpus == 1:
            return this_cpu_array

        gathered_arrays = self._comm.allgather(this_cpu_array)

        return np.concatenate(gathered_arrays, axis=axis)

    def _determine_num_cpu_samples(self, total_num_samples):
        """Determines number of samples to be run on current cpu based on
            total number of samples to be run.
            :param total_num_samples: Total samples to be taken.
            :return: Samples to be taken by this cpu.
        """
        num_cpu_samples = total_num_samples // self._num_cpus

        num_residual_samples = total_num_samples - \
            num_cpu_samples * self._num_cpus

        if self._cpu_rank < num_residual_samples:
            num_cpu_samples += 1

        return num_cpu_samples

    @staticmethod
    def _show_time_estimate(seconds):
        """
        Used to show theoretical simulation time when verbose is enabled.
        :param seconds: int seconds to convert to readable time.
        """
        if isinstance(seconds, np.ndarray):
            seconds = seconds[0]

        time_delta = timedelta(seconds=seconds)

        print 'Estimated simulation time: %s' % str(time_delta)
