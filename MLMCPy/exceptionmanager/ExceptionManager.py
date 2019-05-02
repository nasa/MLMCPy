import numpy as np
from MLMCPy.input import Input
from MLMCPy.model import Model

class ExceptionManager(object):
    @staticmethod
    def check_init_parameters(data, models):
        """
        Inspects parameters given to init method.

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
    def check_simulate_parameters(target_cost):
        """
        Inspects parameters to simulate method.

        :param target_cost: float or int specifying desired simulation cost.
        """
        if target_cost is not None:

            if not isinstance(target_cost, (int, float)):

                raise TypeError('maximum cost must be an int or float.')

            if target_cost <= 0:
                raise ValueError("maximum cost must be greater than zero.")

    @staticmethod
    def check_get_model_inputs_parameters(sample_sizes):
        """
        Inspects parameters in get_model_inputs_to_run_for_each_level().

        :param sample_sizes: List or np.ndarray of int specifying the number of
            sample sizes.
        """
        if not isinstance(sample_sizes, (list, np.ndarray)):
            raise TypeError('sample_sizes must be a list or np.ndarray.')

        for level in range(len(sample_sizes)):
            if not isinstance(sample_sizes[level], int):
                raise TypeError('sample_sizes[%s] must be an int.' % level)

    @staticmethod
    def check_store_model_params(sample_sizes, filenames=None):
        """
        Inspects parameters in store_model_inputs_to_run_for_each_level().

        :param sample_sizes: List or np.ndarray of int specifying the number of
            sample sizes.
        :param filenames: Object that must contain strings of desired file names
            it must match the number of models(levels).
        """
        if not isinstance(sample_sizes, (list, np.ndarray)):
            raise TypeError('sample_sizes must be a list or np.ndarray.')

        for level in range(len(sample_sizes)):
            if not isinstance(sample_sizes[level], int):
                raise TypeError('sample_sizes[%s] must be an int.' % level)

        if filenames is not None:
            if isinstance(filenames, bool) and filenames == True:
                return
            for name in range(len(filenames)):
                if not isinstance(filenames[name], str):
                    raise TypeError('filenames[%s] must be a string.' % name)

    @staticmethod
    def check_compute_estimators_parameter(model_outputs):
        """
        Inspects parameter given to compute_estimators(), and ensures that it
        is a np.ndarray.
        """
        if not isinstance(model_outputs, dict):
            raise TypeError('model_outputs must be a dictionary of output' +
                            'numpy arrays.')

        for key in model_outputs:
            if not isinstance(model_outputs[key], np.ndarray):
                raise TypeError('model_outputs must be a dictionary of output' +
                                'numpy arrays.')