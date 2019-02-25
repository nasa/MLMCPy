import numpy as np

from MLMCPy.model.Model import Model

class CovarianceWrapperModel(Model):
    """
    Adds the product of the inner model outputs to the wrapper output array so
    that covariance may be calculated.
    """
    def __init__(self):
        """
        :param model: An instance of a class inheriting from Model that
            implements the evaluate function.
        """
        self._model = None

    def attach_model(self, model):
        """
        Updates _model to the desired model object.

        :param model: Model object that must inherit from Model class.
        """
        self.__check_init_parameter(model)

        self._model = model

        if hasattr(self._model, 'cost'):
            self.cost = model.cost

    @staticmethod
    def __check_init_parameter(model):

        if not isinstance(model, Model):
            raise TypeError("Model must inherit from class Model.")

    def evaluate(self, sample):
        """
        Evaluates the internal model on the given sample and computes products
        of its outputs
        :param sample: ndarray of 0 or 1 dimensions to be passed to inner model.
        :return: 1d ndarray of indicators.
        """
        self.__check_attached_model(self._model)
        # Run model and collect output value.
        output = self._model.evaluate(sample)

        products = []
        for i, out_i in enumerate(output):
            for out_j in output[i:]:
                products.append(out_i * out_j)

        return np.hstack((output, products))

    @staticmethod
    def __check_attached_model(model):
        if not isinstance(model, Model):
            raise TypeError('Model must be attached.')

    @staticmethod
    def post_process_covariance(expected_values):
        original_output_size = \
            CovarianceWrapperModel._get_inner_model_size(expected_values.size)
        means = expected_values[:original_output_size]
        covariance = np.copy(expected_values[original_output_size:])

        cov_index = 0
        for i, mean_i in enumerate(means):
            for mean_j in means[i:]:
                covariance[cov_index] -= mean_i * mean_j
                cov_index += 1

        return covariance

    @staticmethod
    def _get_inner_model_size(wrapper_output_size):
        inner_size = 0
        next_test_size = -1
        while wrapper_output_size > next_test_size:
            inner_size += 1
            next_test_size = inner_size*(inner_size + 3) / 2
            if wrapper_output_size == next_test_size:
                return inner_size

        raise TypeError("Covariance could not be computed from expected "
                        "values.  Expected values have unrecognized length.")
