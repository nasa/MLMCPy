import numpy as np
from scipy.special import erf as scipy_erf

from Model import Model


class CDFWrapperModel(Model):
    """
    Generates CDF indicators from an inner model that will be used by
    MLMCSimulator to generate a CDF from the inner model outputs.
    """
    def __init__(self, model, grid, smoothing=None, smooth_factor=0.0):
        """
        :param model: An instance of a class inheriting from Model that
            implements the evaluate function.
        :param grid: A 1d ndarray for constructing the indicators that will
            be used in building a CDF.
        :param smoothing: Whether to implement smoothing in order to improve
            the CDF result. TODO: Improve this description.
        :param smooth_factor: factor that controls the amount of smoothing 
            applied. A value of 0.0 provides no smoothing and recovers an 
            indicator function.
        """
        self.__check_init_parameters(model, grid, smoothing, smooth_factor)

        self._model = model
        self._grid = grid
        self._smoothing = smoothing
        self._smooth_factor = smooth_factor
        self._inner_model_outputs = list()

        if hasattr(self._model, 'cost'):
            self.cost = model.cost

    def evaluate(self, sample):
        """
        Evaluates the internal model on the given sample and computes the
            indicators based on the output.
        :param sample: ndarray of 0 or 1 dimensions to be passed to inner model.
        :return: 1d ndarray of indicators.
        """
        # Run model and collect output value.
        model_output = self._model.evaluate(sample)

        if self._smoothing:
            indicators = self._evaluate_smoothed_indicator(model_output)
        else:
            indicators = self._evaluate_indicator(model_output)

        return indicators

    def _evaluate_smoothed_indicator(self, model_output):
        """
        Evaluates an indicator function that has smoothing enabled where the 
        degree of smoothing is controlled by the specified smooth factor. Note
        that 0.0 for this parameter indicates no smoothing and should recover
        the standard indicator
        Currently smoothing is implemented using the Gaussian error function,
        represents case where we've modeled a dirac delta density function with
        a Gaussian kernel with std sigma (sigma is the smooth factor)
        """

        diffs = self._grid - model_output
        scaled_diffs = diffs/(np.sqrt(2.)*self._smooth_factor)
        erf_vals = scipy_erf(scaled_diffs)

        return 0.5*(1 + erf_vals)

    def _evaluate_indicator(self, model_output):
        """
        Evaluate indicator function for given model output. i.e., return
        indicator array of whether or not model_output is less than /equal to
        each point in the supplied grid
        """

        # Compute the indicators.
        indicators = np.zeros(self._grid.size)
        for x, y in enumerate(self._grid):
            indicators[x] = np.count_nonzero(model_output <= y)

        return indicators

    @staticmethod
    def __check_init_parameters(model, grid, smoothing, smooth_factor):

        if not isinstance(model, Model):
            raise TypeError("Model must inherit from class Model.")

        if not isinstance(grid, np.ndarray):
            raise TypeError("Grid must be a ndarray.")

        if len(grid.shape) > 1:
            raise ValueError("Grid must be one dimensional.")

        if grid.size < 3:
            raise ValueError("Grid must have at least three elements.")

        if smoothing is not None and not isinstance(smoothing, bool):
            raise TypeError("Smoothing must be boolean.")
        
        if smooth_factor < 0.0:
            raise ValueError("Smoothing factor must be non-negative")

