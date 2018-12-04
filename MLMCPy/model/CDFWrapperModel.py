import numpy as np

from Model import Model


class CDFWrapperModel(Model):
    """
    Generates CDF indicators from an inner model that will be used by
    MLMCSimulator to generate a CDF from the inner model outputs.
    """
    def __init__(self, model, grid, smoothing=None):
        """
        :param model: An instance of a class inheriting from Model that
            implements the evaluate function.
        :param grid: A 1d ndarray for constructing the indicators that will
            be used in building a CDF.
        :param smoothing: Whether to implement smoothing in order to improve
            the CDF result. TODO: Improve this description.
        """
        self.__check_init_parameters(model, grid, smoothing)

        self._model = model
        self._grid = grid
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
        output = self._model.evaluate(sample)

        # Compute the indicators.
        indicators = np.zeros(self._grid.size)
        for x, y in enumerate(self._grid):
            indicators[x] = np.count_nonzero(output <= y)

        return indicators

    @staticmethod
    def __check_init_parameters(model, grid, smoothing):

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
