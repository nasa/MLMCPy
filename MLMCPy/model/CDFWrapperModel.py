import numpy as np

from MLMCPy.model.WrapperModel import WrapperModel
from MLMCPy.model.Model import Model

class CDFWrapperModel(WrapperModel):
    """
    Generates CDF indicators from an inner model that will be used by
    MLMCSimulator to generate a CDF from the inner model outputs.
    """
    def __init__(self, grid, smoothing=None):
        """
        :param model: An instance of a class inheriting from Model that
            implements the evaluate function.
        :param grid: A 1d ndarray for constructing the indicators that will
            be used in building a CDF.
        :param smoothing: Whether to implement smoothing in order to improve
            the CDF result. TODO: Improve this description.
        """
        self.__check_init_parameters(grid, smoothing)

        self._model = None
        self._grid = grid
        self._inner_model_outputs = list()

    def evaluate(self, sample):
        """
        Evaluates the internal model on the given sample and computes the
            indicators based on the output.
        :param sample: ndarray of 0 or 1 dimensions to be passed to inner model.
        :return: 1d ndarray of indicators.
        """
        self.__check_attached_model(self._model)
        # Run model and collect output value.
        output = self._model.evaluate(sample)

        # Compute the indicators.
        indicators = np.zeros(self._grid.size)
        for x, y in enumerate(self._grid):
            indicators[x] = np.count_nonzero(output <= y)

        return indicators

    def attach_model(self, model):
        """
        Updates _model to the desired model object.

        :param model: Model object that must inherit from Model class.
        """
        self.__check_attach_model_parameter(model)

        self._model = model

        if hasattr(self._model, 'cost'):
            self.cost = self._model.cost

    @staticmethod
    def __check_attach_model_parameter(model):

        if not isinstance(model, Model):
            raise TypeError("model must inherit from class Model.")
    
    @staticmethod
    def __check_attached_model(model):
        if not isinstance(model, Model):
            raise TypeError('Model must be attached.')

    @staticmethod
    def __check_init_parameters(grid, smoothing):
        if not isinstance(grid, np.ndarray):
            raise TypeError("Grid must be a ndarray.")

        if len(grid.shape) > 1:
            raise ValueError("Grid must be one dimensional.")

        if grid.size < 3:
            raise ValueError("Grid must have at least three elements.")

        if smoothing is not None and not isinstance(smoothing, bool):
            raise TypeError("Smoothing must be boolean.")
