import numpy as np

from Model import Model


class DistributionFunction(Model):

    def __init__(self, model, grid, smoothing=None):

        self.__check_init_parameters(model, grid, smoothing)

        self.model = model

    def evaluate(self, sample):

        pass

    @staticmethod
    def __check_init_parameters(model, grid, smoothing):

        pass
