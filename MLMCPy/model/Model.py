import abc


class Model(object):
    """
    Abstract base class for Models which should evaluate sample inputs to
    produce outputs.

    :param inputs: one dimensional ndarray
    :return: two dimensional ndarray
    """
    @abc.abstractmethod
    def evaluate(self, inputs):
        return
