import abc


class Model(object):
    """
    Abstract base class for Models which should evaluate sample inputs to
    produce outputs.
    """
    @abc.abstractmethod
    def evaluate(self, inputs):
        return
