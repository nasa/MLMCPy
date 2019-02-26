import abc

from MLMCPy.model import Model

class WrapperModel(Model):
    """
    Abstract class that inherits from the Model class.
    """
    @abc.abstractmethod
    def attach_model(self, model):
        """
        Abstract method that should take in model object and assign it to the 
        class.
        """
        raise NotImplementedError
