import abc

from MLMCPy.model import Model

class WrapperModel(Model):
    
    def __init__(self):
        self._model = None
       
    @abc.abstractmethod
    def attach_model(self, model):
        raise NotImplementedError
