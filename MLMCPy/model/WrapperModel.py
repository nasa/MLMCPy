from abc import abstractmethod

from MLMCPy.model import Model

class WrapperModel(Model):
    
    def __init__(self, model):
        self._model = model

        if hasattr(self._model, 'cost'):
            self.cost = model.cost
        
    @abstractmethod
    def attach_model(self, model):
        pass
