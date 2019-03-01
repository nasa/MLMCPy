from MLMCPy.model import Model

class WrapperModel(Model):
    """
    Abstract class that inherits from the Model class.
    """
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
            raise TypeError("model must inherit from Model class.")
