import pytest

from MLMCPy.model import WrapperModel

class DummyModel(WrapperModel):
    def __init__(self, cost=100):
        self.cost = cost


def test_attach_model_exception():
    wrapper_model = WrapperModel()

    with pytest.raises(TypeError):
        wrapper_model.attach_model('Not a Model')


def test_cost_attribute():
    wrapper_model = WrapperModel()
    model = DummyModel()

    wrapper_model.attach_model(model)

    assert wrapper_model.cost == 100