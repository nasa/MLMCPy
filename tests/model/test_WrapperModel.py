import pytest

from MLMCPy.model import WrapperModel

class TestCostClass:

    def __init__(self, cost=10):
        self.cost = cost

def test_cost_attribute_initialization():
    cost_class = TestCostClass()
    wrapper_model = WrapperModel(cost_class)

    assert isinstance(wrapper_model.cost, int)