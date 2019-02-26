import pytest

from MLMCPy.model import WrapperModel

def test_attach_model_exception():
    wrapper_model = WrapperModel()

    with pytest.raises(NotImplementedError):
        wrapper_model.attach_model('Not a Model')