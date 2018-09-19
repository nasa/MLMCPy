import pytest

from MLMCPy.input import RandomInput


def test_init_invalid_input():

    with pytest.raises(TypeError):
        RandomInput([])
        RandomInput(1)
        RandomInput('five')

    with pytest.raises(ValueError):
        RandomInput(np.zeros(0))


