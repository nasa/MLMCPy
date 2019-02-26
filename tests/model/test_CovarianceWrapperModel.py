import pytest
import numpy as np

from MLMCPy.model.Model import Model
from MLMCPy.model.CovarianceWrapperModel import CovarianceWrapperModel


class DummyModel(Model):
    def __init__(self, output_size, cost=10):
        self._output_size = output_size
        self.cost = cost

    def evaluate(self, inputs):
        output = np.zeros(self._output_size)
        for i in range(self._output_size):
            if i % 3 == 0:
                output[i] = inputs[0]
            elif i % 3 == 1:
                output[i] = 1 - inputs[0]
        return output


@pytest.mark.parametrize("output_size", range(1, 5))
def test_covariance_correct_size(output_size):
    model = DummyModel(output_size)
    covariance_model = CovarianceWrapperModel(model)

    cov_size = covariance_model.evaluate(np.zeros(1)).size
    expected_cov_size = output_size*(output_size + 3) / 2
    assert cov_size == expected_cov_size


def test_init_exception_for_bad_parameters():
    with pytest.raises(TypeError):
        CovarianceWrapperModel('Not a Model')


def test_model_cost_attribute():
    model = DummyModel(3)
    covariance_model = CovarianceWrapperModel(model)
    print model.cost
    assert covariance_model.cost == 10


def test_covariance_correct_values():
    model = DummyModel(3)
    covariance_model = CovarianceWrapperModel(model)

    cov_output = covariance_model.evaluate(np.array([0.2]))

    np.testing.assert_array_almost_equal(cov_output,
                                         [0.2, 0.8, 0.0, 0.04, 0.16, 0.0,
                                          0.64, 0.0, 0.0])


def test_covariance_post_processing():
    model = DummyModel(3)
    covariance_model = CovarianceWrapperModel(model)
    expected_values = np.array([0.2, 0.8, 0.0, 0.04, 0.16, 0.0, 0.64, 0.0,
                                0.0])
    covariance = covariance_model.post_process_covariance(expected_values)
    expected_covariance = np.zeros(6)

    np.testing.assert_array_almost_equal(covariance, expected_covariance)


def test_raises_error_wrong_sized_expected_values():
    model = DummyModel(3)
    covariance_model = CovarianceWrapperModel(model)
    expected_values = np.array([0.2, 0.8, 0.0])

    with pytest.raises(TypeError):
        _ = covariance_model.post_process_covariance(expected_values)


def test_covariance_end_to_end():
    model = DummyModel(3)
    covariance_model = CovarianceWrapperModel(model)

    MC_output = [covariance_model.evaluate(np.random.random(1))
                 for _ in range(50)]
    MC_output = np.array(MC_output)

    expected_values = np.mean(MC_output, axis=0)
    covariance = covariance_model.post_process_covariance(expected_values)

    assert covariance[0] == pytest.approx(covariance[3])
    assert covariance[1] == pytest.approx(-1*covariance[0])
    assert covariance[2] == pytest.approx(0)
    assert covariance[4] == pytest.approx(0)
    assert covariance[5] == pytest.approx(0)
