from MLMCPy.input.Input import Input


class TestingInput(Input):

    def __init__(self, data):

        self._data = data
        if len(self._data .shape) == 1:
            self._data = self._data.reshape(self._data.shape[0], -1)
        self._index = 0

    def draw_samples(self, num_samples):

        sample = self._data[self._index: self._index + num_samples]

        self._index += num_samples

        return sample

    def reset_sampling(self):

        self._index = 0
