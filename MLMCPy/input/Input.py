import abc


class Input(object):

    @abc.abstractmethod
    def draw_samples(self, num_samples):
        """
        Should return ndarray with num_samples rows.
        """
        return

    @abc.abstractmethod
    def reset_sampling(self):
        """
        If possible, resets sampling index to 0.
        """
        pass
