import abc


class Input(object):
    """
    Abstract base class defining data inputs from which samples can be drawn.
    """
    @abc.abstractmethod
    def draw_samples(self, num_samples):
        """
        Draws requested number of samples from a data source.

        :param num_samples: Number of sample rows to be returned.
        :type num_samples: int
        :return: A ndarray with num_samples rows. Can be one or two dimensional.
        """
        return

    @abc.abstractmethod
    def reset_sampling(self):
        """
        Used to reset the sample index to 0 when drawing from an indexed data
        set such as an ndarray extracted from a data file. Does not need to
        perform any actions for some data sources, for example random
        distributions as in the RandomInput class.
        """
        pass
