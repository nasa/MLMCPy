import abc


class Input(object):

    @abc.abstractmethod
    def draw_samples(self, num_samples):
        return

    def reset_sampling(self):
        return
