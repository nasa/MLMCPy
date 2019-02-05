import numpy as np
from scipy.integrate import odeint

from MLMCPy.model import Model


class SpringMassModel(Model):
    """
    Defines Spring Mass model with 1 free param (stiffness of spring, k). The
    quantity of interest that is returned by the evaluate() function is the
    maximum displacement over the specified time interval
    """

    def __init__(self, mass=1.5, gravity=9.8, state0=None, time_step=None, 
                 cost=None):

        self._mass = mass
        self._gravity = gravity

        # Give default initial conditions & time grid if not specified
        if state0 is None:
            state0 = [0.0, 0.0]
        if time_step is None:
            time_grid = np.arange(0.0, 10.0, 0.1)
        else:
            time_grid = np.arange(0.0, 10.0, time_step)

        self._state0 = state0
        self._t = time_grid
        self.cost = cost

    def simulate(self, stiffness):
        """
        Simulate spring mass system for given spring constant. Returns state
        (position, velocity) at all points in time grid
        """
        return odeint(self._integration_func, self._state0, self._t,
                      args=(stiffness, self._mass, self._gravity))

    def evaluate(self, inputs):
        """
        Returns the max displacement over the course of the simulation.
        MLMCPy convention is that evaluated takes in an array and returns an
        array (even for 1D examples like this one).
        """
        stiffness = inputs[0]
        state = self.simulate(stiffness)
        return np.array([max(state[:, 0])])

    @staticmethod 
    def _integration_func(state, t, k, m, g):
        """
        Return velocity/acceleration given velocity/position and values for
        stiffness and mass. Helper function for numerical integrator
        """

        # unpack the state vector
        x = state[0]
        xd = state[1]

        # compute acceleration xdd
        xdd = ((-k * x) / m) + g

        # return the two state derivatives
        return [xd, xdd]
