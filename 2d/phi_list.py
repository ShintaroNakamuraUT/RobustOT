import numpy as np


# \beta-potentials
class beta_phi_prime:
    def __init__(self, beta):
        self.beta = beta

    def value(self, pi):
        return (1 / (self.beta - 1)) * (pi ** (self.beta - 1) - np.ones_like(pi))


class beta_psi_prime:
    def __init__(self, beta):
        self.beta = beta

    def value(self, theta):
        return ((self.beta - 1) * theta + np.ones_like(theta)) ** (1 / (self.beta - 1))


class beta_psi_two_prime:
    def __init__(self, beta):
        self.beta = beta

    def value(self, theta):
        return ((self.beta - 1) * theta + np.ones_like(theta)) ** (
            1 / (self.beta - 1) - 1
        )
