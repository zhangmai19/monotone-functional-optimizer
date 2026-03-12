import numpy as np


def H_theta(l, theta):
    return 1.0 - np.exp(-l / theta)


def g_phi(p, phi):
    return p ** phi