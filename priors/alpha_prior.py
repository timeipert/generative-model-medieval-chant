import pytensor.tensor as pt
import numpy as np

def transpose_alpha(alpha_base, m):
    index = 3 + m
    return pt.set_subtensor(alpha_base[index], 1.0)


def get_alpha_m(m, size):
    alpha_base = pt.zeros(size) + 0.01
    return transpose_alpha(alpha_base, m)


def get_d2g_alpha(size, start=3, end=6):
    mask = np.full(size, 0.01)
    mask[start:end + 1] = 1.0
    return mask
