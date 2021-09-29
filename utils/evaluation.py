"""evaluation metrics

This module containes functions which implements area under the budget curve (AUBC) [1]_.

Reference
---------

..[1] Xueying ZHAN "A Comparative Survey: Benchmarking for Pool-based Active Learning".

"""

import numpy as np

import pdb

def AUBC(budget, performance):
    """AUBC

    This funciton implements area under the budget curve.

    Parameters
    ----------
    budget: numpy.ndarray, sorted by ascending.
    performance: numpy.ndarray

    """
    return np.trapz(y=performance, x=budget) / budget.shape
