cimport cython
from cython import boundscheck

import numpy as np
cimport numpy as np

ctypedef fused number:
    cython.double
    cython.long

@boundscheck(False)
def euclidean_distance(number[:, :] X, number[:] Y, number[:] XX=None):
    '''
    Computes the euclidean distance between a 2d array of numbers (`X`)
    and a 1d array of numbers (`Y`), which has dimensions of a column in
    `X`. This function computes the euclidean distance between them.
    Another 1d array of numbers (`XX`) can be a precomputed dot product
    of X with itself. This can provide substantial speedup when X is
    large.

    This implementation has ~2/3 of the runtime of the sklearn function
    with a similar name (although with has a slightly different function
    signature).
    '''

    if XX is not None:
        dists = np.copy(XX)
    else:
        dists = np.einsum('ij,ij->i', X, X)

    dists += np.dot(Y, Y)
    dists -= 2*np.dot(X, Y)

    # numerical instability sometimes makes near-zero distances a small
    # negative number here.
    dists[dists < 0] = 0

    return np.array(np.sqrt(dists))
