import math
import numpy as np
import numpy.core.numeric as _nx
from numpy.core import linspace, atleast_1d, atleast_2d, transpose
#from numpy.core.numeric import (
#    ones, zeros, arange, concatenate, array, asarray, asanyarray, empty,
#    empty_like, ndarray, around, floor, ceil, take, dot, where, intp, multiply,
#    integer, isscalar, absolute, AxisError
#    )
from numpy.core.umath import (
    pi, add, arctan2, frompyfunc, cos, less_equal, sqrt, sin,
    mod, exp, log10, not_equal, subtract
    )
from numpy.core.fromnumeric import (
    ravel, nonzero, sort, partition, mean, any, sum
    )
from numpy.core.numerictypes import typecodes, number
from numpy.lib.twodim_base import diag

from numpy.core.multiarray import (
    _insert, add_docstring, digitize, bincount, normalize_axis_index,
    interp as compiled_interp, interp_complex as compiled_interp_complex
    )
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc
from numpy.compat import long
from numpy.compat.py3k import basestring
from numpy.core import iinfo, transpose

from numpy.core.numeric import (
    absolute, asanyarray, arange, zeros, greater_equal, multiply, ones,
    asarray, where, int8, int16, int32, int64, empty, promote_types, diagonal,
    nonzero
    )

import operator


class nd_grid(object):
    """
    Construct a multi-dimensional "meshgrid".

    ``grid = nd_grid()`` creates an instance which will return a mesh-grid
    when indexed.  The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each returned
    argument is greater than 1.

    Parameters
    ----------
    sparse : bool, optional
        Whether the grid is sparse or not. Default is False.

    Notes
    -----
    Two instances of `nd_grid` are made available in the NumPy namespace,
    `mgrid` and `ogrid`::

        mgrid = nd_grid(sparse=False)
        ogrid = nd_grid(sparse=True)

    Users should use these pre-defined instances instead of using `nd_grid`
    directly.

    Examples
    --------
    >>> mgrid = np.lib.index_tricks.nd_grid()
    >>> mgrid[0:5,0:5]
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])
    >>> mgrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    >>> ogrid = np.lib.index_tricks.nd_grid(sparse=True)
    >>> ogrid[0:5,0:5]
    [array([[0],
            [1],
            [2],
            [3],
            [4]]), array([[0, 1, 2, 3, 4]])]

    """

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, key):
        try:
            size = []
            typ = int
            for k in range(len(key)):
                step = key[k].step
                start = key[k].start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, complex):
                    size.append(int(abs(step)))
                    typ = float
                else:
                    size.append(
                        int(math.ceil((key[k].stop - start)/(step*1.0))))
                if (isinstance(step, float) or
                        isinstance(start, float) or
                        isinstance(key[k].stop, float)):
                    typ = float
            if self.sparse:
                nn = [_nx.arange(_x, dtype=_t)
                        for _x, _t in zip(size, (typ,)*len(size))]
            else:
                nn = _nx.indices(size, typ)
            for k in range(len(size)):
                step = key[k].step
                start = key[k].start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, complex):
                    step = int(abs(step))
                    if step != 1:
                        step = (key[k].stop - start)/float(step-1)
                nn[k] = (nn[k]*step+start)
            if self.sparse:
                slobj = [_nx.newaxis]*len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None, None)
                    nn[k] = nn[k][slobj]
                    slobj[k] = _nx.newaxis
            return nn
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, complex):
                step = abs(step)
                length = int(step)
                if step != 1:
                    step = (key.stop-start)/float(step-1)
                stop = key.stop + step
                return _nx.arange(0, length, 1, float)*step + start
            else:
                return _nx.arange(start, stop, step)

    def __len__(self):
        return 0

