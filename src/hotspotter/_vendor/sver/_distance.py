import numpy as np

TEMP_VEC_DTYPE = np.float64


def ori_distance(ori1, ori2, out=None):
    from ._util_math import TAU

    return cyclic_distance(ori1, ori2, modulo=TAU, out=out)


def cyclic_distance(arr1, arr2, modulo, out=None):
    arr_diff = np.subtract(arr1, arr2, out=out)
    abs_diff = np.abs(arr_diff, out=out)
    mod_diff1 = np.mod(abs_diff, modulo, out=out)
    mod_diff2 = np.subtract(modulo, mod_diff1)
    arr_dist = np.minimum(mod_diff1, mod_diff2, out=out)
    return arr_dist


def det_distance(det1, det2):
    det_dist = det1 / det2
    _flip_flag = det_dist < 1
    det_dist[_flip_flag] = np.reciprocal(det_dist[_flip_flag])
    return det_dist


def L2_sqrd(hist1, hist2, dtype=TEMP_VEC_DTYPE):
    hist1_ = np.asarray(hist1, dtype)
    hist2_ = np.asarray(hist2, dtype)
    return ((hist1_ - hist2_) ** 2).sum(-1)
