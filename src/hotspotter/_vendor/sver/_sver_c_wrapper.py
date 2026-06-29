import ctypes as C
import os
import platform
import sys
import warnings
from os.path import dirname, join

import numpy as np

c_double_p = C.POINTER(C.c_double)

kpts_dtype = np.float64
WIN32 = platform.system() == "Windows"
fm_dtype = np.int32 if WIN32 else np.int64
fs_dtype = np.float64
FLAGS_RW = "aligned, c_contiguous, writeable"
FLAGS_RO = "aligned, c_contiguous"

kpts_t = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags=FLAGS_RO)
fm_t = np.ctypeslib.ndpointer(dtype=fm_dtype, ndim=2, flags=FLAGS_RO)
fs_t = np.ctypeslib.ndpointer(dtype=fs_dtype, ndim=1, flags=FLAGS_RO)


def inliers_t(ndim):
    return np.ctypeslib.ndpointer(dtype=np.bool_, ndim=ndim, flags=FLAGS_RW)


def errs_t(ndim):
    return np.ctypeslib.ndpointer(dtype=np.float64, ndim=ndim, flags=FLAGS_RW)


def mats_t(ndim):
    return np.ctypeslib.ndpointer(dtype=np.float64, ndim=ndim, flags=FLAGS_RW)


def get_lib_ext():
    if WIN32:
        return ".dll"
    elif sys.platform == "darwin":
        return ".dylib"
    else:
        return ".so"


_LIB_DIR = join(dirname(__file__), "_sver_cpp")
_LIB_FNAME = join(_LIB_DIR, "libsver" + get_lib_ext())

if os.path.exists(_LIB_FNAME):
    try:
        c_sver = C.cdll[_LIB_FNAME]
    except Exception:
        print("Failed to open lib_fname = %r" % (_LIB_FNAME,))
        if os.path.exists(_LIB_FNAME):
            print("  library exists but cannot be loaded")
        raise
    c_getaffineinliers = c_sver["get_affine_inliers"]
    c_getaffineinliers.restype = C.c_int
    c_getaffineinliers.argtypes = [
        kpts_t,
        C.c_size_t,
        kpts_t,
        C.c_size_t,
        fm_t,
        fs_t,
        C.c_size_t,
        C.c_double,
        C.c_double,
        C.c_double,
        inliers_t(2),
        errs_t(3),
        mats_t(3),
    ]
    c_getbestaffineinliers = c_sver["get_best_affine_inliers"]
    c_getbestaffineinliers.restype = C.c_int
    c_getbestaffineinliers.argtypes = [
        kpts_t,
        C.c_size_t,
        kpts_t,
        C.c_size_t,
        fm_t,
        fs_t,
        C.c_size_t,
        C.c_double,
        C.c_double,
        C.c_double,
        inliers_t(1),
        errs_t(2),
        mats_t(2),
    ]
else:
    warnings.warn("Unable to load C library for sver at %s" % _LIB_FNAME)
    c_sver = None


def get_affine_inliers_cpp(
    kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh
):
    num_matches = len(fm)
    fm = np.ascontiguousarray(fm, dtype=fm_dtype)
    out_inlier_flags = np.empty((num_matches, num_matches), np.bool_)
    out_errors = np.empty((num_matches, 3, num_matches), np.float64)
    out_mats = np.empty((num_matches, 3, 3), np.float64)
    c_getaffineinliers(
        kpts1,
        kpts1.size,
        kpts2,
        kpts2.size,
        fm,
        fs,
        len(fm),
        xy_thresh_sqrd,
        scale_thresh_sqrd,
        ori_thresh,
        out_inlier_flags,
        out_errors,
        out_mats,
    )
    out_inliers = [np.where(row)[0] for row in out_inlier_flags]
    out_errors = list(map(tuple, out_errors))
    return out_inliers, out_errors, out_mats


def get_best_affine_inliers_cpp(
    kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh
):
    fm = np.ascontiguousarray(fm, dtype=fm_dtype)
    out_inlier_flags = np.empty((len(fm),), np.bool_)
    out_errors = np.empty((3, len(fm)), np.float64)
    out_mat = np.empty((3, 3), np.float64)
    c_getbestaffineinliers(
        kpts1,
        6 * len(kpts1),
        kpts2,
        6 * len(kpts2),
        fm,
        fs,
        len(fm),
        xy_thresh_sqrd,
        scale_thresh_sqrd,
        ori_thresh,
        out_inlier_flags,
        out_errors,
        out_mat,
    )
    out_inliers = np.where(out_inlier_flags)[0]
    out_errors = tuple(out_errors)
    return out_inliers, out_errors, out_mat
