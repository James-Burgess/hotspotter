import ctypes as C
import numpy as np
from collections import OrderedDict
from os.path import realpath, dirname

from . import _ctypes_interface

kpts_dtype = np.float32
vecs_dtype = np.uint8
img_dtype = np.uint8

obj_t = C.c_void_p
str_t = C.c_char_p
int_t = C.c_int
bool_t = C.c_bool
float_t = C.c_float

FLAGS_RW = str("aligned, c_contiguous, writeable")
FLAGS_RO = str("aligned, c_contiguous")
kpts_t = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags=FLAGS_RW)
vecs_t = np.ctypeslib.ndpointer(dtype=vecs_dtype, ndim=2, flags=FLAGS_RW)
img_t = np.ctypeslib.ndpointer(dtype=img_dtype, ndim=3, flags=FLAGS_RO)
img32_dtype = np.float32
img32_t = np.ctypeslib.ndpointer(dtype=img32_dtype, ndim=3, flags=FLAGS_RO)
int_array_t = np.ctypeslib.ndpointer(dtype=int_t, ndim=1, flags=FLAGS_RW)
str_list_t = C.POINTER(str_t)

HESAFF_TYPED_PARAMS = [
    (int_t, "numberOfScales", 3),
    (float_t, "threshold", 16.0 / 3.0),
    (float_t, "edgeEigenValueRatio", 10.0),
    (int_t, "border", 5),
    (int_t, "maxPyramidLevels", -1),
    (int_t, "maxIterations", 16),
    (float_t, "convergenceThreshold", 0.05),
    (int_t, "smmWindowSize", 19),
    (float_t, "mrSize", 3.0 * np.sqrt(3.0)),
    (int_t, "spatialBins", 4),
    (int_t, "orientationBins", 8),
    (float_t, "maxBinValue", 0.2),
    (float_t, "initialSigma", 1.6),
    (int_t, "patchSize", 41),
    (float_t, "scale_min", -1.0),
    (float_t, "scale_max", -1.0),
    (bool_t, "rotation_invariance", False),
    (bool_t, "augment_orientation", False),
    (float_t, "ori_maxima_thresh", 0.8),
    (bool_t, "affine_invariance", True),
    (bool_t, "only_count", False),
    (bool_t, "use_dense", False),
    (int_t, "dense_stride", 32),
    (float_t, "siftPower", 1.0),
]

HESAFF_PARAM_DICT = OrderedDict(
    [(key, val) for (type_, key, val) in HESAFF_TYPED_PARAMS]
)
HESAFF_PARAM_TYPES = [type_ for (type_, key, val) in HESAFF_TYPED_PARAMS]


def _load_hesaff_clib():
    root_dir = realpath(dirname(__file__))
    libname = "hesaff"
    clib, def_cfunc, lib_fpath = _ctypes_interface.load_clib(libname, root_dir)
    def_cfunc(int_t, "get_cpp_version", [])
    def_cfunc(int_t, "is_debug_mode", [])
    def_cfunc(int_t, "detect", [obj_t])
    def_cfunc(int_t, "get_kpts_dim", [])
    def_cfunc(int_t, "get_desc_dim", [])
    def_cfunc(None, "exportArrays", [obj_t, int_t, kpts_t, vecs_t])
    def_cfunc(None, "extractDesc", [obj_t, int_t, kpts_t, vecs_t])
    def_cfunc(None, "extractPatches", [obj_t, int_t, kpts_t, img32_t])
    def_cfunc(None, "extractDescFromPatches", [int_t, int_t, int_t, img_t, vecs_t])
    def_cfunc(obj_t, "new_hesaff_fpath", [str_t] + HESAFF_PARAM_TYPES)
    def_cfunc(
        obj_t, "new_hesaff_image", [img_t, int_t, int_t, int_t] + HESAFF_PARAM_TYPES
    )
    def_cfunc(None, "free_hesaff", [obj_t])
    return clib, lib_fpath


try:
    HESAFF_CLIB, __LIB_FPATH__ = _load_hesaff_clib()
    KPTS_DIM = HESAFF_CLIB.get_kpts_dim()
    DESC_DIM = HESAFF_CLIB.get_desc_dim()
except (ImportError, AttributeError):
    import warnings

    warnings.warn("Unable to load C library for Hesaff")
    HESAFF_CLIB = None
    __LIB_FPATH__ = None
    KPTS_DIM = None
    DESC_DIM = None


def alloc_kpts(nKpts):
    return np.empty((nKpts, KPTS_DIM), kpts_dtype)


def alloc_vecs(nKpts):
    return np.empty((nKpts, DESC_DIM), vecs_dtype)


def _make_hesaff_cpp_params(kwargs):
    hesaff_params = HESAFF_PARAM_DICT.copy()
    for key, val in kwargs.items():
        if key in hesaff_params:
            hesaff_params[key] = val
        else:
            print("[pyhesaff] WARNING: key=%r is not known" % key)
    return hesaff_params


def _new_image_hesaff(img, **kwargs):
    hesaff_params = _make_hesaff_cpp_params(kwargs)
    hesaff_args = list(hesaff_params.values())
    rows, cols = img.shape[0:2]
    if len(img.shape) == 2:
        channels = 1
    else:
        channels = img.shape[2]
    hesaff_ptr = HESAFF_CLIB.new_hesaff_image(img, rows, cols, channels, *hesaff_args)
    return hesaff_ptr


def get_hesaff_default_params():
    return HESAFF_PARAM_DICT.copy()


def detect_feats_in_image(img, **kwargs):
    hesaff_ptr = _new_image_hesaff(img, **kwargs)
    nKpts = HESAFF_CLIB.detect(hesaff_ptr)
    kpts = alloc_kpts(nKpts)
    vecs = alloc_vecs(nKpts)
    HESAFF_CLIB.exportArrays(hesaff_ptr, nKpts, kpts, vecs)
    HESAFF_CLIB.free_hesaff(hesaff_ptr)
    return kpts, vecs
