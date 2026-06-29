import numpy as np
import numpy.linalg as npl
from ._util_math import TAU
from . import _linalg as linalgtool
from . import _distance

GRAVITY_THETA = TAU / 4
KPTS_DTYPE = np.float32

XDIM = 0
YDIM = 1
SCAX_DIM = 2
SKEW_DIM = 3
SCAY_DIM = 4
ORI_DIM = 5
LOC_DIMS = np.array([XDIM, YDIM])
SHAPE_DIMS = np.array([SCAX_DIM, SKEW_DIM, SCAY_DIM])


def get_xys(kpts):
    return kpts.T[0:2]


def get_invVs(kpts):
    return kpts.T[2:5]


def get_oris(kpts):
    if kpts.shape[1] == 5:
        return np.zeros(len(kpts), dtype=kpts.dtype)
    elif kpts.shape[1] == 6:
        return kpts.T[5]
    else:
        raise AssertionError("[ktool] Invalid kpts.shape = %r" % (kpts.shape,))


def get_sqrd_scales(kpts):
    if len(kpts) == 0:
        return np.empty(0)
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    return np.multiply(_iv11s, _iv22s)


def get_scales(kpts):
    return np.sqrt(get_sqrd_scales(kpts))


def get_ori_mats(kpts):
    _oris = get_oris(kpts)
    R_mats = [linalgtool.rotation_mat2x2(ori) for ori in _oris]
    return R_mats


def get_invV_mats2x2(kpts):
    nKpts = len(kpts)
    _iv11s, _iv21s, _iv22s = get_invVs(kpts)
    _zeros = np.zeros(nKpts)
    invV_arrs2x2 = np.array([[_iv11s, _zeros], [_iv21s, _iv22s]])
    invV_mats2x2 = np.rollaxis(invV_arrs2x2, 2)
    return invV_mats2x2


def get_invVR_mats2x2(kpts):
    if len(kpts) == 0:
        return np.empty((0, 2, 2))
    invV_mats2x2 = get_invV_mats2x2(kpts)
    R_mats2x2 = get_ori_mats(kpts)
    invVR_mats2x2 = np.matmul(invV_mats2x2, R_mats2x2)
    return invVR_mats2x2


def augment_2x2_with_translation(kpts, _mat2x2):
    nKpts = len(kpts)
    _11s = _mat2x2.T[0, 0]
    _12s = _mat2x2.T[1, 0]
    _21s = _mat2x2.T[0, 1]
    _22s = _mat2x2.T[1, 1]
    _13s, _23s = get_xys(kpts)
    _zeros = np.zeros(nKpts)
    _ones = np.ones(nKpts)
    _arrs3x3 = np.array(
        [[_11s, _12s, _13s], [_21s, _22s, _23s], [_zeros, _zeros, _ones]]
    )
    _mats3x3 = np.rollaxis(_arrs3x3, 2)
    return _mats3x3


def get_invV_mats3x3(kpts):
    invV_mats2x2 = get_invV_mats2x2(kpts)
    invV_mats3x3 = augment_2x2_with_translation(kpts, invV_mats2x2)
    return invV_mats3x3


def get_invVR_mats3x3(kpts):
    invVR_mats2x2 = get_invVR_mats2x2(kpts)
    invVR_mats3x3 = augment_2x2_with_translation(kpts, invVR_mats2x2)
    return invVR_mats3x3


def get_RV_mats_3x3(kpts):
    invVR_mats = get_invVR_mats3x3(kpts)
    RV_mats = invert_invV_mats(invVR_mats)
    return RV_mats


def invert_invV_mats(invV_mats):
    try:
        V_mats = npl.inv(invV_mats)
    except npl.LinAlgError:
        V_mats_list = [None for _ in range(len(invV_mats))]
        for ix, invV in enumerate(invV_mats):
            try:
                V_mats_list[ix] = npl.inv(invV)
            except npl.LinAlgError:
                print("ERROR: invV_mats[%d] could not be inverted" % ix)
                V_mats_list[ix] = np.nan * np.ones(invV.shape)
        V_mats = np.array(V_mats_list)
    return V_mats


def get_invVR_mats_xys(invVR_mats):
    return invVR_mats.T[2, 0:2]


def get_invVR_mats_sqrd_scale(invVR_mats):
    return npl.det(invVR_mats[:, 0:2, 0:2])


def get_invVR_mats_oris(invVR_mats):
    _iv11s = invVR_mats.T[0, 0]
    _iv12s = invVR_mats.T[1, 0]
    _oris = (-np.arctan2(_iv12s, _iv11s)) % TAU
    return _oris


def get_invVR_mats_shape(invVR_mats):
    _iv11s = invVR_mats[:, 0, 0]
    _iv12s = invVR_mats[:, 0, 1]
    _iv21s = invVR_mats[:, 1, 0]
    _iv22s = invVR_mats[:, 1, 1]
    return (_iv11s, _iv12s, _iv21s, _iv22s)


def rectify_invV_mats_are_up(invVR_mats):
    _oris = get_invVR_mats_oris(invVR_mats)
    _a, _b, _c, _d = get_invVR_mats_shape(invVR_mats)
    det_ = np.sqrt(np.abs((_a * _d) - (_b * _c)))
    b2a2 = np.sqrt((_b**2) + (_a**2))
    iv11 = b2a2 / det_
    iv21 = ((_d * _b) + (_c * _a)) / (b2a2 * det_)
    iv22 = det_ / b2a2
    invV_mats = invVR_mats.copy()
    invV_mats[:, 0, 0] = iv11 * det_
    invV_mats[:, 0, 1] = 0
    invV_mats[:, 1, 0] = iv21 * det_
    invV_mats[:, 1, 1] = iv22 * det_
    return invV_mats, _oris


def flatten_invV_mats_to_kpts(invV_mats):
    invV_mats, _oris = rectify_invV_mats_are_up(invV_mats)
    _xs = invV_mats[:, 0, 2]
    _ys = invV_mats[:, 1, 2]
    _iv11s = invV_mats[:, 0, 0]
    _iv21s = invV_mats[:, 1, 0]
    _iv22s = invV_mats[:, 1, 1]
    kpts = np.vstack((_xs, _ys, _iv11s, _iv21s, _iv22s, _oris)).T
    return kpts


def get_kpts_wh(kpts, outer=True):
    if outer:
        invV_mats2x2 = get_invVR_mats2x2(kpts)
        corners = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
        warped_corners = np.array([invV.dot(corners) for invV in invV_mats2x2])
        maxx = warped_corners[:, 0, :].max(axis=1)
        minx = warped_corners[:, 0, :].min(axis=1)
        maxy = warped_corners[:, 1, :].max(axis=1)
        miny = warped_corners[:, 1, :].min(axis=1)
    else:
        a = kpts.T[2]
        c = kpts.T[3]
        d = kpts.T[4]
        x_crit_u = np.array([[1], [-1]])
        x_crit_v = np.array([[0], [0]])
        x_crit_x = a * x_crit_u
        x_crit_y = c * x_crit_u + d * x_crit_v
        part = np.sqrt(c**2 + d**2)
        y_crit_thetas1 = -2 * np.arctan((c + part) / d)
        y_crit_thetas2 = -2 * np.arctan((c - part) / d)
        y_crit_thetas = np.vstack((y_crit_thetas1, y_crit_thetas2))
        y_crit_u = np.cos(y_crit_thetas)
        y_crit_v = np.sin(y_crit_thetas)
        y_crit_x = a * y_crit_u
        y_crit_y = c * y_crit_u + d * y_crit_v
        crit_x = np.vstack([y_crit_x, x_crit_x])
        crit_y = np.vstack([y_crit_y, x_crit_y])
        maxx = crit_x.max(axis=0)
        minx = crit_x.min(axis=0)
        maxy = crit_y.max(axis=0)
        miny = crit_y.min(axis=0)
    w = maxx - minx
    h = maxy - miny
    wh_list = np.vstack([w, h]).T
    return wh_list


def get_kpts_image_extent(kpts, outer=False, only_xy=False):
    if len(kpts) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    xs, ys = get_xys(kpts)
    if only_xy:
        minx = xs.min()
        maxx = xs.max()
        miny = ys.min()
        maxy = ys.max()
    else:
        wh_list = get_kpts_wh(kpts, outer=outer)
        radii = np.divide(wh_list, 2, out=wh_list)
        minx = (xs - radii.T[0]).min()
        maxx = (xs + radii.T[0]).max()
        miny = (ys - radii.T[1]).min()
        maxy = (ys + radii.T[1]).max()
    extent = (minx, maxx, miny, maxy)
    return extent


def get_kpts_dlen_sqrd(kpts, outer=False):
    if len(kpts) == 0:
        return 0.0
    extent = get_kpts_image_extent(kpts, outer=outer)
    x1, x2, y1, y2 = extent
    w = x2 - x1
    h = y2 - y1
    return (w**2) + (h**2)
