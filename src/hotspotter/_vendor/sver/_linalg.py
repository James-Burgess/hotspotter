import numpy as np
import numpy.linalg as npl

TRANSFORM_DTYPE = np.float64


def svd(M):
    import cv2

    flags = cv2.SVD_FULL_UV
    S, U, Vt = cv2.SVDecomp(M, flags=flags)
    s = S.flatten()
    return U, s, Vt


def rotation_mat3x3(radians, sin=np.sin, cos=np.cos):
    sin_ = sin(radians)
    cos_ = cos(radians)
    R = np.array(
        (
            (cos_, -sin_, 0),
            (sin_, cos_, 0),
            (0, 0, 1),
        )
    )
    return R


def rotation_mat2x2(theta):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    rot_ = np.array(
        (
            (cos_, -sin_),
            (sin_, cos_),
        )
    )
    return rot_


def translation_mat3x3(x, y, dtype=TRANSFORM_DTYPE):
    T = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=dtype)
    return T


def scale_mat3x3(sx, sy=None, dtype=TRANSFORM_DTYPE):
    sy = sx if sy is None else sy
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=dtype)
    return S


def whiten_xy_points(xy_m):
    mu_xy = xy_m.mean(1)
    std_xy = xy_m.std(1)
    std_xy[std_xy == 0] = 1
    tx, ty = -mu_xy / std_xy
    sx, sy = 1 / std_xy
    T = np.array([(sx, 0, tx), (0, sy, ty), (0, 0, 1)])
    xy_norm = ((xy_m.T - mu_xy) / std_xy).T
    return xy_norm, T


def add_homogenous_coordinate(_xys):
    assert _xys.shape[0] == 2
    _zs = np.ones((1, _xys.shape[1]), dtype=_xys.dtype)
    _xyzs = np.vstack((_xys, _zs))
    return _xyzs


def remove_homogenous_coordinate(_xyzs):
    assert _xyzs.shape[0] == 3
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _xys = np.divide(_xyzs[0:2], _xyzs[None, 2])
    return _xys


def transform_points_with_homography(H, _xys):
    xyz = add_homogenous_coordinate(_xys)
    xyz_t = np.matmul(H, xyz)
    xy_t = remove_homogenous_coordinate(xyz_t)
    return xy_t
