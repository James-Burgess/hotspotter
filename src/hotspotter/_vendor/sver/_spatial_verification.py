import os
import traceback
import warnings

import numpy as np
import numpy.linalg as npl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

from ._util_math import TAU
from . import _keypoint as ktool
from . import _linalg as ltool
from . import _distance

_VERBOSE_SVER = os.environ.get("HS_SVER_VERBOSE", "") == "1"
_NO_C_EXTENSION = os.environ.get("HS_SVER_NO_C", "") == "1"

try:
    from . import _sver_c_wrapper

    HAVE_SVER_C_WRAPPER = not _NO_C_EXTENSION
except Exception as ex:
    HAVE_SVER_C_WRAPPER = False
    if _VERBOSE_SVER:
        print("please build the sver c wrapper: %s" % ex)
        traceback.print_exc()


SV_DTYPE = np.float64
INDEX_DTYPE = np.int32


def build_lstsqrs_Mx9(xy1_mn, xy2_mn):
    x1_mn = xy1_mn[0]
    y1_mn = xy1_mn[1]
    x2_mn = xy2_mn[0]
    y2_mn = xy2_mn[1]
    num_pts = x1_mn.shape[0]
    Mx9 = np.empty((2 * num_pts, 9), dtype=SV_DTYPE)
    for ix in range(num_pts):
        u2 = x2_mn[ix]
        v2 = y2_mn[ix]
        x1 = x1_mn[ix]
        y1 = y1_mn[ix]
        d, e, f = (-x1, -y1, -1)
        g, h, i = (v2 * x1, v2 * y1, v2)
        j, k, l = (x1, y1, 1)
        p, q, r = (-u2 * x1, -u2 * y1, -u2)
        Mx9[ix * 2] = (0, 0, 0, d, e, f, g, h, i)
        Mx9[ix * 2 + 1] = (j, k, l, 0, 0, 0, p, q, r)
    return Mx9


def try_svd(M):
    try:
        USV = npl.svd(M, full_matrices=True, compute_uv=True)
    except MemoryError as ex:
        if _VERBOSE_SVER:
            print("[sver] Caught MemErr during full SVD. Trying sparse SVD.")
            traceback.print_exc()
        M_sparse = sps.lil_matrix(M)
        USV = spsl.svds(M_sparse)
    except npl.LinAlgError as ex:
        if _VERBOSE_SVER:
            print("[sver] svd did not converge")
            traceback.print_exc()
        raise
    except Exception as ex:
        if _VERBOSE_SVER:
            print("[sver] svd error")
            traceback.print_exc()
        raise
    return USV


def build_affine_lstsqrs_Mx6(xy1_man, xy2_man):
    x1_mn = xy1_man[0]
    y1_mn = xy1_man[1]
    x2_mn = xy2_man[0]
    y2_mn = xy2_man[1]
    num_pts = x1_mn.shape[0]
    Mx6 = np.empty((2 * num_pts, 6), dtype=SV_DTYPE)
    b = np.empty((2 * num_pts, 1), dtype=SV_DTYPE)
    for ix in range(num_pts):
        x1 = x1_mn[ix]
        x2 = x2_mn[ix]
        y1 = y1_mn[ix]
        y2 = y2_mn[ix]
        Mx6[ix * 2] = (x1, y1, 0, 0, 1, 0)
        Mx6[ix * 2 + 1] = (0, 0, x1, y1, 0, 1)
        b[ix * 2] = x2
        b[ix * 2 + 1] = y2
    U, s, Vt = try_svd(Mx6)
    Sinv = np.zeros((len(Vt), len(U)))
    Sinv[np.diag_indices(len(s))] = 1 / s
    a = Vt.T.dot(Sinv).dot(U.T).dot(b).T[0]
    A = np.array([[a[0], a[1], a[4]], [a[2], a[3], a[5]], [0, 0, 1]])
    return A


def compute_affine(xy1_man, xy2_man):
    A = build_affine_lstsqrs_Mx6(xy1_man, xy2_man)
    return A


def compute_homog(xy1_mn, xy2_mn):
    Mx9 = build_lstsqrs_Mx9(xy1_mn, xy2_mn)
    U, S, V = try_svd(Mx9)
    h = V[8]
    H = np.vstack((h[0:3], h[3:6], h[6:9]))
    return H


def get_normalized_affine_inliers(kpts1, kpts2, fm, aff_inliers):
    fm_affine = fm.take(aff_inliers, axis=0)
    kpts1_ma = kpts1.take(fm_affine.T[0], axis=0)
    kpts2_ma = kpts2.take(fm_affine.T[1], axis=0)
    xy1_ma = ktool.get_xys(kpts1_ma)
    xy2_ma = ktool.get_xys(kpts2_ma)
    xy1_man, T1 = ltool.whiten_xy_points(xy1_ma)
    xy2_man, T2 = ltool.whiten_xy_points(xy2_ma)
    return xy1_man, xy2_man, T1, T2


def unnormalize_transform(M_prime, T1, T2):
    M = npl.solve(T2, M_prime).dot(T1)
    M /= M[2, 2]
    return M


def estimate_refined_transform(kpts1, kpts2, fm, aff_inliers, refine_method="homog"):
    import cv2

    xy1_man, xy2_man, T1, T2 = get_normalized_affine_inliers(
        kpts1, kpts2, fm, aff_inliers
    )
    if refine_method == "homog":
        H_prime = compute_homog(xy1_man, xy2_man)
    elif refine_method == "affine":
        H_prime = compute_affine(xy1_man, xy2_man)
    elif refine_method == "cv2-homog":
        H_prime, mask = cv2.findHomography(xy1_man.T, xy2_man.T, method=0)
    elif refine_method == "cv2-ransac-homog":
        H_prime, mask = cv2.findHomography(
            xy1_man.T, xy2_man.T, method=cv2.RANSAC, ransacReprojThreshold=3
        )
    elif refine_method == "cv2-lmeds-homog":
        H_prime, mask = cv2.findHomography(xy1_man.T, xy2_man.T, method=cv2.LMEDS)
    else:
        raise NotImplementedError("[sver] Unknown refine_method=%r" % (refine_method,))
    H = unnormalize_transform(H_prime, T1, T2)
    rank = npl.matrix_rank(H)
    if rank != 3:
        raise npl.LinAlgError("Rank deficient homography")
    return H


def _test_hypothesis_inliers(
    Aff, invVR1s_m, xy2_m, det2_m, ori2_m, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh
):
    invVR1s_mt = np.matmul(Aff, invVR1s_m)
    _xy1_mt = ktool.get_invVR_mats_xys(invVR1s_mt)
    _det1_mt = ktool.get_invVR_mats_sqrd_scale(invVR1s_mt)
    _ori1_mt = ktool.get_invVR_mats_oris(invVR1s_mt)
    xy_err = _distance.L2_sqrd(xy2_m.T, _xy1_mt.T, dtype=SV_DTYPE)
    scale_err = _distance.det_distance(_det1_mt, det2_m)
    ori_err = _distance.ori_distance(_ori1_mt, ori2_m)
    xy_inliers_flag = np.less(xy_err, xy_thresh_sqrd)
    scale_inliers_flag = np.less(scale_err, scale_thresh_sqrd)
    ori_inliers_flag = np.less(ori_err, ori_thresh)
    hypo_inliers_flag = xy_inliers_flag
    np.logical_and(hypo_inliers_flag, ori_inliers_flag, out=hypo_inliers_flag)
    np.logical_and(hypo_inliers_flag, scale_inliers_flag, out=hypo_inliers_flag)
    hypo_inliers = np.where(hypo_inliers_flag)[0]
    hypo_errors = (xy_err, ori_err, scale_err)
    return hypo_inliers, hypo_errors


def get_affine_inliers(
    kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh
):
    kpts1_m = kpts1.take(fm.T[0], axis=0)
    kpts2_m = kpts2.take(fm.T[1], axis=0)
    invVR2s_m = ktool.get_invVR_mats3x3(kpts2_m)
    invVR1s_m = ktool.get_invVR_mats3x3(kpts1_m)
    RV1s_m = ktool.invert_invV_mats(invVR1s_m)
    Aff_mats = np.matmul(invVR2s_m, RV1s_m)
    xy2_m = ktool.get_xys(kpts2_m)
    det2_m = ktool.get_sqrd_scales(kpts2_m)
    ori2_m = ktool.get_oris(kpts2_m)
    inliers_and_errors_list = [
        _test_hypothesis_inliers(
            Aff,
            invVR1s_m,
            xy2_m,
            det2_m,
            ori2_m,
            xy_thresh_sqrd,
            scale_thresh_sqrd,
            ori_thresh,
        )
        for Aff in Aff_mats
    ]
    aff_inliers_list = [tup[0] for tup in inliers_and_errors_list]
    aff_errors_list = [tup[1] for tup in inliers_and_errors_list]
    return aff_inliers_list, aff_errors_list, Aff_mats


def get_best_affine_inliers(
    kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh, forcepy=False
):
    if HAVE_SVER_C_WRAPPER and not forcepy:
        (
            aff_inliers_list,
            aff_errors_list,
            Aff_mats,
        ) = _sver_c_wrapper.get_affine_inliers_cpp(
            kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh
        )
    else:
        aff_inliers_list, aff_errors_list, Aff_mats = get_affine_inliers(
            kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh
        )
    weight_list = np.array([fs.take(inliers).sum() for inliers in aff_inliers_list])
    best_index = weight_list.argmax()
    aff_inliers = aff_inliers_list[best_index]
    aff_errors = aff_errors_list[best_index]
    Aff = Aff_mats[best_index]
    return aff_inliers, aff_errors, Aff


def test_homog_errors(
    H,
    kpts1,
    kpts2,
    fm,
    xy_thresh_sqrd,
    scale_thresh,
    ori_thresh,
    full_homog_checks=True,
):
    kpts1_m = kpts1.take(fm.T[0], axis=0)
    kpts2_m = kpts2.take(fm.T[1], axis=0)
    xy1_m = ktool.get_xys(kpts1_m)
    xy1_mt = ltool.transform_points_with_homography(H, xy1_m)
    xy2_m = ktool.get_xys(kpts2_m)
    xy_err = _distance.L2_sqrd(xy1_mt.T, xy2_m.T)
    if full_homog_checks:
        oris1_m = ktool.get_oris(kpts1_m)
        scales1_m = ktool.get_scales(kpts1_m)
        dxy1_m = np.vstack((np.sin(oris1_m), -np.cos(oris1_m)))
        scaled_dxy1_m = dxy1_m * scales1_m[None, :]
        off_xy1_m = xy1_m + scaled_dxy1_m
        off_xy1_mt = ltool.transform_points_with_homography(H, off_xy1_m)
        scaled_dxy1_mt = xy1_mt - off_xy1_mt
        scales1_mt = npl.norm(scaled_dxy1_mt, axis=0)
        dxy1_mt = scaled_dxy1_mt / scales1_mt
        oris1_mt = np.arctan2(dxy1_mt[1], dxy1_mt[0]) - ktool.GRAVITY_THETA
        _det1_mt = scales1_mt**2
        det2_m = ktool.get_sqrd_scales(kpts2_m)
        ori2_m = ktool.get_oris(kpts2_m)
        scale_err = _distance.det_distance(_det1_mt, det2_m)
        ori_err = _distance.ori_distance(oris1_mt, ori2_m)
        xy_inliers_flag = np.less(xy_err, xy_thresh_sqrd)
        scale_inliers_flag = np.less(scale_err, scale_thresh)
        ori_inliers_flag = np.less(ori_err, ori_thresh)
        hypo_inliers_flag = xy_inliers_flag
        np.logical_and(hypo_inliers_flag, ori_inliers_flag, out=hypo_inliers_flag)
        np.logical_and(hypo_inliers_flag, scale_inliers_flag, out=hypo_inliers_flag)
        refined_inliers = np.where(hypo_inliers_flag)[0].astype(INDEX_DTYPE)
        refined_errors = (xy_err, ori_err, scale_err)
    else:
        refined_inliers = np.where(xy_err < xy_thresh_sqrd)[0].astype(INDEX_DTYPE)
        refined_errors = (xy_err, None, None)
    homog_tup1 = (refined_inliers, refined_errors, H)
    return homog_tup1


def test_affine_errors(
    H, kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh_sqrd, ori_thresh
):
    kpts1_m = kpts1.take(fm.T[0], axis=0)
    kpts2_m = kpts2.take(fm.T[1], axis=0)
    invVR1s_m = ktool.get_invVR_mats3x3(kpts1_m)
    xy2_m = ktool.get_xys(kpts2_m)
    det2_m = ktool.get_sqrd_scales(kpts2_m)
    ori2_m = ktool.get_oris(kpts2_m)
    refined_inliers, refined_errors = _test_hypothesis_inliers(
        H,
        invVR1s_m,
        xy2_m,
        det2_m,
        ori2_m,
        xy_thresh_sqrd,
        scale_thresh_sqrd,
        ori_thresh,
    )
    refined_tup1 = (refined_inliers, refined_errors, H)
    return refined_tup1


def refine_inliers(
    kpts1,
    kpts2,
    fm,
    aff_inliers,
    xy_thresh_sqrd,
    scale_thresh=2.0,
    ori_thresh=1.57,
    full_homog_checks=True,
    refine_method="homog",
):
    H = estimate_refined_transform(
        kpts1, kpts2, fm, aff_inliers, refine_method=refine_method
    )
    if refine_method.endswith("homog"):
        homog_tup1 = test_homog_errors(
            H,
            kpts1,
            kpts2,
            fm,
            xy_thresh_sqrd,
            scale_thresh,
            ori_thresh,
            full_homog_checks,
        )
    elif refine_method == "affine":
        homog_tup1 = test_affine_errors(
            H, kpts1, kpts2, fm, xy_thresh_sqrd, scale_thresh, ori_thresh
        )
    return homog_tup1


def get_best_affine_inliers_(
    kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh
):
    if HAVE_SVER_C_WRAPPER:
        aff_inliers, aff_errors, Aff = _sver_c_wrapper.get_best_affine_inliers_cpp(
            kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh
        )
    else:
        if not os.environ.get("HS_QUIET"):
            print("WARNING: sver has not been compiled")
        aff_inliers, aff_errors, Aff = get_best_affine_inliers(
            kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh
        )
    return aff_inliers, aff_errors, Aff


def spatially_verify_kpts(
    kpts1,
    kpts2,
    fm,
    xy_thresh=0.01,
    scale_thresh=2.0,
    ori_thresh=TAU / 4.0,
    dlen_sqrd2=None,
    min_nInliers=4,
    match_weights=None,
    returnAff=False,
    full_homog_checks=True,
    refine_method="homog",
    max_nInliers=5000,
):
    if len(fm) == 0:
        if _VERBOSE_SVER:
            print("[sver] Cannot verify with no matches")
        return None
    kpts1 = kpts1.astype(np.float64, casting="same_kind", copy=False)
    kpts2 = kpts2.astype(np.float64, casting="same_kind", copy=False)
    assert match_weights is not None, "provide at least ones please for match_weights"
    fs = match_weights
    if dlen_sqrd2 is None:
        kpts2_m = kpts2.take(fm.T[1], axis=0)
        dlen_sqrd2 = ktool.get_kpts_dlen_sqrd(kpts2_m)
    xy_thresh_sqrd = dlen_sqrd2 * xy_thresh
    aff_inliers, aff_errors, Aff = get_best_affine_inliers_(
        kpts1, kpts2, fm, fs, xy_thresh_sqrd, scale_thresh, ori_thresh
    )
    if len(aff_inliers) < min_nInliers:
        if _VERBOSE_SVER:
            print(
                "[sver] Failed spatial verification len(aff_inliers) = %r"
                % (len(aff_inliers),)
            )
        return None
    if (refine_method.endswith("homog") and len(aff_inliers) < 7) or len(
        aff_inliers
    ) < 4:
        if _VERBOSE_SVER:
            print(
                "[sver] Failed spatial verification len(aff_inliers) = %r"
                % (len(aff_inliers),)
            )
        return None
    if len(aff_inliers) >= max_nInliers:
        svtup = (aff_inliers, aff_errors, Aff, aff_inliers, aff_errors, Aff)
        return svtup
    try:
        refined_inliers, refined_errors, H = refine_inliers(
            kpts1,
            kpts2,
            fm,
            aff_inliers,
            xy_thresh_sqrd,
            scale_thresh,
            ori_thresh,
            full_homog_checks,
            refine_method=refine_method,
        )
    except npl.LinAlgError as ex:
        if _VERBOSE_SVER:
            print("[sver] numeric error in homog estimation.")
            traceback.print_exc()
        return None
    except ValueError as ex:
        if _VERBOSE_SVER:
            print("[sver] error cv2 in homog estimation.")
            traceback.print_exc()
        return None
    except IndexError:
        raise
    except Exception as ex:
        traceback.print_exc()
        if _VERBOSE_SVER:
            print("[sver] Unknown error in homog estimation.")
        return None
    if _VERBOSE_SVER:
        print("[sver] Succesfully finished spatial verification.")
    if returnAff:
        svtup = (refined_inliers, refined_errors, H, aff_inliers, aff_errors, Aff)
        return svtup
    else:
        svtup = (refined_inliers, refined_errors, H, None, None, None)
        return svtup
