"""Python re-implementation of the Spindle3D measurement.

Spindle3D is the Fiji plugin of Kletter et al., J Cell Biol 2022
(doi:10.1083/jcb.202106170), https://github.com/embl-cba/spindle3d. This module
follows `Spindle3DMorphometry.java` and `util/Utils.java` of v0.8.0 step by step;
comments name the Java method and the line in `Spindle3DMorphometry.java`
(v0.8.0) it corresponds to. Arrays are numpy (z, y, x); Java uses (x, y, z) with
z = DNA / spindle axis, which is numpy axis 0 here.

The deliberate differences from the Java code are marked "DEVIATION" and listed
in the module README; Java quirks that are kept on purpose are marked
"Java quirk".

Input: two 3D arrays (tubulin, DNA) of one crop in grey values (here: per-crop
background-subtracted) and the voxel size (z, y, x) in um.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


class MeasurementError(RuntimeError):
    pass


@dataclass
class JavaSettings:  # Spindle3DSettings.java defaults
    voxel_size_for_analysis: float = 0.25
    metaphase_plate_width_derivative_delta: float = 1.0  # um
    metaphase_plate_length_derivative_delta: float = 2.0  # um
    spindle_fragment_inclusion_zone: float = 3.0  # um
    axial_pole_refinement_radius: float = 1.0  # um
    lateral_pole_refinement_radius: float = 2.0  # um
    voxel_size_for_initial_dna_threshold: float = 1.5  # um
    initial_dna_threshold_factor: float = 0.5
    minimal_dynamic_range: float = 7.0
    # Java computeMaximum() compares voxel coordinates with metaphasePlateLength/2
    # given in um, i.e. the profile radius is L/2 voxels = L/8 um. "java" replicates
    # this (default, matches the published behaviour); "um" uses L/2 um.
    pole_profile_radius: str = "java"


S26 = np.ones((3, 3, 3), bool)  # imglib2 EIGHT_CONNECTED in 3D
S6 = ndi.generate_binary_structure(3, 1)  # imglib2 FOUR_CONNECTED in 3D


# ---------------------------------------------------------------------------
# Utils.java equivalents
# ---------------------------------------------------------------------------
def create_rescaled(img, scaling):
    """createRescaledArrayImg: Gauss3 (sigma 0.5/s, border extension), then
    n-linear sampling of input at j/s, output interval [0, (long)((n-1)*s)]."""
    s = np.asarray(scaling, float)
    blurred = ndi.gaussian_filter(
        np.asarray(img, np.float64), sigma=0.5 / s, mode="nearest"
    )
    out_shape = tuple(int((n - 1) * f) + 1 for n, f in zip(img.shape, s, strict=True))
    # DEVIATION: Java keeps the input pixel type (uint16 -> integer rounding); float32 here
    return ndi.affine_transform(
        blurred,
        matrix=1.0 / s,
        offset=0.0,
        output_shape=out_shape,
        order=1,
        mode="nearest",
    ).astype(np.float32)


def otsu256(values):
    """thresholdOtsu: 256-bin histogram from min to max, Otsu bin, bin centre."""
    v = np.asarray(values, np.float64).ravel()
    if v.size == 0:
        return np.nan
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    w = (hi - lo) / 256.0
    b = np.clip(((v - lo) / w).astype(np.int64), 0, 255)
    h = np.bincount(b, minlength=256).astype(np.float64)
    centers = lo + (np.arange(256) + 0.5) * w
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    var = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(var[:-1]))])


def fill_holes(mask):
    # Ops morphology().fillHoles (3D). DEVIATION: scipy's default 6-connected
    # background flood; the Ops structuring element was not checked.
    return ndi.binary_fill_holes(mask)


def remove_regions_touching_borders(mask, axes):
    """removeRegionsTouchingImageBorders (26-connected). Returns (mask, n_left)."""
    lab, n = ndi.label(mask, structure=S26)
    border = set()
    for ax in axes:
        for idx in (0, mask.shape[ax] - 1):
            border |= set(np.unique(np.take(lab, idx, axis=ax)).tolist())
    border.discard(0)
    if border:
        mask = mask & ~np.isin(lab, list(border))
    return mask, n - len(border)


def keep_largest_region(mask):
    lab, n = ndi.label(mask, structure=S26)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def rotation_to_z(axis):
    """getRotationTransform3D(z, axis): rotation (quaternion from angle/axis) that
    maps `axis` onto z (numpy axis 0)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    z = np.array([1.0, 0.0, 0.0])
    ang = np.arccos(np.clip(a @ z, -1.0, 1.0))
    if ang == 0.0:
        return np.eye(3)
    k = np.cross(a, z)
    if np.linalg.norm(k) == 0.0:  # anti-parallel
        k = np.cross(a, [0.0, 1.0, 0.0])
        if np.linalg.norm(k) == 0:
            k = np.cross(a, [0.0, 0.0, 1.0])
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R


def transformed_view(img, R, c, src_min, order, mode, cval=0.0):
    """createTransformedView: q = R (p - c). Output on the integer bounding box of
    the transformed source interval (corners, truncated like Java's (long) cast).
    src_min: coordinate of img[0,0,0] in its own frame. Returns (array, qmin)."""
    shape = np.array(img.shape)
    corners = (
        np.array(
            [
                [i, j, k]
                for i in (0, shape[0] - 1)
                for j in (0, shape[1] - 1)
                for k in (0, shape[2] - 1)
            ],
            float,
        )
        + src_min
    )
    tc = (corners - c) @ R.T
    qmin = np.trunc(tc.min(0)).astype(int)
    qmax = np.trunc(tc.max(0)).astype(int)
    out_shape = tuple(qmax - qmin + 1)
    # source index = R^T q + c - src_min, q = index + qmin
    offset = R.T @ qmin + c - src_min
    out = ndi.affine_transform(
        img,
        matrix=R.T,
        offset=offset,
        output_shape=out_shape,
        order=order,
        mode=mode,
        cval=cval,
    )
    return out, qmin


def leftmax_rightmin(coords, values):
    coords = np.asarray(coords, float)
    values = np.asarray(values, float)
    left = np.where(coords <= 0)[0]
    right = np.where(coords >= 0)[0]
    if left.size == 0 or right.size == 0:
        raise MeasurementError("profile has no left or right part")
    li = left[np.argmax(values[left])]  # first maximum (strict >)
    ri = right[np.argmin(values[right])]  # first minimum (strict <)
    return coords[li], coords[ri]


def derivative(coords, values, di):
    n = len(values)
    h = di // 2
    idx = range(h + 1, n - (h + 1))
    return (
        np.array([coords[i] for i in idx]),
        np.array([values[i + h] - values[i - h] for i in idx]),
    )


def disc_open(mask2d, radius):
    """open(mask, radius) with HyperSphereShape(radius), zero outside."""
    r = int(radius)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    se = (yy**2 + xx**2) <= r * r
    p = np.pad(mask2d, r + 1)
    o = ndi.binary_opening(p, structure=se)
    return o[r + 1 : -r - 1, r + 1 : -r - 1]


def central_regions_mask(mask, origin, radius):
    """createSpindleMask / Utils.getCentralRegions: 6-connected components of
    `mask` with ANY voxel within HyperSphereShape(radius) (voxels) of `origin`."""
    lab, _ = ndi.label(mask, structure=S6)
    o = np.asarray(origin, int)
    lo = np.maximum(o - radius, 0)
    hi = np.minimum(o + radius + 1, lab.shape)
    sub = lab[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    gz, gy, gx = np.indices(sub.shape)
    ball = (
        (gz + lo[0] - o[0]) ** 2 + (gy + lo[1] - o[1]) ** 2 + (gx + lo[2] - o[2]) ** 2
    ) <= radius**2
    central = np.unique(sub[ball])
    central = central[central > 0]
    return np.isin(lab, central)


def pole_edges_along_z(mask, origin, radius, vs):
    """determineSpindlePolesAlongDnaAxisFromSpindleMask: per z slice the maximum
    of the mask within `radius` (compared with voxel coordinates, as in Java's
    computeMaximum) of the axis, derivative with a 2-voxel step, left pole = first
    maximum at z <= 0, right pole = first minimum at z >= 0. Returns (zl, zr) um."""
    o = np.asarray(origin, int)
    yy, xx = np.indices(mask.shape[1:])
    pdisc = np.hypot(yy - o[1], xx - o[2]) <= radius
    if not pdisc.any():
        pdisc[o[1], o[2]] = True
    zc = (np.arange(mask.shape[0]) - o[0]) * vs
    pvals = np.array([float(mask[z][pdisc].max()) for z in range(mask.shape[0])])
    dcz, dvz = derivative(zc, pvals, 2)
    return leftmax_rightmin(dcz, dvz)


# ---------------------------------------------------------------------------
# Spindle3DMorphometry.measure()
# ---------------------------------------------------------------------------
def measure(
    tubulin,
    dna,
    spacing,
    st: JavaSettings | None = None,
    keep=False,
    threshold_exclude_um: float = 0.0,
):
    """Spindle3DMorphometry.measure() on one crop.

    Args:
        tubulin, dna: 3D arrays (z, y, x) of one crop.
        spacing: voxel size (z, y, x) in um.
        st: Spindle3D settings (default: Java v0.8.0).
        keep: also return the aligned tubulin (A_tub) for centrosome detection.
        threshold_exclude_um: DEVIATION if > 0: the DNA mask dilated by this
            margin is kept out of the spindle threshold (0 = Java).

    Raises MeasurementError where the Java code throws.
    """
    st = st if st is not None else JavaSettings()
    vs = st.voxel_size_for_analysis
    out = {"status": "ok", "voxel_size": vs}
    # createIsotropicallyResampledImages (l. 695)
    scal = np.asarray(spacing, float) / vs
    tub_iso = create_rescaled(tubulin, scal)
    dna_iso = create_rescaled(dna, scal)
    out["iso_shape"] = tub_iso.shape

    # measureInitialThreshold (l. 734)
    ds = create_rescaled(dna_iso, [vs / st.voxel_size_for_initial_dna_threshold] * 3)
    mn, mx = float(ds.min()), float(ds.max())
    thr0 = (mx - mn) * st.initial_dna_threshold_factor + mn
    out["initial_dna_threshold"] = thr0
    if thr0 < st.minimal_dynamic_range:
        raise MeasurementError(
            "Analysis interrupted: Too low dynamic range in DNA image"
        )

    # createInitialDnaMask (l. 775; border removal lateral only: Java dims x, y)
    m0 = fill_holes(dna_iso > thr0)
    m0, n_left = remove_regions_touching_borders(m0, axes=(1, 2))
    if n_left == 0:
        raise MeasurementError(
            "All initial DNA regions were touching the image border!"
        )
    m0 = keep_largest_region(m0)

    # fitEllipsoid (l. 831; 3D ImageSuite moments: normalised 2nd moments, radius = sqrt(5*eig))
    co = np.argwhere(m0).astype(np.float64)
    c = co.mean(0)
    w, V = np.linalg.eigh(np.cov(co - c, rowvar=False, ddof=0))  # ascending
    shortest_axis = V[:, 0]
    lengths = (
        2.0 * np.sqrt(5.0 * np.clip(w, 0, None)) * vs
    )  # shortest, middle, longest (um)
    S_len, L_len = lengths[0], lengths[2]
    out.update(
        dna_center_iso_px=c,
        dna_shortest_axis=shortest_axis,
        ellipsoid_lengths_um=lengths,
    )

    # createShortestAxisAlignmentTransform + transformImages (l. 803; NN, border extension)
    R = rotation_to_z(shortest_axis)
    zero = np.zeros(3)
    A_tub, qmin = transformed_view(tub_iso, R, c, zero, order=0, mode="nearest")
    A_dna, _ = transformed_view(dna_iso, R, c, zero, order=0, mode="nearest")
    o = -qmin  # array index of the DNA centre (origin)
    out.update(R=R, qmin=qmin)

    # measureDnaWidth(alignedDNA, 2*longest, 2*shortest) (l. 846)
    zs = range(int(-S_len / vs), int(S_len / vs) + 1)
    yy, xx = np.indices(A_dna.shape[1:])
    rr = np.hypot(yy - o[1], xx - o[2])
    disc = rr <= L_len / vs
    prof_c, prof_v = [], []
    for z in zs:
        zi = z + o[0]
        prof_c.append(z * vs)
        prof_v.append(
            float(A_dna[zi][disc].mean()) if 0 <= zi < A_dna.shape[0] else np.nan
        )
    dc, dv = derivative(
        prof_c, prof_v, int(np.ceil(st.metaphase_plate_width_derivative_delta / vs))
    )
    left, right = leftmax_rightmin(dc, dv)
    plate_width = right - left

    # measureDnaLength (l. 880): max projection over z in [(long)(-S/vs), (long)(S/vs)], radial profile
    z0, z1 = int(-S_len / vs) + o[0], int(S_len / vs) + o[0]
    proj = A_dna[max(z0, 0) : min(z1, A_dna.shape[0] - 1) + 1].max(0)
    max_bin = int(L_len / vs)
    ny, nx = proj.shape
    Y, X = np.indices(proj.shape)
    Xr = (X - o[2]).ravel().astype(float)
    Yr = (Y - o[1]).ravel().astype(float)
    # Java quirk (replicated): cursor.localize() is called before cursor.next(), so
    # each value is binned with the PREVIOUS pixel's position (x fastest).
    px = np.concatenate([[Xr[0] - 1], Xr[:-1]])
    py = np.concatenate([[Yr[0]], Yr[:-1]])
    dist = np.hypot(px, py)
    bins = dist.astype(np.int64)
    vals = proj.ravel()
    sel = bins < max_bin
    cnt = np.bincount(bins[sel], minlength=max_bin)[:max_bin]
    sm = np.bincount(bins[sel], weights=vals[sel], minlength=max_bin)[:max_bin]
    rprof = sm / np.where(cnt > 0, cnt, np.nan)
    rcoord = np.arange(max_bin) * vs
    dc2, dv2 = derivative(
        rcoord, rprof, int(np.ceil(st.metaphase_plate_length_derivative_delta / vs))
    )
    if dv2.size == 0:
        raise MeasurementError("DNA radial profile too short")
    radius = dc2[int(np.nanargmin(dv2))]
    plate_length = 2.0 * radius
    chromatin_dilation = 1.0 - rprof[0] / np.nanmax(rprof)
    out.update(
        metaphase_plate_width_um=plate_width,
        metaphase_plate_length_um=plate_length,
        chromatin_dilation=float(chromatin_dilation),
    )

    # measureDnaThreshold (l. 371): box lateral +-(int)(L/2/vs), axial +-(int)(W/vs), border extension
    hl, ha = int(plate_length / 2.0 / vs), int(plate_width / vs)
    zi = np.clip(np.arange(-ha, ha + 1) + o[0], 0, A_dna.shape[0] - 1)
    yi = np.clip(np.arange(-hl, hl + 1) + o[1], 0, A_dna.shape[1] - 1)
    xi = np.clip(np.arange(-hl, hl + 1) + o[2], 0, A_dna.shape[2] - 1)
    dna_thr = otsu256(A_dna[np.ix_(zi, yi, xi)])
    out["dna_threshold"] = dna_thr

    # createDnaMaskAndMeasureDnaVolume (l. 937)
    dmask = fill_holes(A_dna > dna_thr)
    dmask, n_left = remove_regions_touching_borders(dmask, axes=(0, 1, 2))
    if n_left == 0:
        raise MeasurementError("All DNA regions were touching the image border!")
    dmask = keep_largest_region(dmask)
    out["chromatin_volume_um3"] = float(dmask.sum() * vs**3)

    # measureSpindleThreshold (l. 433)
    # DEVIATION (optional, threshold_exclude_um > 0): the 1-voxel rim is taken
    # around the DNA mask dilated by threshold_exclude_um (physical, isotropic here)
    # instead of around the DNA mask itself, and the box is enlarged by the same
    # margin, so chromatin (and a possible DNA leak into the tubulin channel at
    # the plate rim) does not enter the threshold estimate.
    m_ex = float(threshold_exclude_um or 0.0)
    lat_half = int((plate_length / 2.0 + 2.0 + m_ex) / vs)
    ax_half = int((plate_width / 2.0 + 2.0 + m_ex) / vs)
    sl = (
        slice(max(o[0] - ax_half, 0), o[0] + ax_half + 1),
        slice(max(o[1] - lat_half, 0), o[1] + lat_half + 1),
        slice(max(o[2] - lat_half, 0), o[2] + lat_half + 1),
    )
    java_rim = ndi.binary_dilation(dmask, structure=S6) & ~dmask  # HyperSphereShape(1)
    out["threshold_java_rim"] = otsu256(A_tub[sl][java_rim[sl]])  # diagnostic
    if m_ex > 0:
        excl = ndi.distance_transform_edt(~dmask) * vs <= m_ex
        rim = ndi.binary_dilation(excl, structure=S6) & ~excl
    else:
        excl, rim = dmask, java_rim
    out["threshold_exclusion_A"] = excl
    out["threshold_shell_A"] = rim
    peri = rim[sl]
    tub_c = A_tub[sl]
    _, Yc, Xc = np.indices(tub_c.shape)
    lat2 = ((Yc + sl[1].start - o[1]) * vs) ** 2 + ((Xc + sl[2].start - o[2]) * vs) ** 2
    # Java: (L/2 - 2)^2 without clipping at 0 (replicated)
    inside = lat2 < (plate_length / 2.0 - 2.0) ** 2
    pv = tub_c[peri]
    sv = tub_c[peri & inside]
    cv = tub_c[peri & ~inside]
    sp_thr = otsu256(pv)
    if not np.isfinite(sp_thr):
        raise MeasurementError("Spindle threshold was NaN")
    msp, mcy = (
        (float(sv.mean()) if sv.size else np.nan),
        (float(cv.mean()) if cv.size else np.nan),
    )
    ssp, scy = (
        (float(sv.std()) if sv.size else np.nan),
        (float(cv.std()) if cv.size else np.nan),
    )
    out.update(
        spindle_threshold=sp_thr,
        tubulin_cytoplasm_mean=mcy,
        spindle_snr=float((msp - mcy) / np.sqrt(scy**2 + ssp**2))
        if sv.size and cv.size
        else np.nan,
    )
    # Java checks measurements.spindleThreshold (still NaN at that point) against
    # minimalDynamicRange, so the tubulin dynamic-range check never triggers (replicated).

    # createSpindleMask (l. 1073): > threshold, 6-connected, regions with ANY voxel within
    # HyperSphereShape((long)(3.0/vs)) around the DNA centre
    smask = central_regions_mask(
        A_tub > sp_thr, o, int(st.spindle_fragment_inclusion_zone / vs)
    )
    # smoothSpindle is false by default -> no opening

    # determineSpindlePolesAlongDnaAxisFromSpindleMask (l. 968)
    rad = (
        plate_length / 2.0
        if st.pole_profile_radius == "java"
        else plate_length / 2.0 / vs
    )
    zl, zr = pole_edges_along_z(smask, o, rad, vs)
    init_poles = [
        np.array([zl, 0.0, 0.0]),
        np.array([zr, 0.0, 0.0]),
    ]  # um, (z, y, x) in A frame

    # refineSpindlePoles (l. 1021): blur sigma 0.75 um; box span (8, 16, 16) voxels: offsets
    # z -4..+3, y/x -8..+7; brightest blurred voxel inside the mask
    blurred = ndi.gaussian_filter(A_tub.astype(np.float64), 0.75 / vs, mode="nearest")
    lat_span = int(2.0 * st.lateral_pole_refinement_radius / vs)
    ax_span = int(2.0 * st.axial_pole_refinement_radius / vs)
    span = np.array([ax_span, lat_span, lat_span])
    bmin = -(span // 2)
    poles, on_edge, box_off = [], [], []
    for p in init_poles:
        pv_ = np.trunc(p / vs).astype(int)  # asLongs
        idx0 = pv_ + bmin + o
        bloc = None
        lo_b = np.maximum(idx0, 0)
        hi_b = np.minimum(idx0 + span, smask.shape)
        bsub = blurred[lo_b[0] : hi_b[0], lo_b[1] : hi_b[1], lo_b[2] : hi_b[2]]
        msub = smask[lo_b[0] : hi_b[0], lo_b[1] : hi_b[1], lo_b[2] : hi_b[2]]
        if msub.any():
            k = int(np.argmax(np.where(msub, bsub, -np.inf)))
            bloc = np.array(np.unravel_index(k, bsub.shape)) + lo_b
        if bloc is None:
            raise MeasurementError("Could not find maximum within mask.")
        off = bloc - idx0
        edge = (off == 0) | (off == span - 1)
        on_edge.append((bool(edge[0]), bool(edge[1] or edge[2])))  # (axial, lateral)
        box_off.append(
            (off + bmin).tolist()
        )  # offset from the initial pole voxel (z, y, x)
        poles.append((bloc - o).astype(float) * vs)  # um, A frame
    p0, p1 = poles
    length = float(np.linalg.norm(p1 - p0))
    mid = 0.5 * (p0 + p1)
    out.update(
        init_poles_um=init_poles,
        poles_um=poles,
        pole_on_box_edge=on_edge,
        pole_box_offsets=box_off,
        spindle_length_um=length,
        spindle_center_to_plate_center_um=float(np.linalg.norm(mid)),
    )

    # createSpindlePolesTransformAndAlignImages (l. 575): q2 = R2 (q - m), axis = pole0 - pole1
    R2 = rotation_to_z((p0 - p1) / np.linalg.norm(p0 - p1)) if length > 0 else np.eye(3)
    S_mask, q2min = transformed_view(
        smask.astype(np.float32),
        R2,
        mid / vs,
        qmin.astype(float),
        order=0,
        mode="constant",
        cval=0.0,
    )
    S_mask = S_mask > 0.5
    out["spindle_volume_um3"] = float(S_mask.sum() * vs**3)

    # measureSpindleWidth (l. 398, 650): max projection along the spindle axis, open(2), widths
    # through the spindle centre every 10 degrees (Java loop replicated)
    P = disc_open(S_mask.max(0), 2)
    oc = -q2min  # index of the spindle centre
    xs = np.arange(P.shape[1]) - oc[2]  # Java dim 0 (x) range of the projected mask
    widths = []
    ang, d_ang = 0.0, np.pi / 18
    while ang < np.pi:
        # value at rotated position (x, 0) = source at T^-1 (x, 0) = (x cos, -x sin)
        sx = np.floor(xs * np.cos(ang) + 0.5).astype(int) + oc[2]
        sy = np.floor(-xs * np.sin(ang) + 0.5).astype(int) + oc[1]
        ok = (sx >= 0) & (sx < P.shape[1]) & (sy >= 0) & (sy < P.shape[0])
        widths.append(int(P[sy[ok], sx[ok]].sum()))
        ang += d_ang
    widths = np.sort(widths)
    wavg = float(np.mean(widths) * vs)
    out.update(
        spindle_width_avg_um=wavg,
        spindle_width_min_um=float(widths[0] * vs),
        spindle_width_max_um=float(widths[-1] * vs),
        n_width_angles=len(widths),
        spindle_aspect_ratio=length / wavg if wavg > 0 else np.nan,
    )

    # intensities (DEVIATION: evaluated in the DNA-aligned frame, not re-sampled
    # n-linearly into the spindle-aligned frame)
    tv = A_tub[smask]
    out["tubulin_spindle_mean"] = float(tv.mean()) if tv.size else np.nan
    out["tubulin_spindle_cov"] = (
        float(tv.std() / (tv.mean() - sp_thr)) if tv.size else np.nan
    )

    # measureSpindleAxisToCoverslipPlaneAngle (l. 675): 90 - angle(z, R^-1 (p1 - p0)), |cos|
    v_iso = R.T @ (p1 - p0)
    cosz = abs(v_iso[0]) / np.linalg.norm(v_iso) if length > 0 else np.nan
    out["spindle_angle_deg"] = float(90.0 - np.degrees(np.arccos(min(cosz, 1.0))))
    # extra (not in Java): angle between pole axis and the DNA plate normal (A-frame z)
    out["spindle_axis_dna_offset_deg"] = (
        float(np.degrees(np.arccos(min(abs((p1 - p0)[0]) / length, 1.0))))
        if length > 0
        else np.nan
    )

    out.update(A_tub=A_tub, dna_mask_A=dmask, spindle_mask_A=smask, origin_index=o)
    if not keep:
        out.pop("A_tub")
    return out


def a_to_iso_points(points_um_A, res):
    """A-frame points (um, origin = DNA centre) -> iso crop voxel index (z, y, x)."""
    vs = res["voxel_size"]
    R, c = res["R"], res["dna_center_iso_px"]
    return np.array([R.T @ (np.asarray(p) / vs) + c for p in points_um_A])


def a_mask_to_iso(mask_A, res):
    """Nearest-neighbour resampling of an A-frame mask into the iso crop grid."""
    R, c, qmin = res["R"], res["dna_center_iso_px"], res["qmin"]
    # iso index p -> A index = R (p - c) - qmin
    out = ndi.affine_transform(
        mask_A.astype(np.float32),
        matrix=R,
        offset=-R @ c - qmin,
        output_shape=tuple(res["iso_shape"]),
        order=0,
        mode="constant",
        cval=0.0,
    )
    return out > 0.5
