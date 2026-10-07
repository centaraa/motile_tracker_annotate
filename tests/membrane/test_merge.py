import numpy as np
import pytest

from motile_tracker.membrane import MembraneMergeParams, merge_frame
from motile_tracker.membrane.merge import (
    QC_DIVIDING,
    QC_NO_MEMBRANE,
    QC_OK,
    QC_SHARED,
    QC_SPLIT,
    contact_areas,
)


def _box(shape, sl, value, out=None):
    out = np.zeros(shape, dtype=np.uint32) if out is None else out
    out[sl] = value
    return out


def test_contact_areas_respects_spacing():
    labels = np.zeros((4, 4, 4), dtype=np.uint32)
    labels[:, :, :2] = 1
    labels[:, :, 2:] = 2
    pairs, surface = contact_areas(labels, spacing=(2.0, 1.0, 1.0))
    # Shared face along x: 4 (z) * 4 (y) voxel faces of 2 µm * 1 µm.
    assert pairs == {(1, 2): pytest.approx(32.0)}
    # Only the shared face counts; image-border faces are ignored.
    assert surface[1] == pytest.approx(32.0)


def test_oversegmented_cell_is_merged_onto_nucleus():
    shape = (10, 20, 20)
    # One cell cut into three slabs; the nucleus sits in the middle slab.
    membrane = np.zeros(shape, dtype=np.uint32)
    membrane[:, 2:18, 2:7] = 1
    membrane[:, 2:18, 7:12] = 2
    membrane[:, 2:18, 12:18] = 3
    nuclei = _box(shape, (slice(3, 7), slice(8, 12), slice(8, 11)), 42)

    res = merge_frame(membrane, nuclei)

    assert set(np.unique(res.labels)) == {0, 42}
    assert (res.labels == 42).sum() == (membrane > 0).sum()
    f = res.features[42]
    assert f["membrane_n_fragments"] == 3
    assert f["membrane_qc"] == QC_OK
    assert f["membrane_n_nuclei"] == 1


def test_cannot_link_keeps_two_seeded_cells_apart():
    shape = (6, 10, 20)
    membrane = np.zeros(shape, dtype=np.uint32)
    membrane[:, :, :10] = 1
    membrane[:, :, 10:] = 2
    nuclei = _box(shape, (slice(2, 4), slice(4, 6), slice(3, 6)), 7)
    nuclei = _box(shape, (slice(2, 4), slice(4, 6), slice(13, 16)), 8, nuclei)

    res = merge_frame(membrane, nuclei, params=MembraneMergeParams(min_shared_frac=0))

    assert set(np.unique(res.labels)) == {7, 8}
    assert res.features[7]["membrane_n_fragments"] == 1


def test_small_contact_does_not_merge():
    shape = (6, 20, 20)
    membrane = np.zeros(shape, dtype=np.uint32)
    membrane[:, 2:18, 2:10] = 1
    membrane[2:3, 9:10, 10:18] = 2  # touches fragment 1 through one voxel row
    nuclei = _box(shape, (slice(2, 4), slice(8, 12), slice(4, 7)), 5)

    res = merge_frame(
        membrane,
        nuclei,
        params=MembraneMergeParams(min_contact_area_um2=5, fill_gaps=False),
    )

    assert res.features[5]["membrane_n_fragments"] == 1
    assert not res.labels[membrane == 2].any()


def test_distance_limit_blocks_far_fragment():
    shape = (4, 4, 40)
    membrane = np.zeros(shape, dtype=np.uint32)
    membrane[..., :5] = 1
    membrane[..., 5:40] = 2
    nuclei = _box(shape, (slice(1, 3), slice(1, 3), slice(1, 4)), 3)

    res = merge_frame(
        membrane,
        nuclei,
        spacing=(1.0, 1.0, 1.0),
        params=MembraneMergeParams(
            min_shared_frac=0, max_dist_um=5.0, merge_embedded=False
        ),
    )
    assert res.features[3]["membrane_n_fragments"] == 1


def test_undersegmented_fragment_is_split_by_nuclei():
    shape = (6, 10, 30)
    membrane = _box(shape, (slice(None), slice(None), slice(None)), 1)
    nuclei = _box(shape, (slice(2, 4), slice(4, 6), slice(3, 6)), 10)
    nuclei = _box(shape, (slice(2, 4), slice(4, 6), slice(24, 27)), 11, nuclei)

    res = merge_frame(membrane, nuclei)

    assert set(np.unique(res.labels)) == {10, 11}
    assert res.features[10]["membrane_qc"] == QC_SPLIT
    assert res.features[10]["membrane_n_nuclei"] == 2
    assert res.labels[3, 5, 1] == 10
    assert res.labels[3, 5, 28] == 11


def test_dividing_pair_is_split_and_flagged():
    shape = (6, 10, 30)
    membrane = _box(shape, (slice(None), slice(None), slice(None)), 1)
    nuclei = _box(shape, (slice(2, 4), slice(4, 6), slice(3, 6)), 10)
    nuclei = _box(shape, (slice(2, 4), slice(4, 6), slice(24, 27)), 11, nuclei)

    res = merge_frame(membrane, nuclei, dividing_pairs=[(10, 11)])

    assert set(np.unique(res.labels)) == {10, 11}
    assert res.labels[3, 5, 1] == 10 and res.labels[3, 5, 28] == 11
    for n in (10, 11):
        assert res.features[n]["membrane_qc"] == QC_DIVIDING
        assert res.features[n]["membrane_id"] == n


def test_dividing_flag_also_when_raw_already_separated():
    shape = (6, 10, 30)
    membrane = _box(shape, (slice(None), slice(None), slice(0, 15)), 1)
    membrane = _box(shape, (slice(None), slice(None), slice(15, 30)), 2, membrane)
    nuclei = _box(shape, (slice(2, 4), slice(4, 6), slice(3, 6)), 10)
    nuclei = _box(shape, (slice(2, 4), slice(4, 6), slice(24, 27)), 11, nuclei)

    res = merge_frame(membrane, nuclei, dividing_pairs=[(10, 11)])

    assert res.features[10]["membrane_qc"] == QC_DIVIDING
    assert res.features[10]["membrane_n_nuclei"] == 1


def test_no_split_keeps_shared_label():
    shape = (6, 10, 30)
    membrane = _box(shape, (slice(None), slice(None), slice(None)), 1)
    nuclei = _box(shape, (slice(2, 4), slice(4, 6), slice(3, 6)), 10)
    nuclei = _box(shape, (slice(2, 4), slice(4, 6), slice(24, 27)), 11, nuclei)

    res = merge_frame(
        membrane, nuclei, params=MembraneMergeParams(use_watershed_split=False)
    )

    assert set(np.unique(res.labels)) == {10}
    assert res.features[11]["membrane_qc"] == QC_SHARED
    assert res.features[11]["membrane_id"] == 10


def test_nucleus_without_membrane_and_unseeded_background():
    shape = (4, 10, 10)
    membrane = _box(shape, (slice(None), slice(0, 3), slice(None)), 1)
    nuclei = _box(shape, (slice(1, 3), slice(6, 8), slice(4, 6)), 9)

    res = merge_frame(membrane, nuclei)

    assert not res.labels.any()  # fragment 1 is unseeded
    assert res.features[9]["membrane_qc"] == QC_NO_MEMBRANE
    assert res.features[9]["membrane_id"] == 0


def test_min_cell_volume_drops_tiny_cell():
    shape = (4, 10, 10)
    membrane = _box(shape, (slice(1, 3), slice(4, 7), slice(4, 7)), 1)
    nuclei = _box(shape, (slice(1, 3), slice(5, 6), slice(5, 6)), 4)

    res = merge_frame(
        membrane, nuclei, params=MembraneMergeParams(min_cell_volume_um3=100)
    )
    assert not res.labels.any()
    assert res.features[4]["membrane_qc"] == QC_NO_MEMBRANE


def test_embedded_unseeded_fragment_is_absorbed():
    shape = (12, 12, 12)
    membrane = _box(shape, (slice(1, 11), slice(1, 11), slice(1, 11)), 1)
    membrane[5:7, 5:7, 8:10] = 2  # enclosed, no nucleus, tiny contact score
    nuclei = _box(shape, (slice(3, 5), slice(3, 5), slice(3, 5)), 4)

    params = MembraneMergeParams(min_shared_frac=1.0, fill_gaps=False)
    res = merge_frame(membrane, nuclei, params=params)

    assert res.features[4]["membrane_n_fragments"] == 2
    assert (res.labels[membrane == 2] == 4).all()


def test_unseeded_fragment_touching_background_is_not_absorbed():
    shape = (12, 12, 20)
    membrane = _box(shape, (slice(1, 11), slice(1, 11), slice(1, 10)), 1)
    membrane[3:8, 3:8, 10:15] = 2  # sticks out into background
    nuclei = _box(shape, (slice(3, 5), slice(3, 5), slice(3, 5)), 4)

    params = MembraneMergeParams(min_shared_frac=1.0, fill_gaps=False)
    res = merge_frame(membrane, nuclei, params=params)

    assert res.features[4]["membrane_n_fragments"] == 1
    assert not res.labels[membrane == 2].any()


def test_fill_gaps_closes_holes_and_respects_outline():
    shape = (10, 20, 30)
    membrane = np.zeros(shape, dtype=np.uint32)
    membrane[2:8, 2:18, 2:14] = 1
    membrane[2:8, 2:18, 16:28] = 2  # 2-voxel gap between the two cells
    membrane[4:6, 8:11, 6:9] = 0  # hole inside cell 1
    nuclei = _box(shape, (slice(4, 6), slice(4, 7), slice(3, 6)), 10)
    nuclei = _box(shape, (slice(4, 6), slice(4, 7), slice(20, 23)), 11, nuclei)

    res = merge_frame(membrane, nuclei)

    inside = np.zeros(shape, bool)
    inside[2:8, 2:18, 2:28] = True
    # No hole, no gap; at the gap's mouth on the surface the closing may leave
    # a groove one voxel deep, so check from one voxel in.
    assert (res.labels[3:7, 3:17, 2:28] > 0).all()
    assert not res.labels[~inside].any()  # nothing past the outline
    assert res.labels[5, 9, 7] == 10
    assert res.labels[5, 9, 14] == 10 and res.labels[5, 9, 15] == 11
    assert res.features[10]["membrane_volume"] == (res.labels == 10).sum()


def test_fill_embryo_holes_off_keeps_cavity_empty():
    shape = (12, 12, 12)
    membrane = _box(shape, (slice(1, 11), slice(1, 11), slice(1, 11)), 1)
    membrane[4:8, 4:8, 4:8] = 0  # cavity
    nuclei = _box(shape, (slice(1, 3), slice(1, 3), slice(1, 3)), 4)

    keep = merge_frame(
        membrane,
        nuclei,
        params=MembraneMergeParams(fill_embryo_holes=False, closing_radius_um=0),
    )
    filled = merge_frame(membrane, nuclei)

    assert not keep.labels[5, 5, 5]
    assert filled.labels[5, 5, 5] == 4


def test_convex_hull_mask_matches_shape():
    from motile_tracker.membrane.merge import convex_hull_mask

    mask = np.zeros((9, 9, 9), bool)
    mask[1, 1, 1] = mask[7, 1, 1] = mask[1, 7, 1] = mask[1, 1, 7] = True
    hull = convex_hull_mask(mask)
    # A tetrahedron: x + y + z <= 9 over the corner at (1, 1, 1).
    zz, yy, xx = np.indices(mask.shape)
    expected = (zz >= 1) & (yy >= 1) & (xx >= 1) & (zz + yy + xx <= 9)
    np.testing.assert_array_equal(hull, expected)
    assert convex_hull_mask(np.zeros((4, 4, 4), bool)) is None


def test_convex_hull_outline_fills_fjord_from_inside():
    shape = (12, 20, 20)
    membrane = _box(shape, (slice(2, 10), slice(2, 18), slice(2, 18)), 1)
    membrane[2:10, 7:13, 2:12] = 0  # 6-voxel fjord open to the outside at x=2
    nuclei = _box(shape, (slice(5, 7), slice(4, 6), slice(4, 6)), 3)

    convex = merge_frame(
        membrane, nuclei, params=MembraneMergeParams(outline="convex_hull")
    )
    closed = merge_frame(
        membrane, nuclei, params=MembraneMergeParams(closing_radius_um=2)
    )

    assert (convex.labels[2:10, 2:18, 2:18] == 3).all()  # fjord filled
    assert not convex.labels[0:2].any() and not convex.labels[:, :, 18:].any()
    assert not closed.labels[5, 10, 4]  # "closed" keeps the fjord open


def test_closed_outline_fills_narrow_fjord_but_keeps_wide_dent():
    shape = (20, 40, 40)
    membrane = _box(shape, (slice(3, 17), slice(3, 37), slice(3, 37)), 1)
    membrane[3:17, 10:14, 3:15] = 0  # 4-voxel fjord, 12 deep
    membrane[3:17, 22:34, 3:6] = 0  # 12-voxel wide, 3 deep dent (a bulge gap)
    nuclei = _box(shape, (slice(9, 11), slice(18, 20), slice(18, 20)), 3)

    res = merge_frame(membrane, nuclei, params=MembraneMergeParams(closing_radius_um=4))

    # Filled up to a groove at most one voxel deep at the mouth (x=3).
    assert (res.labels[5:15, 10:14, 4:15] == 3).all()
    assert not res.labels[5:15, 24:32, 3:5].any()  # dent kept
    assert not res.labels[:, :, 37:].any()  # nothing past the outline


def test_close_ball_respects_spacing():
    from motile_tracker.membrane.merge import close_ball

    mask = np.zeros((21, 21, 30), bool)
    mask[:, :, :10] = mask[:, :, 16:] = True  # 6-voxel gap along x
    assert not close_ball(mask, 2.0, (1.0, 1.0, 1.0))[10, 10, 12]
    # The same gap is 3 um wide at 0.5 um/voxel and closes with a 2 um ball.
    assert close_ball(mask, 2.0, (1.0, 1.0, 0.5))[10, 10, 12]
    assert close_ball(mask, 2.0, (1.0, 1.0, 1.0))[mask].all()  # mask kept


def _between_cells():
    """Three cells around a slab between them, which touches all three."""
    shape = (10, 30, 30)
    membrane = np.zeros(shape, dtype=np.uint32)
    membrane[2:8, 2:14, 2:14] = 1
    membrane[2:8, 2:14, 16:28] = 2
    membrane[2:8, 16:28, 2:28] = 3
    membrane[2:8, 14:16, 2:28] = 4  # unseeded slab between the cells
    membrane[2:8, 2:14, 14:16] = 4
    nuclei = _box(shape, (slice(4, 6), slice(6, 9), slice(6, 9)), 10)
    nuclei = _box(shape, (slice(4, 6), slice(6, 9), slice(20, 23)), 11, nuclei)
    nuclei = _box(shape, (slice(4, 6), slice(20, 23), slice(12, 16)), 12, nuclei)
    return membrane, nuclei


def test_fragment_between_cells_is_divided_not_merged_whole():
    membrane, nuclei = _between_cells()

    res = merge_frame(membrane, nuclei, params=MembraneMergeParams(min_shared_frac=0))

    slab = res.labels[membrane == 4]
    assert set(np.unique(slab)) == {10, 11, 12}  # divided among neighbours
    assert all(res.features[n]["membrane_n_fragments"] == 1 for n in (10, 11, 12))


def test_ambiguous_share_zero_merges_fragment_whole():
    membrane, nuclei = _between_cells()

    res = merge_frame(
        membrane,
        nuclei,
        params=MembraneMergeParams(min_shared_frac=0, ambiguous_share=0),
    )

    assert len(set(np.unique(res.labels[membrane == 4]))) == 1


def test_fragment_mostly_on_one_cell_still_merges():
    shape = (10, 30, 30)
    membrane = np.zeros(shape, dtype=np.uint32)
    membrane[2:8, 2:14, 2:14] = 1
    membrane[2:8, 2:14, 16:28] = 2
    membrane[2:8, 2:14, 14:16] = 3  # between 1 and 2, but ...
    membrane[2:8, 14:20, 2:16] = 3  # ... mostly against cell 1
    nuclei = _box(shape, (slice(4, 6), slice(6, 9), slice(6, 9)), 10)
    nuclei = _box(shape, (slice(4, 6), slice(6, 9), slice(20, 23)), 11, nuclei)

    params = MembraneMergeParams(min_shared_frac=0, fill_gaps=False)
    res = merge_frame(membrane, nuclei, params=params)

    assert res.features[10]["membrane_n_fragments"] == 2


def test_nucleus_seeds_fill_inside_dissolved_fragment():
    membrane, nuclei = _between_cells()
    nuclei[4:6, 13:16, 6:9] = 10  # nucleus 10 reaches into the slab

    res = merge_frame(membrane, nuclei, params=MembraneMergeParams(min_shared_frac=0))

    assert (res.labels[nuclei == 10] == 10).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_split_in_bounding_box_equals_full_frame(seed):
    """Splitting inside the region's bounding box is exactly the full split."""
    from scipy import ndimage

    from motile_tracker.membrane.merge import split_region

    rng = np.random.default_rng(seed)
    shape = (30, 60, 70)
    # An irregular region (smoothed noise) away from the frame border, with a
    # few small nuclei inside it.
    noise = ndimage.gaussian_filter(rng.random(shape), 3)
    region = noise > np.quantile(noise, 0.6)
    region[:4] = region[:, :6] = region[:, :, -8:] = False
    lab, _ = ndimage.label(region)
    region = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)  # largest piece
    nuclei = np.zeros(shape, np.uint32)
    coords = np.argwhere(region)
    for k, (z, y, x) in enumerate(coords[rng.choice(len(coords), 5, replace=False)]):
        nuclei[z, y, x : x + 2] = 100 + k
    nuclei[~region] = 0
    nucs = [int(i) for i in np.unique(nuclei) if i]
    spacing = (2.0, 0.5, 0.5)

    full = split_region(region, nuclei, nucs, spacing)
    box = ndimage.find_objects(region.astype(np.uint8))[0]
    cropped = np.zeros_like(full)
    cropped[box] = split_region(region[box], nuclei[box], nucs, spacing)

    np.testing.assert_array_equal(cropped, full)
    assert set(np.unique(full[region])) == set(nucs)
