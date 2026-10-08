# Spindle measurement (`motile_tracker.spindle`)

This module measures the mitotic spindle of tracked nuclei in the last frames before they divide. It records geometry and localisation only: mask, poles, axis, length, width, volume and angles. The values are written as `spindle_*` node features into a copy of the tracks.

The measurement is a Python re-implementation of the Fiji plugin Spindle3D v0.8.0 (Kletter et al., *J Cell Biol* 2022, [doi:10.1083/jcb.202106170](https://doi.org/10.1083/jcb.202106170); <https://github.com/embl-cba/spindle3d>). The differences are listed at the end of this file.

## When and how to run

Run it once at the end, after the nuclei are corrected, like the membrane merge. It reads the nuclei segmentation and the division events from the tracks. Later edits to the nuclei do not update the spindle features, so rerun it after them.

```bash
uv run python scripts/spindle_features.py TRACKS TUBULIN DNA OUT.geff --spacing 1,0.26,0.26 --max-frame 454 --workers 4
uv run python scripts/spindle_view.py OUT.geff TUBULIN DNA --node 9449     # or --best / --list; --dry-run
```

**Inputs:**
- `TRACKS`: the corrected tracks (geff). Node id = label in the segmentation; each node has a `bbox`.
- `TUBULIN`: the microtubule channel (e.g. 488).
- `DNA`: the DNA channel (e.g. 561).
- `TUBULIN` and `DNA` are each a zarr array, a tiff, or a folder of one tiff per timepoint (sorted by name). They must have the same shape as the segmentation and are read one frame at a time.
- `--spacing z,y,x`: the voxel size in µm. The default is the voxel size stored with the tracks; pass it if the geff stores 1.

**Outputs next to `OUT.geff`:**
- `OUT.geff`: the tracks with the `spindle_*` node features.
- `<stem>_spindle.csv`: one row per candidate, with the features plus diagnostics.
- `<stem>_params.json`: the parameters used; `spindle_view.py` reads it.
- `<stem>_threshold_diagnostic.csv`: the spindle thresholds per node. These are internal intensity values, not features.
- `qc_<stem>/`: one PNG per candidate plus `overview.png`. Skip them with `--no-qc`; matplotlib is only needed for the QC images.

**Options:**
- `--timepoints-before N` (default 6): frames per division. These are the division node (frame 1 before the division) and its predecessors.
- `--max-frame T`: only divisions up to frame T.
- `--correction ignore|none` (default `ignore`): see step 4 below.
- `--threshold-exclude-um` (default 1.0).
- `--pole-profile-radius java|um` (default `java`): see the quirks below.
- `--half-size-um` (default 20) and `--z-extra-um` (default 4): the crop size.
- `--workers N`: crops measured in parallel. About 2 to 6 s per crop.

## Pipeline

1. **Candidates.** Every node with two successors (it divides into the next frame), plus up to N − 1 predecessors on its single-predecessor chain. Every candidate is measured; the stage is a classification, not a gate.
2. **Crop.** A box of ±20 µm in y/x and ±24 µm in z around the nucleus centroid. The crop is padded beyond the volume edge instead of clipped.
3. **Background subtraction.** Each channel's background is the median of the crop voxels more than 2 µm from the target nucleus and outside other nuclei.
   - The tubulin image is otherwise used as recorded; other nuclei stay in it.
   - In the DNA image, other nuclei (segmentation grown by 1 µm) are set to 0.
4. **Spindle3D** (`spindle3d.py`). Each step is named after its Java method:
   - **Resampling.** Isotropic 0.25 µm resampling: Gaussian blur σ = 0.5/s, then n-linear sampling of input index j/s on the exact grid.
   - **Initial DNA mask.** Threshold = 0.5·(max − min) + min, measured at 1.5 µm voxels. Then 3D hole filling, removal of laterally border-touching regions, and the largest 26-connected region is kept.
   - **Plate axis.** Ellipsoid from 2nd moments; radius = √(5·eigenvalue), as in 3D ImageSuite.
   - **Alignment.** Rotation about the DNA centre so that the shortest axis becomes z. Nearest-neighbour, border extension, full bounding box.
   - **Plate size.**
     - Plate width: derivative (1 µm step) of the axial DNA profile; left maximum / right minimum.
     - Plate length: twice the radius of the steepest drop (2 µm step) of the radial profile of the DNA projection.
   - **Final DNA mask.** DNA threshold = Otsu (256 bins) in a box around the plate. Then hole filling, border regions removed, and the largest region kept.
   - **Spindle threshold.**
     - Otsu (256 bins) on the tubulin values in a 1-voxel shell.
     - With `--correction ignore` the shell lies around the DNA mask dilated by `--threshold-exclude-um`, so chromatin and a possible DNA leak into the tubulin channel cannot raise it.
     - With `--correction none` it is the Java rim around the DNA mask.
   - **Spindle mask.** Tubulin > threshold; 6-connected components with ANY voxel within 3 µm of the DNA centre. No hole filling, no opening. Centrosomes/asters stay in the mask.
   - **Poles.**
     - Initial poles: per z slice, the maximum of the mask near the axis, then a derivative with a 2-voxel step. Left pole = first maximum at z ≤ 0; right pole = first minimum at z ≥ 0. Both poles therefore cannot land on the same side.
     - Refinement: each pole moves independently to the brightest voxel (σ = 0.75 µm blur) inside the mask in a box of offsets z −4..+3, y/x −8..+7 voxels. It is an error if the box holds no mask voxel.
   - **Length, volume, width, angle.**
     - Length = distance between the refined poles.
     - Volume: voxels of the mask after rotating the pole axis onto z.
     - Width: the mask is projected along the spindle axis and opened with a radius-2 disc. Pixels are counted on lines through the centre every 10°; the result is the mean/min/max.
     - Angle to the coverslip = 90° − angle(z, pole axis), using |cos|.
5. **Stage classification** from the DNA crop. Chromatin = 561 > Otsu within 2 µm of the nucleus mask; shape from the PCA axes of the largest chromatin mass. The first matching rule decides:

| Stage | Rule (default threshold) |
|---|---|
| `split` | 2nd / largest chromatin mass ≥ 0.20 |
| `prometaphase` | not flat: shortest / longest axis > 0.40 |
| `prometaphase` | no plate rim: Spindle3D plate length / chromatin extent < 0.5 (the radial DNA profile has no edge at the rim, e.g. a lobed prometaphase rosette) |
| `prometaphase` | thick plate: Spindle3D plate width > 4.0 µm |
| `unclear` | not a disc: middle / longest axis < 0.60 |
| `metaphase` | otherwise |

   If Spindle3D fails, the two plate rules cannot be tested; the rule text says so.
6. **Centrosomes/asters** are detected separately. They are never required and do not change the mask.
   - Candidates are local maxima of tubulin (σ = 0.4 µm) in the DNA-aligned frame.
   - A candidate must lie ≤ 4 µm from the axis, beyond 60 % of the mask's axial extent and ≤ 6 µm past its end.
   - It must also have peak ≥ spindle threshold and peak / median of a 1.5–3 µm shell ≥ 2. The best candidate per side is kept.
   - Centrosome 1 is the one nearest pole A.

## Features (`spindle_*`)

All positions and lengths are in µm. Positions are in the frame of the original image (z = stack axis, voxel size `--spacing`), after undoing every internal rotation and resampling.

| Feature | Meaning |
|---|---|
| `spindle_status` | `ok`, `failed: <reason>`, or `not_measured` (not a candidate) |
| `spindle_stage`, `spindle_stage_rule`, `spindle_is_metaphase` | stage class, the rule that decided it, 1/0 (−1 if not a candidate) |
| `spindle_frames_before_division` | 1 = the dividing node, 2 = one frame earlier, ... |
| `spindle_length_um` | pole-to-pole distance |
| `spindle_width_avg_um`, `_min_um`, `_max_um` | widths through the centre, perpendicular to the axis |
| `spindle_volume_um3` | spindle mask volume (centrosomes included) |
| `spindle_aspect_ratio` | length / mean width |
| `spindle_angle_deg` | tilt of the axis out of the xy (coverslip) plane, 0–90° |
| `spindle_azimuth_deg` | direction in the xy plane, from +x towards +y, 0–180° |
| `spindle_axis_z/y/x` | unit vector pole A → pole B; sign convention z ≥ 0, else y ≥ 0, else x ≥ 0 |
| `spindle_center_z/y/x_um` | pole midpoint |
| `spindle_axis_dna_offset_deg` | angle between the axis and the plate normal |
| `spindle_plate_length_um`, `spindle_plate_width_um` | plate diameter and thickness (Spindle3D) |
| `spindle_chromatin_volume_um3` | final DNA mask volume |
| `spindle_plate_flatness`, `spindle_plate_disc`, `spindle_split_ratio`, `spindle_plate_rim_ratio` | stage-classification metrics (shortest/longest, middle/longest, 2nd/largest mass, plate length/extent) |
| `spindle_n_centrosomes` | 0, 1 or 2 detected puncta |
| `spindle_centrosome1_z/y/x_um`, `spindle_centrosome2_z/y/x_um` | positions (NaN if absent) |
| `spindle_centrosome_distance_um`, `spindle_centrosome_axis_z/y/x`, `spindle_axis_vs_centrosome_deg` | with 2 centrosomes: distance, unit vector (same sign convention), angle to the spindle axis |
| `spindle_flag_pole_on_box_edge` | QC: a refined pole on a lateral face of the refinement box (it may have jumped sideways) |
| `spindle_flag_axis_vs_plate` | QC: the axis is more than 10° from the plate normal |

Non-candidate nodes get NaN, −1 or `not_measured`. Intensity-derived values (SNR, tubulin mean/CoV, thresholds, chromatin dilation) are not stored.

**Additional CSV diagnostics:**
- `plate_axis_mask_fraction`: mask coverage of the axis core through the plate.
- `spindle_mask_components`.
- `pole_tip_offset_um`: pole to mask tip along the axis; > 0 means the pole is inside the tip.
- `pole_to_nearest_centrosome_um`.
- `pole_box_offsets_zyx_vox`, `pole_on_axial_box_edge`, `centrosome_in_mask`, `seconds`.

## Viewer

`scripts/spindle_view.py` recomputes one node with the stored parameters and shows these layers in µm at the node's position:
- the background-subtracted tubulin (the image used);
- the DNA image;
- the nuclei;
- the threshold exclusion region and shell (contours);
- the DNA and spindle masks;
- the axis, the refined and initial poles, and the detected centrosomes.

The measured values appear as a text overlay. `--best` picks the ok metaphase without QC flags and with two centrosomes whose axis is closest to the centrosome line. `--dry-run` prints instead of opening napari.

## Quirks of Java Spindle3D v0.8.0 kept on purpose

- **Pole profile radius.** Java's `computeMaximum` compares voxel coordinates with metaphasePlateLength/2 given in µm. The axial mask profile is therefore taken within L/8 µm (about 2 µm) of the axis instead of L/2 µm. `--pole-profile-radius um` uses L/2 µm.
- **Radial DNA profile.** `cursor.localize()` is called before `cursor.next()`, so each value is binned with the previous pixel's position.
- **Tubulin dynamic-range check.** It compares `measurements.spindleThreshold`, which is still NaN at that point, so it never triggers.
- **SNR split.** (L/2 − 2)² is not clipped at 0. This only affects the SNR, which is not reported.
- **Width angles.** The width loop runs 19 angles (0° to about 180°) because of floating-point accumulation.
- **Pole refinement.** Each pole moves independently in the asymmetric box (−8..+7, −4..+3 voxels). Tubulin gets brighter towards the plate, so refined poles usually sit on the inner axial face of the box, about 1 µm inside the mask tip. In the Pos_23 test data this was true for every measured node. Only the lateral faces are flagged.

## Differences from the original publication / Java Spindle3D v0.8.0

- **Input.** Java measures a user-cropped image of one spindle. Here a box is cropped automatically around each tracked nucleus, padded with 0 beyond the volume. Both channels are background-subtracted per crop, and other nuclei are set to 0 in the DNA channel.
- **Spindle threshold (default `--correction ignore`).**
  - The Otsu shell lies around the DNA mask dilated by 1 µm instead of directly around it, and the sampling box is enlarged by 1 µm. In this dataset the DNA (561) signal leaks into the microtubule (488) channel and would otherwise raise the threshold at the plate.
  - The tubulin image itself is not modified. Earlier attempts to subtract k·DNA or to inpaint the chromatin region created artefacts and were dropped.
  - `--correction none` is the Java behaviour.
- **Outputs.**
  - Intensity outputs (SNR, tubulin mean/CoV, cytoplasm mean, thresholds, chromatin dilation) are not reported.
  - Additions: the stage classification, centrosome detection and features, the axis unit vector, azimuth, centre and centrosome positions in frame µm, the axis-vs-plate-normal angle, and the QC flags.
- **Angles** use |cos|, so they are sign-free (0–90°). Java's `getAngle` does the same.
- **Numerics.**
  - Intensities are float32; Java keeps the input type (uint16 rounding after resampling).
  - The Gaussian truncation (scipy 4σ vs imglib2 Gauss3) and nearest-neighbour rounding differ slightly.
  - Otsu uses its own 256-bin implementation, not compared line by line with imagej-ops `ComputeOtsuThreshold`.
  - Hole filling uses scipy's 6-connected background.
  - Eigenvector signs are arbitrary, so the DNA-aligned frame may be flipped relative to Java. Poles A and B can then swap, and box-edge ties can differ by one voxel.
  - Tubulin intensities are evaluated in the DNA-aligned frame (not reported anyway).
- **Not implemented:** cell mask / cell volume / surface, the composite output image, and `smoothSpindle` (off by default in Java).

**Background.** An earlier Python port of Spindle3D (Spindle3D_Martino) had several issues: an undefined Otsu threshold on failure, an empty-range return value, an off-by-one in a z slice, border regions that were only counted rather than removed, approximate resampling, a centroid instead of an any-voxel central-region rule, and pole edges without the sign split. This module was written from the Java source instead.
