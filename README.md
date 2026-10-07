# Motile Tracker

[![tests](https://github.com/funkelab/motile_tracker/workflows/tests/badge.svg)](https://github.com/funkelab/motile_tracker/actions)
[![codecov](https://codecov.io/gh/funkelab/motile_tracker/branch/main/graph/badge.svg)](https://codecov.io/gh/funkelab/motile_tracker)


An application for interactive tracking with [motile](https://github.com/funkelab/motile).
The full documentation of the plugin can be found [here](https://funkelab.github.io/motile_tracker/).

Motile is a library that makes it easy to solve tracking problems using optimization
by framing the task as an Integer Linear Program (ILP).
See the motile [documentation](https://funkelab.github.io/motile)
for more details on the concepts and method.

----------------------------------

## Installation

Users can download and install an executable application from the github release, or
install from `pypi` in the environment of their choice (e.g. `venv`, `conda`) with the command
`pip install motile-tracker`.
Currently, the motile_tracker requires python >=3.11.

### Recommended extras

For better performance, you can install optional extras:

- **numba**: Speeds up candidate graph construction significantly.
  ```bash
  pip install motile-tracker[numba]
  ```

- **gurobi**: Uses the Gurobi solver instead of the default open-source solver. Gurobi is
  much faster but requires a license (free for academics).
  ```bash
  pip install motile-tracker[gurobi]
  ```

You can install multiple extras at once: `pip install motile-tracker[numba,gurobi]`

### Gurobi license version mismatch

If you have a Gurobi license and encounter an error about license version mismatch,
you may need to install a specific version of `gurobipy` that matches your license.
Use one of the version-specific extras:

```bash
pip install motile-tracker[gurobi12]  # For Gurobi 12.x licenses
pip install motile-tracker[gurobi13]  # For Gurobi 13.x licenses
```

Developers can clone the GitHub repository and then  use `uv` to install and run the code.
See the developer guide in `DEVELOPER.md` for more information.

## Usage

Start napari and call the main widget via Plugins > Motile > Motile Main Widget.
2D+time and 3D+time sample data can be loaded via File > Open Sample > Motile. You can
track objects in napari Labels or Points layers. For details, please read the
[documentation](https://funkelab.github.io/motile_tracker/).

![motile_tracker_quick_demo](https://github.com/user-attachments/assets/07a4a954-3d2d-4d67-8f75-aec11ee14697)

If you are new to using motile-tracker, you can follow this [tutorial](./assets/motile-tracker_tutorial.pdf) to learn the basics.

## Membrane labels

Membrane segmentations are typically over- and under-segmented. Once the
nuclei tracks are corrected, `scripts/merge_membranes.py` turns raw membrane
labels into one cell per tracked nucleus, using only the labels (no raw
image). Run it as the last step, after nuclei correction: later edits to the
nuclei do not update the membrane results, so rerun it after them.

```bash
uv run python scripts/merge_membranes.py TRACKS MEMBRANE LABELS.zarr --out-geff OUT.geff --workers 8
```

- `TRACKS`: the corrected tracks, a geff store or a motile run folder.
- `MEMBRANE`: raw membrane labels with the same shape as the nuclei
  segmentation, as a zarr array, a tiff or a folder of one tiff per timepoint.
- `LABELS.zarr` receives the merged labels frame by frame; `OUT.geff` gets the
  tracks plus a copy of them. Load `OUT.geff` via "Tracks (geff)" to see a
  read-only `<name>_membrane` overlay, coloured like the nuclei.

Per frame, fragments are merged along their shared surface, never across two
nuclei; fragments enclosed by one cell are absorbed; fragments lying between
cells are divided among them; a label holding several nuclei (including cells
in ana-/telophase) is split between them by distance; and every empty voxel
inside the embryo outline goes to its nearest cell. Each node gets the
features `membrane_id`, `membrane_volume`, `membrane_n_fragments`,
`membrane_merge_score`, `membrane_n_nuclei` and `membrane_qc` (`ok`, `split`,
`dividing` for sisters right after a division, `no_membrane`), which show in
the table and are written to CSV exports.

Useful options: `--start-frame`/`--max-frames` to try a range, `--view` to
inspect the result in napari, `--keep-cavities-from FRAME` to leave enclosed
cavities such as a blastocoel empty from FRAME on, and `--closing-radius` for
the width of inlets the embryo outline closes. Lengths are in µm, using the
voxel size stored with the tracks (voxels if none is stored). Each worker
needs about 2 to 3 GB of memory. `--help` lists all options.

## Package the application into an executable and create the installer

Tagging any branch will automatically trigger the deploy.yml workflow,
which pushes the tagged version to PyPi and creates a github release; draft release if the tag contains "-dev", pre-release if the tag contains "-rc' or a full release otherwise. In case of a draft or pre release, when the user updates the release notes and promotes it to a published release, github will trigger `make_bundle_app.yml` workflow which will create the Linux, Mac and Windows installer and will upload them as release artifacts to github.

## Issues

If you encounter any problems, please
[file an issue](https://github.com/funkelab/motile_tracker/issues)
along with a detailed description.
