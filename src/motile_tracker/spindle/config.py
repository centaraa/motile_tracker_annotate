"""One TOML config for the spindle measurement.

Two sections:

- ``[spindle3d]``: the Spindle3D algorithm (`spindle3d.JavaSettings`); the
  defaults are the published Java v0.8.0 behaviour.
- ``[pipeline]``: everything around it (`core.SpindleParams`): candidates, crop,
  voxel size, the threshold exclusion, stage classification, centrosomes, flags.

Each parameter lives in exactly one section. Defaults are the dataclass defaults;
`default_config.toml` next to this module lists them with comments and is the
same text as `dump_config(SpindleConfig())`. Read with stdlib `tomllib`; written
with the small serializer below (scalars, strings and lists only).
"""

from __future__ import annotations

import dataclasses
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .core import SpindleParams
from .spindle3d import JavaSettings

SECTIONS = {"spindle3d": JavaSettings, "pipeline": SpindleParams}

# one-line comment per parameter: unit, meaning, Java name / reference
COMMENTS = {
    "spindle3d": {
        "voxel_size_for_analysis": "um; isotropic voxel size of the analysis (Java voxelSizeForAnalysis)",
        "metaphase_plate_width_derivative_delta": "um; derivative step of the axial DNA profile, plate width (Java metaphasePlateWidthDerivativeDelta)",
        "metaphase_plate_length_derivative_delta": "um; derivative step of the radial DNA profile, plate length (Java metaphasePlateLengthDerivativeDelta)",
        "spindle_fragment_inclusion_zone": "um; spindle mask components with any voxel this close to the DNA centre are kept (Java spindleFragmentInclusionZone)",
        "axial_pole_refinement_radius": "um; half axial size of the pole refinement box (Java axialPoleRefinementRadius)",
        "lateral_pole_refinement_radius": "um; half lateral size of the pole refinement box (Java lateralPoleRefinementRadius)",
        "voxel_size_for_initial_dna_threshold": "um; voxel size for the initial DNA threshold (Java voxelSizeForInitialDNAThreshold)",
        "initial_dna_threshold_factor": "initial DNA threshold = min + factor * (max - min) (Java initialDnaThresholdFactor)",
        "minimal_dynamic_range": "grey values; stop if the initial DNA threshold is lower (Java minimalDynamicRange)",
        "pole_profile_radius": '"java": axial mask profile within L/2 voxels of the axis as in v0.8.0 computeMaximum; "um": within L/2 um',
    },
    "pipeline": {
        "spacing": "um; voxel size [z, y, x] of the images; [] = the voxel size stored with the tracks",
        "max_frame": "only divisions up to this frame; -1 = all",
        "timepoints_before": "frames measured per division: the dividing node and its predecessors",
        "half_size_um": "um; half size of the crop in y and x around the nucleus",
        "z_extra_um": "um; added to the half size in z (padded beyond the volume)",
        "other_dilate_um": "um; other nuclei are grown by this before they are set to 0 in the DNA crop",
        "correction": '"ignore": chromatin + threshold_exclude_um kept out of the spindle threshold; "none": Java threshold rim',
        "threshold_exclude_um": "um; margin around the DNA mask excluded from the spindle threshold (correction = ignore)",
        "psf_sigma_z_um": "um; axial blur removed from the chromatin thickness for the stage metrics (0 = off)",
        "plate_flat_max": "stage: chromatin shortest/longest axis above this = prometaphase (not flat)",
        "plate_disc_min": "stage: chromatin middle/longest axis below this = unclear (not a disc)",
        "split_min_frac": "stage: second/largest chromatin mass at or above this = split",
        "plate_rim_min_ratio": "stage: Spindle3D plate length / chromatin extent below this = prometaphase (no plate rim)",
        "plate_width_max_um": "um; stage: Spindle3D plate width above this = prometaphase (thick plate)",
        "min_mass_um3": "um3; chromatin masses smaller than this are not counted",
        "centrosome_contrast_min": "centrosome: peak / median of a 1.5-3 um shell at least this",
        "centrosome_lateral_um": "um; centrosome: at most this far from the spindle axis",
        "centrosome_beyond_um": "um; centrosome: at most this far beyond the end of the spindle mask",
        "axis_vs_plate_flag_deg": "deg; flag spindle_flag_axis_vs_plate if the axis is further from the plate normal",
    },
}


@dataclass
class SpindleConfig:
    """Both sections of the config."""

    spindle3d: JavaSettings = field(default_factory=JavaSettings)
    pipeline: SpindleParams = field(default_factory=SpindleParams)

    def to_dict(self) -> dict:
        return {
            "spindle3d": _as_dict(self.spindle3d),
            "pipeline": _as_dict(self.pipeline),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpindleConfig:
        unknown = set(d) - set(SECTIONS)
        if unknown:
            raise ValueError(
                f"unknown config section(s) {sorted(unknown)}; valid: {sorted(SECTIONS)}"
            )
        return cls(**{name: _build(name, d.get(name, {})) for name in SECTIONS})


def _as_dict(obj) -> dict:
    d = dataclasses.asdict(obj)
    return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}


def _coerce(section: str, key: str, value, default):
    where = f"[{section}] {key}"
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be true or false, got {value!r}")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} must be an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{where} must be a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a string, got {value!r}")
        return value
    if isinstance(default, tuple):
        if not isinstance(value, list | tuple) or not all(
            isinstance(v, int | float) and not isinstance(v, bool) for v in value
        ):
            raise ValueError(f"{where} must be a list of numbers, got {value!r}")
        return tuple(float(v) for v in value)
    return value


def _build(section: str, values: dict):
    cls = SECTIONS[section]
    if not isinstance(values, dict):
        raise ValueError(f"[{section}] must be a table")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in [{section}]; valid keys: {sorted(fields)}"
        )
    defaults = cls()
    kwargs = {
        k: _coerce(section, k, v, getattr(defaults, k)) for k, v in values.items()
    }
    return cls(**kwargs)


def read_config_dict(path: str | Path) -> dict:
    """The raw TOML of a config file (only the keys it sets), validated."""
    with open(path, "rb") as fh:
        d = tomllib.load(fh)
    SpindleConfig.from_dict(d)  # raises on unknown sections / keys / bad types
    return d


def load_config(
    path: str | Path | None = None, overrides: dict | None = None
) -> SpindleConfig:
    """Defaults, then the values of `path`, then `overrides` ({section: {key: v}})."""
    d: dict = {name: {} for name in SECTIONS}
    if path is not None:
        for name, values in read_config_dict(path).items():
            d[name].update(values)
    for name, values in (overrides or {}).items():
        if name not in SECTIONS:
            raise ValueError(f"unknown config section {name!r}")
        d.setdefault(name, {}).update(values)
    return SpindleConfig.from_dict(d)


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"cannot write {v!r} to TOML")
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def dump_config(config: SpindleConfig, header: str | None = None) -> str:
    """The complete config as TOML text, with one comment line per parameter."""
    lines = []
    if header:
        lines += [f"# {h}" if h else "#" for h in header.splitlines()]
        lines.append("")
    for name, values in config.to_dict().items():
        lines.append(f"[{name}]")
        for key, value in values.items():
            comment = COMMENTS[name].get(key)
            if comment:
                lines.append(f"# {comment}")
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def write_config(
    config: SpindleConfig, path: str | Path, header: str | None = None
) -> None:
    Path(path).write_text(dump_config(config, header), encoding="utf-8")


DEFAULT_CONFIG_HEADER = (
    "Spindle measurement parameters (motile_tracker.spindle); all values are the defaults.\n"
    "[spindle3d] = Spindle3D v0.8.0 (Java) settings; [pipeline] = crop, candidates,\n"
    "threshold exclusion, stage classification, centrosomes and flags.\n"
    "Pass a copy with --config to scripts/spindle_features.py; CLI options override it."
)


def default_config_path() -> Path:
    """The default config shipped with the package."""
    return Path(__file__).with_name("default_config.toml")
