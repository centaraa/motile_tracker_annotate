import sys
from pathlib import Path

import zarr
from geff_spec import GeffMetadata

container = Path(sys.argv[1])

print("--- top level on disk ---")
for p in sorted(container.iterdir()):
    print(("[dir] " if p.is_dir() else "      ") + p.name)

print("--- zarr, read-only ---")
try:
    root = zarr.open_group(str(container), mode="r")
    print("zarr_format:", root.metadata.zarr_format)
    print("root attrs :", sorted(dict(root.attrs)))
    print("groups     :", sorted(root.group_keys()))
    for name in sorted(root.group_keys()):
        sub = root[name]
        print(f"  {name!r} attrs={sorted(dict(sub.attrs))} groups={sorted(sub.group_keys())}")
except Exception as e:
    print("open failed:", type(e).__name__, e)

print("--- geff metadata ---")
for cand in ["", "tracks", "tracks.geff"]:
    p = container / cand if cand else container
    try:
        m = GeffMetadata.read(str(p))
        print(f"OK   at {cand or '<root>'}")
        print("  extra:", list((m.extra or {}).keys()))
        print("  axes :", [a.name for a in (m.axes or [])])
        print("  props:", sorted(m.node_props_metadata))
    except Exception as e:
        print(f"FAIL at {cand or '<root>'}: {type(e).__name__}: {e}")