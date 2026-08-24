import sys
from pathlib import Path
import polars as pl

from funtracks.import_export import import_from_geff

DEFAULT = r"C:\tmp\241030_p6_saved.geff"

def show(label, fn):
    try:
        print(label, fn())
    except Exception as e:
        print(label, f"<{type(e).name}: {e}>")

path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
print("loading  :", path)
t = import_from_geff(path)
print("loaded OK")
show("nodes    :", lambda: len(list(t.graph_solution.node_ids())))
show("features :", lambda: list(t.features))
show("ndim     :", lambda: t.ndim)
meta = dict(getattr(t.graph_full, "metadata", {}) or {})
print("shape    :", meta.get("shape"), "| legacy:", meta.get("segmentation_shape"))
show(
    "annots   :",
    lambda: t.graph_solution.node_attrs(attr_keys=["mitotic", "score"]).head(10),
)
show(
    "edits    :",
    lambda: t.graph_solution.node_attrs(attr_keys=["mitotic", "score"]).filter(
        pl.col("mitotic") | (pl.col("score") != 0)
    ),
)