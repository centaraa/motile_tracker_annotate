import logging

from ._wgpu_setup import configure_wgpu_backend

# cmap logs a warning for each colormap name that exists in several namespaces
# (e.g. "coolwarm") while the GUI stack is imported; scripts with worker
# processes print it once per worker. It needs no action, so keep it quiet.
logging.getLogger("cmap").setLevel(logging.ERROR)

# must run before fastplotlib/pygfx create a wgpu instance
configure_wgpu_backend()

from .application_menus.main_app import StartupWidget  # noqa: E402, F401
