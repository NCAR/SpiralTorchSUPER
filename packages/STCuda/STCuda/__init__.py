import torch
from . import _C, ops

# If you install this as editable, and then pull new code,
# the non-python part won't automatically recompile.
# But the metadata won't bump either, so we can trip off of that.
import importlib.metadata
from .version import __version__
if (importlib.metadata.version(__name__) != __version__):
    raise RuntimeError(
        "New or updated " + __name__ + ". Please run 'pip install -e packages/" + __name__ + "'\n"
        "Installed: " + importlib.metadata.version(__name__) + "\nLocal: " + __version__ )