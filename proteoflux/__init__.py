from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("proteoflux")
except PackageNotFoundError:
    __version__ = "0+unknown"  # e.g. running from source without install

