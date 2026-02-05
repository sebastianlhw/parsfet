"""TechAbs: VLSI Process Technology Abstraction Framework"""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("parsfet")
except (ImportError, PackageNotFoundError):
    try:
        from ._version import __version__
    except ImportError:
        __version__ = "0.0.0"
