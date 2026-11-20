"""
ammber.

A Python package for parametrizing gibbs free energy curves for phase-field modeling.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # The version is set in pyproject.toml only
    __version__ = version("ammber")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"
__author__ = 'W. Beck Andrews, Xander Mensah, Katsuyo Thornton'
__credits__ = 'University of Michigan'
