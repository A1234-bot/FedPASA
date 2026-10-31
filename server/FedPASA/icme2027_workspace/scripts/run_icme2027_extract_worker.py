"""Compatibility entry point for legacy PIAA's packaging import only."""
import runpy
from pathlib import Path
import packaging
import packaging.version
import pkg_resources

# Old PIAA imports packaging from setuptools' formerly vendored namespace.
# Expose the installed public package without modifying model/data arithmetic.
pkg_resources.packaging=packaging
runpy.run_path(str(Path(__file__).with_name('extract_benchmark_features.py')),run_name='__main__')
