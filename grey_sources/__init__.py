"""grey_sources package

Provides simple utilities to fetch PDFs from shadow libraries using only the Python standard library.

Exports:
- scihub_download, SCIHUB_MIRRORS
- libgen_search, libgen_download
- annas_archive_search, annas_archive_download
"""

from .scihub import scihub_download, SCIHUB_MIRRORS
from .libgen import libgen_search, libgen_download
from .annas_archive import annas_archive_search, annas_archive_download
