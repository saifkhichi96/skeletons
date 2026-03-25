from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

with (ROOT / "pyproject.toml").open("rb") as handle:
    pyproject = tomllib.load(handle)

project = "DifferentialSkeletons"
author = ", ".join(
    author_entry["name"] for author_entry in pyproject["project"].get("authors", [])
)
release = pyproject["project"]["version"]
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = f"{project} {version}"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}
html_static_path: list[str] = []
