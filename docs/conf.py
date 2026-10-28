"""Sphinx configuration for the logit-stream documentation.

Build locally with

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -b html docs docs/_build/html
"""

import logit_stream

project = "logit-stream"
author = "Lee Tucker"
copyright = "2026, Lee Tucker"
release = logit_stream.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
highlight_language = "text"

myst_enable_extensions = ["colon_fence", "dollarmath", "deflist", "attrs_inline"]
myst_heading_anchors = 4

autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = False
napoleon_custom_sections = [("Attributes", "params_style")]
default_role = "py:obj"
# names from hdfe-stream that the docstrings mention but this site does not document
nitpicky = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "hdfe_stream": ("https://leetucker.github.io/hdfe-stream", None),
}
# the intersphinx inventories are fetched at build time; a build without
# network access still succeeds, with those links left as plain text
suppress_warnings = ["intersphinx.external", "config.cache"]

html_theme = "furo"
html_title = f"logit-stream {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/leetucker/logit-stream/",
    "source_branch": "main",
    "source_directory": "docs/",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


# viewcode links a method's [source] by the class it is documented on, so the
# result classes' methods inherited from hdfe_stream.core.results_base need
# the base's tags under the public class's name too (as in hdfe-stream).
def _inherited_tags(app, modname):
    from sphinx.errors import PycodeError
    from sphinx.pycode import ModuleAnalyzer

    try:
        analyzer = ModuleAnalyzer.for_module(modname)
        analyzer.find_tags()
    except PycodeError:
        return None
    tags = dict(analyzer.tags)
    for name in logit_stream.__all__:
        cls = getattr(logit_stream, name, None)
        if not isinstance(cls, type):
            continue
        for base in cls.__mro__[1:]:
            if base.__module__ != modname:
                continue
            prefix = base.__qualname__ + "."
            for tag, entry in analyzer.tags.items():
                if tag.startswith(prefix):
                    tags.setdefault(f"{cls.__name__}.{tag[len(prefix):]}", entry)
    return analyzer.code, tags


def setup(app):
    app.connect("viewcode-find-source", _inherited_tags)
