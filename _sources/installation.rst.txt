Installation
============

Base package
------------

Install the library itself from a local checkout:

.. code-block:: bash

   pip install -e .

For development and tests:

.. code-block:: bash

   pip install -e .[dev]

For documentation tooling:

.. code-block:: bash

   pip install -e .[docs]

Building the docs locally
-------------------------

Once the ``docs`` extra is installed, build the HTML site with either command:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

.. code-block:: bash

   make -C docs html

The built site will be written to ``docs/_build/html``.


Read the Docs
-------------

This repository includes:

- ``docs/conf.py`` for the Sphinx configuration
- ``.readthedocs.yaml`` for Read the Docs
- a ``docs`` optional dependency in ``pyproject.toml``

That means a standard Read the Docs build can install the package with its docs
dependencies and build directly from ``docs/conf.py``.
