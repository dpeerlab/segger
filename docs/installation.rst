Installation
============

segger currently supports **pixi** with **Python 3.11** only.

.. note::
   Conda and Python 3.13 support are coming soon.

Install pixi
-------------

.. code-block:: bash

   curl -fsSL https://pixi.sh/install.sh | sh

See the `pixi documentation <https://pixi.sh/latest/installation/>`_ for other install methods.

Install segger
--------------

By default, cloning the repository checks out ``main``, which may include unreleased changes.
To install a specific released version instead, check out the corresponding tag, e.g.
``v0.2.0``:

.. code-block:: bash

   git clone --branch v0.2.0 https://github.com/dpeerlab/segger.git
   cd segger
   pixi install -e cuda121

See the `releases page <https://github.com/dpeerlab/segger/releases>`_ for all available
versions. To use ``main`` instead, omit ``--branch v0.2.0``.

Run commands inside the environment with ``pixi run -e cuda121 <command>``, or activate it with
``pixi shell -e cuda121``.
