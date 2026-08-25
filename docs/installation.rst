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

.. code-block:: bash

   git clone https://github.com/dpeerlab/segger.git
   cd segger
   pixi install -e cuda121

Run commands inside the environment with ``pixi run -e cuda121 <command>``, or activate it with
``pixi shell -e cuda121``.
