Installation
============

segger requires **CUDA 13** and **Python 3.13**.

Clone the repository
--------------------

.. code-block:: bash

   git clone --branch v0.4.0 https://github.com/dpeerlab/segger.git
   cd segger

``--branch v0.4.0`` pins the clone to the latest release. Omit it to get ``main``, which may
include unreleased changes, or name an older tag from the
`releases page <https://github.com/dpeerlab/segger/releases>`_.

Choose an installation method
-----------------------------

Run one of these from the cloned repository. Times are typical and depend on your network
and package cache.

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Method
     - Typical time
     - Command
   * - uv
     - about 30 seconds
     - ``uv pip install -e .``
   * - pixi
     - about 1.5 minutes
     - ``pixi install``
   * - conda
     - about 5 minutes
     - ``conda env create -n segger -f environment.yml``

Install with uv
---------------

Install uv first if you don't have it:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

See the `uv documentation <https://docs.astral.sh/uv/getting-started/installation/>`_ for other
install methods. Then create an environment and install segger into it:

.. code-block:: bash

   uv venv --python 3.13
   source .venv/bin/activate
   uv pip install -e .

Install with pixi
-----------------

Install pixi first if you don't have it:

.. code-block:: bash

   curl -fsSL https://pixi.sh/install.sh | sh

See the `pixi documentation <https://pixi.sh/latest/installation/>`_ for other install methods.

.. code-block:: bash

   pixi install
   pixi shell

To run one command without activating the environment, use ``pixi run <command>``.

Install with conda
------------------

.. code-block:: bash

   conda env create -n segger -f environment.yml
   conda activate segger

Install without cloning
-----------------------

Install straight from GitHub:

.. code-block:: bash

   pip install \
     --extra-index-url https://pypi.nvidia.com \
     --extra-index-url https://download.pytorch.org/whl/cu130 \
     git+https://github.com/dpeerlab/segger.git

Older CUDA versions
-------------------

.. note::
   segger no longer supports CUDA 12. If CUDA 13 is not available to you, use the
   ``v0.3.0`` release, which runs on CUDA 12.1 and Python 3.11 via pixi:

   .. code-block:: bash

      git clone --branch v0.3.0 https://github.com/dpeerlab/segger.git
      cd segger
      pixi install -e cuda121
      pixi shell -e cuda121

   For one command, use ``pixi run -e cuda121 <command>``. That release still depends on
   ``cuSpatial``, so it can't share an environment with packages built against CUDA 13.
