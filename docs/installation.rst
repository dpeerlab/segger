Installation
============

segger requires **CUDA 13** and **Python 3.13**. ``cuspa`` is built from source, so a CUDA
compiler (``nvcc``) must be on ``PATH``. Both ``conda`` and ``pixi`` install ``cuda-nvcc``
for you.

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/dpeerlab/segger.git
   cd segger

Cloning checks out ``main``, which may include unreleased changes. To install a released
version instead, add ``--branch <tag>``; see the
`releases page <https://github.com/dpeerlab/segger/releases>`_ for the available versions.

Install with conda
------------------

.. code-block:: bash

   conda env create -n segger -f environment.yml
   conda activate segger

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

Install without cloning
-----------------------

If you already have a CUDA 13 ``nvcc`` on ``PATH``, install straight from GitHub:

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
