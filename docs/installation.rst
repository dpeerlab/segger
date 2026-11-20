Installation
============

segger requires **CUDA 13** and **Python 3.13**.

Recommended: install from PyPI
------------------------------

Use `uv` for faster installation into an existing or new python environment.

.. code-block:: bash

   uv pip install segger

Alternatively, `pip` installation works as well:

.. code-block:: bash

   pip install segger

Use a container
---------------

Prebuilt images are on Docker Hub as ``sailmskcc/segger``. They include CUDA 13, so the host only
needs an NVIDIA driver (version 580 or newer). Use ``latest`` or pin a release, e.g. ``:1.0.0``.

**Apptainer / Singularity** (HPC clusters). Pull the image once into a SIF file; this takes about
10-20 minutes. Set the cache to a disk with ~20 GB free:

.. code-block:: bash

   export APPTAINER_CACHEDIR=/path/to/cachedir
   apptainer pull segger.sif docker://sailmskcc/segger:latest

Run it on a GPU node. ``--nv`` passes the GPU driver into the container, and ``--bind`` mounts data
outside your home directory:

.. code-block:: bash

   apptainer exec --nv --bind /path/to/data segger.sif segger segment -i ... -o ...

**Docker**. The image's entrypoint is ``segger``:

.. code-block:: bash

   docker run --rm --gpus all --user "$(id -u):$(id -g)" -v /path/to/data:/data \
     sailmskcc/segger:latest segment -i /data/input -o /data/output

Clone the repository
--------------------

.. code-block:: bash

   git clone https://github.com/dpeerlab/segger.git
   cd segger

To install an older release instead, name its tag with ``--branch``, for example ``v0.3.1`` for CUDA 12 support (see `Older CUDA versions`_ below):

.. code-block:: bash

   git clone --branch v0.3.1 https://github.com/dpeerlab/segger.git

Tags are listed on the `releases page <https://github.com/dpeerlab/segger/releases>`_.

Choose an installation method
-----------------------------

Run one of these from the cloned repository. These only work on v1.0.0 and later.

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

Install to a ``uv`` python environment
--------------------------------------

Install uv first if you don't have it:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Then create an environment and install segger into it:

.. code-block:: bash

   uv venv --python 3.13
   source .venv/bin/activate
   uv pip install -e .

Install with pixi
-----------------

Install pixi first if you don't have it:

.. code-block:: bash

   curl -fsSL https://pixi.sh/install.sh | sh
   pixi install
   pixi shell


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
   ``v0.3.1`` release, which runs on CUDA 12.1 and Python 3.11 via pixi:

   .. code-block:: bash

      git clone --branch v0.3.1 https://github.com/dpeerlab/segger.git
      cd segger
      pixi install -e cuda121
      pixi shell -e cuda121

   For one command, use ``pixi run -e cuda121 <command>``. That release still depends on
   ``cuSpatial``, so it can't share an environment with packages built against CUDA 13.
