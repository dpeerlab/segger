# segger

GNN-based cell segmentation of spatial transcriptomics data.

Full documentation (installation, quickstart, outputs, API reference, notebook usage): [docs](docs/index.rst)

## Installation

pixi with Python 3.11 (conda and Python 3.13 support coming soon):

```bash
curl -fsSL https://pixi.sh/install.sh | sh
git clone https://github.com/dpeerlab/segger.git
cd segger
pixi install -e cuda121
```

## Usage

```bash
segger segment -i /path/to/your/ist/data/ -o /path/to/save/outputs/
segger export  -s /path/to/save/outputs/segger_segmentation.parquet -o /path/to/export/
```

## Preprint

TODO: link

## Citation

TODO: citation
