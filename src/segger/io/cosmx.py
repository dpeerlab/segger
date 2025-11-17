from skimage.transform import AffineTransform
from typing import Literal
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import tifffile
import shapely
import os

from .utils import masks_to_contours, contours_to_polygons
from .fields import CosMxBoundaryFields

TOL_FRAC = 1. / 50  # Fraction of area to simplify by

# NOTE: In CosMX, there is a bug in their segmentation where cell masks overlap
# with compartment masks from other cells (e.g. a cell mask A overlaps with 
# nuclear mask for cell B).


def get_cosmx_polygons(
    data_dir: os.PathLike,
    boundary_type: Literal['cell', 'nucleus'],
) -> gpd.GeoDataFrame:
    """
    Extract cell or nuclear polygons from CosMX segmentation outputs.

    Parameters
    ----------
    data_dir : os.PathLike
        Directory containing CellLabels, Compartments, and FOV positions.
    boundary_type : {'cell', 'nucleus'}
        Type of boundary to extract: 'cell' or 'nucleus'.

    Returns
    -------
    polygons : gpd.GeoDataFrame
        GeoDataFrame of polygons indexed by unique cell IDs, with FOV info.

    Raises
    ------
    ValueError
        If `boundary_type` is not valid or FOV file is missing columns.
    FileNotFoundError
        If expected TIFF files are missing.
    """
    # Explicitly coerce to Paths
    data_dir = Path(data_dir)
    _preflight_checks(data_dir)

    fields = CosMxBoundaryFields()
    comp_labels_dir = next(data_dir.glob(fields.compartment_labels_dirname))
    cell_labels_dir = next(data_dir.glob(fields.cell_labels_dirname))
    fov_pos_file = next(data_dir.glob(fields.fov_positions_filename))

    # Check file and directory structures
    fov_info = pd.read_csv(fov_pos_file, index_col='FOV')

    # Add 'Slide' column if doesn't exist
    if 'Slide' not in fov_info:
        fov_info['Slide'] = 1

    # Check compartment type
    if boundary_type == 'cell':
        valid_codes = [
            fields.nucleus_value,
            fields.membrane_value,
            fields.cytoplasmic_value,
        ]
    elif boundary_type == 'nucleus':
        valid_codes = [fields.nucleus_value]
    else:
        msg = (
            f"Invalid compartment '{boundary_type}'. "
            f"Choose 'cell' or 'nucleus'."
        )
        raise ValueError(msg)

    # Assemble polygons per FOV
    polygons = []
    for fov, row in fov_info.iterrows():
        fov_id = str.zfill(str(fov), 3)
        cell_path = cell_labels_dir / f'CellLabels_F{fov_id}.tif'
        comp_path = comp_labels_dir / f'CompartmentLabels_F{fov_id}.tif'

        # Get shapely polygons from cell masks
        cell_labels = tifffile.imread(cell_path)
        comp_labels = tifffile.imread(comp_path)
        masks = np.where(np.isin(comp_labels, valid_codes), cell_labels, 0)

        contours = masks_to_contours(masks).swapaxes(0, -1)
        fov_poly = contours_to_polygons(*contours)

        # Remove redundant vertices
        tol = np.sqrt(fov_poly.area).mean() * TOL_FRAC # scale by avg cell size
        fov_poly.geometry = fov_poly.geometry.simplify(tolerance=tol)

        # FOV coords -> Global coords
        tx = row['X_mm'] * 1e3 / fields.mpp
        ty = row['Y_mm'] * 1e3 / fields.mpp
        # Flip y-axis and Translate to global position
        transform = AffineTransform(scale=[1, -1], translation=[tx, ty])
        fov_poly.geometry = shapely.transform(fov_poly.geometry, transform)

        prefix = f"c_{row['Slide']}_{fov}_"  # match CosMX ID structure
        fov_poly.index = prefix + fov_poly.index.astype(str)
        polygons.append(fov_poly)
    
    polygons = pd.concat(polygons)
    tx = fov_info['X_mm'].max() * 1e3 / fields.mpp
    ty = fov_info['Y_mm'].max() * 1e3 / fields.mpp
    #transform = AffineTransform(translation=[tx, ty])
    #polygons.geometry = shapely.transform(polygons.geometry, transform)
    
    return polygons


def _preflight_checks(
    data_dir: Path,
) -> None:
    """
    Ensure input directories and FOV info file contain expected files and 
    columns.
    """
    fields = CosMxBoundaryFields()

    # Check all files exist
    for pat in [
        fields.compartment_labels_dirname,
        fields.cell_labels_dirname,
        fields.fov_positions_filename,
    ]:
        try:
            next(data_dir.glob(pat))
        except StopIteration:
            msg = (
                f"No file or directory with pattern '{pat}' "
                f"found in {data_dir}."
            )
            raise FileNotFoundError(msg)

    fov_info = pd.read_csv(
        next(data_dir.glob(fields.fov_positions_filename)),
        index_col='FOV'
    )
    required_cols = {'X_mm', 'Y_mm'}
    missing_cols = required_cols - set(fov_info.columns)
    if missing_cols:
        raise ValueError(
            f"Missing columns in FOV info: {', '.join(missing_cols)}"
        )

    expected_fovs = [str.zfill(str(fov), 3) for fov in fov_info.index]
    expected_files = lambda prefix: {
        f"{prefix}_F{fov_id}.tif" for fov_id in expected_fovs
    }

    for dirname, prefix in [
        (fields.cell_labels_dirname, "CellLabels"),
        (fields.compartment_labels_dirname, "CompartmentLabels")
    ]:
        directory = next(data_dir.glob(dirname))
        actual = {f.name for f in directory.glob("*.tif")}
        expected = expected_files(prefix)
        missing = expected - actual
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} {prefix} TIFFs:\n" +
                "\n".join(sorted(missing))
            )
