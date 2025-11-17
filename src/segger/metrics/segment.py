import shapely
import pyarrow
import numpy as np
import cupy as cp
import cudf
import cuspatial
import pandas as pd
import sys
import os


class TranscriptColumns():
    """
    _summary_
    """
    x = 'x_location'
    y = 'y_location'
    id = 'codeword_index'
    label = 'feature_name'
    xy = [x,y]

class BoundaryColumns():
    """
    _summary_
    """
    x = 'vertex_x'
    y = 'vertex_y'
    id = 'label_id'
    label = 'cell_id'
    xy = [x,y]


def get_xy_bounds(
    filepath,
    x: str,
    y: str,
) -> shapely.Polygon:
    # Get index of x- and y-columns
    metadata = pyarrow.parquet.read_metadata(filepath)
    for i in range(metadata.row_group(0).num_columns):
        column = metadata.row_group(0).column(i)
        if column.path_in_schema == x:
            x_idx = i
        elif column.path_in_schema == y:
            y_idx = i
    # Find min and max values across all row groups
    x_max = -1
    x_min = sys.maxsize
    y_max = -1
    y_min = sys.maxsize
    for i in range(metadata.num_row_groups):
        group = metadata.row_group(i)
        x_min = min(x_min, group.column(x_idx).statistics.min)
        x_max = max(x_max, group.column(x_idx).statistics.max)
        y_min = min(y_min, group.column(y_idx).statistics.min)
        y_max = max(y_max, group.column(y_idx).statistics.max)
    bounds = shapely.box(x_min, y_min, x_max, y_max)
    return bounds


def read_parquet_region(
    filepath,
    x: str,
    y: str,
    bounds: shapely.Polygon = None,
    extra_columns: list[str] = [],
    extra_filters: list[str] = [],
    row_group_chunksize: int = 10,
):
    # Find bounds of full file if not supplied
    if bounds is None:
        bounds = get_xy_bounds(filepath, x, y)
    
    # Load pre-filtered data from Parquet file
    filters = [[
        (x, '>', bounds.bounds[0]),
        (y, '>', bounds.bounds[1]),
        (x, '<', bounds.bounds[2]),
        (y, '<', bounds.bounds[3]),
    ] + extra_filters]

    columns = [x, y] + extra_columns
    region = dask_cudf.read_parquet(
        filepath,
        split_row_groups=row_group_chunksize,
        filters=filters,
        columns=columns,
    )
    return region


def get_polygons_from_xy(
    boundaries: cudf.DataFrame,
):
    # Directly convert to GeoSeries from cuDF
    names = BoundaryColumns
    vertices = boundaries[names.xy].astype('double')
    ids = boundaries[names.id].values
    splits = cp.where(ids[:-1] != ids[1:])[0] + 1
    geometry_offset = cp.hstack([0, splits, len(ids)])
    part_offset = ring_offset = cp.arange(len(np.unique(ids)) + 1)
    polygons = cuspatial.GeoSeries.from_polygons_xy(
        vertices.interleave_columns(),
        geometry_offset,
        part_offset,
        ring_offset,
    )
    del vertices
    gc.collect()
    return polygons


def get_points_from_xy(
    transcripts: cudf.DataFrame,
):
    # Directly convert to GeoSeries from cuDF
    names = TranscriptColumns
    coords = transcripts[names.xy].astype('double')
    points = cuspatial.GeoSeries.from_points_xy(coords.interleave_columns())
    del coords
    gc.collect()
    return points


def filter_boundaries(
    boundaries: cudf.DataFrame,
    inset: shapely.Polygon,
    outset: shapely.Polygon,
):
    # Determine overlaps of boundary polygons
    names = BoundaryColumns
    def in_region(region):
        in_x = boundaries[names.x].between(region.bounds[0], region.bounds[2])
        in_y = boundaries[names.y].between(region.bounds[1], region.bounds[3])
        return in_x & in_y
    x1, y1, x4, y4 = outset.bounds
    x2, y2, x3, y3 = inset.bounds
    boundaries['top'] = in_region(shapely.box(x1, y1, x4, y2))
    boundaries['left'] = in_region(shapely.box(x1, y1, x2, y4))
    boundaries['right'] = in_region(shapely.box(x3, y1, x4, y4))
    boundaries['bottom'] = in_region(shapely.box(x1, y3, x4, y4))
    boundaries['center'] = in_region(inset)
    # Filter boundary polygons
    # Include overlaps with top and left, not bottom and right
    gb = boundaries.groupby(names.id, sort=False)
    total = gb['center'].transform('size')
    in_top = gb['top'].transform('sum')
    in_left = gb['left'].transform('sum')
    in_right = gb['right'].transform('sum')
    in_bottom = gb['bottom'].transform('sum')
    in_center = gb['center'].transform('sum')
    keep = in_center == total
    keep |= ((in_center > 0) & (in_left > 0) & (in_bottom == 0))
    keep |= ((in_center > 0) & (in_top > 0) & (in_right == 0))
    inset_boundaries = boundaries.loc[keep]
    return inset_boundaries


def buffer_polygons(
    polygons: cuspatial.GeoSeries,
    distance: float,
):
    polygons_gp = polygons.to_geopandas()
    buffered = polygons_gp.buffer(distance)
    polygons_buffered = cuspatial.GeoSeries(buffered)
    del polygons_gp, buffered
    return polygons_buffered


def get_quadtree_kwargs(
    bounds: shapely.Polygon,
    user_quadtree_kwargs: dict = None,
):
    if user_quadtree_kwargs is None:
        user_quadtree_kwargs = {}
    kwargs = dict(max_depth=10, max_size=10000)
    kwargs.update(user_quadtree_kwargs)
    if 'scale' not in kwargs:
        kwargs['scale'] = max(
            bounds.bounds[2] - bounds.bounds[0],
            bounds.bounds[3] - bounds.bounds[1],
        ) // (1 << kwargs['max_depth']) + 1
    kwargs.update(dict(
        x_min=bounds.bounds[0],
        y_min=bounds.bounds[1],
        x_max=bounds.bounds[2],
        y_max=bounds.bounds[3],
    ))
    return kwargs


def get_expression_matrix(
    points: cuspatial.GeoSeries,
    points_idx: np.ndarray,
    polygons: cuspatial.GeoSeries,
    polygons_idx: np.ndarray,
    bounds: shapely.Polygon,
    quadtree_kwargs: dict = None,
):  
    # Keyword arguments reused below
    kwargs = get_quadtree_kwargs(bounds, quadtree_kwargs)
    
    # Build quadtree on points
    keys_to_pts, quadtree = cuspatial.quadtree_on_points(points, **kwargs)
    
    # Create bounding box and quadtree lookup
    kwargs.pop('max_size')  # not used below
    bboxes = cuspatial.polygon_bounding_boxes(polygons)
    poly_quad_pairs = cuspatial.join_quadtree_and_bounding_boxes(
        quadtree,
        bboxes,
        **kwargs
    )
    
    # Assign transcripts to cells based on polygon boundaries
    result = cuspatial.quadtree_point_in_polygon(
        poly_quad_pairs,
        quadtree,
        keys_to_pts,
        points,
        polygons,
    )
    
    # Map from transcript index to gene index
    codes = cudf.Series(points_idx)
    col_ind = result['point_index'].map(keys_to_pts).map(codes)
    col_ind = col_ind.to_numpy()
    
    # Get ordered cell IDs from Xenium
    _, row_uniques = pd.factorize(polygons_idx)
    row_ind = result['polygon_index'].map(cudf.Series(row_uniques))
    row_ind = row_ind.to_numpy() - 1  # originally, 1-index
    
    # Construct sparse expression matrix
    X = sp.sparse.csr_array(
        (np.ones(result.shape[0]), (row_ind, col_ind)),
        dtype=np.uint32,
    )
    return X


def get_buffered_counts(
    filepath_transcripts: os.PathLike,
    filepath_boundaries: os.PathLike,
    bounds: shapely.Polygon,
    buffer_distance: float,
    overlap: float = 100,
    quadtree_kwargs: dict = None,
    
):
    # Load transcripts
    outset = bounds.buffer(overlap, join_style='mitre')
    transcripts = read_parquet_region(
        filepath_transcripts,
        TranscriptColumns.x,
        TranscriptColumns.y,
        bounds=outset,
        extra_columns=[TranscriptColumns.id],
        extra_filters=[('qv', '>', 20)],
    ).compute()
    points = get_points_from_xy(transcripts)
    
    # Load boundaries
    boundaries = read_parquet_region(
        filepath_boundaries,
        BoundaryColumns.x,
        BoundaryColumns.y,
        bounds=outset,
        extra_columns=[BoundaryColumns.id]
    ).compute()
    boundaries = filter_boundaries(boundaries, bounds, outset)
    polygons = get_polygons_from_xy(boundaries)
    
    if buffer_distance != 0:
        polygons = buffer_polygons(polygons, buffer_distance)

    # Get sparse expression matrix
    X = get_expression_matrix(
        points,
        transcripts[TranscriptColumns.id].to_numpy(),
        polygons,
        boundaries[BoundaryColumns.id].to_numpy(),
        outset,
        quadtree_kwargs,
    )
    return X


def key_to_coordinate(key):
    # Convert the key to binary and remove the '0b' prefix
    binary_key = bin(key)[2:]
    
    # Make sure the binary string length is even by prepending a '0' if necessary
    if len(binary_key) % 2 != 0:
        binary_key = '0' + binary_key

    # Split the binary string into pairs
    pairs = [binary_key[i:i+2] for i in range(0, len(binary_key), 2)]
    
    # Initialize coordinates
    x, y = 0, 0
    
    # Iterate through each pair to calculate the sum of positions
    for i, pair in enumerate(pairs):
        power_of_2 = 2 ** (len(pairs) - i - 1)
        y += int(pair[0]) * power_of_2
        x += int(pair[1]) * power_of_2
    
    return pd.Series([y, x], index=['y', 'x'], name=key)


def get_quadrant_bounds(
    quadtree: pd.DataFrame,
    bounds: shapely.Polygon,
):
    quadtree = quadtree.copy()
    x_min, y_min, x_max, y_max = bounds.bounds
    width = x_max - x_min
    height = y_max - y_min
    levels = quadtree['level'] + 1
    coords = quadtree['key'].apply(key_to_coordinate)
    quadrant_size_x = width / 2**levels 
    quadrant_size_y = height / 2**levels 
    
    quadtree['x_min'] = x_min + coords['x'] * quadrant_size_x
    quadtree['x_max'] = quadtree['x_min'] + quadrant_size_x
    quadtree['y_min'] = y_min + coords['y'] * quadrant_size_y
    quadtree['y_max'] = quadtree['y_min'] + quadrant_size_y
    return quadtree


def get_transcripts_regions(
    filepath,
    max_size: int = 1e7,
    bounds: shapely.Polygon = None,
):
    # Load transcripts
    if bounds is None:
        bounds = get_xy_bounds(filepath, *TranscriptColumns.xy)
    transcripts = read_parquet_region(
        filepath,
        TranscriptColumns.x,
        TranscriptColumns.y,
        bounds=bounds,
    )
    points = get_points_from_xy(transcripts)
    del transcripts
    gc.collect()
    
    # Build quadtree on points
    kwargs = dict(max_depth=10, max_size=max_size)
    kwargs = get_quadtree_kwargs(bounds, kwargs)
    _, quadtree = cuspatial.quadtree_on_points(points, **kwargs)
    quadtree_df = quadtree.to_pandas()
    del quadtree
    gc.collect()
    
    # Get boundaries of quadtree quadrants
    quadtree_df = get_quadrant_bounds(quadtree_df, bounds)
    regions = quadtree_df.loc[
        ~quadtree_df['is_internal_node'],
        ['x_min', 'y_min', 'x_max', 'y_max']
    ]
    return regions


def get_cell_labels(
    filepath_boundaries,
    row_group_chunksize: int = 10,
):
    id = BoundaryColumns.id
    label = BoundaryColumns.label
    boundaries = dask_cudf.read_parquet(
        filepath_boundaries,
        split_row_groups=row_group_chunksize,
        columns=[id, label],
    )
    boundaries[label] = boundaries[label].str.replace('\x00', '')
    cell_labels = boundaries[label].unique().compute()
    return cell_labels.to_numpy()


def get_buffered_counts_distributed(
    filepath_transcripts, # 
    filepath_boundaries,  # Need adapting for Geoparquet
    filepath_gene_panel,
    buffer_distance,
    client,
):
    # Get equal size regions of space w.r.t. no. transcripts in each region
    # This is a proxy for equally distributing workload
    # First split into quadrants to build quadtree on multiple workers; assumes
    # at least one split will be made in quadtree
    bounds = get_xy_bounds(filepath_transcripts, *TranscriptColumns.xy)
    x_min, y_min, x_max, y_max = bounds.buffer(1).bounds
    x_mid = (x_max - x_min) / 2 + x_min
    y_mid = (y_max - y_min) / 2 + y_min
    quadrants = [
        shapely.box(x_min, y_min, x_mid, y_mid),  # Q1
        shapely.box(x_min, y_mid, x_mid, y_max),  # Q2
        shapely.box(x_mid, y_min, x_max, y_mid),  # Q3
        shapely.box(x_mid, y_mid, x_max, y_max),  # Q4
    ]
    
    # Build quadtree and get boundaries of each quadrant region
    futures = client.map(
        lambda q: get_transcripts_regions(filepath_transcripts, bounds=q),
        quadrants
    )
    regions = pd.concat(client.gather(futures))
    gc.collect()
    
    # Build new counts matrices for each region using buffered (offset) cell
    # boundaries
    # Note: transcripts can be doubly-counted
    futures = client.map(
        lambda region: get_buffered_counts(
            filepath_transcripts,
            filepath_boundaries,
            shapely.box(*region),
            buffer_distance=buffer_distance,
        ),
        regions.values,
    )
    matrices = client.gather(futures)
    gc.collect()
    
    # Combine matrices into one
    output_shape = tuple(np.array([*map(np.shape, matrices)]).max(0))
    X = sp.sparse.csr_array(output_shape, dtype='uint32')
    for matrix in matrices:
        matrix.resize(output_shape)
        X += matrix
    
    # Get gene labels and reorder according to gene panel
    with open(filepath_gene_panel) as f:
        gene_panel = json.load(f)
        targets = gene_panel['payload']['targets']
        index = [t['codewords'][0] for t in targets]
        gene_labels = [t['type']['data']['name'].upper() for t in targets]
    X = X[:, index]
    
    # Get cell labels
    # cell_labels = get_cell_labels(filepath_boundaries)

    # Return AnnData object
    ad = AnnData(
        X=sp.sparse.csr_matrix(X),
        obs=pd.DataFrame(index=np.arange(X.shape[0]).astype(str)),
        var=pd.DataFrame(index=gene_labels),
    )
    return ad