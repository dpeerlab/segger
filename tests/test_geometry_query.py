import numpy as np
import pytest

cudf = pytest.importorskip("cudf")
query = pytest.importorskip("segger.geometry.query")


def test_contains_retries_with_point_chunking_on_oom(monkeypatch):
    points = np.zeros((8, 2), dtype=np.float64)
    polygons = [object()]
    call_sizes = []

    monkeypatch.setattr(query, "polygons_to_geoseries", lambda data, backend: data)
    monkeypatch.setattr(query, "points_to_geoseries", lambda data, backend: data)

    def fake_contains_once(batch_points, *_args, **_kwargs):
        call_sizes.append(len(batch_points))
        if len(batch_points) == 8:
            raise MemoryError("cudaErrorMemoryAllocation")
        return cudf.DataFrame({"index_query": [0], "index_match": [0]})

    monkeypatch.setattr(query, "_points_in_polygons_contains_once", fake_contains_once)

    result = query._points_in_polygons_contains(points, polygons)

    assert call_sizes == [8, 4, 4]
    assert result["index_query"].to_numpy().tolist() == [0, 4]
    assert result["index_match"].to_numpy().tolist() == [0, 0]


def test_contains_does_not_swallow_non_oom_errors(monkeypatch):
    points = np.zeros((4, 2), dtype=np.float64)
    polygons = [object()]

    monkeypatch.setattr(query, "polygons_to_geoseries", lambda data, backend: data)
    monkeypatch.setattr(query, "points_to_geoseries", lambda data, backend: data)

    def fake_contains_once(*_args, **_kwargs):
        raise RuntimeError("not an OOM")

    monkeypatch.setattr(query, "_points_in_polygons_contains_once", fake_contains_once)

    with pytest.raises(RuntimeError, match="not an OOM"):
        query._points_in_polygons_contains(points, polygons)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MemoryError("x"), True),
        (RuntimeError("CUDA error: out_of_memory"), True),
        (RuntimeError("cudaErrorMemoryAllocation"), True),
        (RuntimeError("std::bad_alloc"), True),
        (RuntimeError("some other error"), False),
    ],
)
def test_is_cuda_oom_error(error, expected):
    assert query._is_cuda_oom_error(error) is expected
