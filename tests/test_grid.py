import pytest

from rompy_xbeach.grid import GeoPoint, RegularGrid


def test_geo_point_no_crs():
    geo_point = GeoPoint(x=1, y=2)
    assert geo_point.x == 1
    assert geo_point.y == 2
    assert geo_point.crs is None


@pytest.mark.parametrize("crs", ["EPSG:4326", "epsg:4326", "4326", 4326])
def test_geo_point_with_crs(crs):
    geo_point = GeoPoint(x=1, y=2, crs=crs)
    assert geo_point.crs == 4326


def test_geo_point_reproject():
    geo_point_4326 = GeoPoint(x=174.5, y=-41.5, crs=4326)
    geo_point_2193 = GeoPoint(x=1725195.9271650459, y=5404649.736336306, crs=2193)
    geo_point_4326_reprojected = geo_point_2193.reproject(4326)
    geo_point_2193_reprojected = geo_point_4326.reproject(2193)
    assert geo_point_4326.x == pytest.approx(geo_point_4326_reprojected.x)
    assert geo_point_4326.y == pytest.approx(geo_point_4326_reprojected.y)
    assert geo_point_2193.x == pytest.approx(geo_point_2193_reprojected.x)
    assert geo_point_2193.y == pytest.approx(geo_point_2193_reprojected.y)


def test_geo_point_no_crs_cannot_reproject():
    geo_point = GeoPoint(x=1, y=2)
    with pytest.raises(ValueError):
        geo_point.reproject(4326)


def test_regular_grid():
    geo_point = GeoPoint(x=174.5, y=-41.5, crs=4326)
    grid = RegularGrid(
        ori=geo_point,
        alfa=0,
        dx=100,
        dy=100,
        nx=10,
        ny=10,
        crs=2193,
    )
    assert grid.ori == geo_point
    assert grid.alfa == 0
    assert grid.dx == 100
    assert grid.dy == 100
    assert grid.nx == 10
    assert grid.ny == 10
    assert grid.crs == 2193
    assert grid.model_type == "regular"
    assert grid.x0 == pytest.approx(1725195.9271650459)
    assert grid.y0 == pytest.approx(5404649.736336306)
