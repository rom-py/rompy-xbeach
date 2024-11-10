import pytest

from rompy_xbeach.grid import Ori, RegularGrid


def test_ori_no_crs():
    ori = Ori(x=1, y=2)
    assert ori.x == 1
    assert ori.y == 2
    assert ori.crs is None


@pytest.mark.parametrize("crs", ["EPSG:4326", "epsg:4326", "4326", 4326])
def test_ori_with_crs(crs):
    ori = Ori(x=1, y=2, crs=crs)
    assert ori.crs == 4326


def test_ori_reproject():
    ori_4326 = Ori(x=174.5, y=-41.5, crs=4326)
    ori_2193 = Ori(x=1725195.9271650459, y=5404649.736336306, crs=2193)
    ori_4326_reprojected = ori_2193.reproject(4326)
    ori_2193_reprojected = ori_4326.reproject(2193)
    assert ori_4326.x == pytest.approx(ori_4326_reprojected.x)
    assert ori_4326.y == pytest.approx(ori_4326_reprojected.y)
    assert ori_2193.x == pytest.approx(ori_2193_reprojected.x)
    assert ori_2193.y == pytest.approx(ori_2193_reprojected.y)


def test_ori_no_crs_cannot_reproject():
    ori = Ori(x=1, y=2)
    with pytest.raises(ValueError):
        ori.reproject(4326)


def test_regular_grid():
    ori = Ori(x=174.5, y=-41.5, crs=4326)
    grid = RegularGrid(
        ori=ori,
        alfa=0,
        dx=100,
        dy=100,
        nx=10,
        ny=10,
        crs=2193,
    )
    assert grid.ori == ori
    assert grid.alfa == 0
    assert grid.dx == 100
    assert grid.dy == 100
    assert grid.nx == 10
    assert grid.ny == 10
    assert grid.crs == 2193
    assert grid.model_type == "regular"
    assert grid.x0 == pytest.approx(1725195.9271650459)
    assert grid.y0 == pytest.approx(5404649.736336306)
