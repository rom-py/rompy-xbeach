"""Return-schema declarations do not require input data downloads."""

import pytest
from rompy.core.responses import ArtifactType

from rompy_xbeach.config import Config


def make_config(**kwargs):
    # Grid and forcing generation are irrelevant to output-format selection.
    return Config.model_construct(**kwargs)


@pytest.mark.parametrize(
    "outputformat,names",
    [
        (None, {"XBlog.txt", "dims.dat", "xy.dat"}),
        ("fortran", {"XBlog.txt", "dims.dat", "xy.dat"}),
        ("netcdf", {"XBlog.txt", "xboutput.nc"}),
        ("debug", {"XBlog.txt", "dims.dat", "xy.dat", "xboutput.nc"}),
    ],
)
def test_formats(outputformat, names):
    expected = make_config(outputformat=outputformat).expected_artifacts()
    assert {a.path for a in expected} == names
    assert all(a.size_bytes is None for a in expected)


def test_custom_netcdf_name():
    config = make_config(outputformat="netcdf", ncfilename="./results/custom.nc")
    artifact = config.expected_artifacts()[1]
    assert artifact.path == "results/custom.nc"
    assert artifact.artifact_type == ArtifactType.NETCDF


def test_ncfilename_does_not_enable_netcdf():
    assert "custom.nc" not in {
        a.path for a in make_config(ncfilename="custom.nc").expected_artifacts()
    }


def test_existing_and_missing_outputs(tmp_path):
    config = make_config(outputformat="netcdf")
    (tmp_path / "XBlog.txt").write_text("log")
    (tmp_path / "run_result.json").write_text("{}")
    with pytest.warns(UserWarning, match="xboutput.nc"):
        artifacts = config.validate_outputs(tmp_path)
    assert [(a.path, a.size_bytes) for a in artifacts] == [("XBlog.txt", 3)]
    (tmp_path / "xboutput.nc").write_bytes(b"netcdf")
    artifacts = config.validate_outputs(str(tmp_path))
    assert {a.path for a in artifacts} == {"XBlog.txt", "xboutput.nc"}
    assert config.expected_artifacts()[1].size_bytes is None


def test_binary_auxiliary_outputs_are_discovered(tmp_path):
    config = make_config()
    for name in ("XBlog.txt", "dims.dat", "xy.dat", "zs.dat"):
        (tmp_path / name).touch()
    assert {a.path for a in config.validate_outputs(tmp_path)} == {
        "XBlog.txt",
        "dims.dat",
        "xy.dat",
        "zs.dat",
    }
