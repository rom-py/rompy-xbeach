"""Tests for the XBeach Output component."""

import pytest
import logging
import tempfile
from pathlib import Path
from rompy_xbeach.components.output import Output
from rompy_xbeach.types import OutputVarsEnum


@pytest.fixture(scope="module")
def all_output_vars():
    """Return list of all available output variable values."""
    return [var.value for var in OutputVarsEnum]


@pytest.fixture(scope="module")
def timing_file():
    """Create a temporary timing file for testing."""
    tmpdir = Path(tempfile.mkdtemp())
    timing_file = tmpdir / "test_times.txt"

    # Create timing file in XBeach format
    times = [0, 100, 200, 300, 400, 500]
    content = f"{len(times)}\n" + "\n".join(str(t) for t in times) + "\n"
    timing_file.write_text(content)

    yield timing_file

    # Cleanup
    import shutil

    shutil.rmtree(tmpdir)


# =====================================================================================
# Basic instantiation tests
# =====================================================================================
def test_output_default():
    """Test Output with default values."""
    output = Output()
    assert output.model_type == "output"
    assert output.outputformat == "netcdf"
    assert output.meanvars == []
    assert output.globalvars == []
    assert output.pointvars == []
    assert output.points == []
    assert output.rugauges == []


def test_output_with_meanvars():
    """Test Output with mean variables."""
    output = Output(meanvars=[OutputVarsEnum.H, OutputVarsEnum.U, OutputVarsEnum.V])
    assert len(output.meanvars) == 3
    assert output.meanvars[0] == OutputVarsEnum.H


def test_output_with_globalvars():
    """Test Output with global variables."""
    output = Output(globalvars=[OutputVarsEnum.ZS, OutputVarsEnum.ZB])
    assert len(output.globalvars) == 2


def test_output_with_points():
    """Test Output with point locations."""
    output = Output(
        points=[(0.0, 100.0), (500.0, 200.0)],
        pointvars=[OutputVarsEnum.H, OutputVarsEnum.U],
    )
    assert len(output.points) == 2
    assert output.points[0] == (0.0, 100.0)
    assert len(output.pointvars) == 2


def test_output_with_rugauges():
    """Test Output with runup gauges."""
    output = Output(
        rugauges=[(0.0, 100.0), (500.0, 200.0)],
        pointvars=[OutputVarsEnum.H],
    )
    assert len(output.rugauges) == 2
    assert output.rugauges[0] == (0.0, 100.0)


# =====================================================================================
# Params property tests
# =====================================================================================
def test_params_empty():
    """Test params property with empty Output."""
    output = Output()
    params = output.params
    assert "model_type" not in params
    assert "outputformat" in params
    assert params["outputformat"] == "netcdf"


def test_params_with_meanvars():
    """Test params property with mean variables."""
    output = Output(meanvars=[OutputVarsEnum.H, OutputVarsEnum.U, OutputVarsEnum.V])
    params = output.params
    assert params["nmeanvar"] == 3
    assert params["meanvars"] == ["H", "u", "v"]


def test_params_with_globalvars():
    """Test params property with global variables."""
    output = Output(globalvars=[OutputVarsEnum.ZS, OutputVarsEnum.ZB])
    params = output.params
    assert params["nglobalvar"] == 2
    assert params["globalvars"] == ["zs", "zb"]


def test_params_with_points():
    """Test params property with points."""
    output = Output(
        points=[(0.0, 100.0), (500.0, 200.0)],
        pointvars=[OutputVarsEnum.H],
    )
    params = output.params
    assert params["npoints"] == 2
    assert params["points"] == ["0.0 100.0", "500.0 200.0"]
    assert params["npointvar"] == 1
    assert params["pointvars"] == ["H"]


def test_params_with_rugauges():
    """Test params property with runup gauges."""
    output = Output(
        rugauges=[(100.0, 200.0)],
        pointvars=[OutputVarsEnum.H, OutputVarsEnum.U],
    )
    params = output.params
    assert params["nrugauge"] == 1
    assert params["rugauges"] == ["100.0 200.0"]
    assert params["npointvar"] == 2


def test_params_excludes_none():
    """Test that params excludes None values."""
    output = Output(
        outputformat="netcdf",
        ncfilename="output.nc",
        tstart=None,
        tintg=None,
    )
    params = output.params
    assert "tstart" not in params
    assert "tintg" not in params
    assert "ncfilename" in params


def test_params_empty_lists_excluded():
    """Test that empty lists are excluded from params."""
    output = Output(
        meanvars=[],
        globalvars=[],
        pointvars=[],
    )
    params = output.params
    assert "nmeanvar" not in params
    assert "meanvars" not in params
    assert "nglobalvar" not in params
    assert "globalvars" not in params


def test_params_with_timing_fields():
    """Test params with timing configuration."""
    output = Output(
        tstart=0.0,
        tintg=10.0,
        tintm=3600.0,
        tintp=5.0,
    )
    params = output.params
    assert params["tstart"] == 0.0
    assert params["tintg"] == 10.0
    assert params["tintm"] == 3600.0
    assert params["tintp"] == 5.0


def test_params_with_file_timing(timing_file):
    """Test params with file-based timing.

    Note: DataBlob fields are excluded from .params to prevent field leakage.
    Use .get() to fetch files and get the file paths.
    """
    output = Output(
        tsglobal=dict(source=str(timing_file)),
        tsmean=dict(source=str(timing_file)),
        tspoint=dict(source=str(timing_file)),
    )
    params = output.params

    # DataBlob fields are excluded from params to prevent leakage
    assert "tsglobal" not in params
    assert "tsmean" not in params
    assert "tspoint" not in params

    # But the DataBlob objects are accessible on the instance
    assert output.tsglobal is not None
    assert output.tsmean is not None
    assert output.tspoint is not None


def test_params_timings_bool_to_int():
    """Test that timings boolean is serialized to int."""
    output = Output(timings=True)
    params = output.params
    assert params["timings"] == 1

    output = Output(timings=False)
    params = output.params
    assert params["timings"] == 0


# =====================================================================================
# get() method tests
# =====================================================================================
def test_get_method_without_timing_files(tmp_path):
    """Test get() method when no timing files are specified."""
    output = Output(
        globalvars=["H", "zs"],
        tintg=10.0,
    )

    params = output.get(tmp_path)

    # Should return params without timing files
    assert "tintg" in params
    assert params["tintg"] == 10.0
    assert "tsglobal" not in params

    # No files should be created in destdir
    assert len(list(tmp_path.iterdir())) == 0


def test_get_method_with_single_timing_file(timing_file, tmp_path):
    """Test get() method with a single timing file."""
    output = Output(
        globalvars=["H", "zs"],
        tsglobal=dict(source=str(timing_file)),
    )

    params = output.get(tmp_path)

    # Should have fetched the timing file
    assert "tsglobal" in params

    # params contains just the filename, file should exist in destdir
    fetched_file = tmp_path / params["tsglobal"]
    assert fetched_file.exists()
    assert fetched_file.parent == tmp_path

    # File content should match original
    original_content = timing_file.read_text()
    fetched_content = fetched_file.read_text()
    assert fetched_content == original_content

    # Verify file format
    lines = fetched_content.splitlines()
    assert lines[0] == "6"  # Number of times
    assert lines[1] == "0"  # First time
    assert lines[-1] == "500"  # Last time


def test_get_method_with_multiple_timing_files(timing_file, tmp_path):
    """Test get() method with multiple timing files."""
    output = Output(
        globalvars=["H", "zs"],
        tsglobal=dict(source=str(timing_file)),
        meanvars=["H", "u"],
        tsmean=dict(source=str(timing_file)),
        pointvars=["H"],
        points=[(0.0, 500.0)],
        tspoint=dict(source=str(timing_file)),
    )

    params = output.get(tmp_path)

    # All three timing files should be fetched
    assert "tsglobal" in params
    assert "tsmean" in params
    assert "tspoint" in params

    # params contains just filenames, files should exist in destdir
    global_file = tmp_path / params["tsglobal"]
    mean_file = tmp_path / params["tsmean"]
    point_file = tmp_path / params["tspoint"]

    assert global_file.exists()
    assert mean_file.exists()
    assert point_file.exists()

    assert global_file.parent == tmp_path
    assert mean_file.parent == tmp_path
    assert point_file.parent == tmp_path

    # Content should match original
    original_content = timing_file.read_text()
    assert global_file.read_text() == original_content
    assert mean_file.read_text() == original_content
    assert point_file.read_text() == original_content


def test_get_method_preserves_other_params(timing_file, tmp_path):
    """Test that get() method preserves all other parameters."""
    output = Output(
        outputformat="netcdf",
        ncfilename="test_output.nc",
        globalvars=["H", "zs"],
        meanvars=["H", "u", "v"],
        tstart=0.0,
        tintg=10.0,
        tintm=3600.0,
        tsglobal=dict(source=str(timing_file)),
        timings=True,
    )

    params = output.get(tmp_path)

    # Check that all non-timing params are preserved
    assert params["outputformat"] == "netcdf"
    assert params["ncfilename"] == "test_output.nc"
    assert params["nglobalvar"] == 2
    assert params["nmeanvar"] == 3
    assert params["tstart"] == 0.0
    assert params["tintg"] == 10.0
    assert params["tintm"] == 3600.0
    assert params["timings"] == 1

    # Timing file should be fetched (params contains just filename)
    assert "tsglobal" in params
    assert (tmp_path / params["tsglobal"]).exists()


# =====================================================================================
# Validation tests
# =====================================================================================
@pytest.mark.parametrize(
    "field_name,count,limit",
    [
        ("meanvars", 20, 15),
        ("globalvars", 25, 20),
        ("pointvars", 60, 50),
    ],
)
def test_validation_variable_limits(caplog, field_name, count, limit, all_output_vars):
    """Test warning when variable lists exceed XBeach limits."""
    # Use unique variables from the enum, cycling if needed
    value = (all_output_vars * (count // len(all_output_vars) + 1))[:count]

    with caplog.at_level(logging.WARNING):
        Output(**{field_name: value})
    assert f"More than {limit} {field_name} requested" in caplog.text


@pytest.mark.parametrize(
    "field_name,count,limit",
    [
        ("points", 60, 50),
        ("rugauges", 60, 50),
    ],
)
def test_validation_location_limits(caplog, field_name, count, limit):
    """Test warning when location lists exceed XBeach limits."""
    # Create unique coordinate pairs
    value = [(float(i), float(i)) for i in range(count)]

    with caplog.at_level(logging.WARNING):
        Output(**{field_name: value})
    assert f"More than {limit} {field_name} requested" in caplog.text


def test_validation_pointvars_without_locations(caplog):
    """Test warning when pointvars defined without points or rugauges."""
    with caplog.at_level(logging.WARNING):
        Output(pointvars=[OutputVarsEnum.H, OutputVarsEnum.U])
    assert "pointvars" in caplog.text
    assert "no point locations" in caplog.text


def test_validation_points_without_pointvars(caplog):
    """Test warning when points defined without pointvars."""
    with caplog.at_level(logging.WARNING):
        Output(points=[(0.0, 100.0)])
    assert "Point locations" in caplog.text
    assert "no point output variables" in caplog.text


def test_validation_rugauges_without_pointvars(caplog):
    """Test warning when rugauges defined without pointvars."""
    with caplog.at_level(logging.WARNING):
        Output(rugauges=[(0.0, 100.0)])
    assert "runup gauge locations" in caplog.text
    assert "no point output variables" in caplog.text


def test_validation_no_duplicate_variables():
    """Test that duplicate variables in lists are rejected."""
    # Test meanvars
    with pytest.raises(ValueError, match="Duplicate variables found in meanvars"):
        Output(meanvars=["H", "u", "H"])

    # Test globalvars
    with pytest.raises(ValueError, match="Duplicate variables found in globalvars"):
        Output(globalvars=["zs", "H", "zs"])

    # Test pointvars
    with pytest.raises(ValueError, match="Duplicate variables found in pointvars"):
        Output(pointvars=["H", "u", "v", "u"])


def test_validation_unique_variables_ok():
    """Test that unique variables are accepted."""
    # Should not raise
    output = Output(
        meanvars=["H", "u", "v", "zs"],
        globalvars=["H", "zs"],
        pointvars=["H", "u", "v"],
    )
    assert len(output.meanvars) == 4
    assert len(output.globalvars) == 2
    assert len(output.pointvars) == 3


def test_validation_fixed_and_file_times_global(caplog, timing_file):
    """Test warning when both fixed and file times defined for global output."""
    with caplog.at_level(logging.WARNING):
        Output(
            tintg=10.0,
            tsglobal=dict(source=str(timing_file)),
        )
    assert (
        "Global times defined by both fixed (tintg) and file (tsglobal)" in caplog.text
    )
    assert "supersede" in caplog.text


def test_validation_fixed_and_file_times_mean(caplog, timing_file):
    """Test warning when both fixed and file times defined for mean output."""
    with caplog.at_level(logging.WARNING):
        Output(
            tintm=3600.0,
            tsmean=dict(source=str(timing_file)),
        )
    assert "Mean times defined by both fixed (tintm) and file (tsmean)" in caplog.text


def test_validation_fixed_and_file_times_point(caplog, timing_file):
    """Test warning when both fixed and file times defined for point output."""
    with caplog.at_level(logging.WARNING):
        Output(
            tintp=5.0,
            tspoint=dict(source=str(timing_file)),
        )
    assert "Point times defined by both fixed (tintp) and file (tspoint)" in caplog.text


# =====================================================================================
# Complex configuration tests
# =====================================================================================
def test_output_full_configuration():
    """Test Output with a comprehensive configuration."""
    output = Output(
        outputformat="netcdf",
        ncfilename="xbeach_output.nc",
        outputprecision="double",
        meanvars=[
            OutputVarsEnum.H,
            OutputVarsEnum.U,
            OutputVarsEnum.V,
            OutputVarsEnum.ZS,
            OutputVarsEnum.ZB,
        ],
        globalvars=[
            OutputVarsEnum.H,
            OutputVarsEnum.ZS,
        ],
        points=[(0.0, 500.0), (1000.0, 500.0)],
        pointvars=[OutputVarsEnum.H, OutputVarsEnum.U, OutputVarsEnum.V],
        rugauges=[(500.0, 500.0)],
        tstart=0.0,
        tintg=10.0,
        tintm=3600.0,
        tintp=5.0,
        timings=True,
    )

    params = output.params

    # Check output format
    assert params["outputformat"] == "netcdf"
    assert params["ncfilename"] == "xbeach_output.nc"
    assert params["outputprecision"] == "double"

    # Check mean variables
    assert params["nmeanvar"] == 5
    assert len(params["meanvars"]) == 5

    # Check global variables
    assert params["nglobalvar"] == 2
    assert len(params["globalvars"]) == 2

    # Check points
    assert params["npoints"] == 2
    assert len(params["points"]) == 2

    # Check point variables
    assert params["npointvar"] == 3
    assert len(params["pointvars"]) == 3

    # Check runup gauges
    assert params["nrugauge"] == 1
    assert len(params["rugauges"]) == 1

    # Check timing
    assert params["tstart"] == 0.0
    assert params["tintg"] == 10.0
    assert params["tintm"] == 3600.0
    assert params["tintp"] == 5.0
    assert params["timings"] == 1


def test_output_minimal_configuration():
    """Test Output with minimal configuration."""
    output = Output()
    params = output.params

    # Should only have default values
    assert params["outputformat"] == "netcdf"
    # No variables or locations defined
    assert "nmeanvar" not in params
    assert "nglobalvar" not in params
    assert "npoints" not in params


# =====================================================================================
# Edge cases
# =====================================================================================
def test_output_single_variable():
    """Test Output with single variable in each category."""
    output = Output(
        meanvars=[OutputVarsEnum.H],
        globalvars=[OutputVarsEnum.ZS],
        pointvars=[OutputVarsEnum.U],
        points=[(0.0, 0.0)],
    )
    params = output.params
    assert params["nmeanvar"] == 1
    assert params["nglobalvar"] == 1
    assert params["npointvar"] == 1
    assert params["npoints"] == 1


def test_output_coordinate_formatting():
    """Test that coordinates are formatted correctly as strings."""
    output = Output(
        points=[(123.456, 789.012), (-45.67, -89.01)],
        pointvars=[OutputVarsEnum.H],
    )
    params = output.params
    assert params["points"][0] == "123.456 789.012"
    assert params["points"][1] == "-45.67 -89.01"


def test_output_enum_value_extraction():
    """Test that enum values are correctly extracted."""
    output = Output(
        meanvars=[OutputVarsEnum.THETAMEAN, OutputVarsEnum.URMS],
    )
    params = output.params
    assert "thetamean" in params["meanvars"]
    assert "urms" in params["meanvars"]
    # Should not contain enum representation
    assert "OutputVarsEnum" not in str(params["meanvars"])


def test_output_projection_parameters():
    """Test output projection parameters."""
    output = Output(
        projection="EPSG:4326",
        remdryoutput=True,
        rotate=False,
    )
    params = output.params
    assert params["projection"] == "EPSG:4326"
    assert params["remdryoutput"] == 1
    assert params["rotate"] == 0


def test_output_projection_string():
    """Test projection string with various CRS formats."""
    # EPSG format
    output1 = Output(projection="EPSG:32633")
    assert output1.params["projection"] == "EPSG:32633"

    # PROJ4 format
    output2 = Output(projection="+proj=utm +zone=33 +datum=WGS84")
    assert output2.params["projection"] == "+proj=utm +zone=33 +datum=WGS84"

    # WKT format (partial example)
    output3 = Output(projection='GEOGCS["WGS 84",DATUM["WGS_1984"]]')
    assert 'GEOGCS["WGS 84",DATUM["WGS_1984"]]' in output3.params["projection"]


def test_output_projection_defaults():
    """Test that projection parameters are not included when not specified."""
    output = Output()
    params = output.params
    assert "projection" not in params
    assert "remdryoutput" not in params
    assert "rotate" not in params
