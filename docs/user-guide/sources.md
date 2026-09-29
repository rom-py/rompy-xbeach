# Data sources

A source describes where data lives and how to open it; its `open()` method returns an xarray dataset. The bathymetry and every forcing class take a source in their `source` field. rompy defines the base sources, and rompy-xbeach extends them to carry a coordinate reference system (CRS), which is needed to reproject data onto a projected XBeach grid. This page lists the sources, shows which forcing classes accept which, and covers the options that control which part of the data is used.

```python exec="on" session="sources"
# Hidden setup: quiet logging and a temporary output folder.
import tempfile

from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
OUT_DIR = tempfile.mkdtemp()
```

## The sources

| Source | From | `model_type` | Data |
|---|---|---|---|
| [`SourceGeotiff`][rompy_xbeach.source.SourceGeotiff] | rompy-xbeach | `geotiff` | GeoTIFF rasters; CRS read from the file |
| [`SourceCRSFile`][rompy_xbeach.source.SourceCRSFile] | rompy-xbeach | `file` | NetCDF, Zarr or anything `xarray.open_dataset` opens |
| [`SourceXYZ`][rompy_xbeach.source.SourceXYZ] | rompy-xbeach | `xyz` | x, y, z point clouds, gridded when opened |
| [`SourceCRSIntake`][rompy_xbeach.source.SourceCRSIntake] | rompy-xbeach | `intake` | A dataset in an intake catalogue |
| [`SourceCRSDataset`][rompy_xbeach.source.SourceCRSDataset] | rompy-xbeach | `dataset` | An xarray dataset in memory |
| [`SourceCRSWavespectra`][rompy_xbeach.source.SourceCRSWavespectra] | rompy-xbeach | `wavespectra` | Wave spectra read with a wavespectra reader |
| [`SourceCRSOceantide`][rompy_xbeach.source.SourceCRSOceantide] | rompy-xbeach | `oceantide` | Gridded tidal constituents read with an oceantide reader |
| [`SourceTideConsPointCSV`][rompy_xbeach.source.SourceTideConsPointCSV] | rompy-xbeach | `tide_cons_point_csv` | Tidal constituents at one site, from CSV |
| [`SourceTimeseriesCSV`][rompy.core.source.SourceTimeseriesCSV] | rompy | `csv` | A timeseries in a CSV file |

The CRS-aware sources extend rompy's [`SourceFile`][rompy.core.source.SourceFile], [`SourceIntake`][rompy.core.source.SourceIntake] and [`SourceWavespectra`][rompy.core.source.SourceWavespectra] with three fields: `crs`, and `x_dim` and `y_dim`, the names of the spatial dimensions (default `x` and `y`). `SourceCRSWavespectra` and `SourceCRSOceantide` default to `crs=4326` with `lon` and `lat`, the usual layout of those datasets.

`SourceCRSDataset` and the timeseries source `SourceTimeseriesDataFrame` come from the `rompy-binary-datasources` package, installed with `pip install rompy-xbeach[extra]`. They hold data in memory, so they cannot be written to YAML.

## Which classes accept which sources

| Data interface | Sources |
|---|---|
| [`XBeachBathy`][rompy_xbeach.data.bathy.XBeachBathy] | `geotiff`, `file`, `xyz`, `intake`, `dataset` |
| Wave spectra: `BoundaryStationSpectraJons`, `...Jonstable`, `...Swan` | `wavespectra`, `file`, `intake`, `dataset` |
| Wave parameters: `BoundaryStationParam...`, `BoundaryGridParam...` | `file`, `intake`, `dataset` |
| Wave parameters at a point: `BoundaryPointParam...` | `csv`, and rompy's `file`, `intake` and `dataset` without a CRS |
| `WindGrid`, `WindStation`, `WaterLevelGrid`, `WaterLevelStation` | `file`, `geotiff`, `xyz`, `intake`, `dataset` |
| `WindPoint`, `WaterLevelPoint` | `csv` (and `dataframe` with the extra package) |
| `TideConsGrid` | `oceantide` |
| `TideConsPoint` | `tide_cons_point_csv` |

The forcing classes are described in [Waves](waves.md), [Wind](wind.md) and [Tide and water levels](water-levels.md).

## Gridded data

### GeoTIFF

The CRS is read from the file, and `band` picks the raster band (default 1). The data variable is always called `data`, which is the default variable of `XBeachBathy`.

```python exec="on" source="above" result="text" session="sources"
from rompy_xbeach.source import SourceGeotiff

source = SourceGeotiff(filename="tests/data/bathy.tif")
ds = source.open()
print(ds.rio.crs, dict(ds.sizes), list(ds.data_vars))
```

### NetCDF and other xarray formats

`uri` is opened with `xarray.open_dataset`, and `kwargs` are passed to it. NetCDF files rarely declare a CRS, so you state it, with the names of the x and y dimensions:

```python exec="on" source="above" result="text" session="sources"
from rompy_xbeach.source import SourceCRSFile

source = SourceCRSFile(uri="tests/data/bathy.nc", crs=4326, x_dim="x", y_dim="y")
ds = source.open()
print(ds.rio.crs, dict(ds.sizes))
```

### XYZ point clouds

Survey or LiDAR points are interpolated onto a regular grid of resolution `res` (in CRS units) with `scipy.interpolate.griddata` when opened. `xcol`, `ycol` and `zcol` name the columns, `read_csv_kwargs` is passed to `pandas.read_csv` and `griddata_kwargs` to `griddata` (default `{"method": "linear"}`). The gridded variable is called `data`.

```python exec="on" source="above" result="text" session="sources"
from rompy_xbeach.source import SourceXYZ

source = SourceXYZ(
    filename="tests/data/bathy_xyz.zip",
    crs=4326,
    res=0.0005,
    xcol="easting",
    ycol="northing",
    zcol="elevation",
    read_csv_kwargs={"sep": "\t"},
)
print(dict(source.open().sizes))
```

### Intake catalogues

An intake catalogue lists datasets by name, so configurations can refer to data without hard-coding file paths. Give either `catalog_uri` or `catalog_yaml` (the catalogue as a string), and `dataset_id`:

```python exec="on" source="above" result="text" session="sources"
from rompy_xbeach.source import SourceCRSIntake

source = SourceCRSIntake(
    catalog_uri="tests/data/catalog.yaml", dataset_id="bathy_netcdf", crs=4326
)
print(dict(source.open().sizes))
```

### An xarray dataset in memory

Useful when data has been downloaded or processed in the same script. Needs `rompy-xbeach[extra]`:

```python
import xarray as xr
from rompy_xbeach.source import SourceCRSDataset

source = SourceCRSDataset(obj=xr.open_dataset("tests/data/bathy.nc"), crs=4326)
```

## Specialised sources

### Wave spectra

[`SourceCRSWavespectra`][rompy_xbeach.source.SourceCRSWavespectra] opens spectra with a [wavespectra](https://wavespectra.readthedocs.io) reader such as `read_ww3`, `read_swan` or `read_era5`; `kwargs` are passed to the reader.

```python exec="on" source="above" result="text" session="sources"
from rompy_xbeach.source import SourceCRSWavespectra

source = SourceCRSWavespectra(
    uri="tests/data/ww3-spectra-20230101-short.nc", reader="read_ww3"
)
print(dict(source.open().sizes))
```

### Tidal constituents

[`SourceCRSOceantide`][rompy_xbeach.source.SourceCRSOceantide] reads gridded constituents with an [oceantide](https://github.com/oceanum/oceantide) reader, here an OTIS binary model. [`SourceTideConsPointCSV`][rompy_xbeach.source.SourceTideConsPointCSV] reads amplitude and phase per constituent at one site; `ccol`, `acol` and `pcol` name its columns.

```python exec="on" source="above" result="text" session="sources"
from rompy_xbeach.source import SourceCRSOceantide, SourceTideConsPointCSV

cons_dir = "tests/data/swaus_tide_cons"
source = SourceCRSOceantide(
    reader="read_otis_binary",
    kwargs={
        "gfile": f"{cons_dir}/grid_m2s2n2k2k1o1p1q1mmmf",
        "hfile": f"{cons_dir}/h_m2s2n2k2k1o1p1q1mmmf",
        "ufile": f"{cons_dir}/u_m2s2n2k2k1o1p1q1mmmf",
    },
)
print("Gridded:", source.open().con.values)

source = SourceTideConsPointCSV(filename="tests/data/tide_cons_station.csv")
print("Point:  ", source.open().con.values)
```

### Timeseries

The `...Point` classes take a single timeseries, such as a buoy or tide gauge record, which needs no CRS. rompy's [`SourceTimeseriesCSV`][rompy.core.source.SourceTimeseriesCSV] reads one from CSV; `tcol` names the time column and the other columns become variables.

```python exec="on" source="above" result="text" session="sources"
from rompy.core.source import SourceTimeseriesCSV

source = SourceTimeseriesCSV(filename="tests/data/wind.csv", tcol="time")
print(source.open())
```

## Choosing which data is used

The forcing classes build on rompy's [`DataGrid`][rompy.core.data.DataGrid] and inherit its fields. rompy-xbeach adds the fields that pick a location on the XBeach grid.

| Field | From | Purpose |
|---|---|---|
| `source` | rompy | Where the data comes from |
| `coords` | rompy | Names of the `t`, `x`, `y` and site (`s`) coordinates |
| `variables` | rompy | Variables to read |
| `filter` | rompy | Sort, subset, crop or rename when the data is opened |
| `crop_data`, `time_buffer` | rompy | Crop the data to the run period, plus extra time steps |
| `buffer` | rompy | Spatial margin for cropping; no effect in rompy-xbeach, which samples points |
| `location` | rompy-xbeach | Where the data is taken: `centre` of the grid, or `offshore` boundary midpoint |
| `sel_method`, `sel_method_kwargs` | rompy-xbeach | How the data is interpolated to that location |

Wind and water levels are taken at the grid `centre` by default; wave boundaries always at the `offshore` midpoint.

The `ds` property returns the dataset exactly as the object sees it, with the source, variables and filters applied, which is the quickest way to check a configuration before writing files. `coords` must match the dataset: the default names are `longitude`, `latitude` and `time`, so data with `lon` and `lat` must say so.

```python exec="on" source="above" result="text" session="sources"
from rompy_xbeach.data.waterlevel import WaterLevelGrid

ssh_source = SourceCRSFile(
    uri="tests/data/ssh_gridded.nc", crs=4326, x_dim="lon", y_dim="lat"
)
waterlevel = WaterLevelGrid(
    source=ssh_source, coords={"x": "lon", "y": "lat"}, variables=["ssh"]
)
print(dict(waterlevel.ds.sizes))
```

### Filters

`filter.crop` selects a range along any coordinate when the data is opened, which keeps memory use low with large regional or global datasets. [`Filter`][rompy.core.filters.Filter] also provides `sort`, `subset` and `rename`. Filters are applied after `variables` are selected, so they cannot create the variables a class asks for.

```python exec="on" source="above" result="text" session="sources"
from rompy.core.filters import Filter

cropped = WaterLevelGrid(
    source=ssh_source,
    coords={"x": "lon", "y": "lat"},
    variables=["ssh"],
    filter=Filter(crop={"lon": slice(115.4, 115.8), "lat": slice(-32.8, -32.5)}),
)
print(dict(cropped.ds.sizes))
```

In YAML, a crop is written as `filter: {crop: {lon: {start: 115.4, stop: 115.8}}}`.

### Cropping to the run period

With `crop_data=True` (the default), `get()` limits the data to the run period plus `time_buffer` source time steps on each side (default one before and one after). The margin lets start and end times that fall between source time steps be interpolated. After `get()` the time crop appears in the filter:

```python exec="on" source="above" result="text" session="sources"
from rompy.core.time import TimeRange
from rompy_xbeach.grid import RegularGrid

grid = RegularGrid(
    ori={"x": 115.594239, "y": -32.641104, "crs": 4326},
    alfa=347.0, dx=10.0, dy=15.0, nx=230, ny=220, crs=28350,
)
period = TimeRange(start="2023-01-01T06", end="2023-01-01T12", interval="1h")

waterlevel.get(destdir=OUT_DIR, grid=grid, time=period)
print(waterlevel.filter.crop)
```

The run period must lie within the source's time range, otherwise `get()` stops with an error naming both ranges.

### Interpolating to the location: `sel_method`

Station data (a site dimension, with x and y as variables) and gridded data are sampled differently:

| Data | `sel_method` | Default | Options |
|---|---|---|---|
| Stations | `idw`, `nearest` | `idw` | wavespectra's `sel_idw` / `sel_nearest`; `sel_method_kwargs` such as `tolerance` or `max_sites` |
| Gridded | `sel`, `interp` | `sel` with `{"method": "nearest"}` | xarray's `sel` or `interp`; `sel_method_kwargs` are passed to them |

Both write the same file, with slightly different values. Inverse distance weighting needs neighbouring sites within its tolerance. With a single site, or none close enough, it returns only missing values and `get()` raises an error suggesting `sel_method="nearest"`.

```python exec="on" source="above" result="text" session="sources"
from pathlib import Path

from rompy_xbeach.data.waterlevel import WaterLevelStation

for method in ("idw", "nearest"):
    stations = WaterLevelStation(
        source=SourceCRSFile(
            uri="tests/data/ssh_stations.nc", crs=4326, x_dim="lon", y_dim="lat"
        ),
        coords={"s": "site", "x": "lon", "y": "lat"},
        variables=["ssh"],
        sel_method=method,
    )
    params = stations.get(destdir=OUT_DIR, grid=grid, time=period)
    first_line = (Path(OUT_DIR) / params["zs0file"]).read_text().splitlines()[0]
    print(f"{method:8s} {params['zs0file']}: {first_line}")
```

## See it in the notebooks

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["sources", "input-data"]))
```
