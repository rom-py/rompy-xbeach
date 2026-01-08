# Data Interfaces

Rompy-xbeach provides a comprehensive data interface layer that bridges external data sources with XBeach model inputs. This abstraction handles the complexity of:

- **Coordinate transformations** between different CRS systems
- **Spatial interpolation** from source data to model grids
- **Temporal slicing** to extract data for simulation periods
- **Format conversion** to XBeach-compatible input files

## Architecture

```
External Data Sources          Data Interfaces              XBeach Inputs
┌─────────────────┐           ┌─────────────────┐          ┌─────────────────┐
│ GeoTIFF         │──────────▶│ XBeachBathy     │─────────▶│ dep.txt         │
│ NetCDF          │           │                 │          │ x.txt, y.txt    │
│ XYZ files       │           └─────────────────┘          └─────────────────┘
└─────────────────┘
                              ┌─────────────────┐          ┌─────────────────┐
┌─────────────────┐           │ BoundaryStation │          │ jonswap.txt     │
│ Wave spectra    │──────────▶│ BoundaryGrid    │─────────▶│ jonstable.txt   │
│ Wave params     │           │ BoundaryPoint   │          │ swan_spec.txt   │
└─────────────────┘           └─────────────────┘          └─────────────────┘

┌─────────────────┐           ┌─────────────────┐          ┌─────────────────┐
│ Wind grids      │──────────▶│ WindStation     │─────────▶│ wind.txt        │
│ Wind stations   │           │ WindGrid        │          │                 │
│ CSV timeseries  │           │ WindPoint       │          │                 │
└─────────────────┘           └─────────────────┘          └─────────────────┘

┌─────────────────┐           ┌─────────────────┐          ┌─────────────────┐
│ Tide cons       │──────────▶│ TideConsGrid    │─────────▶│ zs0file.txt     │
│ Water levels    │           │ TideConsPoint   │          │                 │
│ SSH timeseries  │           │ WaterLevelPoint │          │                 │
└─────────────────┘           └─────────────────┘          └─────────────────┘
```

## Key Concepts

### Source Objects

Source objects define how to read external data. They handle:

- File format parsing (GeoTIFF, NetCDF, CSV, etc.)
- CRS information extraction
- Lazy loading for large datasets

```python
from rompy_xbeach.source import SourceGeotiff, SourceCRSDataset

# GeoTIFF bathymetry
source = SourceGeotiff(filename="bathymetry.tif")

# NetCDF with CRS
source = SourceCRSDataset(uri="data.nc", crs="EPSG:4326")
```

### The `get()` Method

All data interfaces implement a `get(destdir, grid, time)` method that:

1. Opens the source data
2. Extracts/interpolates to the grid location
3. Slices to the time period
4. Writes XBeach-compatible files
5. Returns parameters for `params.txt`

```python
# Generate XBeach input files
params = data_interface.get(
    destdir=Path("./run"),
    grid=grid,
    time=TimeRange(start="2024-01-01", end="2024-01-02"),
)
```

### Integration with Config

Data interfaces connect to `Config` through the `DataInterface` aggregator:

```python
from rompy_xbeach.config import Config, DataInterface

config = Config(
    grid=grid,
    bathy=bathy,
    input=DataInterface(
        wave=wave_boundary_data,
        tide=tide_data,
        wind=wind_data,
    ),
)
```

## Sections

| Section | Description |
|---------|-------------|
| [Grid](grid.md) | Grid definition with CRS support and coordinate transformations |
| [Sources](sources.md) | Source objects for loading data with CRS awareness |
| [Bathymetry](bathy.md) | Bathymetry interpolation, extension, and file generation |
| [Wave Boundaries](boundaries.md) | Wave boundary generation from spectra and parameters |
| [Forcing](forcing.md) | Wind and tide forcing from various data sources |

## Quick Reference

### Data Structure Types

| Type | Description | Example Classes |
|------|-------------|-----------------|
| **Grid** | Spatially gridded data | `WindGrid`, `TideConsGrid` |
| **Station** | Multi-point station data | `WindStation`, `BoundaryStationSpectraJons` |
| **Point** | Single-point timeseries | `WindPoint`, `WaterLevelPoint`, `BoundaryPointParamJons` |

### Supported Source Types

| Source | `model_type` | Use Case |
|--------|--------------|----------|
| `SourceGeotiff` | `geotiff` | Bathymetry rasters |
| `SourceCRSDataset` | `dataset` | NetCDF with coordinates |
| `SourceCRSWavespectra` | `wavespectra` | Wave spectra files |
| `SourceCRSOceantide` | `oceantide` | Tide constituent data |
| `SourceTimeseriesCSV` | `timeseries_csv` | CSV timeseries |
| `SourceXYZ` | `xyz` | XYZ point data |
