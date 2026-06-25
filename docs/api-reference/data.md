# Data API

API reference for data interfaces that generate boundary conditions from external sources.

## Wave Boundaries

### JONS Boundary Type

::: rompy_xbeach.data.boundary.BoundaryStationSpectraJons
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryStationParamJons
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryPointParamJons
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryGridParamJons
    options:
      show_root_heading: true
      show_source: false

### JONSTABLE Boundary Type

::: rompy_xbeach.data.boundary.BoundaryStationSpectraJonstable
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryStationParamJonstable
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryPointParamJonstable
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryGridParamJonstable
    options:
      show_root_heading: true
      show_source: false

### SWAN Boundary Type

::: rompy_xbeach.data.boundary.BoundaryStationSpectraSwan
    options:
      show_root_heading: true
      show_source: false

### File-Based Boundaries

For using pre-existing boundary files:

::: rompy_xbeach.data.boundary.BoundaryFileJons
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryFileJonstable
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryFileSwan
    options:
      show_root_heading: true
      show_source: false

### Non-Spectral Boundaries

::: rompy_xbeach.data.boundary.BoundaryStat
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryBichrom
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryStatTable
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryTs1
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryTs2
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryTsNonh
    options:
      show_root_heading: true
      show_source: false

### Special Boundaries

::: rompy_xbeach.data.boundary.BoundaryOff
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.BoundaryReuse
    options:
      show_root_heading: true
      show_source: false

### Base Parameter Classes

::: rompy_xbeach.data.boundary.WaveBoundaryParams
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.boundary.SpectralWaveBoundaryParams
    options:
      show_root_heading: true
      show_source: false

---

## Tide / Water Level

### Tidal Constituents

::: rompy_xbeach.data.waterlevel.TideConsGrid
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.waterlevel.TideConsPoint
    options:
      show_root_heading: true
      show_source: false

### Water Level Timeseries

::: rompy_xbeach.data.waterlevel.WaterLevelGrid
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.waterlevel.WaterLevelStation
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.waterlevel.WaterLevelPoint
    options:
      show_root_heading: true
      show_source: false

---

## Wind

::: rompy_xbeach.data.wind.WindGrid
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.wind.WindStation
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.data.wind.WindPoint
    options:
      show_root_heading: true
      show_source: false
