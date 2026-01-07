# Output Component

The `Output` component controls what XBeach writes to output files, including format, timing, and variable selection.

## Overview

```python
from rompy_xbeach.components.output import Output

output = Output(
    outputformat="netcdf",
    tintg=3600,      # Global output every hour
    tintm=3600,      # Mean output every hour
    tintp=60,        # Point output every minute
)
```

## Output Format

```python
output = Output(
    outputformat="netcdf",  # or "fortran"
)
```

| Value | Description |
|-------|-------------|
| `netcdf` | NetCDF format (recommended) |
| `fortran` | Fortran binary format |

## Output Timing

Control when outputs are written:

```python
output = Output(
    tstart=0,        # Start output at t=0
    tintg=3600,      # Global output interval (s)
    tintm=3600,      # Mean output interval (s)
    tintp=60,        # Point output interval (s)
    tintc=1,         # Cross-shore output interval (s)
)
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tstart` | Time to start output (s) | 0 |
| `tintg` | Global (spatial) output interval (s) | 1 |
| `tintm` | Time-averaged output interval (s) | tstop |
| `tintp` | Point output interval (s) | 1 |
| `tintc` | Cross-shore output interval (s) | 1 |

## Output Variables

### Global Variables

Spatial fields written at `tintg` intervals:

```python
output = Output(
    nglobalvar=3,
    globalvar=["zs", "zb", "H"],
)
```

Common global variables:

| Variable | Description |
|----------|-------------|
| `zs` | Water level |
| `zb` | Bed level |
| `H` | Wave height |
| `ue` | Eulerian velocity (x) |
| `ve` | Eulerian velocity (y) |
| `sedero` | Cumulative erosion/deposition |
| `hh` | Water depth |

### Mean Variables

Time-averaged fields written at `tintm` intervals:

```python
output = Output(
    nmeanvar=2,
    meanvar=["zs", "H"],
)
```

### Point Variables

Time series at specific locations:

```python
output = Output(
    npointvar=3,
    pointvar=["zs", "H", "ue"],
    npoints=2,
    xpointsw=[100, 200],
    ypointsw=[50, 50],
)
```

## Runup Gauges

Track runup at cross-shore transects:

```python
output = Output(
    nrugauge=3,           # Number of runup gauges
    nrugdepth=2,          # Number of depth thresholds
    rugdepth=[0.01, 0.1], # Depth thresholds (m)
)
```

## Hotstart Output

Write hotstart files for restarting simulations:

```python
output = Output(
    writehotstart=True,   # Enable hotstart writing
    tinth=3600,           # Hotstart output interval (s)
)
```

If `tinth=0` or not specified, hotstart is only written at simulation end.

## Custom Timing Files

For non-uniform output timing:

```python
output = Output(
    tsglobal=dict(
        source="/path/to/global_times.txt",
    ),
    tsmean=dict(
        source="/path/to/mean_times.txt",
    ),
)
```

## Complete Example

```python
from rompy_xbeach.components.output import Output

output = Output(
    # Format
    outputformat="netcdf",
    
    # Timing
    tstart=0,
    tintg=3600,      # Hourly spatial output
    tintm=3600,      # Hourly means
    tintp=60,        # Minute point output
    
    # Global variables
    nglobalvar=5,
    globalvar=["zs", "zb", "H", "ue", "ve"],
    
    # Mean variables
    nmeanvar=3,
    meanvar=["zs", "H", "sedero"],
    
    # Point output
    npointvar=3,
    pointvar=["zs", "H", "ue"],
    npoints=2,
    xpointsw=[100, 200],
    ypointsw=[50, 50],
    
    # Runup gauges
    nrugauge=3,
    nrugdepth=1,
    rugdepth=[0.01],
    
    # Hotstart
    writehotstart=True,
    tinth=7200,      # Every 2 hours
)
```

## YAML Configuration

```yaml
output:
  outputformat: netcdf
  
  tstart: 0
  tintg: 3600
  tintm: 3600
  tintp: 60
  
  nglobalvar: 5
  globalvar:
    - zs
    - zb
    - H
    - ue
    - ve
    
  nmeanvar: 3
  meanvar:
    - zs
    - H
    - sedero
    
  writehotstart: true
  tinth: 7200
```

## API Reference

See [Output API Reference](../api-reference/components.md#output) for the complete API documentation.
