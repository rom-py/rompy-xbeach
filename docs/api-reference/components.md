# Components API

API reference for all rompy-xbeach components.

## Physics

### Physics (Aggregator)

::: rompy_xbeach.components.physics.Physics
    options:
      show_root_heading: true
      show_source: false

### Wave Models

::: rompy_xbeach.components.physics.wavemodel.Stationary
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.components.physics.wavemodel.Surfbeat
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.components.physics.wavemodel.Nonh
    options:
      show_root_heading: true
      show_source: false

### Breaker Formulations

::: rompy_xbeach.components.physics.wavemodel.Roelvink1
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.components.physics.wavemodel.Roelvink2
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.components.physics.wavemodel.RoelvinkDaly
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.components.physics.wavemodel.Baldock
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.components.physics.wavemodel.Janssen
    options:
      show_root_heading: true
      show_source: false

### Friction

::: rompy_xbeach.components.physics.friction.BedFriction
    options:
      show_root_heading: true
      show_source: false

### Viscosity

::: rompy_xbeach.components.physics.friction.Viscosity
    options:
      show_root_heading: true
      show_source: false

### Vegetation

::: rompy_xbeach.components.physics.vegetation.Vegetation
    options:
      show_root_heading: true
      show_source: false

### Numerics

::: rompy_xbeach.components.physics.numerics.WaveNumerics
    options:
      show_root_heading: true
      show_source: false

::: rompy_xbeach.components.physics.numerics.FlowNumerics
    options:
      show_root_heading: true
      show_source: false

### Constants

::: rompy_xbeach.components.physics.constants.PhysicalConstants
    options:
      show_root_heading: true
      show_source: false

---

## Sediment

### Sediment (Aggregator)

::: rompy_xbeach.components.sediment.Sediment
    options:
      show_root_heading: true
      show_source: false

### Transport

::: rompy_xbeach.components.sediment.transport.SedimentTransport
    options:
      show_root_heading: true
      show_source: false

### Morphology

::: rompy_xbeach.components.sediment.morphology.Morphology
    options:
      show_root_heading: true
      show_source: false

### Bed Composition

::: rompy_xbeach.components.sediment.composition.BedComposition
    options:
      show_root_heading: true
      show_source: false

### Groundwater

::: rompy_xbeach.components.sediment.groundwater.GroundwaterFlow
    options:
      show_root_heading: true
      show_source: false

---

## Output

::: rompy_xbeach.components.output.Output
    options:
      show_root_heading: true
      show_source: false

---

## Boundaries

### Flow Boundaries

::: rompy_xbeach.components.boundary.parameters.FlowBoundaryConditions
    options:
      show_root_heading: true
      show_source: false

### Tide Boundaries

::: rompy_xbeach.components.boundary.parameters.TideBoundaryConditions
    options:
      show_root_heading: true
      show_source: false

### Wave Boundaries

For manual wave boundary specification, see [Wave Boundaries Data API](../data.md#wave-boundaries).

---

## Hotstart

::: rompy_xbeach.components.hotstart.Hotstart
    options:
      show_root_heading: true
      show_source: false

---

## MPI

::: rompy_xbeach.components.mpi.Mpi
    options:
      show_root_heading: true
      show_source: false
