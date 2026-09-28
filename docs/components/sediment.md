# Sediment Component

The `Sediment` component controls sediment transport, morphological updating, bed composition, and groundwater flow.

## Overview

```python
from rompy_xbeach.components.sediment import Sediment
from rompy_xbeach.components.sediment.transport import SedimentTransport
from rompy_xbeach.components.sediment.morphology import Morphology
from rompy_xbeach.components.sediment.composition import BedComposition
from rompy_xbeach.components.sediment.bed import BedUpdate
from rompy_xbeach.components.sediment.groundwater import GroundwaterFlow

sediment = Sediment(
    sedtrans=SedimentTransport(form="vanthiel_vanrijn"),
    morphology=Morphology(morfac=10),
    bed_composition=BedComposition(D50=0.0002),
)
```

## Sub-Components

### Sediment Transport

Controls the sediment transport formulation:

```python
from rompy_xbeach.components.sediment.transport import SedimentTransport

transport = SedimentTransport(
    # Transport formulation
    form="vanthiel_vanrijn",
    
    # Wave shape effects
    waveform="vanthiel",
    
    # Stirring mechanisms
    turb="wave_averaged",  # Turbulence formulation
    sws=True,              # Short wave stirring
    lws=True,              # Long wave stirring
    lwt=False,             # Long wave turbulence
    
    # Calibration factors
    facua=0.1,             # Onshore transport factor
    facAs=0.1,             # Skewness factor
    facSk=0.1,             # Asymmetry factor
)
```

**Transport formulations (`form`):**

| Value | Description |
|-------|-------------|
| `soulsby_vanrijn` | Soulsby-Van Rijn (1997) |
| `vanthiel_vanrijn` | Van Thiel-Van Rijn (2009) |
| `vanrijn1993` | Van Rijn (1993) |

**Wave shape formulations (`waveform`):**

| Value | Description |
|-------|-------------|
| `ruessink_vanrijn` | Ruessink et al. (2012) |
| `vanthiel` | Van Thiel de Vries (2009) |

**Turbulence formulations (`turb`):**

| Value | Description |
|-------|-------------|
| `bore_averaged` | Bore-averaged turbulence |
| `wave_averaged` | Wave-averaged turbulence |
| `none` | No turbulence |

### Morphology

Controls morphological acceleration and timing:

```python
from rompy_xbeach.components.sediment.morphology import Morphology

morphology = Morphology(
    morfac=10,           # Morphological acceleration factor
    morstart=3600,       # Start morphology after 1 hour (s)
    morstop=86400,       # Stop morphology after 1 day (s)
    
    # Avalanching
    wetslp=0.3,          # Critical wet slope
    dryslp=1.0,          # Critical dry slope
    
    # Structures
    struct=False,        # Enable structures
    ne_layer=None,       # Non-erodible layer file
)
```

**Key parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| [`morfac`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.morfac) | Morphological acceleration factor | 1 |
| [`morstart`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.morstart) | Time to start morphology (s) | 0 |
| [`morstop`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.morstop) | Time to stop morphology (s) | tstop |
| [`wetslp`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.wetslp) | Critical wet slope for avalanching | 0.3 |
| [`dryslp`](../api-reference/components.md#rompy_xbeach.components.sediment.morphology.Morphology.dryslp) | Critical dry slope for avalanching | 1.0 |

### Bed Composition

Defines sediment properties:

```python
from rompy_xbeach.components.sediment.composition import BedComposition

bed = BedComposition(
    # Grain sizes (lists for each sediment class)
    D50=[0.0002],        # Median grain size (m)
    D90=[0.0003],        # 90th percentile (m)
    D15=[0.00015],       # 15th percentile (m)
    
    # Multiple grain classes
    ngd=1,               # Number of grain classes
    
    # Sediment properties
    rhos=2650,           # Sediment density (kg/m³)
    por=0.4,             # Porosity
    
    # Bed layers
    nd=3,                # Number of bed layers
    dzg1=0.1,            # Top layer thickness (m)
    dzg2=0.1,            # Middle layer thickness (m)
    dzg3=0.1,            # Bottom layer thickness (m)
    
    # Calibration
    sedcal=1.0,          # Sediment transport calibration
    ucrcal=1.0,          # Critical velocity calibration
)
```

**Validation:**

- `D90` must be greater than `D50`
- `D50` must be greater than `D15` (if specified)
- Grain size lists must match `ngd`

### Bed Update

Controls bed level updating:

```python
from rompy_xbeach.components.sediment.bed import BedUpdate

bed_update = BedUpdate(
    fwfile=None,         # Wave friction file
    fwcutoff=1000,       # Wave friction cutoff depth (m)
)
```

### Groundwater Flow

```python
from rompy_xbeach.components.sediment.groundwater import GroundwaterFlow

groundwater = GroundwaterFlow(
    gwflow=True,         # Enable groundwater
    gwnonh=False,        # Non-hydrostatic groundwater
    kx=0.0001,           # Horizontal permeability (m/s)
    ky=0.0001,           # Vertical permeability (m/s)
    gwheadmodel=1,       # Head boundary model
)
```

## Complete Example

```python
from rompy_xbeach.components.sediment import Sediment
from rompy_xbeach.components.sediment.transport import SedimentTransport
from rompy_xbeach.components.sediment.morphology import Morphology
from rompy_xbeach.components.sediment.composition import BedComposition
from rompy_xbeach.components.sediment.groundwater import GroundwaterFlow

sediment = Sediment(
    # Enable sediment transport with custom formulation
    sedtrans=SedimentTransport(
        form="vanthiel_vanrijn",
        waveform="vanthiel",
        turb="wave_averaged",
        sws=True,
        lws=True,
        facua=0.1,
    ),
    
    # Morphological acceleration
    morphology=Morphology(
        morfac=10,
        morstart=3600,
        wetslp=0.3,
        dryslp=1.0,
    ),
    
    # Sediment properties
    bed_composition=BedComposition(
        D50=[0.0002],
        D90=[0.0003],
        rhos=2650,
        por=0.4,
    ),
    
    # Groundwater (optional)
    groundwater=GroundwaterFlow(
        gwflow=True,
        kx=0.0001,
    ),
)
```

## YAML Configuration

```yaml
sediment:
  transport:
    form: vanthiel_vanrijn
    waveform: vanthiel
    turb: wave_averaged
    sws: true
    lws: true
    facua: 0.1
    
  morphology:
    morfac: 10
    morstart: 3600
    wetslp: 0.3
    dryslp: 1.0
    
  bed_composition:
    D50: 0.0002
    D90: 0.0003
    rhos: 2650
    por: 0.4
```

## API Reference

See [Sediment API Reference](../api-reference/components.md#sediment) for the complete API documentation.
