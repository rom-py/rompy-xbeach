# Sediment and morphology

[`Sediment`][rompy_xbeach.components.sediment.sediment.Sediment] holds the XBeach settings for sediment transport and bed change: the transport formulation and its calibration, the transport numerics, quasi-3D transport, morphological acceleration and avalanching, the bed composition and layering, and groundwater flow. It is the `sediment` field of [`Config`][rompy_xbeach.config.Config]. As in the other components, a field left as `None` is not written, and XBeach uses its own default.

```python exec="on" session="sediment"
# Hidden setup: quiet logging and a temporary folder for files.
import tempfile
from pathlib import Path

from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
TMP = Path(tempfile.mkdtemp())
```

## Switching transport and morphology on or off

`Config.sediment` defaults to an empty `Sediment()`, which writes nothing, and `sediment=None` leaves the component out; both leave the choice to XBeach. XBeach 1.24 computes sediment transport by default with the stationary and surfbeat wave models (not with `Nonh`), and bed updating (`morphology`) and avalanching follow `sedtrans` unless set.

!!! warning "Hydrodynamics-only runs"
    A run with the stationary or surfbeat model and no sediment settings updates the bed. To keep the bed fixed, switch transport and morphology off explicitly.

`sedtrans`, `morphology` and `q3d` each take `True`, `False` or a component with settings:

```python exec="on" source="above" result="text" session="sediment"
from rompy_xbeach.components.sediment import Sediment

sediment = Sediment(sedtrans=False, morphology=False)
print(sediment.get(destdir=TMP))

sediment = Sediment(sedtrans=True, morphology=True)
print(sediment.get(destdir=TMP))
```

## Transport formulation and calibration

[`SedimentTransport`][rompy_xbeach.components.sediment.transport.SedimentTransport] switches transport on and sets how it is computed:

| Setting | Fields |
|---|---|
| Equilibrium concentration formula | `form`: `soulsby_vanrijn`, `vanthiel_vanrijn` or `vanrijn1993` |
| Wave shape | `waveform`: `ruessink_vanrijn` or `vanthiel` |
| Onshore transport by wave skewness and asymmetry | `facua`, or `facSk` and `facAs` separately |
| Bed slope effects | `bdslpeffmag`, `bdslpeffdir`, `bdslpeffini`, `bdslpeffdirfac`, `facsl` |
| Stirring and turbulence | `sws`, `lws`, `lwt`, `turb`, `turbadv`, `Tbfac`, `BRfac` |
| Bed and suspended load factors | `bed`, `sus`, `bulk` |
| Other calibration | `Tsmin`, `tsfac`, `facDc`, `smax`, `z0`, `jetfac`, `bermslope`, `dilatancy`, ... |

`facua` is the usual calibration parameter for beach and dune profiles: larger values move more sediment onshore.

```python exec="on" source="above" result="text" session="sediment"
from rompy_xbeach.components.sediment.transport import SedimentTransport

sediment = Sediment(
    sedtrans=SedimentTransport(
        form="vanthiel_vanrijn",
        facua=0.15,
        bdslpeffmag="roelvink_total",
    )
)
print(sediment.get(destdir=TMP))
```

rompy-xbeach logs a warning for `turb="bore_averaged"` with `waveform="ruessink_vanrijn"`, which XBeach cannot combine.

### Transport numerics and quasi-3D transport

[`TransportNumerics`][rompy_xbeach.components.sediment.transport.TransportNumerics] (`numerics`) sets the maximum concentration `cmax`, the advection scheme weight `thetanum`, the adaptation time limiter (`dtlimts`, `oldTsmin`) and `sourcesink`, which computes bed change from source and sink terms. [`Quasi3D`][rompy_xbeach.components.sediment.transport.Quasi3D] (`q3d`) switches on the quasi-3D model for the vertical structure of flow and suspended sediment, with `kmax` sigma layers.

```python exec="on" source="above" result="text" session="sediment"
from rompy_xbeach.components.sediment.transport import Quasi3D, TransportNumerics

sediment = Sediment(
    numerics=TransportNumerics(cmax=0.1, thetanum=0.5),
    q3d=Quasi3D(kmax=20),
)
print(sediment.get(destdir=TMP))
```

## Morphological acceleration

[`Morphology`][rompy_xbeach.components.sediment.morphology.Morphology] switches bed updating on and sets its timing. `morfac` multiplies the bed change of each time step, so one hour of hydrodynamics gives `morfac` hours of bed change. `morstart` delays bed updating until the hydrodynamics have spun up, and `morstop` ends it.

```python exec="on" source="above" result="text" session="sediment"
from rompy_xbeach.components.sediment.morphology import Morphology

sediment = Sediment(morphology=Morphology(morfac=10.0, morstart=3600.0, morstop=36000.0))
print(sediment.get(destdir=TMP))
```

With `morfacopt` on (the XBeach default), XBeach reads all times in morphological time: `tstop` (set from the run period), `morstart`, `morstop`, the output times and the time axis of the boundary conditions are divided by `morfac` internally. A run period of 10 days with `morfac=10` therefore computes 1 day of hydrodynamics, and the forcing is compressed to fit. With `morfacopt=False` the times stay hydrodynamic.

## Avalanching

Wet and dry slopes steeper than a critical value collapse. `wetslp` applies below water and `dryslp` above, with `hswitch` the depth that separates them; `dzmax` limits the bed change per time step. Avalanching is switched with `Physics(avalanching=...)`, and follows `morphology` in XBeach when not set.

```python exec="on" source="above" result="text" session="sediment"
sediment = Sediment(morphology=Morphology(wetslp=0.3, dryslp=1.0, hswitch=0.1))
print(sediment.get(destdir=TMP))
```

## Non-erodible layers

Rock, revetments or reefs are represented by `struct=True` and an `ne_layer` file, which gives the thickness of erodible sediment above the hard layer at each grid point, in the bathymetry file layout. The file is given as an [`XBeachDataBlob`][rompy_xbeach.types.XBeachDataBlob], copied into the workspace and written by name.

```python exec="on" source="above" result="text" session="sediment"
import numpy as np

from rompy_xbeach.types import XBeachDataBlob

thickness = np.full((110, 115), 5.0)
thickness[:, 90:] = 0.0  # a revetment near the back of the domain
np.savetxt(TMP / "ne_layer.txt", thickness, fmt="%.1f")

sediment = Sediment(
    morphology=Morphology(struct=True, ne_layer=XBeachDataBlob(source=TMP / "ne_layer.txt"))
)
print(sediment.get(destdir=TMP / "workspace"))
```

## Bed composition

[`BedComposition`][rompy_xbeach.components.sediment.composition.BedComposition] (`bed_composition`) describes the sediment: the grain sizes `D50`, `D90` (and `D15`, used with `dilatancy`), the density `rhos`, the porosity `por`, the number of bed layers `nd` and their thicknesses `dzg1`, `dzg2` and `dzg3`.

```python exec="on" source="above" result="text" session="sediment"
from rompy_xbeach.components.sediment.composition import BedComposition

sediment = Sediment(bed_composition=BedComposition(D50=0.0003, D90=0.0005, por=0.4))
print(sediment.get(destdir=TMP))
```

Graded sediment uses `ngd` classes, with one value per class in `D50`, `D90`, `D15`, `sedcal` and `ucrcal`. The lists are written space-separated. Their lengths must match `ngd`, and each D90 must exceed its D50:

```python exec="on" source="above" result="text" session="sediment"
sediment = Sediment(
    bed_composition=BedComposition(
        ngd=2, D50=[0.0002, 0.0008], D90=[0.0003, 0.0012], nd=5, dzg1=0.05
    )
)
print(sediment.get(destdir=TMP))
```

```python exec="on" source="above" result="text" session="sediment"
from pydantic import ValidationError

try:
    BedComposition(D50=0.0005, D90=0.0003)
except ValidationError as err:
    print(err.errors()[0]["msg"])
```

With more than one class, XBeach also reads the fraction of each class in each bed layer from the files `gdist1.inp`, `gdist2.inp`, and so on. rompy-xbeach does not write these, so add them to the workspace yourself, for example through the configuration's template.

## Bed layers and prescribed bed updates

[`BedUpdate`][rompy_xbeach.components.sediment.bed.BedUpdate] (`bed_update`) controls how the bed layers split and merge during bed updating (`frac_dz`, `split`, `merge`, `nd_var`). It can also impose bed levels at given times from a `setbathyfile`, for example for nourishments or dredging. `nsetbathy`, the number of bed levels in the file, must then be given, and the process switched on with `Physics(setbathy=True)`. Computed bed changes are overwritten by the prescribed ones, so morphology is normally switched off with prescribed bed updates.

```python exec="on" source="above" result="text" session="sediment"
from rompy_xbeach.components.sediment.bed import BedUpdate

sediment = Sediment(bed_update=BedUpdate(frac_dz=0.7, split=1.01, merge=0.01))
print(sediment.get(destdir=TMP))
```

## Groundwater

[`GroundwaterFlow`][rompy_xbeach.components.sediment.groundwater.GroundwaterFlow] (`groundwater`) describes the aquifer: its bottom level (`aquiferbot` or an `aquiferbotfile`), the initial groundwater level (`gw0` or a `gw0file`), the permeabilities `kx`, `ky` and `kz`, and the flow scheme. Groundwater matters for swash infiltration and gravel beaches. The process itself is switched on with `Physics(gwflow=True)`.

```python exec="on" source="above" result="text" session="sediment"
from rompy_xbeach.components.sediment.groundwater import GroundwaterFlow

sediment = Sediment(groundwater=GroundwaterFlow(aquiferbot=-5.0, gw0=0.0, kx=0.001))
print(sediment.get(destdir=TMP))
```

## A storm erosion setup

A typical configuration for a sandy beach, and its YAML equivalent under the `sediment` key of the model config:

```python exec="on" source="above" result="text" session="sediment"
sediment = Sediment(
    sedtrans=SedimentTransport(form="vanthiel_vanrijn", facua=0.1),
    morphology=Morphology(morfac=5.0, morstart=3600.0, wetslp=0.3),
    bed_composition=BedComposition(D50=0.0003, D90=0.0005),
)
print(sediment.get(destdir=TMP))
```

```python exec="on" source="above" result="text" session="sediment"
import yaml

sediment = Sediment.model_validate(
    yaml.safe_load(
        """
        sedtrans:
          form: vanthiel_vanrijn
          facua: 0.1
        morphology:
          morfac: 5.0
          morstart: 3600.0
          wetslp: 0.3
        bed_composition:
          D50: 0.0003
          D90: 0.0005
        """
    )
)
print(sediment.get(destdir=TMP))
```

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["sediment"]))
```
