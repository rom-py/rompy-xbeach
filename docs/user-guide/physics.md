# Physics

[`Physics`][rompy_xbeach.components.physics.physics.Physics] holds the XBeach settings for the hydrodynamic processes: the wave model and its breaker formulation, the roller, bed friction, viscosity, vegetation, wave-current interaction, wind stress, the flow and wave numerics and the physical constants. It is the `physics` field of [`Config`][rompy_xbeach.config.Config] and the only component that must be given. Every field except the wave model defaults to `None`, which leaves the parameter out of `params.txt` so that XBeach uses its own default.

```python exec="on" session="physics"
# Hidden setup: quiet logging and a temporary folder for files.
import tempfile
from pathlib import Path

from rompy.logging import config as logging_config

logging_config.update(level="WARNING")
TMP = Path(tempfile.mkdtemp())
```

## Choosing the wave model

[`wavemodel`][rompy_xbeach.components.physics.physics.Physics.wavemodel] is required and takes one of three classes:

| Class | Writes | Resolves | Typical use |
|---|---|---|---|
| [`Stationary`][rompy_xbeach.components.physics.wavemodel.Stationary] | `wavemodel = stationary` | Wave-averaged energy, no wave groups or infragravity waves | Mild conditions, quick studies |
| [`Surfbeat`][rompy_xbeach.components.physics.wavemodel.Surfbeat] | `wavemodel = surfbeat` | Short-wave groups and the infragravity waves they force | Storm impact, dune erosion |
| [`Nonh`][rompy_xbeach.components.physics.wavemodel.Nonh] | `wavemodel = nonh` | Individual waves (non-hydrostatic) | Swash, overtopping, laboratory scale |

`get()` returns the parameters a component writes to `params.txt`:

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.physics.wavemodel import Surfbeat

physics = Physics(wavemodel=Surfbeat())
print(physics.get(destdir=TMP))
```

The wave model must also suit the wave boundary: [`Config`][rompy_xbeach.config.Config] rejects combinations XBeach refuses, such as bichromatic waves without surfbeat. See [Waves](waves.md).

### The non-hydrostatic model

`Nonh` resolves the waves itself, so the short-wave driver must be off: set `swave=False`. XBeach turns `swave` off by default in this mode, but rompy-xbeach logs a warning when `swave` is left unset or set to `True`. `Nonh` has its own settings for breaking (`maxbrsteep`, `secbrsteep`, `reformsteep`, `nhbreaker`), for the pressure solver (`solver`, `solver_acc`, `solver_maxit`, `solver_urelax`) and for dispersion (`dispc`, `Topt`, `nhlay`, `kdmin`).

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.wavemodel import Nonh

physics = Physics(wavemodel=Nonh(maxbrsteep=0.6, nhbreaker=2), swave=False)
print(physics.get(destdir=TMP))
```

## Breaker formulations

The breaker formulation belongs to the wave model, in its [`breaktype`][rompy_xbeach.components.physics.wavemodel.Surfbeat.breaktype] field, so only the combinations XBeach supports can be built. It is written as the XBeach keyword `break`.

| Wave model | Breakers | XBeach default |
|---|---|---|
| `Stationary` | [`Baldock`][rompy_xbeach.components.physics.wavemodel.Baldock], [`Janssen`][rompy_xbeach.components.physics.wavemodel.Janssen] | `baldock` |
| `Surfbeat` | [`Roelvink1`][rompy_xbeach.components.physics.wavemodel.Roelvink1], [`Roelvink2`][rompy_xbeach.components.physics.wavemodel.Roelvink2], [`RoelvinkDaly`][rompy_xbeach.components.physics.wavemodel.RoelvinkDaly] | `roelvink_daly` |
| `Nonh` | none: breaking is set by the `Nonh` steepness fields | |

The breaker classes share the dissipation settings of [`WaveDissipation`][rompy_xbeach.components.physics.wavemodel.WaveDissipation] (`breakerdelay`, `breakviscfac`, `breakvisclen`, `delta`, `facrun`, `facsd`, `fwcutoff`, `gammax`, `shoaldelay` and `wavfric`) and add their own:

| Class | Writes `break =` | Own fields |
|---|---|---|
| `Baldock` | `baldock` | `gamma` |
| `Janssen` | `janssen` | none |
| `Roelvink1` | `roelvink1` | `alpha`, `gamma`, `n` |
| `Roelvink2` | `roelvink2` | `alpha`, `gamma`, `n` |
| `RoelvinkDaly` | `roelvink_daly` | `gamma2` |

`RoelvinkDaly` has no `gamma` field, so its start-of-breaking parameter keeps the XBeach default.

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.wavemodel import Roelvink2, RoelvinkDaly

physics = Physics(wavemodel=Surfbeat(breaktype=Roelvink2(gamma=0.55, alpha=1.0)))
print(physics.get(destdir=TMP))

physics = Physics(wavemodel=Surfbeat(breaktype=RoelvinkDaly(gamma2=0.3)))
print(physics.get(destdir=TMP))
```

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.wavemodel import Janssen, Stationary

physics = Physics(wavemodel=Stationary(breaktype=Janssen(gammax=2.0)))
print(physics.get(destdir=TMP))
```

Values are range-checked when the object is built:

```python exec="on" source="above" result="text" session="physics"
from pydantic import ValidationError

try:
    Roelvink2(gamma=2.0)
except ValidationError as err:
    print(err.errors()[0]["msg"])
```

In YAML the field name is `breaktype` (the name `break` is only used in `params.txt`):

```python exec="on" source="above" result="text" session="physics"
import yaml

physics = Physics.model_validate(
    yaml.safe_load(
        """
        wavemodel:
          model_type: surfbeat
          breaktype:
            model_type: roelvink_daly
            gamma2: 0.3
        """
    )
)
print(physics.get(destdir=TMP))
```

### Short-wave friction

A breaker can also carry a [`ShortWaveFriction`][rompy_xbeach.components.physics.wavemodel.ShortWaveFriction] in its `wavfric` field, with either a constant friction coefficient `fw` or a spatially varying `fwfile`, not both. XBeach's default is `fw = 0`, no short-wave friction.

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.wavemodel import Roelvink1, ShortWaveFriction

physics = Physics(wavemodel=Surfbeat(breaktype=Roelvink1(wavfric=ShortWaveFriction(fw=0.05))))
print(physics.get(destdir=TMP))
```

## The roller

The roller model stores breaking-wave energy in a surface roller before it is dissipated, which moves wave setup and longshore currents shorewards. XBeach turns it on by default. `roller=False` switches it off; a [`Roller`][rompy_xbeach.components.physics.wavemodel.Roller] switches it on and sets the breaker slope `beta` and the feedback switch `rfb`.

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.wavemodel import Roller

physics = Physics(wavemodel=Surfbeat(), roller=Roller(beta=0.1))
print(physics.get(destdir=TMP))
```

## Process switches

Some processes are simple switches, others accept either a switch or a component with settings. `True` switches the process on with XBeach defaults, `False` switches it off, and a component switches it on and sets its parameters:

| Field | Accepts | XBeach default |
|---|---|---|
| `roller` | `bool` or [`Roller`][rompy_xbeach.components.physics.wavemodel.Roller] | on |
| `viscosity` | `bool` or [`Viscosity`][rompy_xbeach.components.physics.friction.Viscosity] | on |
| `vegetation` | `bool` or [`Vegetation`][rompy_xbeach.components.physics.vegetation.Vegetation] | off |
| `wci` | `bool` or [`WaveCurrentInteraction`][rompy_xbeach.components.physics.wci.WaveCurrentInteraction] | off |
| `wind` | `bool` or [`Wind`][rompy_xbeach.components.physics.wind.Wind] | off |
| `swave`, `lwave`, `flow`, `advection` | `bool` | on (`swave` off with `Nonh`) |
| `avalanching` | `bool` | same as `morphology` |
| `single_dir` | `bool` | on for surfbeat on 2D grids |
| `snells`, `swrunup`, `gwflow`, `cyclic`, `ships`, `setbathy` | `bool` | off |

The defaults are those of XBeach 1.24. Set the switches that matter for your study explicitly: they are then written to `params.txt` and do not depend on the XBeach version.

```python exec="on" source="above" result="text" session="physics"
physics = Physics(wavemodel=Surfbeat(), wind=True, avalanching=True, single_dir=False)
print(physics.get(destdir=TMP))
```

Some switches pair with settings elsewhere: `gwflow` with [`Sediment.groundwater`](sediment.md#groundwater), `setbathy` with the prescribed bed updates in [`Sediment.bed_update`](sediment.md#bed-layers-and-prescribed-bed-updates), and `wind` with the wind forcing in [`input.wind`](wind.md).

## Bed friction

[`bedfriction`][rompy_xbeach.components.physics.physics.Physics.bedfriction] takes one of five formulations, written as `bedfriction = <name>`. XBeach uses Manning when it is not set.

| Class | Writes | `bedfriccoef` | XBeach default coefficient |
|---|---|---|---|
| [`Cf`][rompy_xbeach.components.physics.friction.Cf] | `cf` | Dimensionless friction coefficient | 0.003 |
| [`Chezy`][rompy_xbeach.components.physics.friction.Chezy] | `chezy` | Chezy value C (m<sup>1/2</sup>/s) | 55 |
| [`Manning`][rompy_xbeach.components.physics.friction.Manning] | `manning` | Manning n (s/m<sup>1/3</sup>) | 0.02 |
| [`WhiteColebrook`][rompy_xbeach.components.physics.friction.WhiteColebrook] | `white-colebrook` | Nikuradse roughness k<sub>s</sub> (m) | 0.01 |
| [`WhiteColebrookGrainsize`][rompy_xbeach.components.physics.friction.WhiteColebrookGrainsize] | `white-colebrook-grainsize` | none: computed from the sediment D90 | |

`Manning`, `WhiteColebrook` and `WhiteColebrookGrainsize` also take `mincf` and `maxcf`, which bound the resulting dimensionless friction coefficient, for example in very shallow water.

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.friction import Chezy, Manning

physics = Physics(wavemodel=Surfbeat(), bedfriction=Chezy(bedfriccoef=55.0))
print(physics.get(destdir=TMP))

physics = Physics(
    wavemodel=Surfbeat(),
    bedfriction=Manning(bedfriccoef=0.02, mincf=0.001, maxcf=0.05),
)
print(physics.get(destdir=TMP))
```

All formulations accept the XBeach-G friction modifiers `friction_acceleration` (`none`, `mccall` or `nielsen`), `friction_infiltration`, `friction_turbulence` and `gamma_turb`:

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.friction import WhiteColebrookGrainsize

physics = Physics(
    wavemodel=Surfbeat(),
    bedfriction=WhiteColebrookGrainsize(friction_turbulence=True, gamma_turb=1.0),
)
print(physics.get(destdir=TMP))
```

### Spatially varying friction

`bedfricfile` reads the coefficient from a file with one value per grid point, laid out like the bathymetry file. It replaces `bedfriccoef`, and giving both raises an error. The file is given as an [`XBeachDataBlob`][rompy_xbeach.types.XBeachDataBlob]; it is copied into the workspace and its name is written to `params.txt`.

```python exec="on" source="above" result="text" session="physics"
import numpy as np

from rompy_xbeach.types import XBeachDataBlob

values = np.full((110, 115), 0.02)  # one value per grid point
values[40:70, 60:90] = 0.06  # a rough patch
np.savetxt(TMP / "manning.txt", values, fmt="%.3f")

physics = Physics(
    wavemodel=Surfbeat(),
    bedfriction=Manning(bedfricfile=XBeachDataBlob(source=TMP / "manning.txt")),
)
print(physics.get(destdir=TMP / "workspace"))
```

## Viscosity

XBeach computes horizontal viscosity with the Smagorinsky model by default, where `nuh` is the Smagorinsky coefficient. With `smag=False`, `nuh` is a constant viscosity in m²/s. [`Viscosity`][rompy_xbeach.components.physics.friction.Viscosity] also sets the roller-induced viscosity factor `nuhfac` and the longshore enhancement `nuhv`.

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.friction import Viscosity

physics = Physics(wavemodel=Surfbeat(), viscosity=Viscosity(smag=False, nuh=0.5))
print(physics.get(destdir=TMP))
```

## Vegetation

[`Vegetation`][rompy_xbeach.components.physics.vegetation.Vegetation] adds wave and flow damping by vegetation. XBeach reads a species list (`veggiefile`), which names one file per species, and a map (`veggiemapfile`) with the species number at each grid point (0 for none). Both are copied into the workspace. The species files named in the list are not: put them in the workspace yourself, for example with the configuration's template.

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.vegetation import Vegetation

(TMP / "seagrass.txt").write_text("nsec = 1\nah = 0.2\nbv = 0.005\nN = 1000\nCd = 1.0\n")
(TMP / "veggiefile.txt").write_text("seagrass.txt\n")
np.savetxt(TMP / "vegmap.txt", (values > 0.02).astype(int), fmt="%d")

physics = Physics(
    wavemodel=Surfbeat(),
    vegetation=Vegetation(
        veggiefile=XBeachDataBlob(source=TMP / "veggiefile.txt"),
        veggiemapfile=XBeachDataBlob(source=TMP / "vegmap.txt"),
        vegnonlin=True,
    ),
)
print(physics.get(destdir=TMP / "workspace"))
```

## Wave-current interaction

With wave-current interaction, currents refract and Doppler-shift the waves, as at rip channels or tidal inlets. [`WaveCurrentInteraction`][rompy_xbeach.components.physics.wci.WaveCurrentInteraction] sets the current averaging time `cats` (in mean wave periods) and the depth range `hwci` to `hwcimax` over which the interaction is computed.

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.wci import WaveCurrentInteraction

physics = Physics(wavemodel=Surfbeat(), wci=WaveCurrentInteraction(cats=4.0, hwci=0.1))
print(physics.get(destdir=TMP))
```

## Wind stress

XBeach 1.24 ignores wind unless `wind = 1`, so wind forcing given in `input.wind` needs `Physics(wind=True)`. [`Wind`][rompy_xbeach.components.physics.wind.Wind] also sets the drag coefficient `Cd`; the air density `rhoa` is in the physical constants. The forcing itself is covered in [Wind](wind.md).

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.wind import Wind

physics = Physics(wavemodel=Surfbeat(), wind=Wind(Cd=0.0015))
print(physics.get(destdir=TMP))
```

## Numerics

[`FlowNumerics`][rompy_xbeach.components.physics.numerics.FlowNumerics] (`flow_numerics`) controls the shallow water solver: the CFL number `cfl`, the wet-dry threshold `eps`, the Stokes drift thresholds `hmin`, `deltahmin` and `oldhmin`, second-order advection `secorder`, a fixed time step `dtset` and others. [`WaveNumerics`][rompy_xbeach.components.physics.numerics.WaveNumerics] (`wave_numerics`) sets the wave advection `scheme` (`upwind_1`, `lax_wendroff`, `upwind_2` or `warmbeam`), the stationary solver's `maxiter` and `maxerror`, and `wavint`, the interval between wave computations in the stationary model.

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.numerics import FlowNumerics, WaveNumerics

physics = Physics(
    wavemodel=Stationary(),
    flow_numerics=FlowNumerics(cfl=0.7, eps=0.005),
    wave_numerics=WaveNumerics(scheme="upwind_2", wavint=600.0),
)
print(physics.get(destdir=TMP))
```

The XBeach defaults suit most field-scale models. Lower `cfl` first when a run becomes unstable.

## Physical constants and Coriolis

[`PhysicalConstants`][rompy_xbeach.components.physics.constants.PhysicalConstants] (`constants`) sets gravity `g`, water density `rho`, air density `rhoa` and `depthscale`, which scales the depth thresholds of laboratory-scale models. [`Coriolis`][rompy_xbeach.components.physics.constants.Coriolis] (`coriolis`) sets the latitude `lat` and the Earth's angular velocity `wearth`; XBeach uses a latitude of 0, so no Coriolis force, unless `lat` is given.

```python exec="on" source="above" result="text" session="physics"
from rompy_xbeach.components.physics.constants import Coriolis, PhysicalConstants

physics = Physics(
    wavemodel=Surfbeat(),
    constants=PhysicalConstants(rho=1025.0),
    coriolis=Coriolis(lat=-32.6),
)
print(physics.get(destdir=TMP))
```

## In YAML

The same settings in a YAML configuration, under the `physics` key of the model config. Each class choice is made with its `model_type`:

```python exec="on" source="above" result="text" session="physics"
physics = Physics.model_validate(
    yaml.safe_load(
        """
        wavemodel:
          model_type: surfbeat
          breaktype:
            model_type: roelvink2
            gamma: 0.55
        bedfriction:
          model_type: manning
          bedfriccoef: 0.02
        roller:
          beta: 0.1
        wind: true
        flow_numerics:
          cfl: 0.7
        """
    )
)
print(physics.get(destdir=TMP))
```

```python exec="on"
from rompy_docs.notebooks import see_also

print(see_also("xbeach", ["physics", "components"]))
```
