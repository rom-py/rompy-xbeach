# Parameter index

Every XBeach parameter rompy-xbeach can write to `params.txt`, and the field that sets it. The index is generated from the code when the docs are built.

A parameter is only written when its field is set. Most fields default to `None`, and XBeach then uses its own default, given in the description.

## Parameters set by components

Fields of the model settings (`physics`, `sediment`, `output`, the flow and tide boundaries, `mpi`, `hotstart`) and of the wave boundaries. Fields that select a variant, such as `wavemodel`, write the name of the chosen variant.

```python exec="on"
import sys

sys.path.insert(0, "scripts")
from parameter_index import parameter_table

print(parameter_table())
```

## Parameters derived from the run and data

rompy-xbeach writes these from the grid, the bathymetry, the forcing and the run period. The values come from an example model generated when the docs were built.

```python exec="on"
import sys

sys.path.insert(0, "scripts")
from parameter_index import derived_table

print(derived_table())
```
