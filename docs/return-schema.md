# Return-schema reference

This integration requires the core canonical return-schema API used in this
workspace. XBeach's `break` alias remains accepted by the core typo checker;
other misspelled fields still receive typo suggestions. Modern XBeach builds
also need an explicit hydrodynamic selection: set `wavemodel="surfbeat"`,
`"stationary"` or `"nonh"`. This optional field is written to `params.txt`;
leaving it unset preserves older configurations but may be rejected by newer
executables. Surfbeat builds using single-direction mode also require
`dtheta_s` (short-wave directional resolution, e.g. `10.0` degrees).
Regular grids explicitly write `vardx=0`, so the executable uses
the generated uniform spacing instead of expecting `xfile`/`yfile` inputs.

`Config.expected_artifacts()` returns core `Artifact` objects with normalized
workspace-relative POSIX paths. It does not fetch forcing data or run XBeach.

| `outputformat` | Declared outputs (in addition to `XBlog.txt`) |
| --- | --- |
| unset or `fortran` | `dims.dat`, `xy.dat` |
| `netcdf` | `ncfilename`, defaulting to `xboutput.nc` |
| `debug` | Both Fortran metadata and the NetCDF file |

Debug expectations assume a NetCDF-enabled executable. Setting `ncfilename`
alone does not enable NetCDF output. Variable-specific binary outputs depend
on solver defaults and are discovered rather than guessed. Fortran metadata
is classified as `OTHER`, the run log as `TEXT`, and NetCDF as `NETCDF`.

`validate_outputs(workspace)` returns existing files with byte sizes and warns
for missing declarations. Missing files are never reported as observed, so core
`ModelRun.run()` can populate `expected_outputs` and `missing_outputs` correctly.
Discovery also includes input/control and auxiliary files; it excludes core
result sidecars, `.rompy-postprocess` and symlinks escaping the workspace.
Absolute output names and parent traversal are rejected by the core artifact
contract. Presence checks do not validate file contents or model correctness.

Output conventions: [XBeach manual](https://xbeach.readthedocs.io/en/latest/xbeach_manual.html).
