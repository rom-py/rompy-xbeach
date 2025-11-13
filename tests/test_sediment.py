"""Tests for Sediment component."""

import pytest
from rompy_xbeach.components.sediment import Sediment
from rompy_xbeach.components.sediment.bed import BedComposition, Quasi3D
from rompy_xbeach.components.sediment.morphology import (
    Avalanching,
    Morphology,
    PrescribedBathymetry,
)
from rompy_xbeach.components.sediment.transport import (
    SedimentTransport,
    TransportNumerics,
)


def test_sediment_transport():
    """Test SedimentTransport model with various parameters."""
    transport = SedimentTransport(
        form="vanthiel_vanrijn",
        waveform="vanthiel",
        facAs=0.2,
        facSk=0.15,
        z0=0.006,
        bdslpeffmag="roelvink_total",
        bdslpeffdir="talmon",
        bdslpeffdirfac=1.0,
        facsl=1.6,
        reposeangle=30.0,
        sws=True,
        lws=True,
        lwt=False,
        turb="wave_averaged",
        bed=1.0,
        sus=1.0,
        bulk=False,
        Tsmin=0.5,
        tsfac=0.1,
        facDc=1.0,
        Tbfac=1.0,
        smax=2.0,
    )
    params = transport.params
    # Formulation params
    assert params["form"] == "vanthiel_vanrijn"
    assert params["waveform"] == "vanthiel"
    assert params["facAs"] == 0.2
    assert params["facSk"] == 0.15
    assert params["z0"] == 0.006
    # Bed slope params
    assert params["bdslpeffmag"] == "roelvink_total"
    assert params["bdslpeffdir"] == "talmon"
    assert params["facsl"] == 1.6
    assert params["reposeangle"] == 30.0
    # Process params
    assert params["sws"] == 1
    assert params["lws"] == 1
    assert params["lwt"] == 0
    assert params["turb"] == "wave_averaged"
    assert params["bed"] == 1.0
    assert params["sus"] == 1.0
    assert params["bulk"] == 0
    # Calibration params
    assert params["Tsmin"] == 0.5
    assert params["tsfac"] == 0.1
    assert params["facDc"] == 1.0
    assert params["Tbfac"] == 1.0
    assert params["smax"] == 2.0


def test_transport_numerics():
    """Test TransportNumerics model."""
    numerics = TransportNumerics(
        cmax=0.1,
        sourcesink=False,
        thetanum=1.0,
        dtlimts=1.0,
    )
    params = numerics.params
    assert params["cmax"] == 0.1
    assert params["sourcesink"] == 0
    assert params["thetanum"] == 1.0
    assert params["dtlimts"] == 1.0


def test_morphology():
    """Test Morphology model."""
    morphology = Morphology(
        morfac=10.0,
        morfacopt=True,
        morstart=0.0,
        morstop=3600.0,
        struct=False,
    )
    params = morphology.params
    assert params["morfac"] == 10.0
    assert params["morfacopt"] == 1
    assert params["morstart"] == 0.0
    assert params["morstop"] == 3600.0
    assert params["struct"] == 0


def test_avalanching():
    """Test Avalanching model."""
    avalanching = Avalanching(
        dryslp=1.0,
        wetslp=0.3,
        hswitch=0.1,
        dzmax=0.05,
    )
    params = avalanching.params
    assert params["dryslp"] == 1.0
    assert params["wetslp"] == 0.3
    assert params["hswitch"] == 0.1
    assert params["dzmax"] == 0.05


def test_prescribed_bathymetry():
    """Test PrescribedBathymetry model."""
    prescribed = PrescribedBathymetry(
        nsetbathy=10,
    )
    params = prescribed.params
    assert params["nsetbathy"] == 10


def test_bed_composition():
    """Test BedComposition model."""
    bed_comp = BedComposition(
        frac_dz=0.7,
        split=1.01,
        merge=0.01,
        nd_var=2,
    )
    params = bed_comp.params
    assert params["frac_dz"] == 0.7
    assert params["split"] == 1.01
    assert params["merge"] == 0.01
    assert params["nd_var"] == 2


def test_quasi3d():
    """Test Quasi3D model."""
    quasi3d = Quasi3D(
        kmax=100,
        sigfac=1.3,
        deltar=0.025,
        rwave=2.0,
        vonkar=0.4,
    )
    params = quasi3d.params
    assert params["kmax"] == 100
    assert params["sigfac"] == 1.3
    assert params["deltar"] == 0.025
    assert params["rwave"] == 2.0
    assert params["vonkar"] == 0.4


def test_sediment_component_empty():
    """Test Sediment component with no parameters."""
    sediment = Sediment()
    params = sediment.params
    assert len(params) == 0


def test_sediment_component_with_all_subcomponents():
    """Test Sediment component with all subcomponents."""
    sediment = Sediment(
        transport=SedimentTransport(
            form="vanthiel_vanrijn",
            facua=0.15,
            bdslpeffmag="roelvink_total",
            facsl=1.6,
            sws=True,
            lws=True,
            Tsmin=0.5,
            tsfac=0.1,
        ),
        numerics=TransportNumerics(
            cmax=0.1,
        ),
        morphology=Morphology(
            morfac=10.0,
            morstart=0.0,
        ),
        avalanching=Avalanching(
            dryslp=1.0,
            wetslp=0.3,
        ),
        bed_composition=BedComposition(
            frac_dz=0.7,
        ),
        quasi3d=Quasi3D(
            kmax=50,
        ),
    )

    # Use get() to flatten nested components
    params = sediment.get(destdir="/tmp")

    # Check transport params
    assert params["form"] == "vanthiel_vanrijn"
    assert params["facua"] == 0.15
    assert params["bdslpeffmag"] == "roelvink_total"
    assert params["facsl"] == 1.6
    assert params["sws"] == 1
    assert params["lws"] == 1
    assert params["Tsmin"] == 0.5
    assert params["tsfac"] == 0.1

    # Check numerics params
    assert params["cmax"] == 0.1

    # Check morphology params
    assert params["morfac"] == 10.0
    assert params["morstart"] == 0.0

    # Check avalanching params
    assert params["dryslp"] == 1.0
    assert params["wetslp"] == 0.3

    # Check bed composition params
    assert params["frac_dz"] == 0.7

    # Check quasi3d params
    assert params["kmax"] == 50


def test_validation_ranges():
    """Test that validation ranges work correctly."""
    # Test valid ranges
    transport = SedimentTransport(facAs=0.5, facSk=0.3)
    assert transport.facAs == 0.5

    # Test invalid ranges
    with pytest.raises(ValueError):
        SedimentTransport(facAs=1.5)  # > 1.0

    with pytest.raises(ValueError):
        Morphology(morfac=2000.0)  # > 1000.0

    with pytest.raises(ValueError):
        Avalanching(dryslp=3.0)  # > 2.0

    with pytest.raises(ValueError):
        TransportNumerics(cmax=1.5)  # > 1.0

    with pytest.raises(ValueError):
        Quasi3D(kmax=2000)  # > 1000


def test_sediment_minimal_configuration():
    """Test Sediment with minimal configuration."""
    sediment = Sediment(morphology=Morphology(morfac=5.0))
    params = sediment.get(destdir="/tmp")
    assert params["morfac"] == 5.0
    assert len(params) == 1


def test_sediment_morphology_only():
    """Test Sediment with morphology parameters only."""
    sediment = Sediment(
        morphology=Morphology(
            morfac=10.0,
            morfacopt=True,
            morstart=0.0,
            morstop=7200.0,
        ),
        avalanching=Avalanching(
            dryslp=1.0,
            wetslp=0.3,
        ),
    )
    params = sediment.get(destdir="/tmp")
    assert params["morfac"] == 10.0
    assert params["morfacopt"] == 1
    assert params["morstart"] == 0.0
    assert params["morstop"] == 7200.0
    assert params["dryslp"] == 1.0
    assert params["wetslp"] == 0.3
