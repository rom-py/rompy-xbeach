import os
from pathlib import Path
import yaml
import pytest
from importlib.metadata import entry_points

from rompy.model import ModelRun
from rompy.core.time import TimeRange
from rompy_xbeach.config import Config


HERE = Path(__file__).parent

os.environ["XBEACH_PATH"] = str(HERE.parent.parent)


@pytest.fixture(scope="module")
def kwargs():
    # Load the YAML file
    with open(HERE / "test_config.yml") as f:
        config_data = yaml.load(f, Loader=yaml.Loader)

    # Replace relative paths with absolute paths
    def replace_paths(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if (
                    key in ["filename", "uri", "gfile", "hfile", "ufile", "source"]
                    and isinstance(value, str)
                    and value.startswith("./")
                ):
                    # Convert relative path to absolute path
                    obj[key] = str(HERE / value[2:])
                elif isinstance(value, (dict, list)):
                    replace_paths(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    replace_paths(item)

    replace_paths(config_data)
    yield config_data


def test_config_entrypoint():
    eps = entry_points(group="rompy.config")
    names = [ep.name for ep in eps]
    assert "xbeach" in names


def test_xbeach_config(kwargs):
    config = Config(**kwargs)
    assert config.model_type == "xbeach"


def test_model_generate(kwargs, tmp_path):
    config = Config(**kwargs)
    model = ModelRun(
        run_id="test",
        output_dir=str(tmp_path),
        config=config,
        period=TimeRange(start="2023-01-01T00", end="2023-01-01T12", interval="1h"),
    )
    model.generate()
    assert (tmp_path / model.run_id / "params.txt").is_file()


def test_boolean_params_rendered_as_integers(kwargs, tmp_path):
    """Booleans must be rendered as 0/1 in params.txt, not Python True/False.

    Wave boundary classes don't go through the XBeachBaseModel serializer, so the
    boolean->int conversion is applied at the Config aggregation chokepoint. This
    asserts the conversion reaches the rendered params.txt for a wave boundary
    boolean parameter.
    """
    import copy

    kwargs = copy.deepcopy(kwargs)
    kwargs["input"]["wave"]["wbcScaleEnergy"] = True
    kwargs["input"]["wave"]["wbcRemoveStokes"] = False
    config = Config(**kwargs)
    model = ModelRun(
        run_id="test_bools",
        output_dir=str(tmp_path),
        config=config,
        period=TimeRange(start="2023-01-01T00", end="2023-01-01T12", interval="1h"),
    )
    model.generate()
    params_txt = (tmp_path / model.run_id / "params.txt").read_text()
    assert "wbcScaleEnergy = 1" in params_txt
    assert "wbcRemoveStokes = 0" in params_txt
    # No Python-style booleans should appear anywhere in the rendered file
    assert "= True" not in params_txt
    assert "= False" not in params_txt
