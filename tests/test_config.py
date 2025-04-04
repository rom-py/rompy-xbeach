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
                if key in ['filename', 'uri', 'gfile', 'hfile', 'ufile'] and isinstance(value, str) and value.startswith('./'):
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
        output_dir=tmp_path,
        config=config,
        period=TimeRange(start="2023-01-01T00", end="2023-01-01T12", interval="1h"),
    )
    model.generate()
    assert (tmp_path / model.run_id / "params.txt").is_file()
