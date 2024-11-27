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
    yield yaml.load(open(HERE / "test_config.yml"), Loader=yaml.Loader)


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
