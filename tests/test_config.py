from pathlib import Path
import yaml
import pytest
from importlib.metadata import entry_points

from rompy_xbeach.config import Config


HERE = Path(__file__).parent


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