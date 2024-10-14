from pydantic import ConfigDict

from rompy.core.types import RompyBaseModel
from rompy.core.config import BaseConfig


class XBeachBaseModel(RompyBaseModel):
    """Base class for all XBeach models."""
    model_config = ConfigDict(extra="forbid")


class XBeachBaseConfig(BaseConfig):
    """Base configuration class for all XBeach models."""
    model_config = ConfigDict(extra="forbid")
