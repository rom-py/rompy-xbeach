"""XBEACH Rompy config."""

import logging
from pathlib import Path
from typing import Literal
from pydantic import Field

from rompy.core.config import BaseConfig


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


class Config(BaseConfig):
    """Xbeach config class."""

    model_type: Literal["xbeach"] = Field(
        default="xbeach",
        description="Model type discriminator",
    )
    template: str = Field(
        default=str(HERE / "templates" / "base"),
        description="The model config template",
    )

    def __call__(self, runtime) -> dict:
        """Callable where data and config are interfaced and CMD is rendered."""
        staging_dir = runtime.staging_dir
        # Do something
        ret = {"staging_dir": staging_dir}
        return ret