from enum import Enum
from typing import Any

from pydantic import ConfigDict, model_validator
from rompy.core.config import BaseConfig


class XBeachBaseConfig(BaseConfig):
    """Base configuration class for all XBeach models."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def check_for_typos(cls, data: Any) -> Any:
        """Let valid Pydantic aliases through core's field-name typo check.

        In particular, XBeach's ``break`` parameter aliases ``breaktype``.
        Keep the original keys for Pydantic validation and serialization.
        """
        if isinstance(data, dict):
            aliases = {
                field.alias: name
                for name, field in cls.model_fields.items()
                if field.alias is not None
            }
            super().check_for_typos(
                {aliases.get(key, key): value for key, value in data.items()}
            )
        return data


class WbcEnum(str, Enum):
    """Valid options for wbctype.

    Attributes
    ----------
    PARAMS: "params"
        Wave boundary conditions specified as a constant value.
    JONS: "jons"
        Wave boundary conditions specified as a single Jonswap spectrum.
    JONSTABLE: "jonstable"
        Wave boundary conditions specified as a time-series of wave parameters.
    SWAN: "swan"
        Wave boundary conditions specified as a SWAN 2D spectrum file.
    VARDENS: "vardens"
        Wave boundary conditions specified as a general spectrum file.
    TS_1: "ts_1"
        Wave boundary conditions specified as a variation in time of wave energy (first-order).
    TS_2: "ts_2"
        Wave boundary conditions specified as a variation in time of wave energy (second-order).
    TS_NONH: "ts_nonh"
        Wave boundary conditions specified as a variation in time of the horizontal
        velocity, vertical velocity and the free surface elevation.
    REUSE: "reuse"
        Wave boundary conditions specified from a previous run.
    OFF: "off"
        No wave boundary conditions.

    """

    PARAMS = "params"
    JONS = "jons"
    JONSTABLE = "jonstable"
    SWAN = "swan"
    VARDENS = "vardens"
    TS_1 = "ts_1"
    TS_2 = "ts_2"
    TS_NONH = "ts_nonh"
    REUSE = "reuse"
    OFF = "off"
