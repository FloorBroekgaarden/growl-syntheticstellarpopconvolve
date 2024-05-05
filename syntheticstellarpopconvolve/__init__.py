import copy

from syntheticstellarpopconvolve.convolve import convolve  # noqa: F401
from syntheticstellarpopconvolve.default_convolution_config import (
    default_convolution_config,
)

convolution_config = copy.copy(default_convolution_config)
