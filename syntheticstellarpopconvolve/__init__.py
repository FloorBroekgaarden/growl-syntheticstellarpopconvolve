import copy

from syntheticstellarpopconvolve.convolution_default_settings import (
    convolution_options_defaults,
)
from syntheticstellarpopconvolve.convolve import convolve  # noqa: F401

convolution_config = copy.copy(convolution_options_defaults)
