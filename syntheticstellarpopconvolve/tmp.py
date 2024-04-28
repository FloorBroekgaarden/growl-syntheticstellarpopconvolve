# DH0001_file
import astropy.units as u
from astropy.cosmology import Planck13 as cosmo  # Planck 2013

from syntheticstellarpopconvolve.cosmology_utils import (
    lookback_time_to_redshift,
    redshift_to_lookback_time,
)

print(redshift_to_lookback_time(0.5, cosmology=cosmo))

ding = redshift_to_lookback_time(0.5, cosmology=cosmo)
dong = ding + 1 * u.Gyr
print(lookback_time_to_redshift(dong.value, cosmology=cosmo))

dong = ding + 2 * u.Gyr
print(lookback_time_to_redshift(dong.value, cosmology=cosmo))

dong = ding + 3 * u.Gyr
print(lookback_time_to_redshift(dong.value, cosmology=cosmo))


# .[0.65010332 0.8336452  1.06610848]
# ...current lookback value:  5.1920300256910465 Gyr
