"""What the water's refractive index actually is, and how well it can be known.

`refraction.SWEET_WATER` and `refraction.SALTY_WATER` are the two round numbers
Łuczyński et al. use for field work. They are fine as defaults and wrong as
physics: the index of water is a function of salinity, temperature *and*
wavelength, and this module supplies that function so the paper can say what
assuming a constant costs rather than assuming it costs nothing.

Three facts fall out, and they are the reason this is a module rather than a
notebook cell.

**The two water types do not overlap.** Fresh water spans 1.3332-1.3352 over
5-30 C and sea water 1.3387-1.3423 over 0-30 C and 30-35 PSU, with a clear gap
between. A two-mode preset is therefore a well-posed classification, not a
guess.

**Temperature matters as much as salinity.** About 0.0010 per 10 C against
0.0009 per 5 PSU. A correction that takes salinity from a CTD and ignores the
thermometer has only done half the job.

**Dispersion is comparable to the fresh-to-sea step itself.** Across 450-650 nm
the index moves by 0.0082, against 0.0062 between the two envelopes' midpoints
and 0.0090 between the paper's round constants. So an index quoted without a
wavelength is underspecified, and refining one below about 0.002 is meaningless
until the band is pinned. The laser dot is monochromatic and has no such
problem; the fish silhouette is broadband and does.

`effective_index` is the broadband answer. Water strips red far faster than
blue, so the light that survives to the sensor is blue-shifted, and blue has the
higher index -- the effective index therefore sits *above* the mid-visible value
and climbs with range. **This is only true in clear water.** Coastal water
carries CDOM and particulates that absorb blue instead, which moves the
transmission window toward green-yellow and reverses the drift. The absorption
table here is pure water, so treat the range dependence as clear-water
behaviour, not a universal.
"""

from __future__ import annotations

import math

#: Wavelength of the FishSense laser. Monochromatic, so the dot's index is exact.
LASER_WAVELENGTH_NM = 532.0

#: Mid-visible, the wavelength an unqualified "index of water" usually means.
REFERENCE_WAVELENGTH_NM = 550.0

# Quan & Fry (1995), "Empirical equation for the index of refraction of
# seawater." Stated validity: 0-30 C, 0-35 PSU, 400-700 nm.
_QF = (1.31405, 1.779e-4, -1.05e-6, 1.6e-8, -2.02e-6,
       15.868, 0.01155, -0.00423, -4382.0, 1.1455e6)

#: Pure-water absorption coefficient, 1/m, from Pope & Fry (1997) and Smith &
#: Baker. Used only as a spectral *weight*, so its absolute scale is irrelevant
#: -- what matters is that red is attenuated roughly fifty times faster than
#: blue, which is what shifts the effective index.
PURE_WATER_ABSORPTION = {
    400: 0.0066, 425: 0.0077, 450: 0.0092, 475: 0.0143, 500: 0.0257,
    525: 0.0429, 550: 0.0638, 575: 0.0900, 600: 0.2440, 625: 0.2900,
    650: 0.3400, 675: 0.4200, 700: 0.6240,
}

#: Deployment envelopes as (salinity PSU, temperature C) ranges. Sea water is
#: capped at 35 PSU because that is where Quan & Fry's fit is stated to hold.
FRESH_ENVELOPE = ((0.0, 0.0), (5.0, 30.0))
SEA_ENVELOPE = ((30.0, 35.0), (0.0, 30.0))
BRACKISH_ENVELOPE = ((5.0, 25.0), (5.0, 30.0))


def index(salinity: float, temperature: float,
          wavelength_nm: float = REFERENCE_WAVELENGTH_NM) -> float:
    """Refractive index of seawater, after Quan & Fry (1995).

    `salinity` in PSU, `temperature` in C, `wavelength_nm` in nanometres.
    """
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9 = _QF
    s, t, lam = salinity, temperature, wavelength_nm
    return (n0 + (n1 + n2 * t + n3 * t * t) * s + n4 * t * t
            + (n5 + n6 * s + n7 * t) / lam + n8 / lam ** 2 + n9 / lam ** 3)


def envelope(bounds, samples: int = 6, wavelength_nm: float = REFERENCE_WAVELENGTH_NM):
    """Every index a deployment envelope can present, on a grid.

    `bounds` is ``((salinity_min, salinity_max), (temp_min, temp_max))``.
    """
    (s_lo, s_hi), (t_lo, t_hi) = bounds

    def grid(lo, hi):
        if lo == hi:
            return [lo]
        return [lo + (hi - lo) * i / (samples - 1) for i in range(samples)]

    return [index(s, t, wavelength_nm) for s in grid(s_lo, s_hi) for t in grid(t_lo, t_hi)]


def effective_index(salinity: float, temperature: float, range_m: float,
                    path_factor: float = 2.0) -> float:
    """Attenuation-weighted index for broadband imaging at a given range.

    Weights each wavelength by how much of it survives `path_factor * range_m`
    of water, `path_factor = 2` being the round trip out to the subject and
    back. Sensor response and subject reflectance are taken as flat, which
    isolates the water's contribution; both would push the result further blue,
    so this is a floor on the shift rather than a measurement.

    Averaging the *index* is legitimate because length error is linear in the
    index across the visible band -- see `tests/test_water_index.py`, which
    pins that rather than assuming it.
    """
    numerator = denominator = 0.0
    for lam, absorption in PURE_WATER_ABSORPTION.items():
        weight = math.exp(-absorption * path_factor * range_m)
        numerator += index(salinity, temperature, lam) * weight
        denominator += weight
    return numerator / denominator


def nearest_preset(n: float, presets) -> float:
    """The preset a two-mode correction would select for a true index `n`."""
    return min(presets, key=lambda preset: abs(n - preset))
