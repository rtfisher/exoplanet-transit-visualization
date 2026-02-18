# Exoplanet Transit Simulator

[![Test Exoplanet Transit](https://github.com/rtfisher/exoplanet_transit/actions/workflows/test.yml/badge.svg)](https://github.com/rtfisher/exoplanet_transit/actions/workflows/test.yml)

An interactive Python animation of an exoplanet transit, designed for use with introductory astronomy students.  The simulator shows the geometry of the
transit on the stellar disk alongside the resulting photometric light curve,
and lets students explore how the **impact parameter** shapes both.

## Features

- **Stellar disk view** — quadratic limb-darkened star (warm solar colours)
  with the planet moving across it in real time
- **Transit light curve** — exact geometric model (uniform stellar disk) with
  annotated contact points T₁–T₄
- **Impact parameter slider** — drag *b* from 0 to beyond 1 + Rp/R★ to move
  continuously through full transit → grazing transit → no transit
- **Live status annotation** — reports transit depth in % and ppm, or flags
  grazing / no-transit
- **Dark space-themed UI** built entirely with standard matplotlib

## Transit physics recap

| Regime | Condition | Light curve shape |
|--------|-----------|-------------------|
| Full transit | b < 1 − Rp/R★ | Flat-bottomed; depth = (Rp/R★)² |
| Grazing transit | 1 − Rp/R★ ≤ b < 1 + Rp/R★ | V-shaped; shallower minimum |
| No transit | b ≥ 1 + Rp/R★ | Flat (flux = 1) |

The **impact parameter** is defined as

```
b = (a / R★) cos i
```

where *a* is the semi-major axis and *i* is the orbital inclination.  *b* = 0
means the planet crosses the stellar centre; *b* = 1 means a grazing chord
along the stellar limb.

The **transit depth** for a full transit on a uniform disk is

```
ΔF / F = (Rp / R★)²
```

The default ratio Rp/R★ = 0.15 is slightly larger than the Jupiter/Sun value
(≈ 0.103) to make the transit more visually prominent; it corresponds to an
inflated hot Jupiter.

The **time axis** is normalised:

```
τ = t · v_orb / R★
```

so ingress begins at τ = −√[(1 + Rp/R★)² − b²] regardless of physical
orbital parameters.

Contact points T₁–T₄ mark external first contact, internal second contact,
internal third contact, and external fourth contact respectively.

## Requirements

- Python 3.9+
- NumPy
- Matplotlib

## Installation

```bash
# Using conda (recommended for this course)
conda activate npscipy
python exoplanet_transit.py

# Or install dependencies with pip
pip install numpy matplotlib
python exoplanet_transit.py
```

## Usage

```bash
python exoplanet_transit.py
```

### Interactive controls

| Control | Effect |
|---------|--------|
| **Impact parameter slider** | Drag to change *b* from 0 (central transit) to ~1.2 (no transit) |

The animation loops continuously.  Close the window to exit.

## Running the tests

```bash
# Install pytest if needed
pip install pytest

# Run the full suite
pytest test_exoplanet_transit.py -v
```

## Code structure

| Symbol | Description |
|--------|-------------|
| `_overlap_scalar(d, r1, r2)` | Area of intersection of two circles — the core geometric primitive |
| `transit_lc(t_arr, b)` | Vectorised geometric light curve (uniform stellar disk) |
| `contact_times(b)` | Returns T₁–T₄ contact times for a given impact parameter |
| `if __name__ == '__main__':` block | All matplotlib figure, animation, and slider code |

The physics functions are defined at module level so they can be imported
independently by the test suite without launching a display window.

## License

Educational use permitted.  

## Author

Robert Fisher with Claude
