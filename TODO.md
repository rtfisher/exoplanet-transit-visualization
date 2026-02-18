# TODO — Exoplanet Transit Simulator

Recommended enhancements, roughly in order of difficulty.

---

## Physics improvements

### 1. Limb-darkened light curve model
The stellar *image* already uses quadratic limb darkening for the visuals, but
the light-curve model is uniform-disk.  Replace the geometric overlap integral
with the analytic limb-darkening transit formula (Mandel & Agol 2002) or the
polynomial series (Agol & Winn 2007).  This would show students that the real
V-shaped ingress/egress is noticeably rounder than the sharp geometric model.

### 2. Rp/R★ slider
Add a second slider to vary the planet-to-star radius ratio interactively
(e.g. 0.01 – 0.25).  This directly teaches that transit depth = (Rp/R★)² and
allows comparison of Jupiter-sized vs. Earth-sized planets.

### 3. Realistic unit conversion
Add optional physical-units mode: accept stellar radius in R☉, orbital
period and semi-major axis in days/AU via `astropy.units`, and display the
transit duration in hours.

### 4. Secondary eclipse (occultation)
Extend the animation to a full orbital cycle.  Show the secondary eclipse
(planet passing *behind* the star) and the phase-curve variation between
primary and secondary events.

### 5. Radial-velocity companion panel
Add a third panel showing the stellar radial-velocity curve (Doppler wobble)
in phase with the transit.  Demonstrates how transit photometry and RV
measurements are combined to derive planetary mass and density.

### 6. Stellar spots / faculae
Allow the user to place circular dark spots or bright faculae on the stellar
disk.  When the planet occults a spot the light curve shows a characteristic
bump (spot crossing event) — a key systematic in real transit surveys.

---

## Visualisation improvements

### 7. Save animation to file
Add a **Save GIF / MP4** button (using `matplotlib.animation.PillowWriter`
or `FFMpegWriter`) so the animation can be exported for presentations or
course notes.

### 8. Phase-folded light curve overlay
Show a faint trail of all previously computed light curves on the bottom panel
as the slider is moved, making it easy to compare shapes at different *b* values
side by side.

### 9. Contact-point chord on the stellar disk
Draw the chord between the planet centre at T1 and at T4 on the sky view,
visually linking the geometry to the transit duration formula.

### 10. 3-D orbital geometry inset
Add a small inset showing the 3-D orbital orientation (inclination *i*, line
of sight, stellar disc) so students can see how the impact parameter connects
to the orbital geometry.

---

## Software / educational improvements

### 11. Real exoplanet database presets
Add a drop-down (or radio buttons) populated from a small bundled table of
well-known transiting planets (HD 209458 b, TRAPPIST-1 b, Kepler-7 b, etc.)
that sets Rp/R★ and *b* to the published values.  Could be extended to query
NASA Exoplanet Archive via `requests` at runtime.

### 12. Noise simulation
Overlay Gaussian photon noise on the light curve (with a signal-to-noise
slider) to show how faint planets can be hidden in noise, and to motivate
phase-folding of many transits.

### 13. Jupyter notebook version
Convert the script to an ipywidgets-powered Jupyter notebook so students can
step through the physics derivation cell by cell and interact with the plots
inline in JupyterHub.

### 14. Command-line parameter mode
Support `--b 0.5 --rp 0.103 --save transit.gif` CLI flags (via `argparse`)
for headless batch runs and automated figure generation for problem sets.

---

## Testing / CI improvements

### 15. Limb-darkening coefficient tests
Once the LC model includes limb darkening, add tests that verify:
- `I(μ=1) = 1` (centre of disk)
- `I(μ=0) = 1 − u1 − u2` (limb)
- The integrated intensity equals the normalisation constant used in the model.

### 16. Property-based testing with Hypothesis
Use the `hypothesis` library to generate random valid (b, Rp, Rs) triples and
assert that flux ∈ [0, 1], symmetry holds, and depth ≤ (Rp/Rs)² for all cases.

### 17. Visual regression tests
Use `pytest-mpl` to snapshot the light curve and stellar-disk panels and flag
unexpected visual changes in future refactors.
