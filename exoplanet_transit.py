#!/usr/bin/env python3
"""
exoplanet_transit.py
====================
Interactive animation of an exoplanet transit.

  Top panel    – Stellar disk (quadratic limb darkening) with transiting planet.
  Bottom panel – Geometric (uniform stellar disk) transit light curve.
  Slider       – Impact parameter  b = (a cos i) / R★

Drag the slider to see how b controls the transit shape:
  b = 0              planet crosses the stellar centre (deepest, flattest bottom)
  0 < b < 1 − Rp/R★  full transit (all four contact points occur)
  1 − Rp/R★ ≤ b < 1 + Rp/R★  grazing transit (shallower, V-shaped)
  b ≥ 1 + Rp/R★      no transit (planet misses the star)

Time axis is normalised so that the stellar radius equals one velocity unit:
  τ = t · v_orb / R★

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# ── Physical parameters ────────────────────────────────────────────────────────
R_STAR   = 1.00   # stellar radius (defines the length scale)
R_PLANET = 0.15   # planet-to-star radius ratio  (Rp / R★)
B_INIT   = 0.30   # initial impact parameter

# ── Numerical parameters ───────────────────────────────────────────────────────
N_FRAMES = 220                         # frames per animation loop
T_MIN, T_MAX = -2.5, 2.5              # time axis  [R★ / v_orb]
N_LC  = 600                            # light-curve sample points
N_IMG = 400                            # star-image resolution (pixels per side)

# Limb-darkening coefficients (quadratic law)
LD_U1, LD_U2 = 0.45, 0.25

# ── Physics ────────────────────────────────────────────────────────────────────
def _overlap_scalar(d, r1, r2):
    """
    Area of intersection of two circles with radii r1, r2
    whose centres are separated by distance d.
    Uses the standard two-sector (lens) formula.
    """
    if d >= r1 + r2:
        return 0.0                        # circles are disjoint
    if d + r2 <= r1:
        return np.pi * r2 * r2            # small circle fully inside large
    if d + r1 <= r2:
        return np.pi * r1 * r1
    # Partial (lens-shaped) overlap
    ca = np.clip((d*d + r1*r1 - r2*r2) / (2.0 * d * r1), -1.0, 1.0)
    cb = np.clip((d*d + r2*r2 - r1*r1) / (2.0 * d * r2), -1.0, 1.0)
    sq = max(0.0, (-d + r1 + r2) * (d + r1 - r2) *
                  (d - r1 + r2) * (d + r1 + r2))
    return r1*r1 * np.arccos(ca) + r2*r2 * np.arccos(cb) - 0.5 * np.sqrt(sq)


def transit_lc(t_arr, b, r_s=R_STAR, r_p=R_PLANET):
    """
    Geometric transit light curve for a uniform stellar disk.
    Returns the normalised flux array (1 = out of transit).

    Parameters
    ----------
    t_arr : array_like  – time in units of R★ / v_orb
    b     : float       – impact parameter (planet's closest approach / R★)
    r_s   : float       – stellar radius (default R_STAR)
    r_p   : float       – planet radius (default R_PLANET)
    """
    d  = np.hypot(np.asarray(t_arr, float), b)
    ov = np.zeros_like(d)

    # Three regions: disjoint (ov=0 already), planet fully inside, partial overlap
    mask_in  = (d + r_p <= r_s)                  # fully inside
    mask_pt  = (~mask_in) & (d < r_s + r_p)      # partial

    ov[mask_in] = np.pi * r_p * r_p

    if mask_pt.any():
        dp = d[mask_pt]
        ca = np.clip((dp*dp + r_s*r_s - r_p*r_p) / (2*dp*r_s), -1., 1.)
        cb = np.clip((dp*dp + r_p*r_p - r_s*r_s) / (2*dp*r_p), -1., 1.)
        sq = np.maximum(0., (-dp + r_s + r_p) * (dp + r_s - r_p) *
                            (dp - r_s + r_p) * (dp + r_s + r_p))
        ov[mask_pt] = (r_s*r_s * np.arccos(ca) + r_p*r_p * np.arccos(cb)
                       - 0.5 * np.sqrt(sq))

    return 1.0 - ov / (np.pi * r_s * r_s)


def contact_times(b, r_s=R_STAR, r_p=R_PLANET):
    """
    Compute the four transit contact times T1–T4 (in units of R★/v_orb).

    Returns a dict with keys 'T1', 'T2', 'T3', 'T4'.
    T2 and T3 are None for grazing transits (b > r_s - r_p).
    All values are None if b >= r_s + r_p (no transit).
    """
    arg_outer = (r_s + r_p)**2 - b**2
    arg_inner = (r_s - r_p)**2 - b**2

    if arg_outer <= 0:
        return dict(T1=None, T2=None, T3=None, T4=None)

    T1 = -np.sqrt(arg_outer)
    T4 =  np.sqrt(arg_outer)
    T2 = T3 = None

    if arg_inner > 0:
        T2 = -np.sqrt(arg_inner)
        T3 =  np.sqrt(arg_inner)

    return dict(T1=T1, T2=T2, T3=T3, T4=T4)


# ── GUI / animation ────────────────────────────────────────────────────────────
# Everything below here runs only when the script is executed directly.
# Importing the module for unit-testing will NOT launch a window.

if __name__ == '__main__':

    t_frames = np.linspace(T_MIN, T_MAX, N_FRAMES)
    t_model  = np.linspace(T_MIN, T_MAX, N_LC)

    # ── Pre-render limb-darkened stellar disk ──────────────────────────────────
    _xv = np.linspace(-2.6, 2.6, N_IMG)
    _yv = np.linspace(-1.6, 1.6, N_IMG)
    _XX, _YY = np.meshgrid(_xv, _yv)
    _R  = np.sqrt(_XX**2 + _YY**2)

    # cos(theta) from centre: mu = sqrt(1 - (r/R★)^2)
    _mu = np.sqrt(np.maximum(0., 1. - np.minimum(1., _R / R_STAR)**2))

    # Quadratic limb-darkening law:  I(mu) = 1 - u1*(1-mu) - u2*(1-mu)^2
    _I  = np.where(_R <= R_STAR,
                   np.clip(1. - LD_U1*(1. - _mu) - LD_U2*(1. - _mu)**2, 0., 1.),
                   0.)

    # Build RGBA image (warm solar colour)
    _star_rgba         = np.zeros((N_IMG, N_IMG, 4), dtype=np.float32)
    _star_rgba[..., 0] = np.clip(_I * 0.98, 0., 1.)      # R channel
    _star_rgba[..., 1] = np.clip(_I * 0.75, 0., 1.)      # G channel
    _star_rgba[..., 2] = np.clip(_I * 0.22, 0., 1.)      # B channel
    _star_rgba[..., 3] = (_R <= R_STAR).astype(np.float32)  # alpha mask

    # ── Figure layout ──────────────────────────────────────────────────────────
    BG = '#0c0c1e'   # deep-space background

    fig = plt.figure(figsize=(9, 8.5), facecolor=BG)
    gs  = gridspec.GridSpec(
        3, 1,
        height_ratios=[2.5, 2.1, 0.38],
        hspace=0.50,
        top=0.91, bottom=0.07, left=0.13, right=0.94,
    )
    ax_sky  = fig.add_subplot(gs[0])    # stellar disk view
    ax_lc   = fig.add_subplot(gs[1])    # light curve
    ax_sld  = fig.add_subplot(gs[2])    # slider

    for ax in (ax_sky, ax_lc):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a5c')

    # ── Stellar-disk panel ─────────────────────────────────────────────────────
    ax_sky.set_xlim(-2.6, 2.6)
    ax_sky.set_ylim(-1.6, 1.6)
    ax_sky.set_xlabel(r'$x\ /\ R_\star$',  color='#9aa0bc', fontsize=11)
    ax_sky.set_ylabel(r'$y\ /\ R_\star$',  color='#9aa0bc', fontsize=11)
    ax_sky.set_title('Stellar Disk View',   color='white',   fontsize=12, pad=6)
    ax_sky.tick_params(colors='#778899')

    # Star RGBA image (static — drawn once).
    # aspect='auto' is intentionally omitted; set_aspect('equal') is called
    # after all patches are added so it cannot be overridden by imshow.
    ax_sky.imshow(_star_rgba,
                  extent=[-2.6, 2.6, -1.6, 1.6],
                  origin='lower',
                  zorder=2, interpolation='bilinear')

    # Faint stellar limb outline
    ax_sky.add_patch(plt.Circle((0, 0), R_STAR,
                                 fill=False, ec='#aa8844',
                                 lw=0.8, alpha=0.5, zorder=3))

    # Planet size legend note
    ax_sky.text(-2.4, 1.45,
                fr'$R_p/R_\star = {R_PLANET}$',
                color='#aabbdd', fontsize=9, va='top')

    # Impact-parameter guide line (horizontal dashed line at y = b).
    # Extend well beyond T_MIN/T_MAX so it remains visible even if
    # adjustable='datalim' expands the x-axis.
    b_guide, = ax_sky.plot(
        [-6.0, 6.0], [B_INIT, B_INIT],
        ls='--', color='#6699ff', lw=1.0, alpha=0.75, zorder=4)

    b_label = ax_sky.text(
        2.35, B_INIT + 0.11, f'b = {B_INIT:.2f}',
        color='#6699ff', fontsize=9.5, ha='right', va='bottom', zorder=5)

    # Planet disk (animated)
    planet = plt.Circle(
        (t_frames[0], B_INIT), R_PLANET,
        color='#111128', ec='#7799ee', lw=1.3, zorder=6)
    ax_sky.add_patch(planet)

    # Equal aspect ratio — called LAST so imshow cannot override it.
    # adjustable='datalim' expands the data range to fill the axes box
    # rather than leaving whitespace, so the panel is fully used.
    ax_sky.set_aspect('equal', adjustable='datalim')
    ax_sky.set_xlim(-2.6, 2.6)
    ax_sky.set_ylim(-1.6, 1.6)

    # ── Light-curve panel ──────────────────────────────────────────────────────
    ax_lc.set_xlim(T_MIN, T_MAX)
    ax_lc.set_xlabel(r'Time  $\tau = t \cdot v_{\rm orb}\ /\ R_\star$',
                     color='#9aa0bc', fontsize=11)
    ax_lc.set_ylabel('Relative Flux', color='#9aa0bc', fontsize=11)
    ax_lc.set_title('Transit Light Curve  (uniform stellar disk)',
                    color='white', fontsize=12, pad=6)
    ax_lc.tick_params(colors='#778899')

    # Baseline (out-of-transit flux = 1)
    ax_lc.axhline(1.0, color='#3a3a5c', lw=0.9, ls=':', zorder=1)

    # Light-curve model line
    flux0    = transit_lc(t_model, B_INIT)
    lc_line, = ax_lc.plot(t_model, flux0, color='#ffcc44', lw=2.0, zorder=2)

    # Vertical time marker (updated each frame)
    t_vline  = ax_lc.axvline(t_frames[0], color='#ff5544',
                               lw=1.4, ls='--', alpha=0.70, zorder=4)

    # Dot tracking current flux
    t_dot,   = ax_lc.plot(
        [t_frames[0]], [flux0[0]],
        'o', ms=7, color='#ff5544', zorder=5,
        label='Current position')

    # Contact-point annotations (T1–T4 marks) — updated by _draw_contacts()
    contact_lines = []   # axvline Line2D objects for T1–T4
    contact_texts = []   # Text objects for T1–T4 labels

    # Status text (transit type + depth)
    info_txt = ax_lc.text(
        0.02, 0.06, '', transform=ax_lc.transAxes,
        color='#aabbcc', fontsize=9.5, va='bottom')


    def _set_ylim(flux):
        """Set y-axis limits to nicely frame the light curve."""
        depth = 1.0 - flux.min()
        lo = max(1.0 - 1.6 * depth - 4e-4, 0.955)
        ax_lc.set_ylim(lo, 1.007)


    def _status(b):
        depth_pct = R_PLANET**2 * 100.0
        depth_ppm = R_PLANET**2 * 1e6
        if b < 1.0 - R_PLANET - 1e-6:
            return (f'Full transit   '
                    f'depth = {depth_pct:.2f}%  ({depth_ppm:.0f} ppm)')
        elif b < 1.0 + R_PLANET + 1e-6:
            return 'Grazing transit  –  shallower, V-shaped minimum'
        else:
            return 'No transit  –  planet misses the stellar disk'


    def _draw_contacts(b):
        """Draw (or clear) T1–T4 vertical contact-point lines."""
        for ln in contact_lines:
            ln.remove()
        contact_lines.clear()
        for tx in contact_texts:
            tx.remove()
        contact_texts.clear()

        ct = contact_times(b)
        if ct['T1'] is None:
            return   # no transit

        labels = ['T₁', 'T₄']
        times  = [ct['T1'], ct['T4']]
        if ct['T2'] is not None:
            labels = ['T₁', 'T₂', 'T₃', 'T₄']
            times  = [ct['T1'], ct['T2'], ct['T3'], ct['T4']]

        ylims = ax_lc.get_ylim()
        for xc, label in zip(times, labels):
            if T_MIN < xc < T_MAX:
                ln = ax_lc.axvline(xc, color='#335577', lw=0.9,
                                    ls=':', alpha=0.8, zorder=1)
                tx = ax_lc.text(xc, ylims[0] + 0.002 * (ylims[1] - ylims[0]),
                                label, color='#6688aa', fontsize=7.5,
                                ha='center', va='bottom')
                contact_lines.append(ln)
                contact_texts.append(tx)


    _set_ylim(flux0)
    info_txt.set_text(_status(B_INIT))
    _draw_contacts(B_INIT)

    # ── Slider ─────────────────────────────────────────────────────────────────
    ax_sld.set_facecolor('#14142a')
    slider_b = Slider(
        ax_sld,
        label='Impact parameter  b',
        valmin=0.0,
        valmax=1.0 + R_PLANET + 0.05,   # slightly beyond 1+Rp to show no-transit
        valinit=B_INIT,
        color='#334499',
    )
    slider_b.label.set_color('white')
    slider_b.label.set_fontsize(10)
    slider_b.valtext.set_color('#88aaff')

    # Mark the transition thresholds on the slider axis (in data / slider coords)
    for bmark in [1.0 - R_PLANET, 1.0 + R_PLANET]:
        ax_sld.axvline(bmark, color='#556688', lw=0.8, ls=':', alpha=0.9)

    # ── Shared mutable state ───────────────────────────────────────────────────
    state = {'b': B_INIT}


    def _on_slider(val):
        """Called whenever the slider is moved."""
        b = float(slider_b.val)
        state['b'] = b

        # Recompute and redraw the light curve
        flux = transit_lc(t_model, b)
        lc_line.set_ydata(flux)
        _set_ylim(flux)

        # Update contact-point markers (clears and redraws them)
        _draw_contacts(b)

        # Update impact-parameter guide line and label in the sky panel
        b_guide.set_ydata([b, b])
        y_lbl = b + 0.11 if b < 1.45 else b - 0.17   # keep label inside axes
        b_label.set_y(y_lbl)
        b_label.set_text(f'b = {b:.2f}')

        # Update status annotation
        info_txt.set_text(_status(b))

        fig.canvas.draw_idle()


    slider_b.on_changed(_on_slider)

    # ── Animation ──────────────────────────────────────────────────────────────
    def _animate(frame):
        b = state['b']
        x = t_frames[frame]

        # Move planet across the stellar disk
        planet.set_center((x, b))

        # Compute current flux at this exact position
        d    = float(np.hypot(x, b))
        curr = 1.0 - _overlap_scalar(d, R_STAR, R_PLANET) / (np.pi * R_STAR**2)

        # Update time marker on the light curve
        t_dot.set_data([x], [curr])
        t_vline.set_xdata([x, x])

        return planet, t_dot, t_vline


    ani = FuncAnimation(
        fig, _animate,
        frames=N_FRAMES,
        interval=28,       # ms per frame  (~35 fps)
        blit=False,        # must be False so slider updates redraw immediately
        repeat=True,
    )

    # ── Title ──────────────────────────────────────────────────────────────────
    fig.suptitle('Exoplanet Transit Simulator',
                 color='white', fontsize=14, fontweight='bold', y=0.97)

    plt.show()
