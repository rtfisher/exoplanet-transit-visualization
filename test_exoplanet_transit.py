#!/usr/bin/env python3
"""
test_exoplanet_transit.py
=========================
pytest suite for the physics functions in exoplanet_transit.py.

Tested functions
----------------
_overlap_scalar(d, r1, r2)  – circle–circle intersection area
transit_lc(t_arr, b)        – geometric transit light curve
contact_times(b)            – T1–T4 contact-point times

Run with:
    pytest test_exoplanet_transit.py -v
"""

import pytest
import numpy as np

# Use a non-interactive backend so CI runners without a display work fine.
import matplotlib
matplotlib.use('Agg')

from exoplanet_transit import (
    _overlap_scalar,
    transit_lc,
    contact_times,
    R_STAR,
    R_PLANET,
)


# =============================================================================
# Helpers / shared constants
# =============================================================================

RS = R_STAR
RP = R_PLANET


# =============================================================================
# _overlap_scalar  –  circle–circle intersection area
# =============================================================================

class TestOverlapScalar:
    """Geometric properties of the two-circle intersection formula."""

    def test_disjoint_circles(self):
        """Circles that do not touch have zero overlap."""
        assert _overlap_scalar(3.0, 1.0, 1.0) == pytest.approx(0.0)

    def test_external_tangency(self):
        """Circles touching at exactly one point have zero overlap."""
        assert _overlap_scalar(2.0, 1.0, 1.0) == pytest.approx(0.0, abs=1e-12)

    def test_small_circle_fully_inside(self):
        """Small circle completely inside large circle → overlap = π r₂²."""
        r2 = 0.3
        # d + r2 ≤ r1 → planet fully inside star
        area = _overlap_scalar(0.0, 1.0, r2)
        assert area == pytest.approx(np.pi * r2**2, rel=1e-10)

    def test_concentric_equal_circles(self):
        """Concentric equal circles overlap completely → π r²."""
        r = 0.5
        assert _overlap_scalar(0.0, r, r) == pytest.approx(np.pi * r**2, rel=1e-10)

    def test_symmetry_in_radii(self):
        """Overlap area is symmetric: swap r1 and r2 gives the same result."""
        d, r1, r2 = 0.8, 1.0, 0.4
        assert (_overlap_scalar(d, r1, r2)
                == pytest.approx(_overlap_scalar(d, r2, r1), rel=1e-10))

    def test_partial_overlap_positive(self):
        """Partial overlap must be strictly positive."""
        area = _overlap_scalar(0.9, 1.0, 0.4)
        assert area > 0.0

    def test_partial_overlap_less_than_small_disk(self):
        """Partial overlap cannot exceed the smaller disk area."""
        r2 = 0.4
        area = _overlap_scalar(0.9, 1.0, r2)
        assert area < np.pi * r2**2

    def test_partial_overlap_less_than_large_disk(self):
        """Partial overlap cannot exceed the larger disk area."""
        r1 = 1.0
        area = _overlap_scalar(0.9, r1, 0.4)
        assert area < np.pi * r1**2

    def test_internal_tangency_equals_small_disk_area(self):
        """Internal tangency (d = r1 − r2) → overlap = π r₂²."""
        r1, r2 = 1.0, 0.3
        d = r1 - r2   # exactly touching from inside
        area = _overlap_scalar(d, r1, r2)
        assert area == pytest.approx(np.pi * r2**2, rel=1e-8)

    def test_nonnegative_for_many_configurations(self):
        """Overlap is never negative for any distance."""
        r1, r2 = 1.0, 0.2
        for d in np.linspace(0.0, 2.0, 50):
            assert _overlap_scalar(d, r1, r2) >= 0.0

    def test_monotone_decrease_with_separation(self):
        """Overlap decreases (weakly) as centres move further apart."""
        r1, r2 = 1.0, 0.4
        d_vals = np.linspace(0.0, r1 + r2, 30)
        areas  = [_overlap_scalar(d, r1, r2) for d in d_vals]
        for a, b in zip(areas, areas[1:]):
            assert a >= b - 1e-12


# =============================================================================
# transit_lc  –  geometric light curve
# =============================================================================

class TestTransitLc:
    """Physical and mathematical properties of the transit light curve."""

    # ── Out-of-transit baseline ────────────────────────────────────────────────

    def test_flux_unity_far_from_star(self):
        """Far from the star, flux must equal 1."""
        t_far = np.array([-10.0, -5.0, 5.0, 10.0])
        flux  = transit_lc(t_far, b=0.0)
        np.testing.assert_allclose(flux, 1.0, atol=1e-12)

    def test_flux_unity_before_first_contact(self):
        """Flux = 1 before first contact T₁ = −√[(1+Rp)² − b²]."""
        b  = 0.3
        T1 = -np.sqrt((RS + RP)**2 - b**2)
        t_before = np.linspace(T1 - 2.0, T1 - 0.01, 20)
        flux = transit_lc(t_before, b)
        np.testing.assert_allclose(flux, 1.0, atol=1e-12)

    def test_flux_unity_after_fourth_contact(self):
        """Flux = 1 after fourth contact T₄ = +√[(1+Rp)² − b²]."""
        b  = 0.3
        T4 = np.sqrt((RS + RP)**2 - b**2)
        t_after = np.linspace(T4 + 0.01, T4 + 2.0, 20)
        flux = transit_lc(t_after, b)
        np.testing.assert_allclose(flux, 1.0, atol=1e-12)

    # ── Transit depth ──────────────────────────────────────────────────────────

    def test_central_transit_depth(self):
        """
        For b = 0 at mid-transit (t = 0), depth = (Rp/Rs)².
        This is the maximum depth for a uniform-disk model.
        """
        flux_mid = transit_lc(np.array([0.0]), b=0.0)[0]
        expected_depth = (RP / RS)**2
        assert 1.0 - flux_mid == pytest.approx(expected_depth, rel=1e-6)

    def test_depth_increases_with_smaller_b(self):
        """Transit depth is greatest for b = 0 and decreases as |b| grows."""
        depths = []
        for b in [0.0, 0.2, 0.4, 0.6]:
            f = transit_lc(np.array([0.0]), b)[0]
            depths.append(1.0 - f)
        # Each successive depth should be ≤ the previous one
        for d1, d2 in zip(depths, depths[1:]):
            assert d1 >= d2 - 1e-10

    def test_no_transit_for_large_b(self):
        """For b ≥ Rs + Rp the planet misses the star entirely; flux stays 1."""
        b_no_transit = RS + RP + 0.05
        t = np.linspace(-3.0, 3.0, 200)
        flux = transit_lc(t, b_no_transit)
        np.testing.assert_allclose(flux, 1.0, atol=1e-12)

    def test_grazing_transit_shallower_than_full(self):
        """A grazing transit produces a shallower minimum than a full transit."""
        b_full    = 0.0
        b_grazing = RS - RP + 0.05   # just beyond the full-transit boundary
        f_full    = transit_lc(np.array([0.0]), b_full)[0]
        f_grazing = transit_lc(np.array([0.0]), b_grazing)[0]
        assert f_full < f_grazing   # full transit is deeper

    # ── Symmetry ───────────────────────────────────────────────────────────────

    def test_time_symmetry(self):
        """Light curve must be symmetric in time: flux(t) = flux(−t)."""
        b = 0.3
        t_pos = np.linspace(0.01, 2.0, 50)
        t_neg = -t_pos
        np.testing.assert_allclose(
            transit_lc(t_pos, b),
            transit_lc(t_neg, b),
            rtol=1e-10,
        )

    def test_impact_parameter_sign_symmetry(self):
        """transit_lc(t, b) = transit_lc(t, −b): geometry is symmetric in b."""
        t = np.linspace(-2.0, 2.0, 100)
        for b in [0.1, 0.4, 0.7, 1.0]:
            np.testing.assert_allclose(
                transit_lc(t,  b),
                transit_lc(t, -b),
                rtol=1e-10,
            )

    # ── Monotonicity ───────────────────────────────────────────────────────────

    def test_ingress_monotone_decreasing(self):
        """Flux must decrease monotonically during ingress (T1 → T2)."""
        b  = 0.2
        T1 = -np.sqrt((RS + RP)**2 - b**2)
        T2 = -np.sqrt((RS - RP)**2 - b**2)
        t_ingress = np.linspace(T1 + 1e-4, T2 - 1e-4, 40)
        flux = transit_lc(t_ingress, b)
        diffs = np.diff(flux)
        assert np.all(diffs <= 1e-12), "Flux should decrease during ingress"

    def test_egress_monotone_increasing(self):
        """Flux must increase monotonically during egress (T3 → T4)."""
        b  = 0.2
        T3 =  np.sqrt((RS - RP)**2 - b**2)
        T4 =  np.sqrt((RS + RP)**2 - b**2)
        t_egress = np.linspace(T3 + 1e-4, T4 - 1e-4, 40)
        flux = transit_lc(t_egress, b)
        diffs = np.diff(flux)
        assert np.all(diffs >= -1e-12), "Flux should increase during egress"

    def test_flat_bottom_for_central_transit(self):
        """For b = 0 the flux is constant between T2 and T3."""
        b  = 0.0
        T2 = -(RS - RP)
        T3 =  (RS - RP)
        t_flat = np.linspace(T2 + 1e-3, T3 - 1e-3, 50)
        flux   = transit_lc(t_flat, b)
        np.testing.assert_allclose(flux, flux[0], rtol=1e-8)

    # ── Flux bounds ────────────────────────────────────────────────────────────

    def test_flux_never_exceeds_unity(self):
        """Flux ≤ 1 everywhere (planet can only block, never amplify)."""
        t = np.linspace(-3.0, 3.0, 300)
        for b in [0.0, 0.3, 0.7, 1.0, 1.2]:
            flux = transit_lc(t, b)
            assert np.all(flux <= 1.0 + 1e-12)

    def test_flux_always_positive(self):
        """Flux > 0 everywhere (even for an enormous planet)."""
        t    = np.linspace(-3.0, 3.0, 300)
        flux = transit_lc(t, b=0.0, r_p=0.99 * RS)
        assert np.all(flux >= 0.0)

    # ── Radius-ratio scaling ───────────────────────────────────────────────────

    def test_depth_scales_as_radius_ratio_squared(self):
        """
        For a uniform disk, mid-transit depth = (Rp/Rs)² for any Rp < Rs.
        """
        for rp in [0.05, 0.10, 0.15, 0.20]:
            f_mid = transit_lc(np.array([0.0]), b=0.0, r_p=rp)[0]
            expected = (rp / RS)**2
            assert 1.0 - f_mid == pytest.approx(expected, rel=1e-5), \
                f"depth mismatch for r_p={rp}"

    # ── Input handling ─────────────────────────────────────────────────────────

    def test_single_element_array_input(self):
        """transit_lc with a 1-element array returns shape (1,)."""
        flux = transit_lc(np.array([0.0]), b=0.0)
        assert flux.shape == (1,)

    def test_empty_array_input(self):
        """transit_lc with empty array returns empty array."""
        flux = transit_lc(np.array([]), b=0.0)
        assert flux.shape == (0,)


# =============================================================================
# contact_times  –  T1–T4 contact-point times
# =============================================================================

class TestContactTimes:
    """Analytical properties of the four contact-point times."""

    def test_no_transit_returns_none(self):
        """b ≥ Rs + Rp → all contact times are None."""
        ct = contact_times(RS + RP + 0.1)
        assert all(v is None for v in ct.values())

    def test_full_transit_has_four_contacts(self):
        """Full transit (b < Rs − Rp) returns all four times."""
        ct = contact_times(0.0)
        assert all(v is not None for v in ct.values())

    def test_grazing_transit_missing_inner_contacts(self):
        """Grazing transit (Rs − Rp < b < Rs + Rp) has only T1 and T4."""
        b_grazing = RS - RP + 0.01
        ct = contact_times(b_grazing)
        assert ct['T1'] is not None
        assert ct['T4'] is not None
        assert ct['T2'] is None
        assert ct['T3'] is None

    def test_contact_time_ordering(self):
        """T1 < T2 < T3 < T4 for a full transit."""
        ct = contact_times(0.3)
        assert ct['T1'] < ct['T2'] < ct['T3'] < ct['T4']

    def test_symmetry_T1_T4(self):
        """T4 = −T1 by symmetry of the orbit geometry."""
        for b in [0.0, 0.2, 0.5, 0.8]:
            ct = contact_times(b)
            if ct['T1'] is not None:
                assert ct['T4'] == pytest.approx(-ct['T1'], rel=1e-10)

    def test_symmetry_T2_T3(self):
        """T3 = −T2 by symmetry."""
        for b in [0.0, 0.2, 0.5]:
            ct = contact_times(b)
            if ct['T2'] is not None:
                assert ct['T3'] == pytest.approx(-ct['T2'], rel=1e-10)

    def test_outer_contact_formula(self):
        """T1 = −√[(Rs + Rp)² − b²] by definition."""
        b = 0.4
        ct = contact_times(b)
        expected_T1 = -np.sqrt((RS + RP)**2 - b**2)
        assert ct['T1'] == pytest.approx(expected_T1, rel=1e-10)

    def test_inner_contact_formula(self):
        """T2 = −√[(Rs − Rp)² − b²] by definition."""
        b = 0.4
        ct = contact_times(b)
        expected_T2 = -np.sqrt((RS - RP)**2 - b**2)
        assert ct['T2'] == pytest.approx(expected_T2, rel=1e-10)

    def test_flux_unity_at_T1_and_T4(self):
        """Flux = 1 at first and fourth contact (planet just touching limb)."""
        b = 0.3
        ct = contact_times(b)
        for key in ('T1', 'T4'):
            flux_at_contact = transit_lc(np.array([ct[key]]), b)[0]
            assert flux_at_contact == pytest.approx(1.0, abs=1e-5)

    def test_flux_depth_at_T2_and_T3(self):
        """
        At second and third contact the full disk area is just blocked:
        depth equals (Rp/Rs)² for b = 0.
        """
        b = 0.0
        ct = contact_times(b)
        expected_flux = 1.0 - RP**2 / RS**2
        for key in ('T2', 'T3'):
            flux_at_contact = transit_lc(np.array([ct[key]]), b)[0]
            assert flux_at_contact == pytest.approx(expected_flux, rel=1e-4)

    def test_transit_duration_decreases_with_b(self):
        """Total transit duration T4 − T1 decreases as b increases."""
        durations = []
        for b in [0.0, 0.2, 0.4, 0.6, 0.8]:
            ct = contact_times(b)
            if ct['T1'] is not None:
                durations.append(ct['T4'] - ct['T1'])
        for d1, d2 in zip(durations, durations[1:]):
            assert d1 > d2 - 1e-10


# =============================================================================
# Integration: consistency between contact_times and transit_lc
# =============================================================================

class TestConsistency:
    """Cross-checks between contact_times and the actual light curve."""

    def test_mid_transit_flux_consistent_with_depth_formula(self):
        """
        For a full transit, the flat-bottom flux should equal 1 − (Rp/Rs)².
        """
        b = 0.2
        ct = contact_times(b)
        t_mid = np.array([0.5 * (ct['T2'] + ct['T3'])])
        flux_mid = transit_lc(t_mid, b)[0]
        expected  = 1.0 - (RP / RS)**2
        assert flux_mid == pytest.approx(expected, rel=1e-5)

    def test_light_curve_is_continuous(self):
        """No sudden jumps in flux across the full time range."""
        t    = np.linspace(-3.0, 3.0, 1000)
        flux = transit_lc(t, b=0.3)
        max_jump = np.max(np.abs(np.diff(flux)))
        assert max_jump < 0.02, f"Discontinuous flux jump of {max_jump} detected"

    def test_flux_at_all_contact_times(self):
        """
        Verify the physical flux values at all four contact points for
        a standard full transit.
        """
        b  = 0.0
        ct = contact_times(b)
        # At T1 and T4: planet just touching → flux ≈ 1
        for key in ('T1', 'T4'):
            f = transit_lc(np.array([ct[key]]), b)[0]
            assert f == pytest.approx(1.0, abs=1e-4)
        # At T2 and T3: planet fully inside → flux = 1 − (Rp/Rs)²
        f_inside = 1.0 - (RP / RS)**2
        for key in ('T2', 'T3'):
            f = transit_lc(np.array([ct[key]]), b)[0]
            assert f == pytest.approx(f_inside, rel=1e-4)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
