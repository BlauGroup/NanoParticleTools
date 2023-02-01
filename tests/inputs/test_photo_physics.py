from NanoParticleTools.inputs.photo_physics import (
    gaussian,
    get_absorption_cross_section_from_line_strength,
    get_transition_rate_from_line_strength,
    get_critical_energy_gap,
    get_MD_line_strength_from_icc,
    magnetic_dipole_operation,
    get_absorption_cross_section_from_MD_line_strength,
    get_oscillator_strength_from_MD_line_strength,
    get_rate_from_MD_line_strength,
    gaussian_overlap_integral,
    phonon_assisted_energy_transfer_constant,
    energy_transfer_constant
)
import pytest
import numpy as np


def test_gaussian():
    assert gaussian(0, 0, 1) == pytest.approx(0.3989422804014327)
    assert np.allclose(gaussian(np.array([0, 5]), 1, 2),
                       np.array([0.17603266, 0.02699548]))

