from NanoParticleTools.species_data.species import (
    EnergyLevel,
    Transition,
    Dopant,
)
import numpy as np
import pytest


def test_energy_level():
    energy_level1 = EnergyLevel('Er', '4I15/2', 0.0)

    assert energy_level1.element == 'Er'
    assert energy_level1.label == '4I15/2'
    assert energy_level1.energy == 0.0

    energy_level2 = EnergyLevel('Er', '4I13/2', 6495.0)
    assert energy_level2.element == 'Er'
    assert energy_level2.label == '4I13/2'
    assert energy_level2.energy == 6495.0
    assert repr(energy_level2) == 'Er - 4I13/2 - 6495.0'


def test_transition():
    energy_level1 = EnergyLevel('Er', '4I15/2', 0.0)
    energy_level2 = EnergyLevel('Er', '4I13/2', 6495.0)
    transition = Transition(energy_level1, energy_level2, 0.5)
    assert transition.initial_level == energy_level1
    assert transition.final_level == energy_level2
    assert transition.line_strength == 0.5
    assert repr(transition) == '4I15/2 -> 4I13/2'


def test_dopant():
    dopant = Dopant('Yb', 0.1)

    assert len(dopant.energy_levels) == 2
    assert dopant.n_levels == 2
    assert dopant.symbol == 'Yb'
    assert dopant.molar_concentration == 0.1
    assert dopant.volume_concentration == pytest.approx(1.3813172234688942)
    assert dopant.absFWHM == [400.0, 400.0]
    assert np.allclose(dopant.slj, np.array([[0.5, 3.0, 3.5], [0.5, 3.0,
                                                               2.5]]))
    assert np.allclose(dopant.eigenvector_sl, np.array([[0.5, 3.0]]))
    assert dopant.judd_ofelt_parameters == []
    assert dopant.check_intrinsic_data()

    # Check Er without specifying n_levels
    dopant = Dopant('Er', 0.02)

    assert len(dopant.energy_levels) == 34
    assert dopant.n_levels == 34
    assert dopant.judd_ofelt_parameters == [
        1.33e-20, 8.6100002e-21, 7.6600002e-21
    ]

    # Check Er with specifying n_levels
    dopant = Dopant('Er', 0.02, 30)

    assert len(dopant.energy_levels) == 30
    assert dopant.n_levels == 30
    assert dopant.check_intrinsic_data()

    # Check too high of a concentration
    with pytest.raises(ValueError):
        dopant = Dopant('Er', 1.2)


def test_line_strengths():
    dopant = Dopant('Er', 0.02, 4)
    line_strengths = dopant.get_line_strength_matrix()
    print(line_strengths)
    # yapf:disable
    expected_line_strengths = [
        [0.0000000e+00, 1.9410000e-20, 5.4019999e-21, 2.4879999e-21],
        [1.9410000e-20, 0.0000000e+00, 1.6229999e-20, 8.8570001e-21],
        [5.4019999e-21, 1.6229999e-20, 0.0000000e+00, 2.5289999e-21],
        [2.4879999e-21, 8.8570001e-21, 2.5289999e-21, 0.0000000e+00],
    ]
    # yapf:enable
    assert np.allclose(line_strengths, expected_line_strengths)


def test_surfaces():
    for key, item in Dopant.SURFACE_DOPANT_NAMES_TO_SYMBOLS.items():
        print(key, item)
        surface = Dopant(key, 0.1)
        surface_element_def = Dopant(item, 0.1)

        assert surface.symbol == surface_element_def.symbol

    with pytest.raises(FileNotFoundError):
        surface = Dopant('Surface1', 0.5)

    surface = Dopant('Surface3', 0.5)
    assert surface.check_intrinsic_data()
    assert surface.judd_ofelt_parameters == []
    assert surface.intermediate_coupling_coefficients == []
    assert surface.eigenvector_sl == []
