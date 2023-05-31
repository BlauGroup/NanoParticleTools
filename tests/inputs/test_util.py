from NanoParticleTools.inputs.util import (
    specie_energy_level_to_combined_energy_level,
    combined_energy_level_to_specie_energy_level,
    combined_energy_level_to_specie_id, combined_energy_level_to_specie_name,
    get_non_radiative_interactions, get_radiative_interactions,
    get_magnetic_dipole_interactions, get_energy_transfer_interactions,
    get_all_interactions, get_sites, get_species, get_formula_by_constraint)
from NanoParticleTools.inputs import (SpectralKinetics,
                                                   DopedNanoparticle,
                                                   SphericalConstraint)

from NanoParticleTools.species_data.species import Dopant
import json
from pathlib import Path
import os
import pytest

MODULE_DIR = Path(__file__).absolute().parent
TEST_FILE_DIR = MODULE_DIR / '..' / 'test_files/spectral_kinetics/Yb10_Er2'


def initialize_sk():
    with open(os.path.join(TEST_FILE_DIR, 'sk_config.json'), 'r') as f:
        config_dict = json.load(f)
    dopants = [Dopant('Er', 0.02, 34), Dopant('Yb', 0.1, 2)]
    sk = SpectralKinetics(dopants, **config_dict)
    return sk


SPECTRAL_KINETICS = initialize_sk()


def test_specie_energy_level_to_combined_energy_level():
    dopants = [Dopant('Yb', 0.5), Dopant('Er', 0.15)]
    assert specie_energy_level_to_combined_energy_level('Yb', 0, dopants) == 0
    assert specie_energy_level_to_combined_energy_level('Yb', 1, dopants) == 1
    assert specie_energy_level_to_combined_energy_level('Er', 0, dopants) == 2
    assert specie_energy_level_to_combined_energy_level('Er', 33,
                                                        dopants) == 35
    assert specie_energy_level_to_combined_energy_level(1, 15, dopants) == 17
    assert specie_energy_level_to_combined_energy_level(
        dopants[1], 10, dopants) == 12
    with pytest.raises(ValueError):
        specie_energy_level_to_combined_energy_level('Er', 35, dopants)

    with pytest.raises(ValueError):
        specie_energy_level_to_combined_energy_level({'Yb': 1}, 10, dopants)


def test_combined_energy_level_to_specie_energy_level():
    assert combined_energy_level_to_specie_energy_level(SPECTRAL_KINETICS,
                                                        10) == 10
    assert combined_energy_level_to_specie_energy_level(SPECTRAL_KINETICS,
                                                        33) == 33
    assert combined_energy_level_to_specie_energy_level(SPECTRAL_KINETICS,
                                                        34) == 0
    assert combined_energy_level_to_specie_energy_level(SPECTRAL_KINETICS,
                                                        35) == 1
    with pytest.raises(KeyError):
        combined_energy_level_to_specie_energy_level(SPECTRAL_KINETICS, 39)

    with pytest.raises(KeyError):
        combined_energy_level_to_specie_energy_level(SPECTRAL_KINETICS, -2)


def test_combined_energy_level_to_specie_id():
    assert combined_energy_level_to_specie_id(SPECTRAL_KINETICS, 10) == 0
    assert combined_energy_level_to_specie_id(SPECTRAL_KINETICS, 35) == 1
    with pytest.raises(KeyError):
        combined_energy_level_to_specie_id(SPECTRAL_KINETICS, 39)

    with pytest.raises(KeyError):
        combined_energy_level_to_specie_id(SPECTRAL_KINETICS, -2)


def test_combined_energy_level_to_specie_name():
    assert combined_energy_level_to_specie_name(SPECTRAL_KINETICS, 10) == 'Er'
    assert combined_energy_level_to_specie_name(SPECTRAL_KINETICS, 34) == 'Yb'


def test_get_non_radiative_interactions():
    nr_interactions = get_non_radiative_interactions(SPECTRAL_KINETICS)
    assert len(nr_interactions) == 51


def test_get_radiative_interactions():
    rad_interactions = get_radiative_interactions(SPECTRAL_KINETICS)
    assert len(rad_interactions) == 579


def test_get_magnetic_dipole_interactions():
    md_interactions = get_magnetic_dipole_interactions(SPECTRAL_KINETICS)
    assert len(md_interactions) == 195


def test_get_energy_transfer_interactions():
    et_interactions = get_energy_transfer_interactions(SPECTRAL_KINETICS)
    assert len(et_interactions) == 19189


def test_get_all_interactions():
    all_interactions = get_all_interactions(SPECTRAL_KINETICS)
    assert len(all_interactions) == 20014


def test_get_sites():
    constraints = [
        SphericalConstraint(10),
        SphericalConstraint(20),
        SphericalConstraint(25)
    ]
    dopant_specifications = [(0, 0.5, 'Yb', 'Y'), (0, 0.12, 'Er', 'Y'),
                             (2, 0.2, 'Er', 'Y')]
    dnp = DopedNanoparticle(constraints, dopant_specifications)
    dnp.generate()

    sites = get_sites(dnp, SPECTRAL_KINETICS)
    assert len(sites) == 120

    with pytest.raises(ValueError):
        constraints = [
            SphericalConstraint(10),
            SphericalConstraint(20),
            SphericalConstraint(25)
        ]
        dopant_specifications = [(0, 0.5, 'Yb', 'Y'), (0, 0.12, 'Er', 'Y'),
                                 (2, 0.2, 'Nd', 'Y')]
        dnp = DopedNanoparticle(constraints, dopant_specifications)
        dnp.generate()

        get_sites(dnp, SPECTRAL_KINETICS)


def test_get_species():
    species = get_species(SPECTRAL_KINETICS)
    assert len(species) == 2


def test_get_formula_by_constraint():
    constraints = [
        SphericalConstraint(10),
        SphericalConstraint(20),
        SphericalConstraint(25)
    ]
    dopant_specifications = [(0, 0.5, 'Yb', 'Y'), (0, 0.12, 'Er', 'Y'),
                             (2, 0.2, 'Nd', 'Y')]
    dnp = DopedNanoparticle(constraints, dopant_specifications)
    dnp.generate()

    formulas = get_formula_by_constraint(dnp)
    assert formulas == ['Yb28Er7', '', 'Nd85']
