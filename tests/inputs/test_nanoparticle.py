from NanoParticleTools.inputs.nanoparticle import (
    SphericalConstraint, PrismConstraint, CubeConstraint, DopedNanoparticle,
    get_nayf4_structure, get_wse2_structure, get_disordered_nayf4_structure)
from pymatgen.core import Element, DummySpecies, Composition, Species
import numpy as np
import pytest


def test_spherical_constraint():
    constraint = SphericalConstraint(30)
    points = [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 30, 30], [30, 30, 30],
              [17, 17, 17], [18, 18, 18]]

    assert constraint.get_host_structure() == get_nayf4_structure()
    assert constraint.radius == 30
    assert constraint.bounding_box() == [30, 30, 30]

    assert np.all(
        constraint.sites_in_bounds(points) == np.array(
            [True, True, True, False, False, True, False]))

    structure = get_wse2_structure()
    constraint = SphericalConstraint(20, structure)
    assert constraint.get_host_structure() == structure


def test_prism_constraint():
    constraint = PrismConstraint(60, 60, 80)
    points = [[30, 0, 0], [0, 30, 0], [0, 0, 40], [0, 0, 41], [31, 0, 0],
              [30, 30, 40]]
    assert constraint.bounding_box() == [60, 60, 80]

    assert np.all(
        constraint.sites_in_bounds(points) == np.array(
            [True, True, True, False, False, True]))


def test_cube_constraint():
    constraint = CubeConstraint(60)
    points = [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 30, 30], [30, 30, 30],
              [31, 31, 31], [0, 0, 31]]
    assert constraint.bounding_box() == [60, 60, 60]

    assert np.all(
        constraint.sites_in_bounds(points) == np.array(
            [True, True, True, True, True, False, False]))


def test_constraint_as_dict():
    constraint = CubeConstraint(60)
    constraint_dict = constraint.as_dict()

    # check the structure is not present
    assert constraint_dict['host_structure'] is None
    assert constraint_dict['a'] == pytest.approx(60)

    structure = get_wse2_structure()
    constraint = SphericalConstraint(20, host_structure=structure)
    constraint_dict = constraint.as_dict()
    assert constraint_dict['host_structure'] is not None
    assert constraint_dict['radius'] == pytest.approx(20)

    new_constraint = SphericalConstraint.from_dict(constraint_dict)
    assert new_constraint.host_structure is not None
    assert new_constraint.host_structure.composition.reduced_formula == 'WSe2'

    structure = get_disordered_nayf4_structure(False, False)
    constraint = SphericalConstraint(20, host_structure=structure)
    constraint_dict = constraint.as_dict()

    new_constraint = SphericalConstraint.from_dict(constraint_dict)
    assert constraint_dict['host_structure'] is not None
    assert new_constraint.host_structure.composition.reduced_formula == 'NaYF3'


def test_doped_nanoparticle():
    # Test simple structure generation
    constraints = [SphericalConstraint(10)]
    dopants = [(0, 0.1, 'Yb', 'Y')]
    dnp = DopedNanoparticle(constraints, dopants)

    # TODO: test this only if we want longer running test
    dnp.generate()
    assert len(dnp.dopant_sites) == 6
    assert len(dnp.sites) == 332
    assert dnp.composition == Composition({
        'Na': 60,
        'Y': 50,
        'Yb': 6,
        'F': 216
    })
    assert dnp.dopant_composition == Composition({'Yb': 6})

    # Test accelerated generation
    dnp = DopedNanoparticle(constraints, dopants, prune_hosts=True)
    dnp.generate()
    assert len(dnp.dopant_sites) == 6
    assert len(dnp.sites) == 56


def test_nanoparticle_composition():
    constraints = [SphericalConstraint(20)]
    dopants = [(0, 0.1, 'Yb', 'Y'), (0, 0.2, 'Dy', 'Y')]
    dnp = DopedNanoparticle(constraints, dopants)

    with pytest.raises(RuntimeError):
        dnp.composition

    with pytest.raises(RuntimeError):
        dnp.dopant_composition

    dnp.generate()

    assert dnp.composition == Composition({
        'Na': 459,
        'Y': 314,
        'Yb': 45,
        'Dy': 90,
        'F': 1791
    })
    assert dnp.dopant_composition == Composition({'Yb': 45, 'Dy': 90})


def test_host_species_dopant():
    # Test simple structure generation
    constraints = [SphericalConstraint(10)]
    dopants = [(0, 0.1, 'Na', 'Y')]

    with pytest.raises(ValueError):
        dnp = DopedNanoparticle(constraints, dopants)

    dopants = [(0, 0.1, 'Na', 'Y'), (0, 0.2, 'Yb', 'Y')]
    dnp = DopedNanoparticle(constraints, dopants)
    dnp.generate()

    assert dnp.composition == Composition({
        'Na': 66,
        'Y': 40,
        'Yb': 10,
        'F': 216
    })
    assert dnp.dopant_composition == Composition({'Yb': 10})


def test_doped_near_one():
    with pytest.raises(ValueError):
        constraints = [SphericalConstraint(10)]
        dopants = [(0, 0.1, 'Yb', 'Y'), (0, 1, 'Er', 'Y')]
        dnp = DopedNanoparticle(constraints, dopants)

    with pytest.raises(ValueError):
        constraints = [SphericalConstraint(10)]
        dopants = [(0, 2e-4, 'Yb', 'Y'), (0, 1, 'Er', 'Y')]
        dnp = DopedNanoparticle(constraints, dopants)

    constraints = [SphericalConstraint(10)]
    dopants = [(0, 0.33334, 'Yb', 'Y'), (0, 0.666667, 'Er', 'Y')]
    dnp = DopedNanoparticle(constraints, dopants)
    assert dnp.dopant_specification[0][1] == pytest.approx(0.333337, abs=1e-6)
    assert dnp.dopant_specification[1][1] == pytest.approx(0.666662, abs=1e-6)

    constraints = [SphericalConstraint(10), SphericalConstraint(20)]
    dopants = [(0, 0.33334, 'Yb', 'Y'), (0, 0.666667, 'Er', 'Y'),
               (1, .20001, 'Yb', 'Y'), (1, 0.8, 'Er', 'Y')]
    dnp = DopedNanoparticle(constraints, dopants)
    assert dnp.dopant_specification[0][1] == pytest.approx(0.333337, abs=1e-6)
    assert dnp.dopant_specification[1][1] == pytest.approx(0.666662, abs=1e-6)
    assert dnp.dopant_specification[2][1] == pytest.approx(0.200007, abs=1e-6)
    assert dnp.dopant_specification[3][1] == pytest.approx(0.799992, abs=1e-6)


def test_nanoparticle_with_empty():
    constraints = [
        SphericalConstraint(40),
        SphericalConstraint(85),
        SphericalConstraint(102.5)
    ]
    dopant_specifications = [(0, 0.5, 'Yb', 'Y'), (0, 0.2, 'Er', 'Y'),
                             (2, 0.5, 'Yb', 'Y'), (2, 0.1, 'Nd', 'Y')]

    dnp = DopedNanoparticle(constraints,
                            dopant_specifications,
                            prune_hosts=True)
    dnp.generate()
    assert len(dnp.dopant_sites) == 18015
    assert len(dnp.sites) == 59790
    assert dnp.dopant_concentrations()['Yb'] == 0.24609466465964208
    assert dnp.dopant_concentrations()['Er'] == 0.011975246696772036
    assert dnp.dopant_concentrations()['Nd'] == 0.04323465462451915


def test_empty_nanoparticle():
    """
    All of these tests are expected to throw an error
    """
    # Empty constraint list
    constraints = []
    dopants = [(0, 0.1, 'Yb', 'Y')]
    with pytest.raises(ValueError):
        dnp = DopedNanoparticle(constraints, dopants)

    # No dopants specified
    constraints = [SphericalConstraint(100)]
    dopants = []
    with pytest.raises(ValueError):
        dnp = DopedNanoparticle(constraints, dopants)

    constraints = [SphericalConstraint(100)]
    dopants = [(0, 0.0, 'Er', 'Y'), (0, 0.0, 'Nd', 'Y'), (0, 0.0, 'Yb', 'Y')]
    with pytest.raises(ValueError):
        dnp = DopedNanoparticle(constraints, dopants)

    constraints = [SphericalConstraint(100)]
    dopants = [(0, 0, 'Er', 'Y')]
    with pytest.raises(ValueError):
        dnp = DopedNanoparticle(constraints, dopants)

    # Too many dopants specified
    constraints = [SphericalConstraint(100)]
    dopants = [(0, 0.5, 'Er', 'Y'), (0, 0.5, 'Nd', 'Y'), (0, 0.5, 'Yb', 'Y')]
    with pytest.raises(ValueError):
        dnp = DopedNanoparticle(constraints, dopants)

    constraints = [SphericalConstraint(100)]
    dopants = [(0, 1.001, 'Er', 'Y')]
    with pytest.raises(ValueError):
        dnp = DopedNanoparticle(constraints, dopants)


def test_write_to_file(tmp_path):
    """
    Note, this does not check the contents of the file, only if we can write
    the file.
    """
    constraints = [SphericalConstraint(10)]
    dopants = [(0, 0.1, 'Na', 'Y'), (0, 0.2, 'Yb', 'Y')]
    dnp = DopedNanoparticle(constraints, dopants)

    with pytest.raises(RuntimeError):
        dnp.to_file('xyz', f'{tmp_path}/nanoparticle.xyz')

    with pytest.raises(RuntimeError):
        dnp.dopants_to_file('xyz', f'{tmp_path}/dopant_nanoparticle.xyz')

    dnp.generate()

    dnp.to_file('xyz', f'{tmp_path}/nanoparticle.xyz')
    dnp.dopants_to_file('xyz', f'{tmp_path}/dopant_nanoparticle.xyz')


def test_get_nayf4_structure():
    struct = get_nayf4_structure()

    assert ([str(el) for el in struct.species] == [
        'Na', 'Na', 'Na', 'Y', 'Y', 'Y', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
        'F', 'F', 'F', 'F', 'F'
    ])
    lattice = np.array([[6.067, 0.0, 0.0], [-3.0335, 5.25417612, 0.0],
                        [0.0, 0.0, 7.103]])
    assert np.allclose(struct.lattice.matrix, lattice)


def test_get_wse2_structure():
    struct = get_wse2_structure()

    assert ([str(el)
             for el in struct.species] == ['Se', 'Se', 'Se', 'Se', 'W', 'W'])

    lattice = np.array([[3.327, 0.0, 0], [-1.6635, 2.88126652, 0],
                        [0.0, 0.0, 15.069]])
    assert np.allclose(struct.lattice.matrix, lattice)


def test_disordered_structure():
    # Test non-charge decorated structure
    struct = get_disordered_nayf4_structure(False, False)

    assert ([str(el) for el in struct.species
             ] == ['Na', 'Na', 'Y', 'Y', 'F', 'F', 'F', 'F', 'F', 'F'])
    assert np.allclose(struct.lattice.abc, [5.9688, 5.9688, 3.5090])
    expected_coords = [[0.00000000e+00, 0.00000000e+00, 3.33355000e-01],
                       [0.00000000e+00, 0.00000000e+00, 2.08785500e+00],
                       [-2.98440000e-04, 3.44626059e+00, 2.63175000e+00],
                       [2.98469844e+00, 1.72287184e+00, 8.77250000e-01],
                       [-1.53278784e+00, 3.53258510e+00, 8.77250000e-01],
                       [6.91485480e-01, 2.07540667e+00, 8.77250000e-01],
                       [2.14309764e+00, 4.38859343e-01, 2.63175000e+00],
                       [8.41302360e-01, 4.73027309e+00, 8.77250000e-01],
                       [2.29291452e+00, 3.09372576e+00, 2.63175000e+00],
                       [4.51718784e+00, 1.63654733e+00, 2.63175000e+00]]
    assert np.allclose(struct.cart_coords, expected_coords)

    # Test charge decorated structure
    struct = get_disordered_nayf4_structure(True, False)
    assert ([str(el) for el in struct.species] == [
        'Na+', 'Na+', 'Y3+', 'Y3+', 'F-', 'F-', 'F-', 'F-', 'F-', 'F-'
    ])

    # Test the inclusion of partial occupancy
    struct = get_disordered_nayf4_structure(False, True)

    # yapf: disable
    expected_species = [{Element('Na'): 0.25, DummySpecies(): 0.75},
                        {Element('Na'): 0.25, DummySpecies(): 0.75},
                        {Element('Na'): 0.25, DummySpecies(): 0.75},
                        {Element('Na'): 0.25, DummySpecies(): 0.75},
                        {Element('Y'): 0.75, Element('Na'): 0.25},
                        {Element('Y'): 0.75, Element('Na'): 0.25},
                        {Element('F'): 1}, {Element('F'): 1}, {Element('F'): 1},
                        {Element('F'): 1}, {Element('F'): 1}, {Element('F'): 1}]
    # yapf: enable

    assert [dict(site.species) for site in struct.sites] == expected_species

    # Test the inclusion of partial occupancy and charges
    struct = get_disordered_nayf4_structure(True, True)

    # yapf: disable
    expected_species = [{Species('Na', 1): 0.25, DummySpecies(): 0.75},
                        {Species('Na', 1): 0.25, DummySpecies(): 0.75},
                        {Species('Na', 1): 0.25, DummySpecies(): 0.75},
                        {Species('Na', 1): 0.25, DummySpecies(): 0.75},
                        {Species('Y', 3): 0.75, Species('Na', 1): 0.25},
                        {Species('Y', 3): 0.75, Species('Na', 1): 0.25},
                        {Species('F', -1): 1}, {Species('F', -1): 1}, {Species('F', -1): 1},
                        {Species('F', -1): 1}, {Species('F', -1): 1}, {Species('F', -1): 1}]
    # yapf: enable

    assert [dict(site.species) for site in struct.sites] == expected_species
