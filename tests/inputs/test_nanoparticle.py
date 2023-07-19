from NanoParticleTools.inputs.nanoparticle import (
    SphericalConstraint,
    PrismConstraint,
    CubeConstraint,
    DopedNanoparticle,
    get_nayf4_structure,
    get_wse2_structure
)
import numpy as np
import pytest


def test_spherical_constraint():
    constraint = SphericalConstraint(30)
    points = [[30, 0, 0],
              [0, 30, 0],
              [0, 0, 30],
              [0, 30, 30],
              [30, 30, 30],
              [17, 17, 17],
              [18, 18, 18]]

    assert constraint.host_structure == get_nayf4_structure()
    assert constraint.radius == 30
    assert constraint.bounding_box() == [30, 30, 30]

    assert np.all(constraint.sites_in_bounds(points) ==
                  np.array([True, True, True, False, False, True, False]))

    structure = get_wse2_structure()
    constraint = SphericalConstraint(20, structure)
    assert constraint.host_structure == structure


def test_prism_constraint():
    constraint = PrismConstraint(60, 60, 80)
    points = [[30, 0, 0],
              [0, 30, 0],
              [0, 0, 40],
              [0, 0, 41],
              [31, 0, 0],
              [30, 30, 40]]
    assert constraint.bounding_box() == [60, 60, 80]

    assert np.all(constraint.sites_in_bounds(points) ==
                  np.array([True, True, True, False, False, True]))


def test_cube_constraint():
    constraint = CubeConstraint(60)
    points = [[30, 0, 0],
              [0, 30, 0],
              [0, 0, 30],
              [0, 30, 30],
              [30, 30, 30],
              [31, 31, 31],
              [0, 0, 31]]
    assert constraint.bounding_box() == [60, 60, 60]

    assert np.all(constraint.sites_in_bounds(points) ==
                  np.array([True, True, True, True, True, False, False]))


def test_doped_nanoparticle():
    # Test simple structure generation
    constraints = [SphericalConstraint(10)]
    dopants = [(0, 0.1, 'Yb', 'Y')]
    dnp = DopedNanoparticle(constraints, dopants)

    # TODO: test this only if we want longer running test
    dnp.generate()
    assert len(dnp.dopant_sites) == 6
    assert len(dnp.sites) == 332

    # Test accelerated generation
    dnp = DopedNanoparticle(constraints, dopants, prune_hosts=True)
    dnp.generate()
    assert len(dnp.dopant_sites) == 6
    assert len(dnp.sites) == 56


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


def test_get_nayf4_structure():
    struct = get_nayf4_structure()

    assert ([str(el) for el in struct.species] ==
            ['Na', 'Na', 'Na', 'Y', 'Y', 'Y', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'])
    lattice = np.array([[6.067, 0.0, 0.0],
                        [-3.0335, 5.25417612, 0.0],
                        [0.0, 0.0, 7.103]])
    assert np.allclose(struct.lattice.matrix, lattice)


def test_get_wse2_structure():
    struct = get_wse2_structure()

    assert ([str(el) for el in struct.species] == ['Se', 'Se', 'Se', 'Se', 'W', 'W'])

    lattice = np.array([[3.327, 0.0, 0],
                        [-1.6635, 2.88126652, 0],
                        [0.0, 0.0, 15.069]])
    assert np.allclose(struct.lattice.matrix, lattice)
