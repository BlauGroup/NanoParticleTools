from NanoParticleTools.util.helper import (
    dopant_specifications_to_concentrations,
    dopant_concentration_to_specifications)

import pytest


def test_dopant_specification_to_concentration():
    specs = [(0, 0.5, 'Yb', 'Y'), (0, 0.15, 'Er', 'Y'), (1, 0.35, 'Yb', 'Y'),
             (3, 0.91, 'Nd', 'Y')]

    concentrations = dopant_specifications_to_concentrations(
        specs, 4, include_zeros=True)
    expected_concentrations = [{
        'Yb': 0.5,
        'Er': 0.15,
        'Nd': 0
    }, {
        'Yb': 0.35,
        'Er': 0,
        'Nd': 0
    }, {
        'Yb': 0,
        'Er': 0,
        'Nd': 0
    }, {
        'Yb': 0,
        'Er': 0,
        'Nd': 0.91
    }]
    assert concentrations == expected_concentrations

    concentrations = dopant_specifications_to_concentrations(
        specs, 4, include_zeros=False)
    expected_concentrations = [{
        'Yb': 0.5,
        'Er': 0.15
    }, {
        'Yb': 0.35
    }, {}, {
        'Nd': 0.91,
    }]
    assert concentrations == expected_concentrations

    with pytest.warns(UserWarning):
        concentrations = dopant_specifications_to_concentrations(
            specs, 4, include_zeros=False, possible_elements=['Yb'])
    expected_concentrations = [{
        'Yb': 0.5,
    }, {
        'Yb': 0.35
    }, {}, {}]
    assert concentrations == expected_concentrations

    with pytest.raises(ValueError):
        concentrations = dopant_specifications_to_concentrations(
            specs, 3, include_zeros=False)


def test_dopant_concentration_to_specification():
    concentrations = [{
        'Yb': 0.5,
        'Er': 0.15,
        'Nd': 0
    }, {
        'Yb': 0.35,
        'Er': 0,
        'Nd': 0
    }, {
        'Yb': 0,
        'Er': 0,
        'Nd': 0
    }, {
        'Yb': 0,
        'Er': 0,
        'Nd': 0.91
    }]

    specs = dopant_concentration_to_specifications(concentrations,
                                                   include_zeros=False)
    expected_specs = [(0, 0.5, 'Yb', 'Y'), (0, 0.15, 'Er', 'Y'),
                      (1, 0.35, 'Yb', 'Y'), (3, 0.91, 'Nd', 'Y')]
    assert specs == expected_specs

    concentrations = [{'Yb': 0.5, 'Er': 0.15}, {'Yb': 0.35}, {}, {'Nd': 0.91}]
    specs = dopant_concentration_to_specifications(concentrations,
                                                   include_zeros=False)
    expected_specs = [(0, 0.5, 'Yb', 'Y'), (0, 0.15, 'Er', 'Y'),
                      (1, 0.35, 'Yb', 'Y'), (3, 0.91, 'Nd', 'Y')]
    assert specs == expected_specs

    concentrations = [{
        'Yb': 0.5,
        'Er': 0.15,
        'Nd': 0
    }, {
        'Yb': 0.35,
        'Er': 0,
        'Nd': 0
    }, {
        'Yb': 0,
        'Er': 0,
        'Nd': 0
    }, {
        'Yb': 0,
        'Er': 0,
        'Nd': 0.91
    }]
    specs = dopant_concentration_to_specifications(concentrations,
                                                   include_zeros=False)
    expected_specs = [(0, 0.5, 'Yb', 'Y'), (0, 0.15, 'Er', 'Y'),
                      (1, 0.35, 'Yb', 'Y'), (3, 0.91, 'Nd', 'Y')]
    assert specs == expected_specs

    concentrations = [{
        'Yb': 0.5,
        'Er': 0.15,
        'Nd': 0
    }, {
        'Yb': 0.35,
        'Er': 0,
        'Nd': 0
    }, {
        'Yb': 0,
        'Er': 0,
        'Nd': 0
    }, {
        'Yb': 0,
        'Er': 0,
        'Nd': 0.91
    }]
    specs = dopant_concentration_to_specifications(concentrations,
                                                   include_zeros=True)
    expected_specs = [(0, 0.5, 'Yb', 'Y'), (0, 0.15, 'Er', 'Y'),
                      (0, 0, 'Nd', 'Y'), (1, 0.35, 'Yb', 'Y'),
                      (1, 0, 'Er', 'Y'), (1, 0, 'Nd', 'Y'), (2, 0, 'Yb', 'Y'),
                      (2, 0, 'Er', 'Y'), (2, 0, 'Nd', 'Y'), (3, 0, 'Yb', 'Y'),
                      (3, 0, 'Er', 'Y'), (3, 0.91, 'Nd', 'Y')]
    assert specs == expected_specs
