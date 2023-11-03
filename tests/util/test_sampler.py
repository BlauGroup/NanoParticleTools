from NanoParticleTools.util.sampler import NanoParticleSampler
import numpy as np


def test_default_sampler():
    sampler = NanoParticleSampler(seed=0)

    assert sampler.min_core_radius == 40
    assert sampler.max_core_radius == 40
    assert sampler.min_shell_thickness == 10
    assert sampler.max_shell_thickness == 25
    assert sampler.min_concentration == 0
    assert sampler.max_concentration == 0.4
    assert sampler.concentration_constraint == 0.5
    assert sampler._rng is None

    rng = sampler.rng
    assert sampler._rng is not None

    # get 100 core sizes and check that they are all 40
    core_sizes = [sampler.random_nanoparticle_core_size() for i in range(100)]
    assert np.alltrue(np.array(core_sizes) == 40)

    # get 100 thicknesses and check that they are all in the range
    thicknesses = [sampler.random_nanoparticle_layer_thickness() for i in range(100)]
    assert np.alltrue(np.array(thicknesses) >= 10)
    assert np.alltrue(np.array(thicknesses) <= 25)

    # get 100 concentrations and check that they are all in the range
    concentrations = [sampler.random_doping_concentration() for i in range(100)]
    assert np.alltrue(np.array(concentrations) >= 0)
    assert np.alltrue(np.array(concentrations) <= 0.4)


def test_custom_sampler():
    sampler = NanoParticleSampler(seed=0,
                                  min_core_radius=1,
                                  max_core_radius=100,
                                  min_shell_thickness=5,
                                  max_shell_thickness=100,
                                  min_concentration=0.4,
                                  max_concentration=1,
                                  concentration_constraint=1)

    assert sampler.min_core_radius == 1
    assert sampler.max_core_radius == 100
    assert sampler.min_shell_thickness == 5
    assert sampler.max_shell_thickness == 100
    assert sampler.min_concentration == 0.4
    assert sampler.max_concentration == 1
    assert sampler.concentration_constraint == 1

    # get 100 core sizes and check that they are all 40
    core_sizes = [sampler.random_nanoparticle_core_size() for i in range(100)]
    assert np.alltrue(np.array(core_sizes) >= 1)
    assert np.alltrue(np.array(core_sizes) <= 100)

    # get 100 thicknesses and check that they are all in the range
    thicknesses = [sampler.random_nanoparticle_layer_thickness() for i in range(100)]
    assert np.alltrue(np.array(thicknesses) >= 5)
    assert np.alltrue(np.array(thicknesses) <= 100)

    # get 100 concentrations and check that they are all in the range
    concentrations = [sampler.random_doping_concentration() for i in range(100)]
    assert np.alltrue(np.array(concentrations) >= 0.4)
    assert np.alltrue(np.array(concentrations) <= 1.0)


def test_generate_samples():
    sampler = NanoParticleSampler(seed=0)

    samples = sampler.generate_samples(10, 800, 1e5, ['Er', 'Nd', 'Yb'])
    assert len(samples) == 10
    assert samples[0][0] == 800
    assert samples[0][1] == 1e5

    # TODO: check that the samples are valid (i.e valid total concs and # of layers)
