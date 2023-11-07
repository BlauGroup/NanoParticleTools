from NanoParticleTools.flows.flows import get_npmc_flow
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint

from jobflow import run_locally
from maggma.stores import MemoryStore
from jobflow import JobStore

import pytest
"""
Currently, we skip this test, since there is no easy way to compile NPMC
for multiple systems (esp. in github actions testing workflows).

TODO: Revisit this when a RNMC/NPMC conda package is created.
"""


@pytest.mark.skip(reason='No way to run this currently')
def test_flow():
    constraints = [SphericalConstraint(20)]
    dopant_specifications = [(0, 0.1, 'Yb', 'Y'), (0, 0.02, 'Er', 'Y')]

    npmc_args = {
        'npmc_command': 'NPMC',
        'num_sims': 2,
        'base_seed': 1000,
        'thread_count': 8,
        'simulation_length': 1000,
    }
    spectral_kinetics_args = {
        'excitation_power': 1e12,
        'excitation_wavelength': 980
    }

    flow = get_npmc_flow(constraints=constraints,
                         dopant_specifications=dopant_specifications,
                         doping_seed=0,
                         spectral_kinetics_args=spectral_kinetics_args,
                         npmc_args=npmc_args,
                         output_dir='./scratch')

    # Store the output data locally in a MemoryStore
    docs_store = MemoryStore()
    data_store = MemoryStore()
    store = JobStore(docs_store,
                     additional_stores={'trajectories': data_store})

    responses = run_locally(flow, store=store, ensure_success=True)

    assert data_store.count() == 2
