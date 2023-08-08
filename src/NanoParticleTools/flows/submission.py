import numpy as np
import uuid
from copy import deepcopy
from jobflow import JobStore
from fireworks import LaunchPad
from atomate.common.powerups import add_priority
from jobflow.managers.fireworks import flow_to_workflow
from NanoParticleTools.flows.flows import get_npmc_flow


def submit_job(constraints,
               dopant_specifications,
               lp: LaunchPad,
               store: JobStore,
               excitation_wavelength=800,
               excitation_power=1e5,
               metadata: dict = None,
               priority: int = None,
               doping_seed: int = None,
               base_seed: int = None,
               npmc_runner_kwargs: dict = None):

    # if no seeds are specified, generate random ones
    if doping_seed is None:
        doping_seed = np.random.randint(65536)
    if base_seed is None:
        base_seed = np.random.randint(65536)

    npmc_args = {
        'npmc_command': 'NPMC',
        'num_sims': 4,
        'base_seed': base_seed,
        'thread_count': 4,
        'simulation_time': 0.01,
    }
    if npmc_runner_kwargs is not None:
        npmc_runner_kwargs = {}
    npmc_args.update(npmc_runner_kwargs)

    spectral_kinetics_args = {
        'excitation_wavelength': excitation_wavelength,
        'excitation_power': excitation_power
    }

    initial_state_db_args = {'interaction_radius_bound': 3}

    np_uuid = str(uuid.uuid4())

    if metadata is None:
        _metadata = {}
    else:
        _metadata = deepcopy(metadata)
    _metadata.update({'nanoparticle_identifier': np_uuid})
    flow = get_npmc_flow(constraints=constraints,
                         dopant_specifications=dopant_specifications,
                         doping_seed=doping_seed,
                         spectral_kinetics_args=spectral_kinetics_args,
                         initial_state_db_args=initial_state_db_args,
                         npmc_args=npmc_args,
                         output_dir='./scratch',
                         metadata=_metadata)

    wf = flow_to_workflow(flow, store=store)
    if priority is not None:
        wf = add_priority(wf, priority)
    lp.add_wf(wf)
