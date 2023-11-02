from NanoParticleTools.inputs.nanoparticle import NanoParticleConstraint
from NanoParticleTools.flows.flows import get_npmc_flow

import numpy as np
import uuid
from copy import deepcopy
from jobflow import JobStore
from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow


def submit_job(constraints: list[NanoParticleConstraint],
               dopant_specifications: list[tuple[int, float, str, str]],
               lp: LaunchPad,
               store: JobStore,
               excitation_wavelength=800,
               excitation_power=1e5,
               metadata: dict = None,
               priority: int = None,
               doping_seed: int = None,
               base_seed: int = None,
               npmc_runner_kwargs: dict = None,
               initial_state_db_kwargs: dict = None,
               add_to_launchpad: bool = True):
    """

    Args:
        constraints: A list of constraints which are used to specify the control
            volumes of the nanoparticle.
        dopant_specifications: A list of tuples which specify the dopants (and
            their quantity) for each contraint in the nanoparticle.
        lp: The fireworks launchpad to which the workflow will be added and run
        store: Store for which the simulation results will be stored.
        excitation_wavelength: Excitation wavelength for these simulations
        excitation_power: Excitation power for these simulations
        metadata: Metadata to be added to the workflow and saved along with the output
        priority: The fireworks priority for this workflow
        doping_seed: The random seed used for placing dopants in the NP lattice
        base_seed: The base random seed used for sampling events in MC.
            If N simulations are run, the seeds used will be base_seed to base_seed + N
        npmc_runner_kwargs: Keyword arguments to be passed to the NPMC runner
        initial_state_db_kwargs: Keyword arguments to be passed to the NPMCInput.
            This is used to specify the interaction radius bound or to change the
            relative weights of single site transitions vs two site (Energy Transfer)
            transitions.
        add_to_launchpad: Whether or not to add the workflow to the launchpad.
            Setting this to false is useful for debugging, since it will return
            the workflow and not modify the launchpad.
    """

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
    if npmc_runner_kwargs is None:
        npmc_runner_kwargs = {}
    npmc_args.update(npmc_runner_kwargs)

    spectral_kinetics_args = {
        'excitation_wavelength': excitation_wavelength,
        'excitation_power': excitation_power
    }

    if initial_state_db_kwargs is None:
        initial_state_db_kwargs = {'interaction_radius_bound': 3}

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
                         initial_state_db_args=initial_state_db_kwargs,
                         npmc_args=npmc_args,
                         output_dir='./scratch',
                         metadata=_metadata)

    wf = flow_to_workflow(flow, store=store)
    if priority is not None:
        wf = add_priority(wf, priority)

    if add_to_launchpad:
        lp.add_wf(wf)

    return wf


def add_priority(original_wf, root_priority, child_priority=None):
    """
    Note: To avoid package bloat, since atomate heavily used in this
        library, this function is copied from atomate.common.powerups.
          
    Adds priority to a workflow

    Args:
        original_wf (Workflow): original WF
        root_priority (int): priority of first (root) job(s)
        child_priority(int): priority of all child jobs. Defaults to
            root_priority

    Returns:
       Workflow: priority-decorated workflow
    """
    child_priority = child_priority or root_priority
    root_fw_ids = original_wf.root_fw_ids
    for fw in original_wf.fws:
        if fw.fw_id in root_fw_ids:
            fw.spec["_priority"] = root_priority
        else:
            fw.spec["_priority"] = child_priority
    return original_wf
