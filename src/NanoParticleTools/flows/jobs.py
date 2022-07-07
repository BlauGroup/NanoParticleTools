import json
import os
from typing import Sequence, Tuple, Optional, Union, List

from jobflow import job
from monty.json import MontyEncoder

from NanoParticleTools.analysis import SimulationReplayer
from NanoParticleTools.core import NPMCInput, NPMCRunner, create_interupt_state_sql, create_interupt_cutoff_sql
from NanoParticleTools.inputs.nanoparticle import DopedNanoparticle, NanoParticleConstraint
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.species_data.species import Dopant
from NanoParticleTools.inputs.util import get_all_interactions
from pymatgen.core import Composition
from collections import Counter
import numpy as np
import sqlite3


# Save 'trajectory_doc' to the trajectories store (as specified in the JobStore)
@job(trajectories='trajectory_doc')
def npmc_job(constraints: Sequence[NanoParticleConstraint],
             dopant_specifications: Sequence[Tuple[int, float, str, str]],
             doping_seed: int,
             output_dir: Optional[str] = '.',
             initial_states: Optional[Union[Sequence[int], None]] = None,
             spectral_kinetics_args: Optional[dict] = {},
             initial_state_db_args: Optional[dict] = {},
             npmc_args: Optional[dict] = {},
             override: Optional[bool] = False,
             population_record_interval: Optional[float] = 1e-5,
             **kwargs) -> List[dict]:
    """

    :param constraints: Constraints from which to build the Nanoparticle.
        Ex. constraints = [SphericalConstraint(20), SphericalConstraint(30)]
    :param dopant_specifications:
        List of tuples specifying (constraint_index, mole fraction desired, dopant species, replaced species)
        Ex. dopant_specification = [(0, 0.1, 'Yb', 'Y'), (0, 0.02, 'Er', 'Y'), (1, 0.1, 'Gd', 'Y')]
    :param doping_seed: Random generator seed for placing dopants to ensure deterministic nanoparticle generation
    :param output_dir: Subdirectory to save output.
        Note: In most cases, jobflow/Fireworks will automatically run in a new directory,
              so this should not be necessary
    :param initial_states: Initial states for the Monte Carlo simulation.
        Typically only supplied for a lifetime experiment
    :param spectral_kinetics_args: a dictionary specifying the parameters for the SpectralKinetics object
        to be constructed. For more information, check the documentation for
        NanoParticleTools.inputs.spectral_kinetics.SpectralKinetics.__init__()
        Ex. spectral_kinetics_args = {'excitation_power': 1e12, 'excitation_wavelength':980}
    :param initial_state_db_args: a dictionary specifying the parameters to populate the initial_state_database with.
        Ex. initial_state_db_args = {'interaction_radius_bound': 3}
    :param npmc_args: a dictionary specifying the parameters to run NPMC with. For more information,
        check the documentation for NanoParticleTools.core.NPMCRunner.run()
    :param override: If an existing NPMC run exists, override that run (deletes existing run in the folder
    :param population_record_interval: Interval to record the population of states (in s)
    :return: List of trajectory documents
    """

    files = {'output_dir': output_dir,
             'initial_state_db_path': os.path.join(output_dir, 'initial_state.sqlite'),
             'np_db_path': os.path.join(output_dir, 'np.sqlite'),
             'npmc_input': os.path.join(output_dir, 'npmc_input.json')}

    # Generate Nanoparticle
    nanoparticle = DopedNanoparticle(constraints, dopant_specifications, doping_seed)
    nanoparticle.generate()

    # Initialize Spectral Kinetics class to calculate transition rates
    dopants = [Dopant(key, concentration) for key, concentration in nanoparticle.dopant_concentrations.items()]
    spectral_kinetics = SpectralKinetics(dopants, **spectral_kinetics_args)

    # Create an NPMCInput class
    npmc_input = NPMCInput(spectral_kinetics, nanoparticle, initial_states)

    # Write files
    _initial_state_db_args = {'one_site_interaction_factor': 1,
                              'two_site_interaction_factor': 1,
                              'interaction_radius_bound': 3,
                              'distance_factor_type': 'inverse_cubic'}
    _initial_state_db_args.update(initial_state_db_args)

    # Check if output dir exists. If so, look for the
    if override==False and os.path.exists(output_dir):
        # Check if the required files are in the directory (inital_state.sqlite, np.sqlite, npmc_input.json)
        np_present = os.path.exists(files['np_db_path'])
        initial_state_present = os.path.exists(files['initial_state_db_path'])
        npmc_input_present = os.path.exists(files['npmc_input'])

        # Check if the inputs match
        with sqlite3.connect(files['np_db_path']) as con:
            cur = con.cursor()
            num_dopant_site_db = len(list(cur.execute('SELECT * from sites')))
            num_dopant_sites = len(nanoparticle.dopant_sites)
            if num_dopant_sites != num_dopant_site_db:
                raise RuntimeError('Existing run found, num sites does not match. Simulation must begin from scratch')

        ## Check the number of interactions

        with sqlite3.connect(files['np_db_path']) as con:
            cur = con.cursor()
            num_interactions_db = len(list(cur.execute('SELECT * from interactions')))
            num_interactions = len(get_all_interactions(spectral_kinetics))
            if num_interactions != num_interactions_db:
                raise RuntimeError('Existing run found, number of interactions does not match. Simulation must begin from scratch')


        if np_present and initial_state_present and npmc_input_present:
            # Check if the 'interupt_state' table is found in the sqlite. If not, create it
            with sqlite3.connect(files['initial_state_db_path']) as con:

                cur = con.cursor()

                try:
                    cur.execute('SELECT * from interupt_state')
                except:
                    print(f'creating interupt_state and interupt_cutoff table')
                    cur.execute(create_interupt_state_sql)
                    cur.execute(create_interupt_cutoff_sql)
                cur.close()
        else:
            raise RuntimeError('Existing run found, but some files are missing. ')

    elif override or os.path.exists(output_dir)==False:
        # Directories of written files
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        npmc_input.generate_initial_state_database(files['initial_state_db_path'], **_initial_state_db_args)
        npmc_input.generate_nano_particle_database(files['np_db_path'])
        with open(files['npmc_input'], 'w') as f:
            json.dump(npmc_input, f, cls=MontyEncoder)
    else:
        raise RuntimeError('Existing run found. Override is set to false, terminating')

    # Initialize the wrapper class to run NPMC
    npmc_runner = NPMCRunner(np_db_path=files['np_db_path'],
                             initial_state_db_path=files['initial_state_db_path'])

    # Actually run NPMC
    npmc_runner.run(**npmc_args)

    # # TODO: figure out why the nanoparticle sites gets cleared.
    # nanoparticle.generate() # generate nanoparticle, since it's state is cleared

    # Initialize a simulation replayer and run analysis
    simulation_replayer = SimulationReplayer(files['initial_state_db_path'], files['npmc_input'])
    data = simulation_replayer.run(population_record_interval) # simulation_time, event_statistics, new_x, new_population_evolution, new_site_evolution

    # Check that desired number of simulations was run
    if len(data[0].keys()) != npmc_args['num_sims']:
        raise RuntimeError(f'Run did not successfully complete. Expected {npmc_args["num_sims"]} trajectories, '
                           f'found {len(data[0].keys())}.')

    # get population by shell

    # Generate documents        
    result_docs = simulation_replayer.generate_docs(data)
    for doc in result_docs:
        doc['initial_state_db_args'] = _initial_state_db_args
    
    return result_docs


