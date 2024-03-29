import json
import os
from typing import Sequence, Tuple, List, Set

from jobflow import job
from monty.json import MontyEncoder

from NanoParticleTools.analysis import SimulationReplayer
from NanoParticleTools.core import (NPMCInput, NPMCRunner,
                                    create_interupt_state_sql,
                                    create_interupt_cutoff_sql)
from NanoParticleTools.inputs import (SpectralKinetics, DopedNanoparticle,
                                      NanoParticleConstraint)
from NanoParticleTools.species_data.species import Dopant
import sqlite3
import shutil
import logging

LOGGER = logging.getLogger('NPMC_Job')


# Save 'trajectory_doc' to the trajectories store
# (as specified in the JobStore)
@job(trajectories='trajectory_doc')
def npmc_job(constraints: Sequence[NanoParticleConstraint],
             dopant_specifications: Sequence[Tuple[int, float, str, str]],
             doping_seed: int,
             output_dir: str = '.',
             initial_states: Sequence[int] | None = None,
             spectral_kinetics_args: dict | None = None,
             initial_state_db_args: dict | None = None,
             npmc_args: dict | None = None,
             override: bool = False,
             population_record_interval: float = 1e-5,
             metadata: dict | None = None,
             **kwargs) -> List[dict]:
    """
    :param constraints: Constraints from which to build the Nanoparticle.
        Ex. constraints = [SphericalConstraint(20), SphericalConstraint(30)]
    :param dopant_specifications:
        List of tuples specifying (constraint_index, mole fraction desired,
        dopant species, replaced species)
        Ex. dopant_specification =
            [(0, 0.1, 'Yb', 'Y'), (0, 0.02, 'Er', 'Y'), (1, 0.1, 'Gd', 'Y')]
    :param doping_seed: Random generator seed for placing dopants to ensure
        deterministic nanoparticle generation
    :param output_dir: Subdirectory to save output.
        Note: In most cases, jobflow/Fireworks will automatically run in a
              new directory, so this should not be necessary
    :param initial_states: Initial states for the Monte Carlo simulation.
        Typically only supplied for a lifetime experiment
    :param spectral_kinetics_args: a dictionary specifying the parameters
        for the SpectralKinetics object to be constructed. For more
        information, check the documentation for
        NanoParticleTools.inputs.spectral_kinetics.SpectralKinetics

        Ex. spectral_kinetics_args = {'excitation_power': 1e12,
        'excitation_wavelength':980}
    :param initial_state_db_args: a dictionary specifying the parameters to
        populate the initial_state_database with.
        Ex. initial_state_db_args = {'interaction_radius_bound': 3}
    :param npmc_args: a dictionary specifying the parameters to run NPMC with.
        For more information, check the documentation for
        NanoParticleTools.core.NPMCRunner.run()
    :param override: If an existing NPMC run exists, override that run.
        Note: If set true, this will delete existing run in the folder
    :param population_record_interval: Interval to record the population of
        states (in s)
    :return: List of trajectory documents
    """
    if spectral_kinetics_args is None:
        spectral_kinetics_args = {}
    if initial_state_db_args is None:
        initial_state_db_args = {}
    if npmc_args is None:
        npmc_args = {}
    if metadata is None:
        metadata = {}

    files = {
        'output_dir': output_dir,
        'initial_state_db_path': os.path.join(output_dir,
                                              'initial_state.sqlite'),
        'np_db_path': os.path.join(output_dir, 'np.sqlite'),
        'npmc_input': os.path.join(output_dir, 'npmc_input.json')
    }
    _initial_state_db_args = {
        'one_site_interaction_factor': 1,
        'two_site_interaction_factor': 1,
        'interaction_radius_bound': 3,
        'distance_factor_type': 'inverse_cubic'
    }
    _initial_state_db_args.update(initial_state_db_args)

    # Check if output dir exists. If so, check if the input files match
    fresh_start = False
    if os.path.exists(output_dir):
        # Check if the required files are in the directory
        # (inital_state.sqlite, np.sqlite, npmc_input.json)
        np_present = os.path.exists(files['np_db_path'])
        initial_state_present = os.path.exists(files['initial_state_db_path'])
        npmc_input_present = os.path.exists(files['npmc_input'])

        # check if the initial_state db has all required tables
        with sqlite3.connect(files['initial_state_db_path']) as con:
            cur = con.cursor()
            expected_tables = ['factors', 'initial_state']
            all_tables_exist, missing = tables_exist(cur, expected_tables)
            if not all_tables_exist:
                LOGGER.info(f'Existing run found, but missing {missing} table.'
                            f'Re-initializing the simulation')
                fresh_start = True

            cur.close()

        # Check if the inputs match
        with sqlite3.connect(files['np_db_path']) as con:
            cur = con.cursor()
            # Check if the sites and interaction table exists
            expected_tables = ['metadata', 'species', 'sites', 'interactions']
            all_tables_exist, missing = tables_exist(cur, expected_tables)
            if not all_tables_exist:
                LOGGER.info(f'Existing run found, but missing {missing} table.'
                            f'Re-initializing the simulation')
                fresh_start = True
            # # Check the number of sites
            # num_dopant_site_db = len(list(cur.execute('SELECT * from sites')))
            # num_dopant_sites = len(nanoparticle.dopant_sites)
            # if num_dopant_sites != num_dopant_site_db:
            #     Logger.info(
            #         'Existing run found, num sites does not match.'
            #         ' Simulation must begin from scratch')

            # # Check the number of interactions
            # num_interactions_db = len(
            #     list(cur.execute('SELECT * from interactions')))
            # num_interactions = len(get_all_interactions(spectral_kinetics))
            # if num_interactions != num_interactions_db:
            #     Logger.info(
            #         'Existing run found, number of interactions does not '
            #         'match. Simulation must begin from scratch')

            cur.close()

        if np_present and initial_state_present and npmc_input_present:
            # Check if the 'interupt_state' table is found in the sqlite.
            # If not, create it
            with sqlite3.connect(files['initial_state_db_path']) as con:
                cur = con.cursor()

                table_exist, _ = tables_exist(cur, ['interupt_state'])
                if not table_exist:
                    LOGGER.info(
                        'creating interupt_state and interupt_cutoff table')
                    cur.execute(create_interupt_state_sql)
                    cur.execute(create_interupt_cutoff_sql)
                cur.close()
        else:
            LOGGER.info('Existing run found, but some files are missing. ')
            fresh_start = True

    if fresh_start or os.path.exists(output_dir) is False:
        if override or os.path.exists(output_dir) is False:
            LOGGER.info('Writing new input files')
            # Generate Nanoparticle
            nanoparticle = DopedNanoparticle(constraints,
                                             dopant_specifications,
                                             doping_seed,
                                             prune_hosts=True)
            nanoparticle.generate()

            # Initialize Spectral Kinetics class to calculate transition rates
            dopants = [
                Dopant(key, concentration) for key, concentration in
                nanoparticle.dopant_concentrations().items()
            ]
            spectral_kinetics = SpectralKinetics(dopants,
                                                 **spectral_kinetics_args)

            # Create an NPMCInput class
            npmc_input = NPMCInput(spectral_kinetics, nanoparticle,
                                   initial_states)

            # Write files
            if os.path.exists(output_dir):
                # delete the directory, so we can start from scratch
                shutil.rmtree(output_dir)

            # Make the directory
            os.mkdir(output_dir)

            npmc_input.generate_initial_state_database(
                files['initial_state_db_path'], **_initial_state_db_args)
            npmc_input.generate_nano_particle_database(files['np_db_path'])
            with open(files['npmc_input'], 'w') as f:
                json.dump(npmc_input, f, cls=MontyEncoder)
        else:
            raise RuntimeError(
                'Existing run found. Override is set to false, terminating')

    # Initialize the wrapper class to run NPMC
    npmc_runner = NPMCRunner(
        np_db_path=files['np_db_path'],
        initial_state_db_path=files['initial_state_db_path'])

    # Actually run NPMC
    LOGGER.info('Invoking C++ MC simulation')
    npmc_runner.run(**npmc_args)

    # TODO: figure out why the nanoparticle sites gets cleared.
    # generate nanoparticle, since it's state is cleared
    # nanoparticle.generate()

    # Initialize a simulation replayer and run analysis
    simulation_replayer = SimulationReplayer(files['initial_state_db_path'],
                                             files['npmc_input'])
    # data contains a tuple of (simulation_time, event_statistics, new_x,
    # new_population_evolution, new_site_evolution)
    data = simulation_replayer.run(population_record_interval)

    # Check that desired number of simulations was run
    if len(data[0].keys()) != npmc_args['num_sims']:
        raise RuntimeError(f'Run did not successfully complete.'
                           f' Expected {npmc_args["num_sims"]} trajectories, '
                           f'found {len(data[0].keys())}.')

    # Check that all the simulations are complete
    if 'simulation_time' in npmc_args.keys():
        for seed, simulation_time in data[0].items():
            if simulation_time < npmc_args['simulation_time']:
                raise RuntimeError(
                    f'Run did not successfully complete.'
                    f' Simulation {seed} did not complete. Simulated'
                    f' {simulation_time} s of {npmc_args["simulation_time"]} s'
                )

    # get population by shell

    # Generate documents
    result_docs = simulation_replayer.generate_docs(data)
    for i, _ in enumerate(result_docs):
        result_docs[i]['initial_state_db_args'] = _initial_state_db_args
        result_docs[i]['metadata'] = metadata

        # Add metadata to trajectory doc
        result_docs[i]['trajectory_doc']['metadata'] = metadata

    return result_docs


def tables_exist(cur, tables: List | Set):
    if isinstance(tables, list):
        expected_tables = set(tables)
    sql_search = " OR ".join([f"name='{table}'" for table in expected_tables])
    existing_tables = list(
        cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND ({sql_search})"
        ))
    existing_tables = {table[0] for table in existing_tables}
    if existing_tables != expected_tables:
        return False, expected_tables - existing_tables
    return True, {}
