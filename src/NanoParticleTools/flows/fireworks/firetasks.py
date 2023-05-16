from fireworks import Firework, Workflow, FiretaskBase, explicit_serialize
import json
import os
from typing import Sequence, Tuple, Optional, Union, List

from jobflow import job
from monty.json import MontyEncoder

from NanoParticleTools.analysis import SimulationReplayer
from NanoParticleTools.core import (NPMCInput, NPMCRunner,
                                    create_interupt_state_sql,
                                    create_interupt_cutoff_sql)
from NanoParticleTools.inputs.nanoparticle import (DopedNanoparticle,
                                                   NanoParticleConstraint)
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.species_data.species import Dopant
from NanoParticleTools.inputs.util import get_all_interactions
import sqlite3

@explicit_serialize
class NPMCFiretask(FiretaskBase):
    required_params = ['constraints', 'dopant_specifications', "doping_seed"]

    optional_params = [
        "output_dir", "initial_states", "spectral_kinetics_args",
        "initial_state_db_args", "npmc_args", "override",
        "population_record_interval", "metadata"
    ]

    def run_task(self, fw_spec):
        constraints = self['constraints']
        dopant_specifications = self['dopant_specifications']
        doping_seed = self['doping_seed']

        output_dir = self.get('output_dir', '.')
        initial_states = self.get('initial_states', None)
        spectral_kinetics_args = self.get('spectral_kinetics_args', {})
        initial_state_db_args = self.get('initial_state_db_args', {})
        npmc_args = self.get('npmc_args', {})
        override = self.get('override', False)
        population_record_interval = self.get('population_record_interval',
                                              1e-5)
        metadata = self.get('metadata', {})

        files = {
            'output_dir':
            output_dir,
            'initial_state_db_path':
            os.path.join(output_dir, 'initial_state.sqlite'),
            'np_db_path':
            os.path.join(output_dir, 'np.sqlite'),
            'npmc_input':
            os.path.join(output_dir, 'npmc_input.json')
        }

        # Generate Nanoparticle
        nanoparticle = DopedNanoparticle(constraints, dopant_specifications,
                                         doping_seed)
        nanoparticle.generate()

        # Initialize Spectral Kinetics class to calculate transition rates
        dopants = [
            Dopant(key, concentration) for key, concentration in
            nanoparticle.dopant_concentrations().items()
        ]
        spectral_kinetics = SpectralKinetics(dopants, **spectral_kinetics_args)

        # Create an NPMCInput class
        npmc_input = NPMCInput(spectral_kinetics, nanoparticle, initial_states)

        # Write files
        _initial_state_db_args = {
            'one_site_interaction_factor': 1,
            'two_site_interaction_factor': 1,
            'interaction_radius_bound': 3,
            'distance_factor_type': 'inverse_cubic'
        }
        _initial_state_db_args.update(initial_state_db_args)

        # Check if output dir exists. If so, look for the
        if override is False and os.path.exists(output_dir):
            # Check if the required files are in the directory
            # (inital_state.sqlite, np.sqlite, npmc_input.json)
            np_present = os.path.exists(files['np_db_path'])
            initial_state_present = os.path.exists(
                files['initial_state_db_path'])
            npmc_input_present = os.path.exists(files['npmc_input'])

            # Check if the inputs match
            with sqlite3.connect(files['np_db_path']) as con:
                cur = con.cursor()
                num_dopant_site_db = len(
                    list(cur.execute('SELECT * from sites')))
                num_dopant_sites = len(nanoparticle.dopant_sites)
                if num_dopant_sites != num_dopant_site_db:
                    raise RuntimeError(
                        'Existing run found, num sites does not match.'
                        ' Simulation must begin from scratch')

            # Check the number of interactions
            with sqlite3.connect(files['np_db_path']) as con:
                cur = con.cursor()
                num_interactions_db = len(
                    list(cur.execute('SELECT * from interactions')))
                num_interactions = len(get_all_interactions(spectral_kinetics))
                if num_interactions != num_interactions_db:
                    raise RuntimeError(
                        'Existing run found, number of interactions does not '
                        'match. Simulation must begin from scratch')

            if np_present and initial_state_present and npmc_input_present:
                # Check if the 'interupt_state' table is found in the sqlite.
                # If not, create it
                with sqlite3.connect(files['initial_state_db_path']) as con:

                    cur = con.cursor()

                    try:
                        cur.execute('SELECT * from interupt_state')
                    except Exception:
                        print(
                            'creating interupt_state and interupt_cutoff table'
                        )
                        cur.execute(create_interupt_state_sql)
                        cur.execute(create_interupt_cutoff_sql)
                    cur.close()
            else:
                raise RuntimeError(
                    'Existing run found, but some files are missing. ')

        elif override or os.path.exists(output_dir) is False:
            # Directories of written files
            if not os.path.exists(output_dir):
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
        npmc_runner.run(**npmc_args)

        # TODO:Implement analysis and insertion into database
        # # TODO: figure out why the nanoparticle sites gets cleared.
        # # generate nanoparticle, since it's state is cleared
        # # nanoparticle.generate()

        # # Initialize a simulation replayer and run analysis
        # simulation_replayer = SimulationReplayer(
        #     files['initial_state_db_path'], files['npmc_input'])
        # # data contains a tuple of (simulation_time, event_statistics, new_x,
        # # new_population_evolution, new_site_evolution)
        # data = simulation_replayer.run(population_record_interval)

        # # Check that desired number of simulations was run
        # if len(data[0].keys()) != npmc_args['num_sims']:
        #     raise RuntimeError(
        #         f'Run did not successfully complete.'
        #         f' Expected {npmc_args["num_sims"]} trajectories, '
        #         f'found {len(data[0].keys())}.')

        # # get population by shell

        # # Generate documents
        # result_docs = simulation_replayer.generate_docs(data)
        # for i, _ in enumerate(result_docs):
        #     result_docs[i]['initial_state_db_args'] = _initial_state_db_args
        #     result_docs[i]['metadata'] = metadata

        #     # Add metadata to trajectory doc
        #     result_docs[i]['trajectory_doc']['metadata'] = metadata

        # return result_docs

