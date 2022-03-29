import os
import json
from monty.json import MontyEncoder, MontyDecoder
from typing import Sequence, Tuple, Optional, Union

from NanoParticleTools.core import NPMCInput, NPMCRunner
from NanoParticleTools.inputs.nanoparticle import DopedNanoparticle, NanoParticleConstraint
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.inputs.util import get_all_interactions, get_sites, get_species
from NanoParticleTools.species_data.species import Dopant
from jobflow import job

from NanoParticleTools.analysis import SimulationReplayer


@job
def write_inputs(constraints: Sequence[NanoParticleConstraint],
                 dopant_specifications: Sequence[Tuple[int, float, str, str]],
                 seed: int,
                 output_dir: Optional[str] = '.',
                 initial_states: Optional[Union[Sequence[int], None]] = None,
                 **kwargs) -> dict:
    # Generate Nanoparticle
    nanoparticle = DopedNanoparticle(constraints, dopant_specifications, seed)
    nanoparticle.generate()

    # Initialize Spectral Kinetics class to calculate transition rates
    dopants = [Dopant(key, concentration) for key, concentration in nanoparticle.dopant_concentrations.items()]
    sk = SpectralKinetics(dopants, **kwargs)

    # Create an NPMCInput class
    npmc_input = NPMCInput(sk, nanoparticle, initial_states)

    # Directories of written files
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    files = {'output_dir': output_dir,
             'initial_state_db_path': os.path.join(output_dir, 'initial_state.sqlite'),
             'np_db_path': os.path.join(output_dir, 'np.sqlite'),
             'npmc_input': os.path.join(output_dir, 'npmc_input.json')}

    # Write files
    npmc_input.generate_initial_state_database(files['initial_state_db_path'])
    npmc_input.generate_nano_particle_database(files['np_db_path'])
    with open(files['npmc_input'], 'w') as f:
        json.dump(npmc_input, f, cls=MontyEncoder)

    return files


@job
def run_npmc(files,
             **kwargs):
    # Initialize the wrapper class to run NPMC
    npmc_runner = NPMCRunner(np_db_path=files['np_db_path'],
                             initial_state_db_path=files['initial_state_db_path'])

    # Actually run NPMC
    npmc_runner.run(**kwargs)

    return files


@job
def run_analysis(files):
    # Initialize a simulation replayer
    simulation_replayer = SimulationReplayer.from_run_directory(files['output_dir'])

    _d = {'summary': simulation_replayer.get_summaries()}
    return
