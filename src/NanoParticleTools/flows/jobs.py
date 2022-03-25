import os
import pickle
from typing import Sequence, Tuple, Optional

from NanoParticleTools.core import NPMCInput, NPMCRunner
from NanoParticleTools.inputs.nanoparticle import DopedNanoparticle, NanoParticleConstraint
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.inputs.util import get_all_interactions, get_sites, get_species
from NanoParticleTools.species_data.species import Dopant
from jobflow import job

from NanoParticleTools.simulation_loader import SimulationReplayer


@job
def write_inputs(constraints: Sequence[NanoParticleConstraint],
                 dopant_specifications: Sequence[Tuple[int, float, str, str]],
                 seed: int,
                 output_dir: Optional[str] = '.') -> dict:
    # Generate Nanoparticle
    nanoparticle = DopedNanoparticle.from_constraints(constraints, seed)
    for dopant_specification in dopant_specifications:
        nanoparticle.add_dopant(*dopant_specification)

    # Initialize Spectral Kinetics class to calculate transition rates
    dopants = [Dopant(key, concentration) for key, concentration in nanoparticle.dopant_concentrations.items()]
    sk = SpectralKinetics(dopants)
    sk.set_kinetic_parameters()

    # Gather and format data for NPMCInput
    interactions = get_all_interactions(sk)
    sites = get_sites(nanoparticle, sk)
    species = get_species(sk)

    # Create an NPMCInput class
    npmc_input = NPMCInput(interactions, sites, species)

    # Directories of written files
    files = {'initial_state_db_path': os.path.join(output_dir, 'initial_state.sqlite'),
             'np_db_path': os.path.join(output_dir, 'np.sqlite'),
             'npmc_input': os.path.join(output_dir, 'npmc_input.pickle')}

    # Write files
    npmc_input.generate_initial_state_database(files['initial_state_db_path'])
    npmc_input.generate_nano_particle_database(files['np_db_path'])
    with open(files['npmc_input'], 'wb') as f:
        pickle.dump(npmc_input, f)

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
    # Load the npmc class from the pickle
    with open(files['npmc_input'], 'rb') as f:
        npmc_input = pickle.load(f)

    # Load the NPMC trajectories
    npmc_input.load_trajectories(files['initial_state_db_path'])

    # Initialize a simulation replayer
    simulation_replayer = SimulationReplayer(npmc_input)
    return simulation_replayer.get_summary_dict()
