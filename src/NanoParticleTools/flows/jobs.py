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


@job(trajectories='trajectory_doc')
def run_analysis(files):
    # Initialize a simulation replayer
    simulation_replayer = SimulationReplayer.from_run_directory(files['output_dir'])

    # Re-generate nanoparticle
    nanoparticle = simulation_replayer.npmc_input.nanoparticle
    nanoparticle.generate()

    # shortened reference to spectral kinetics
    spectral_kinetics = simulation_replayer.npmc_input.spectral_kinetics

    results = []
    for seed, trajectory in simulation_replayer.trajectories.items():
        _d = {'simulation_seed': trajectory.seed,
              'simulation_length': len(trajectory.trajectory),
              'n_dopant_sites': len(nanoparticle.dopant_sites),
              'n_dopants': len(spectral_kinetics.dopants),
              'total_n_levels': spectral_kinetics.total_n_levels,
              'dopant_concentration': nanoparticle._dopant_concentration,
              'overall_dopant_concentration': nanoparticle.dopant_concentrations,
              'dopants': [str(dopant.symbol) for dopant in spectral_kinetics.dopants],
              }

        dopant_amount = {}
        for dopant in nanoparticle.dopant_sites:
            try:
                dopant_amount[str(dopant.specie)] += 1
            except:
                dopant_amount[str(dopant.specie)] = 1
        _d['dopant_composition'] = dopant_amount

        _input_d = {'constraints': nanoparticle.constraints,
                    'dopant_seed': nanoparticle.seed,
                    'dopant_specifications': nanoparticle.dopant_specification,
                    'excitation_power': spectral_kinetics.excitation_power,
                    'excitation_wavelength': spectral_kinetics.excitation_wavelength,
                    'n_levels': [dopant.n_levels for dopant in spectral_kinetics.dopants],
                    }
        _d['input'] = _input_d
        _output_d = {'simulation_time': trajectory.simulation_time,
                     'summary': trajectory.get_summary()
                     }
        _output_d['x_populations'], _output_d['y_populations'] = trajectory.get_population_evolution()
        _d['output'] = _output_d

        results.append({'trajectory_doc': _d})
    return results