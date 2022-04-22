import json
import os
from typing import Sequence, Tuple, Optional, Union, List

from jobflow import job
from monty.json import MontyEncoder

from NanoParticleTools.analysis import SimulationReplayer
from NanoParticleTools.core import NPMCInput, NPMCRunner
from NanoParticleTools.inputs.nanoparticle import DopedNanoparticle, NanoParticleConstraint
from NanoParticleTools.inputs.spectral_kinetics import SpectralKinetics
from NanoParticleTools.species_data.species import Dopant
from pymatgen.core import Composition
from collections import Counter
import numpy as np


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
    :return: List of trajectory documents
    """

    # Generate Nanoparticle
    nanoparticle = DopedNanoparticle(constraints, dopant_specifications, doping_seed)
    nanoparticle.generate()

    # Initialize Spectral Kinetics class to calculate transition rates
    dopants = [Dopant(key, concentration) for key, concentration in nanoparticle.dopant_concentrations.items()]
    spectral_kinetics = SpectralKinetics(dopants, **spectral_kinetics_args)

    # Create an NPMCInput class
    npmc_input = NPMCInput(spectral_kinetics, nanoparticle, initial_states)

    # Directories of written files
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    files = {'output_dir': output_dir,
             'initial_state_db_path': os.path.join(output_dir, 'initial_state.sqlite'),
             'np_db_path': os.path.join(output_dir, 'np.sqlite'),
             'npmc_input': os.path.join(output_dir, 'npmc_input.json')}

    # Write files
    _initial_state_db_args = {'one_site_interaction_factor': 1,
                              'two_site_interaction_factor': 1,
                              'interaction_radius_bound': 3,
                              'distance_factor_type': 'inverse_cubic'}
    _initial_state_db_args.update(initial_state_db_args)
    npmc_input.generate_initial_state_database(files['initial_state_db_path'], **_initial_state_db_args)
    npmc_input.generate_nano_particle_database(files['np_db_path'])
    with open(files['npmc_input'], 'w') as f:
        json.dump(npmc_input, f, cls=MontyEncoder)

    # Initialize the wrapper class to run NPMC
    npmc_runner = NPMCRunner(np_db_path=files['np_db_path'],
                             initial_state_db_path=files['initial_state_db_path'])

    # Actually run NPMC
    npmc_runner.run(**npmc_args)

    # Initialize a simulation replayer
    simulation_replayer = SimulationReplayer.from_run_directory(files['output_dir'])

    # TODO: figure out why the nanoparticle sites gets cleared.
    nanoparticle.generate() # generate nanoparticle, since it's state is cleared

    # Analysis of trajectories and generate documents
    results = []
    for seed, trajectory in simulation_replayer.trajectories.items():
        dopant_amount = {}
        for dopant in nanoparticle.dopant_sites:
            try:
                dopant_amount[str(dopant.specie)] += 1
            except:
                dopant_amount[str(dopant.specie)] = 1
        c = Composition(dopant_amount)
        nanostructure = '-'.join(["core" if i == 0 else "shell" for i, _ in enumerate(nanoparticle.constraints)])
        nanostructure_size = '-'.join(
            [f"{int(max(c.bounding_box()))}A_core" if i == 0 else f"{int(max(c.bounding_box()))}A_shell" for i, c in
             enumerate(nanoparticle.constraints)])
        formula_by_constraint = get_formula_by_constraint(nanoparticle)
        _d = {'simulation_seed': trajectory.seed,
              'dopant_seed': nanoparticle.seed,
              'simulation_length': len(trajectory.trajectory),
              'simulation_time': trajectory.simulation_time,
              'n_constraints': len(nanoparticle.constraints),
              'n_dopant_sites': len(nanoparticle.dopant_sites),
              'n_dopants': len(spectral_kinetics.dopants),
              'formula': c.formula.replace(" ", ""),
              'nanostructure': nanostructure,
              'nanostructure_size': nanostructure_size,
              'total_n_levels': spectral_kinetics.total_n_levels,
              'formula_by_constraint': formula_by_constraint,
              'dopants': [str(dopant.symbol) for dopant in spectral_kinetics.dopants],
              'dopant_concentration': nanoparticle._dopant_concentration,
              'overall_dopant_concentration': nanoparticle.dopant_concentrations,
              'excitation_power': spectral_kinetics.excitation_power,
              'excitation_wavelength': spectral_kinetics.excitation_wavelength,
              'dopant_composition': dopant_amount,
              }

        # Add the input parameters to the trajectory document
        _input_d = {'constraints': nanoparticle.constraints,
                    'dopant_specifications': nanoparticle.dopant_specification,
                    'n_levels': [dopant.n_levels for dopant in spectral_kinetics.dopants],
                    'initial_state_db_args': _initial_state_db_args
                    }
        _d['input'] = _input_d

        # Add output to the trajectory document
        _output_d = {}
        summary = trajectory.get_summary()
        keys = ['interaction_id', 'number_of_sites', 'species_id_1', 'species_id_2', 'left_state_1', 'left_state_2',
                'right_state_1', 'right_state_2', 'interaction_type', 'rate_coefficient', 'dNdT', 'dNdT per atom',
                'occurences', 'occurences per atom']
        _output_d['summary_keys'] = keys
        _output_d['summary'] = [[interaction[key] for key in keys] for interaction in summary]

        x, overall_population_evolution, population_by_constraint, site_evolution = trajectory.get_state_evolution()
        _output_d['x_populations'] = x
        _output_d['y_overall_populations'] = overall_population_evolution
        _output_d['y_constraint_populations'] = population_by_constraint
        _output_d['final_states'] = site_evolution[-1, :]

        _d['output'] = _output_d

        # Use the "trajectory_doc" key to ensure that each gets saved as a separate trajectory
        results.append({'trajectory_doc': _d})
    return results


def get_formula_by_constraint(nanoparticle):
    sites = np.array([site.coords for site in nanoparticle.dopant_sites])
    species_names = np.array([str(site.specie) for site in nanoparticle.dopant_sites])

    # Get the indices for dopants within each constraint
    site_indices_by_constraint = []
    for constraint in nanoparticle.constraints:

        indices_inside_constraint = set(np.where(constraint.sites_in_bounds(sites))[0])

        for l in site_indices_by_constraint:
            indices_inside_constraint = indices_inside_constraint - set(l)
        site_indices_by_constraint.append(sorted(list(indices_inside_constraint)))

    formula_by_constraint = []
    for indices in site_indices_by_constraint:
        formula = Composition(Counter(species_names[indices])).formula.replace(" ", "")
        formula_by_constraint.append(formula)
    return formula_by_constraint