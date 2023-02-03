from NanoParticleTools.flows.flows import get_npmc_flow
from jobflow import JobStore
from jobflow import run_locally
from maggma.stores import MemoryStore
import time
from multiprocessing import Pool
from bson import uuid
from typing import Tuple, List, Dict
from NanoParticleTools.inputs.nanoparticle import NanoParticleConstraint


def run_single_npmc(data: Tuple) -> Tuple[Tuple, float]:
    """
    This function runs a single NPMC simulation and returns its runtime.
    This function is intended to be used for testing of runtime when running
    parallel simulations (not for production).

    Args:
        data (Tuple): Tuple containing the following:
            constraints (List[NanoParticleConstraint]): Constraints for the
                NPMC simulation
            dopant_specifications (List[Tuple]): Specifications on how dopants
                should be applied to the lattice
            npmc_args (Dict): Arguments to pass to the NPMC simulation
            spectral_kinetics_args (Dict): Arguments to pass to the spectral
                kinetics module

    Returns:
        Tuple[Tuple, float]: Tuple containing the following:
            Tuple: The original input tuple
            float: The runtime of the simulation
    """
    constraints = data[0]
    dopant_specifications = data[1]
    npmc_args = data[2]
    spectral_kinetics_args = data[3]

    dir_name = f'./scratch_{str(uuid.uuid4())}'
    flow = get_npmc_flow(constraints=constraints,
                         dopant_specifications=dopant_specifications,
                         doping_seed=0,
                         spectral_kinetics_args=spectral_kinetics_args,
                         npmc_args=npmc_args,
                         output_dir=dir_name)

    # Store the output data locally in a memorystore
    docs_store = MemoryStore()
    data_store = MemoryStore()
    store = JobStore(docs_store,
                     additional_stores={'trajectories': data_store})

    tic = time.time()
    _ = run_locally(flow, store=store, ensure_success=True)
    runtime = time.time() - tic
    print(
        f'FINISHED RUNNING: nsims = {npmc_args["num_sims"]}'
        f' sim_length={npmc_args["simulation_length"]} in {runtime} seconds'
    )
    return data, runtime


def run_multiple_npmc(constraints: List[NanoParticleConstraint],
                      dopant_specifications: List[Tuple],
                      spectral_kinetics_args: Dict,
                      npmc_command: str,
                      max_threads: int,
                      num_workers: int = 20) -> List[List]:
    """
    Runs a series of NPMC simulations over a matrix of # of parallel
    simulations and length of simulation. This function is used to benchmark
    the performance of the NPMC code on a system.

    Args:
        constraints (List[NanoParticleConstraint]): The constraints that
            define the nanoparticle.
        dopant_specifications (List[Tuple]): Specifications on how dopants
            should be applied to the lattice
        spectral_kinetics_args (Dict): The arguments to pass to the spectral
            kinetics module
        npmc_command (str): The command to run the NPMC executable
        max_threads (int): The maximum number of threads available on the
            system
        num_workers (int, optional): The number of simulations to run in
            parallel. This should be set to the number of cores available on
            the system.
            Defaults to 20.

    Returns:
        List[List]: The simulation run statistics.
    """
    params = []
    for num_sims in [1, 4, 16, 64, 256]:
        for simulation_length in [1000, 10000, 50000, 100000]:
            npmc_args = {
                'npmc_command': npmc_command,
                'num_sims': num_sims,
                'base_seed': 1000,
                'thread_count': min(num_sims, max_threads),
                'simulation_length': simulation_length,
            }
            params.append((constraints, dopant_specifications, npmc_args,
                           spectral_kinetics_args))

    if num_workers == 'slurm':
        import os
        num_workers = int(os.environ['SLURM_JOB_NUM_NODES'])
    p = Pool(num_workers)
    results = p.map(run_single_npmc, params)

    p.close()
    p.join()

    output = []
    for result in results:
        num_sims = result[0][2]['num_sims']
        simulation_length = result[0][2]['simulation_length']
        output.append([num_sims, simulation_length, result[1]])
    return output
