from jobflow.managers.fireworks import flow_to_workflow
from NanoParticleTools.flows.flows import get_npmc_flow
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint
from fireworks import LaunchPad
from maggma.stores import MongoStore
from jobflow import JobStore
import shutil
import os
from jobflow import run_locally
from maggma.stores import MemoryStore
from jobflow import JobStore
import time
from multiprocessing import Pool
from bson import uuid

def run_single_npmc(data):
    constraints = data[0]
    dopant_specifications = data[1]
    npmc_args = data[2]
    spectral_kinetics_args = data[3]

    dir_name = f'./scratch_{str(uuid.uuid4())}'
    flow = get_npmc_flow(constraints = constraints,
                         dopant_specifications = dopant_specifications,
                         doping_seed = 0,
                         spectral_kinetics_args = spectral_kinetics_args,
                         npmc_args = npmc_args,
                         output_dir = dir_name)

    # Store the output data locally in a memorystore
    docs_store = MemoryStore()
    data_store = MemoryStore()
    store = JobStore(docs_store, additional_stores={'trajectories': data_store})

    tic = time.time()
    responses = run_locally(flow, store=store, ensure_success=True)
    runtime = time.time()-tic
    print(f'FINISHED RUNNING: nsims = {npmc_args["num_sims"]} sim_length={npmc_args["simulation_length"]} in {runtime} seconds')
    return data, runtime




def run_multiple_npmc(constraints, dopant_specifications, spectral_kinetics_args, npmc_command, max_threads, num_workers=20):
    params = []
    for num_sims in [1, 4, 16, 64, 256]:
        for simulation_length in [1000, 10000, 50000, 100000]:
            npmc_args = {'npmc_command': npmc_command,
                         'num_sims': num_sims,
                         'base_seed': 1000,
                         'thread_count': min(num_sims, max_threads),
                         'simulation_length': simulation_length,
                         }
            params.append((constraints, dopant_specifications, npmc_args, spectral_kinetics_args))

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
