NanoParticleTools tools is a python module that facilitates monte carlo simulation of Upconverting Nanoparticles (UCNP) using [RNMC](https://github.com/BlauGroup/RNMC).

# Using NanoParticleTools
NanoParticleTools provides functionality to generate inputs for running Monte Carlo Simulations on nanoparticles and analyzing outputs. Monte Carlo simulation uses NMPC within the [RNMC](https://github.com/BlauGroup/RNMC) package. While NanoParticleTools provides wrapper functions to run the C++ based simulator, [RNMC](https://github.com/BlauGroup/RNMC) must be installed to perform simulations.

## Installation
To install NanoParticleTools to a python environment, clone the repository and use one of the following commands from within the NanoParticleTools directory
```bash
python setup.py develop
```
or 
```bash
pip install .
```

### NixOS
A NixOS environment is also provided for an alternative setup method. This environment includes access to a compiled RNMC executable. To access the Nix development shell
```
nix develop
```
*Note: To use the NixOS environment, you must have root access on the system you are running on (i.e. This is usually not the case on supercomputers).*

## Running Simulations
```python
from NanoParticleTools.flows.flows import get_npmc_flow
from NanoParticleTools.inputs.nanoparticle import SphericalConstraint

constraints = [SphericalConstraint(20)]
dopant_specifications = [(0, 0.1, 'Yb', 'Y'),
                         (0, 0.02, 'Er', 'Y')]

npmc_args = {'npmc_command': <NPMC_command>,
             'num_sims':2,
             'base_seed': 1000,
             'thread_count': 8,
             'simulation_length': 1000,
             }
spectral_kinetics_args = {'excitation_power': 1e12,
                          'excitation_wavelength':980}

flow = get_npmc_flow(constraints = constraints,
                     dopant_specifications = dopant_specifications,
                     doping_seed = 0,
                     spectral_kinetics_args = spectral_kinetics_args,
                     npmc_args = npmc_args,
                     output_dir = './scratch')
```

```python
from jobflow import run_locally
from maggma.stores import MemoryStore
from jobflow import JobStore

# Store the output data locally in a MemoryStore
docs_store = MemoryStore()
data_store = MemoryStore()
store = JobStore(docs_store, additional_stores={'trajectories': data_store})

responses = run_locally(flow, store=store, ensure_success=True)
```



# Contributing 
If you wish to make changes to NanoParticle tools, it may be wise to install the package in development mode. After cloning the package, use the following command.
```bash
python -m pip install -e .
```
Modifications should now be reflected when you run any functions in NanoParticleTools.

Further guidance on contributing via Pull Requests will be added in the near future.
